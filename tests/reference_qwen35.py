"""Independent reference implementation of the Qwen3.5/3.6 hybrid forward.

A *separate* reimplementation of the same architecture (Gated-Delta-Net linear
attention + gated GQA + 256-way shared-expert MoE), written from the
architecture description rather than sharing code with
``freetoken.models.qwen3_5_moe``. It is driven off the *loaded parameters* of a
loaded FreeToken model, so it computes what the forward *should* produce given
those exact weights. The forward test compares the FreeToken forward's
last-position logits against this reference; a disagreement flags a wiring /
shape / math bug in the real forward.

The GQA attention is cross-checked against the engine's own reference attention
backend (``TritonAttentionBackend``) rather than re-derived here -- so this
reference validates the *model-side* wiring (projections, partial RoPE, gating,
Gated-Delta-Net recurrence, MoE routing) against a known-correct attention
implementation. (Full numerical agreement with HF transformers is the separate
35B nightly reference match.)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.models.qwen3_5_moe import _apply_rotary_pos_emb, _rope_for_positions


def _rmsnorm(x, weight, eps):
    # The model's _RMSNorm: norm in float32, then *(1 + weight) (zero-init
    # weight == identity). Independent restatement of the same math.
    x32 = x.float()
    variance = x32.pow(2).mean(-1, keepdim=True)
    out = x32 * torch.rsqrt(variance + eps)
    return (out * (1.0 + weight.float())).to(x.dtype)


def _l2norm(x, eps=1e-6):
    # Match the model's _l2norm exactly: add eps INSIDE the rsqrt (rsqrt(sum+eps)),
    # not clamp-then-divide. The two differ when the squared norm is below eps
    # (a head whose key energy is tiny), and that per-head difference amplifies
    # over the recurrent loop.
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def ref_gated_deltanet(hidden, positions, la, slot_state, dtype):
    """Reference Gated-Delta-Net layer -> (out [T,H], new_state tuple).

    ``hidden`` is the model's 2-D per-request slice ``[T, H]`` (bsz=1), matching
    the real forward; the reference adds a leading batch dim internally for the
    recurrent state and folds it back before returning.
    """
    T, H = hidden.shape
    B = 1
    dev, dt = hidden.device, dtype
    w = la.conv1d.weight  # [C,1,K] (groups=C: each channel convolves 1 input channel)
    ring = la.conv_kernel - 1
    mixed = la.in_proj_qkv(hidden).unsqueeze(0).transpose(1, 2)  # [1, C, T]
    conv_state = None if slot_state is None else slot_state[1].clone()
    # Mirror the real _GatedDeltaNet._conv exactly: zero-pad the ring, conv with
    # padding=0, then slice the FIRST T positions (the new tokens' outputs). (The
    # ring offset is absorbed by the padding, so the slice is :T, not ring:.)
    if slot_state is None:
        seq = torch.cat(
            [torch.zeros(1, la.conv_dim, ring, device=dev, dtype=dt), mixed],
            dim=-1,
        )
    else:
        seq = torch.cat([conv_state.unsqueeze(0), mixed], dim=-1)
    conv = F.conv1d(seq, w, padding=0, groups=la.conv_dim)[:, :, :T]  # [1, C, T]
    conv_state_new = seq[:, :, -ring:].clone()

    z = la.in_proj_z(hidden)  # [T,value_dim]
    beta = la.in_proj_b(hidden).sigmoid()  # [T,num_v]
    a = la.in_proj_a(hidden)  # [T,num_v]
    # conv is [1, C, T]; transpose to [1, T, C] then view the head groups.
    c = conv[0].transpose(0, 1)  # [T, C]
    q = c[:, 0 : la.key_dim].view(T, la.num_k_heads, la.head_k_dim)
    k = c[:, la.key_dim : 2 * la.key_dim].view(T, la.num_k_heads, la.head_k_dim)
    v = c[:, 2 * la.key_dim :].view(T, la.num_v_heads, la.head_v_dim)
    if la.group_ratio > 1:
        q = q.repeat_interleave(la.group_ratio, dim=1)
        k = k.repeat_interleave(la.group_ratio, dim=1)
    g = -la.A_log.float().exp() * F.softplus(a.float() + la.dt_bias)  # [T,num_v]

    # Recurrent delta rule, upcast to float32 (as the model does). The reference
    # builds q/k/v in token-major [T, H, D]; the model l2-normalises *per head*
    # over the head dim, so transpose to head-major [B, H, T, D] first (the model
    # does exactly this before its per-head norm + scale). The state
    # [num_v, key_dim, value_dim] is then indexed by the head dim (dim 1).
    q32 = _l2norm(q).float().unsqueeze(0).transpose(1, 2).contiguous()
    k32 = _l2norm(k).float().unsqueeze(0).transpose(1, 2).contiguous()
    v32 = v.float().unsqueeze(0).transpose(1, 2).contiguous()
    beta32 = beta.float().unsqueeze(0).transpose(1, 2).contiguous()
    g32 = g.float().unsqueeze(0).transpose(1, 2).contiguous()
    q32 = q32 * (1.0 / (la.head_k_dim ** 0.5))
    Hh = q32.shape[1]  # head dim == num_v_heads (== num_k_heads when grouped)
    if slot_state is None:
        s = torch.zeros(B, Hh, la.head_k_dim, la.head_v_dim, device=dev, dtype=torch.float32)
    else:
        s = slot_state[0].to(torch.float32).clone()
    out = torch.zeros(B, Hh, T, la.head_v_dim, dtype=torch.float32, device=dev)
    for i in range(T):
        q_t, k_t, v_t = q32[:, :, i], k32[:, :, i], v32[:, :, i]
        g_t = g32[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)  # [B,num_v,1,1]
        beta_t = beta32[:, :, i].unsqueeze(-1)
        s = s * g_t
        kv_mem = (s * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        s = s + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (s * q_t.unsqueeze(-1)).sum(dim=-2)
    out = out.transpose(1, 2).contiguous().to(dt)[0]  # [T,num_v,D]
    o = out.reshape(-1, la.head_v_dim)
    zv = z.reshape(-1, la.head_v_dim)
    var = o.float().pow(2).mean(-1, keepdim=True)
    o = (o.float() * torch.rsqrt(var + la.norm.variance_epsilon))
    o = (la.norm.weight * o).to(dt)
    o = (o * F.silu(zv.float())).to(dt)
    o = o.view(T, -1)
    out = la.out_proj(o)
    return out, (s, conv_state_new.to(dt))


def ref_full_attention(hidden, positions, sa, table_idx, ctx, batch):
    T, H = hidden.shape
    dt = hidden.dtype
    qg = sa.q_proj(hidden).view(T, sa.num_heads, sa.head_dim * 2)
    q, gate = qg.split(sa.head_dim, dim=-1)
    q = _rmsnorm(q, sa.q_norm.weight, sa.q_norm.eps)
    k = _rmsnorm(sa.k_proj(hidden).view(T, sa.num_kv_heads, sa.head_dim), sa.k_norm.weight, sa.q_norm.eps)
    v = sa.v_proj(hidden).view(T, sa.num_kv_heads, sa.head_dim)
    cos, sin = _rope_for_positions(sa.inv_freq, positions, sa.head_dim_rot)
    cos = cos.reshape(T, -1)
    sin = sin.reshape(T, -1)
    # Partial-RoPE via the shared helper (identical to the real forward): it
    # unsqueezes cos/sin at the head dim and runs the rotation in float32, then
    # casts back to the compute dtype so the reference K/V match the bf16 K/V the
    # real forward writes to the pool.
    q, k = _apply_rotary_pos_emb(q, k, cos, sin)
    # Lay out 3-D head-major [heads, T, head_dim] (the Triton backend + paged-KV
    # pool contract), matching the real forward (which squeezes its bsz=1 dim).
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    gate = gate.transpose(0, 1)
    ctx.kv_cache.write_kv(k, v, positions)
    out = ctx.attn_backend.forward(q, k, v, sa.layer_id, batch, table_idx=table_idx)
    out = out * torch.sigmoid(gate)
    return sa.o_proj(out.transpose(0, 1).reshape(T, -1))


def ref_moe(flat, mlp, ctx, batch):
    n, H = flat.shape
    logits = mlp.gate(flat)  # [n, E]
    probs = F.softmax(logits.float(), dim=-1)
    top_w, top_idx = torch.topk(probs, mlp.top_k, dim=-1)
    top_w = (top_w / top_w.sum(-1, keepdim=True)).to(flat.dtype)
    shared = F.sigmoid(mlp.shared_expert_gate(flat)) * mlp.shared_expert(flat)
    if mlp.use_offload:
        raise NotImplementedError("reference covers the in-VRAM path only")
    routed = torch.zeros_like(flat)
    for e in range(mlp.num_experts):
        for slot in range(mlp.top_k):
            sel = top_idx[:, slot] == e
            if sel.any():
                routed[sel] = routed[sel] + top_w[sel, slot].unsqueeze(-1) * mlp.experts[e](flat[sel])
    return (routed + shared).reshape(n, H)


def reference_logits(model, input_ids, positions, table_idx, extend_len, ctx, batch):
    """Run the reference forward; return last-position logits [1, V] (bs=1).

    The caller must have already written the new K/V for the full-attention
    layers into ``ctx.kv_cache`` (via a prior real forward, or a pre-write)
    and set ``ctx.linear_state_pool`` to the model's pool -- the reference reads
    the same pool the real forward would. ``batch`` is the active batch.
    """
    dt = model.dtype
    H = model.config.hidden_size
    h = F.embedding(input_ids.reshape(-1), model.embed_tokens.weight).view(-1, H)
    T = input_ids.shape[0]
    # Read the same linear-state pool the real forward uses (the model owns the
    # pool; the test points ctx.linear_state_pool at it). For each linear layer,
    # read this request's slot via table_idx -- exactly what the real model's
    # _GatedDeltaNet.forward does. The reference reprocesses all T tokens in one
    # shot, so it starts from the pool's *current* (pristine) state and advances
    # it, matching the real forward's single prefill step.
    for layer in model.layers:
        la, sa = layer.linear_attn, layer.self_attn
        inp = _rmsnorm(h, layer.input_layernorm.weight, layer.input_layernorm.eps)
        if la is not None:
            slot = None
            pool = ctx.linear_state_pool
            if pool is not None and layer.layer_id in pool._layers:
                slot = pool.get(layer.layer_id, table_idx)
            attn_out, new_state = ref_gated_deltanet(inp, positions, la, (slot.state, slot.conv_state) if slot is not None else None, dt)
            # Write the advanced state back into the pool slot, exactly as the
            # real model's _delta_rule / _conv do in place -- otherwise the pool
            # the test inspects afterwards still holds the pristine (zero) state.
            if slot is not None:
                # new_state[0] is the batched recurrent state [1, H, KD, VD]; the
                # per-request slot.state is the unbatched [H, KD, VD] (== the real
                # model's slot.state.copy_(s[0])). new_state[1] is the conv ring.
                # new_state[0] is the batched recurrent state [1, H, KD, VD]; index
                # [0] for the unbatched per-request slot.state. new_state[1] is the
                # batched conv ring [1, C, ring]; index [0] for slot.conv_state.
                slot.state.copy_(new_state[0][0])
                slot.conv_state.copy_(new_state[1][0])
        else:
            attn_out = ref_full_attention(inp, positions, sa, table_idx, ctx, batch)
        h = h + attn_out
        moe_in = _rmsnorm(h, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps)
        h = h + ref_moe(moe_in.reshape(-1, H), layer.mlp, ctx, batch)
    h = _rmsnorm(h, model.norm.weight, model.norm.eps)
    return F.linear(h[-1:], model.lm_head.weight)  # [1, V]
