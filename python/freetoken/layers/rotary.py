"""Pure-torch rotary position embeddings (RoPE).

Upstream NVIDIA path: python/freetoken/layers/rotary.py. Upstream applies RoPE
through a CUDA/Triton kernel (``apply_rope_with_cos_sin_cache_inplace``, resolved
from ``flashinfer`` or ``freetoken.kernel.triton.rope``); that kernel cannot be
built on the Intel XPU (no triton-xpu on the oneAPI 2025.2 torch). The port
reproduces the upstream public API exactly (``RotaryEmbedding`` + ``get_rope`` +
``set_rope_device``) but performs the rotation with the equivalent pure-torch ops
instead of the Triton kernel -- the same NeoX half-split (or GPT-J interleaved)
pairing, in float32, applied in place to the first ``rotary_dim`` dims of each
head with the rest passed through.

Lazy ``import torch`` inside methods keeps the package import torch-free for the
CPU venv; the cos/sin cache is built inside the ``_ROPE_DEVICE`` context (set via
:func:`set_rope_device`) when the torch default device is ``meta``, exactly as
upstream does, so lazy meta construction works.

State: the cos/sin cache is a derived artefact (a pure function of the
constructor args), never a checkpoint weight, so it is stored under an
underscore-prefixed name and the ``BaseOP`` state-dict machinery skips it. The
``inv_freq`` is a plain ``BaseOP`` attribute (a non-``_`` Tensor):
``StateLessOP.state_dict`` still returns ``{}`` for it (RoPE is stateless w.r.t.
checkpoint weights) and ``StateLessOP.load_state_dict`` is a no-op for an empty
dict, so the derived table is simply never (re)loaded.
"""
from __future__ import annotations

import functools
import math
from contextlib import nullcontext
from typing import Any, Callable, Dict, Optional, Tuple

from freetoken.layers.base import StateLessOP

_ROPE_DEVICE = None

_VALID_HEAD_SIZES = (64, 128, 256, 512)


def set_rope_device(device: object) -> None:
    """Set the device the ``get_rope`` cache builds its cos/sin cache on.

    Upstream reads the torch default device at cache-build time; when the
    process was started on a meta device (``torch.set_default_device("meta")``
    for lazy meta init) the cache would land on meta and the later real forward
    would find no storage. Setting a concrete device here forces the cache to
    that device. Mirrors upstream ``set_rope_device``.
    """
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


def _llama3_post_process(inv_freq: "object", config: Dict[str, Any]) -> "object":
    """LLaMA-3 low/high-frequency correction of the inverse frequencies (upstream).

    Dims whose raw frequency falls outside the ``[low, high]`` band (derived from
    ``low_freq_factor``/``high_freq_factor`` and the original context length) are
    left at the base frequency; the rest are divided by ``factor`` (equivalently,
    the raw frequency is multiplied by ``factor``).
    """
    import torch

    old_ctx = config["original_max_position_embeddings"]
    factor = config.get("factor") or config.get("scale_factor") or 1.0
    low = config.get("low_freq_factor", 1)
    high = config.get("high_freq_factor", 4)
    if factor <= 1.0 or low >= high:
        return inv_freq
    f = inv_freq.to(torch.float32)
    orig = 1.0 / f
    low_mask = orig > (old_ctx / low)
    high_mask = orig < (old_ctx / high)
    new = f * factor
    # low-frequency and high-frequency dims keep the base frequency; the rest are
    # scaled by ``factor``.
    new = torch.where(high_mask, f, new)
    new = torch.where(low_mask, f, new)
    return new


def _yarn_post_process(inv_freq: "object", config: Dict[str, Any]) -> "object":
    """YaRN low/high-frequency ramp correction of the inverse frequencies.

    Dims between the YaRN ``low`` and ``high`` correction boundaries are blended
    from the base frequency toward the base/factor frequency by a smooth ramp, so
    low-frequency dims keep the original period and high-frequency dims are
    scaled by ``factor``.
    """
    import torch

    orig_ctx = config["original_max_position_embeddings"]
    factor = config.get("factor") or config.get("scale_factor") or 1.0
    base = config.get("base", 10000.0)
    beta_fast = config.get("beta_fast", 32.0)
    beta_slow = config.get("beta_slow", 1.0)
    if factor <= 1.0 or beta_fast == beta_slow:
        return inv_freq
    f = inv_freq.to(torch.float32).reshape(-1)
    n = f.shape[0]
    dim = n * 2

    def correction_dim(x: float) -> float:
        return dim * math.log(orig_ctx / x) / math.log(base)

    low = max(correction_dim(beta_fast), 0.0)
    high = min(correction_dim(beta_slow), dim - 1.0)
    if low <= high:
        # index d in [0, n); map onto the dim space via (d * dim / n) == 2d.
        idx = torch.arange(n, dtype=torch.float32) * (dim / n)
        smooth = (high - idx) / (high - low)
        f = f * (1.0 + smooth * (factor - 1.0))
    return f.reshape(inv_freq.shape)


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
    is_neox: bool = True,
) -> "RotaryEmbedding":
    """Dispatch on ``rope_scaling['rope_type']`` to the matching construction path.

    ``None``/``"default"`` -> plain; ``"proportional"`` -> proportional inv_freq;
    ``"llama3"`` -> llama3 post-process; ``"yarn"`` -> yarn post-process + an
    attention factor. Anything else raises (upstream behaviour).
    """
    post_process: Optional[Callable[..., object]] = None
    proportional = False
    attention_factor = 1.0
    if rope_scaling is not None:
        scaling_type = rope_scaling.get("rope_type")
        if scaling_type in (None, "default"):
            pass
        elif scaling_type == "proportional":
            proportional = True
        elif scaling_type == "llama3":
            post_process = lambda f: _llama3_post_process(f, rope_scaling)  # noqa: E731
        elif scaling_type == "yarn":
            post_process = lambda f: _yarn_post_process(f, rope_scaling)  # noqa: E731
            mscale = rope_scaling.get("mscale")
            beta_fast = rope_scaling.get("beta_fast", 32.0)
            beta_slow = rope_scaling.get("beta_slow", 1.0)
            if mscale is None:
                if beta_fast == beta_slow:
                    mscale = math.sqrt(
                        1.0 + math.log(beta_fast / 32.0) * (beta_fast / (beta_fast - beta_slow))
                    )
                else:
                    mscale = 1.0
            attention_factor = 0.1 * mscale * math.log(rope_scaling.get("factor", 1.0) + 1.0) + 1.0
        else:
            raise ValueError(f"Unknown RoPE type {scaling_type!r}")
    return RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position,
        base=base,
        post_process=post_process,
        proportional=proportional,
        attention_factor=attention_factor,
        is_neox=is_neox,
    )


@functools.lru_cache(maxsize=16)
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
    is_neox: bool = True,
) -> "RotaryEmbedding":
    """Build (or reuse, by args) a :class:`RotaryEmbedding`.

    Cached on its args so that every decoder layer shares one RoPE instance
    (upstream behaviour). ``rope_scaling`` is a tuple of (key, value) pairs so
    it is hashable/cachable; converted to a dict before dispatch. If the torch
    default device is ``meta`` a concrete device must have been registered via
    :func:`set_rope_device` (upstream's ``_ROPE_DEVICE`` guard); otherwise the
    cos/sin cache is built on the default device.
    """
    import torch

    scaling_dict = dict(rope_scaling) if rope_scaling else None
    if _ROPE_DEVICE is None and torch.tensor(1.0).device.type == "meta":
        raise RuntimeError(
            "RoPE is being constructed with a meta default device; call "
            "freetoken.layers.rotary.set_rope_device(device) with a concrete "
            "device first (upstream _ROPE_DEVICE guard)."
        )
    ctx = torch.device(_ROPE_DEVICE) if _ROPE_DEVICE is not None else nullcontext()
    with ctx:
        return _get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position,
            base=base,
            rope_scaling=scaling_dict,
            is_neox=is_neox,
        )


class RotaryEmbedding(StateLessOP):
    """Pure-torch RoPE with upstream's public API.

    ``forward(positions, query, key)`` rotates the first ``rotary_dim`` dims of
    each head of ``query``/``key`` **in place** by the per-position cos/sin
    (NeoX half-split when ``is_neox=True``, GPT-J interleaved when False),
    computed in float32 and cast back to the input dtype, then returns
    ``(query, key)``. The rest of each head (``dims >= rotary_dim``) passes
    through untouched (partial RoPE). Mirrors upstream ``RotaryEmbedding``
    except the rotation is performed with torch ops instead of the
    Triton/flashinfer ``apply_rope_with_cos_sin_cache_inplace`` kernel.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: Optional[Callable[..., object]] = None,
        proportional: bool = False,
        attention_factor: float = 1.0,
        is_neox: bool = True,
    ) -> None:
        assert 0 < rotary_dim <= head_size, "0 < rotary_dim <= head_size"
        assert rotary_dim % 2 == 0, "rotary_dim must be even"
        assert head_size in _VALID_HEAD_SIZES, f"head_size must be one of {_VALID_HEAD_SIZES}"
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.is_neox = is_neox

        import torch

        if proportional:
            # inv_freq spans the full head (head_size/2 entries); dims beyond the
            # rotary_dim are masked to 0.0 so they are not rotated (partial rope).
            inv_freq = 1.0 / (
                base ** (torch.arange(0, head_size, 2, dtype=torch.float32) / head_size)
            )
            if rotary_dim < head_size:
                inv_freq[rotary_dim // 2 :] = 0.0
        else:
            inv_freq = 1.0 / (
                base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
            )
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        self.inv_freq = inv_freq

        freqs = torch.einsum(
            "i,j -> ij",
            torch.arange(max_position_embeddings, dtype=torch.float32),
            inv_freq,
        )
        # [max_pos, rotary_dim]: first half is cos, second half is sin (upstream).
        self._cos_sin_cache = torch.cat(
            (freqs.cos() * attention_factor, freqs.sin() * attention_factor), dim=-1
        )

    def _apply(self, positions: "object", query: "object", key: "object") -> None:
        """Rotate ``query``/``key`` (each ``[*, heads, head_size]``) in place at the
        per-token absolute ``positions`` (``[N]``), gathering cos/sin from the
        cache and applying the NeoX/interleaved pairing over the first
        ``rotary_dim`` dims of every head."""
        import torch

        n = positions.shape[0]
        if n == 0:
            return
        pos = positions.to(torch.int64)
        for x in (query, key):
            # Gather the fp32 cache in x's dtype+device so the gathered cos/sin match x.
            # The cache always stores the per-frequency cos in the first half and sin
            # in the second half (cache[pos, :half] = cos, cache[pos, half:rotary_dim]
            # = sin) under BOTH conventions -- the upstream Triton kernel gathers
            # exactly these contiguous halves and only swaps the x-pairing (d0/d1)
            # between NeoX and interleaved. So _apply is convention-agnostic; the
            # convention lives entirely in _rotate's x-gather.
            cache = self._cos_sin_cache.to(x.device, dtype=x.dtype)
            half = self.rotary_dim // 2
            cos = cache[pos, :half]
            sin = cache[pos, half : self.rotary_dim]
            # cos/sin gather to [N, half] (the per-token rows). x is [*N, heads,
            # head_size]; the rotation operates on the last head axis, so pad the
            # heads broadcast axis (unsqueeze the head dim, then expand to *N,1,half)
            # so cos/sin apply identically to every head.
            cos = cos.unsqueeze(1).expand(*x.shape[:-2], 1, half)
            sin = sin.unsqueeze(1).expand(*x.shape[:-2], 1, half)
            self._rotate(x, cos, sin)

    def _rotate(self, x: "object", cos: "object", sin: "object") -> None:
        import torch

        dtype = x.dtype
        xf = x.to(torch.float32)
        half = self.rotary_dim // 2
        # Mirror the upstream Triton kernel's pairing exactly:
        #   is_neox=True  -> d0 = d,        d1 = half + d  (NeoX half-split)
        #   is_neox=False -> d0 = 2d,       d1 = 2d + 1    (GPT-J / HF interleaved)
        # cos/sin are the contiguous cache halves gathered in _apply: cos[d]
        # multiplies dim d0, sin[d] multiplies dim d1 (see kernel lines 54-64).
        # Both x0/x1 are half-width, so the [*, heads, half] cos/sin broadcast.
        x0 = xf[..., :half] if self.is_neox else xf[..., 0::2]
        x1 = xf[..., half : self.rotary_dim] if self.is_neox else xf[..., 1::2]
        out0 = x0 * cos - x1 * sin
        out1 = x1 * cos + x0 * sin
        # Assemble into a fresh tensor. Both conventions avoid in-place strided
        # write-back: on this oneAPI XPU runtime that pattern (``out[..., 0::2] =
        # out0``) produces mis-interleaved values, whereas ``torch.cat`` /
        # ``reshape``-based assembly is correct (and is what the test's faithful
        # references use).
        if self.is_neox:
            # NeoX: out0 at dims [0, half), out1 at [half, rotary_dim).
            out = torch.cat([out0, out1], dim=-1)
        else:
            # Interleaved: out0 at even dims, out1 at odd dims. cat then reshape
            # to [*, heads, half, 2] and transpose the last two dims so the row is
            # (out0_0, out1_0, out0_1, out1_1, ...) == (even, odd, even, odd).
            out = torch.cat([out0, out1], dim=-1).reshape(*x.shape[:-1], half, 2)
            out = out.transpose(-2, -1).reshape(*x.shape)
        if self.rotary_dim < self.head_size:
            out = torch.cat([out, xf[..., self.rotary_dim :]], dim=-1)
        x.copy_(out.to(dtype))

    def forward(self, positions: "object", query: "object", key: "object") -> Tuple[object, object]:
        self._apply(positions, query, key)
        return query, key


__all__ = ["RotaryEmbedding", "get_rope", "set_rope_device"]
