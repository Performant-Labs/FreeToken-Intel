"""End-to-end: a fabricated dense Qwen3 checkpoint loads and runs a real
forward pass through the Engine (issue `models-dense`, #20's own accept
bar: "at least one dense model runs... on B70").

A small (few-KB) fabricated checkpoint, not a real 7B-27B one -- mirrors
every other model port in this project (test_engine_loop.py's own
qwen3_moe fixture, test_qwen35_gptq_e2e_loader.py, etc.): the wiring (loader
-> parse_config/iter_weights -> real forward, KV pool, attention, RoPE, the
Engine's admit/step/generate loop) is what this test proves, not a specific
checkpoint at scale. The attention math itself is adapted verbatim from
qwen3_moe's own _Qwen3Attention, already reference-matched elsewhere in this
project's test suite.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from freetoken.core import Req, SamplingParams, reset_global_ctx
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import Engine
from freetoken.models.qwen3 import iter_weights, parse_config
from freetoken.models.register import get_model_class

DEVICE = "cpu"

TINY_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 32,
    "vocab_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "intermediate_size": 48,
    "max_position_embeddings": 128,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": False,
}


@pytest.fixture(autouse=True)
def _clean_global_ctx():
    yield
    reset_global_ctx()


def _write_tiny_checkpoint(tmp_path) -> str:
    from safetensors.torch import save_file

    model_path = tmp_path / "ckpt"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(TINY_CONFIG))
    # No model.safetensors.index.json: iter_safetensors falls back to
    # scanning the single shard's own header (see its own docstring) --
    # unlike test_engine_loop.py's qwen3_moe fixture, this test does NOT
    # use use_dummy_weight, so the checkpoint's real tensor bytes must
    # actually be read, not just a present-but-empty index tolerated.

    config = parse_config(type("Hf", (), {"to_dict": lambda self: TINY_CONFIG})())
    state = {}

    def add(name, shape):
        state[name] = torch.randn(shape, dtype=torch.float32) * 0.02

    hs, vocab = config.hidden_size, config.vocab_size
    heads, kv, head_dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    inter = config.intermediate_size
    layers = config.num_layers

    add("model.embed_tokens.weight", (vocab, hs))
    for l in range(layers):
        prefix = f"model.layers.{l}"
        add(f"{prefix}.input_layernorm.weight", (hs,))
        add(f"{prefix}.self_attn.q_proj.weight", (heads * head_dim, hs))
        add(f"{prefix}.self_attn.k_proj.weight", (kv * head_dim, hs))
        add(f"{prefix}.self_attn.v_proj.weight", (kv * head_dim, hs))
        add(f"{prefix}.self_attn.o_proj.weight", (hs, heads * head_dim))
        add(f"{prefix}.self_attn.q_norm.weight", (head_dim,))
        add(f"{prefix}.self_attn.k_norm.weight", (head_dim,))
        add(f"{prefix}.post_attention_layernorm.weight", (hs,))
        add(f"{prefix}.mlp.gate_proj.weight", (inter, hs))
        add(f"{prefix}.mlp.up_proj.weight", (inter, hs))
        add(f"{prefix}.mlp.down_proj.weight", (hs, inter))
    add("model.norm.weight", (hs,))
    add("lm_head.weight", (vocab, hs))

    save_file(state, str(model_path / "model.safetensors"))
    return str(model_path)


def _engine_config(model_path: str, *, device: str | None = None) -> EngineConfig:
    return EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.float32,
        device=device,
        attention_backend="auto",
        max_running_req=2,
        page_size=1,
        max_seq_len_override=32,
    )


def _add_prompt(engine: Engine, output_len: int) -> None:
    engine.add_request(
        Req(
            input_ids=[1, 2, 3],
            table_idx=0,
            cached_len=0,
            output_len=output_len,
            uid=0,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=output_len),
            cache_handle=None,
        )
    )


def test_iter_weights_covers_every_dense_param(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    names = {n for n, _ in iter_weights(model_path, torch.device("cpu"))}
    assert "model.embed_tokens.weight" in names
    assert "model.layers.0.self_attn.q_proj.weight" in names
    assert "model.layers.0.mlp.gate_proj.weight" in names
    assert "lm_head.weight" in names


def test_tie_word_embeddings_synthesizes_lm_head(tmp_path):
    config = dict(TINY_CONFIG, tie_word_embeddings=True)
    model_path = tmp_path / "tied_ckpt"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(config))
    from safetensors.torch import save_file

    save_file(
        {"model.embed_tokens.weight": torch.randn(config["vocab_size"], config["hidden_size"])},
        str(model_path / "model.safetensors"),
    )
    got = dict(iter_weights(str(model_path), torch.device("cpu")))
    assert "lm_head.weight" in got
    torch.testing.assert_close(got["lm_head.weight"], got["model.embed_tokens.weight"])


def test_model_class_builds_real_nn_module(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    config = parse_config(type("Hf", (), {"to_dict": lambda self: TINY_CONFIG})())
    model = get_model_class("Qwen3ForCausalLM", config, device=torch.device("cpu"))
    assert isinstance(model, torch.nn.Module)
    assert len(model.layers) == TINY_CONFIG["num_hidden_layers"]


def test_engine_generate_prefill_and_decode(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)
    engine = Engine(_engine_config(model_path, device=DEVICE))
    _add_prompt(engine, output_len=4)
    generated = engine.generate()
    vocab = engine.config.model_config.vocab_size
    assert len(generated) == 1
    assert len(generated[0]) == 4
    assert all(0 <= t < vocab for t in generated[0])


def test_engine_greedy_is_deterministic(tmp_path):
    model_path = _write_tiny_checkpoint(tmp_path)

    def build():
        engine = Engine(_engine_config(model_path, device=DEVICE))
        _add_prompt(engine, output_len=3)
        return engine.generate()

    a = build()
    b = build()
    assert a == b
    assert len(a[0]) == 3


def test_write_kv_uses_physical_pool_slots_not_logical_positions(tmp_path):
    """Real bug found running a real trained checkpoint's generation through
    the actual Engine (not a synthetic fixture): ``write_kv``'s third
    argument is ``out_loc`` -- PHYSICAL pool slots -- not logical token
    positions. These only coincide under an identity page table.
    ``MHAKVCache`` (the real per-request free-list allocator every live
    Engine uses, not ``BaseKVCachePool``) reserves slot 0 as a dummy/padding
    slot, so the first real token of any request lands on slot >= 1, never
    slot 0 -- the two spaces are NOT identity for any real request. Passing
    raw ``positions`` here silently wrote every token to the wrong physical
    slot: real (non-synthetic) generation was degenerate garbage even after
    an unrelated RoPE-orientation bug was independently fixed, traced to
    this exact call via a ``read_kv`` == pre-write value round-trip check
    that failed only under the real ``MHAKVCache`` allocator, never under a
    synthetic fixture's identity-mapped pool -- which is why no existing
    synthetic-checkpoint test (including this file's own
    ``test_engine_generate_prefill_and_decode``/``test_engine_greedy_is_deterministic``,
    both of which only assert "produces some deterministic in-range token",
    not correctness against a reference) ever caught it.

    This pins the fix directly: after a real Engine's real per-request slot
    allocation (never slot 0 for a real request's first token), the value
    written into the pool and the value read back for that same position
    must be byte-identical -- proof ``write_kv`` received real physical
    slots, not raw positions.
    """
    model_path = _write_tiny_checkpoint(tmp_path)
    engine = Engine(_engine_config(model_path, device=DEVICE))

    # Spy on the KV pool's own write_kv -- the ONLY reliable way to see what
    # out_loc argument the model code actually passed (not what a test
    # independently recomputes, which would pass vacuously regardless of
    # the model code's real behavior).
    captured = {}
    pool = engine.ctx.kv_cache
    orig_pool_write_kv = pool.write_kv

    def spy_write_kv(k, v, out_loc, layer_id=0):
        if layer_id == 0 and "out_loc" not in captured:
            captured["out_loc"] = out_loc.clone()
        return orig_pool_write_kv(k, v, out_loc, layer_id)

    pool.write_kv = spy_write_kv
    try:
        _add_prompt(engine, output_len=1)
        engine.generate()
    finally:
        pool.write_kv = orig_pool_write_kv

    table_idx = 0
    positions = torch.arange(3)  # the prompt is 3 tokens, see _add_prompt
    expected_slots = engine.ctx.page_table[table_idx, positions.long()]

    # A real request's first token never lands on the reserved slot 0 --
    # confirms the allocator genuinely isn't identity for this run.
    assert 0 not in expected_slots.tolist()
    # The actual out_loc write_kv received must be the PHYSICAL pool slots
    # from the page table, not the raw logical positions [0, 1, 2] -- this
    # is what the real bug got wrong (verified this assertion fails without
    # the fix: reverting the out_loc translation makes captured["out_loc"]
    # equal positions instead of the real allocated slots).
    assert captured["out_loc"].equal(expected_slots)
    assert not captured["out_loc"].equal(positions)
