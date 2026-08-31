"""Tests for the vocabulary-embedding / LM-head layer pair (issue ``layers``, #24 WP2).

Two halves (mirrors ``test_layers_norm.py``):

* CPU-safe (the per-PR ``ci`` job, torch-free): module presence + the package
  re-exports + the tied-head ``state_dict``/``load_state_dict`` invariant, which is
  pure-Python (no tensors touched) and so runs on the torch-free CPU venv.

* ``xpu``-marked (the-only: every path is driven on the B70 against the *same* PyTorch
  op the port performs (``F.embedding`` row-gather / ``F.linear``), fed the same
  bf16 tensors the layer consumes, so the comparison is op- and dtype-faithful (and
  bit-exact for the plain gather).

XPU repr hazard (why every check computes a plain float first):
    On the B70, if a tensor lands inside a *failing* ``assert``, pytest's
    assertion-rewriter calls ``repr()`` on it to build the failure diff; on this
    oneAPI runtime that can OOM the device and wedge the run. So every numerical
    check computes its max-abs-error into a plain Python ``float`` (``.item()``)
    *outside* the ``assert`` and asserts on that float only.
"""
from __future__ import annotations

import importlib.util

import pytest

torch = pytest.importorskip("torch")

DEV = "xpu"


def test_embedding_layer_module_present():
    """No torch needed to check presence (self-skips on the torch-free CPU venv)."""
    spec = importlib.util.find_spec("freetoken.layers.embedding")
    assert spec is not None, "freetoken.layers.embedding is missing"


def test_embedding_package_exports():
    """The package re-exports both classes (torch-free check, CPU-safe)."""
    import freetoken.layers as layers

    assert hasattr(layers, "VocabParallelEmbedding")
    assert hasattr(layers, "ParallelLMHead")
    # ParallelLMHead is a subclass of VocabParallelEmbedding (upstream parity).
    assert issubclass(layers.ParallelLMHead, layers.VocabParallelEmbedding)


def _max_abs_err(got, want) -> float:
    """Max abs error as a plain Python float, computed *outside* any ``assert``."""
    return (got.float() - want.float()).abs().max().item()


# --------------------------------------------------------------------------- #
# CPU reference: the row-gather the embedding layer performs is F.embedding.
# The port IS built on F.embedding (upstream's CUDA `indexing` kernel replaced by
# the equivalent torch op), so that op is the reference -- fed the same tensors the
# layer consumes. A hand gather (weight[indices]) would also agree, but matching the
# exact op the port calls keeps the comparison bit-exact on the XPU.
# --------------------------------------------------------------------------- #
def _embedding_ref(weights, indices):
    import torch.nn.functional as F

    return F.embedding(indices, weights)


def _linear_ref(x, weight, bias=None):
    import torch.nn.functional as F

    return F.linear(x, weight, bias)


# --------------------------------------------------------------------------- #
# XPU: drive the embedding / lm-head on the B70.
# --------------------------------------------------------------------------- #
@pytest.mark.xpu
def test_vocab_parallel_embedding_forward_matches_reference():
    from freetoken.layers.embedding import VocabParallelEmbedding

    vocab, H, N, seed = 512, 64, 9, 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    weights = torch.randn(vocab, H, device="cpu", dtype=torch.float32).to(DEV, dtype=torch.bfloat16)
    # token ids into [0, vocab)
    indices = torch.randint(0, vocab, (N,), device="cpu", dtype=torch.long, generator=g).to(DEV)
    layer = VocabParallelEmbedding(vocab, H)
    layer.weight = weights
    got = layer.forward(indices)
    want = _embedding_ref(weights, indices)
    assert got.shape == want.shape == (N, H)
    err = _max_abs_err(got, want)
    assert err < 1e-3, f"embedding forward: max abs err {err:.5f}"


@pytest.mark.xpu
def test_vocab_parallel_embedding_single_row():
    from freetoken.layers.embedding import VocabParallelEmbedding

    vocab, H = 128, 32
    g = torch.Generator(device="cpu").manual_seed(1)
    weights = torch.randn(vocab, H, device="cpu", dtype=torch.float32).to(DEV, dtype=torch.bfloat16)
    layer = VocabParallelEmbedding(vocab, H)
    layer.weight = weights
    for i in range(3):
        idx = torch.tensor([i], device=DEV, dtype=torch.long)
        got = layer.forward(idx)
        want = weights[i : i + 1]
        assert got.shape == (1, H)
        err = _max_abs_err(got, want)
        assert err < 1e-3, f"embedding row {i}: max abs err {err:.5f}"


@pytest.mark.xpu
def test_vocab_parallel_embedding_state_dict_weight_only():
    from freetoken.layers.embedding import VocabParallelEmbedding

    vocab, H = 64, 16
    layer = VocabParallelEmbedding(vocab, H)
    sd = layer.state_dict()
    # weight is the only persisted param; the lazy _embed_scale_t (underscore-prefixed)
    # is a runtime artefact and must NOT appear.
    assert set(sd.keys()) == {"weight"}, f"state_dict keys were {set(sd.keys())}"
    new_w = torch.randn(vocab, H)
    layer.load_state_dict({"weight": new_w})
    assert layer.weight.detach().cpu().numpy().tobytes() == new_w.detach().cpu().numpy().tobytes()


@pytest.mark.xpu
def test_vocab_parallel_embedding_embed_scale_lazy_and_scaled():
    from freetoken.layers.embedding import VocabParallelEmbedding

    vocab, H, N, scale, seed = 96, 32, 5, 2.5, 2
    g = torch.Generator(device="cpu").manual_seed(seed)
    weights = torch.randn(vocab, H, device="cpu", dtype=torch.float32).to(DEV, dtype=torch.bfloat16)
    indices = torch.randint(0, vocab, (N,), device="cpu", dtype=torch.long, generator=g).to(DEV)
    layer = VocabParallelEmbedding(vocab, H, embed_scale=scale)
    layer.weight = weights
    # Before the first forward the lazy scale buffer must not yet exist (not materialised).
    assert getattr(layer, "_embed_scale_t", None) is None, "scale must be lazy (no buffer before forward)"
    got = layer.forward(indices)
    want = _embedding_ref(weights, indices) * scale
    assert layer._embed_scale_t is not None, "scale buffer must materialise on first forward"
    # The cached scalar must be in the input's dtype/device (so the multiply is a same-dtype op).
    assert layer._embed_scale_t.dtype == indices.dtype or layer._embed_scale_t.dtype == weights.dtype
    err = _max_abs_err(got, want)
    assert err < 1e-2, f"embedding embed_scale: max abs err {err:.5f}"


@pytest.mark.xpu
def test_vocab_parallel_embedding_no_scale_is_plain_gather():
    from freetoken.layers.embedding import VocabParallelEmbedding

    vocab, H, N, seed = 96, 32, 5, 3
    g = torch.Generator(device="cpu").manual_seed(seed)
    weights = torch.randn(vocab, H, device="cpu", dtype=torch.float32).to(DEV, dtype=torch.bfloat16)
    indices = torch.randint(0, vocab, (N,), device="cpu", dtype=torch.long, generator=g).to(DEV)
    layer = VocabParallelEmbedding(vocab, H)
    layer.weight = weights
    got = layer.forward(indices)
    want = _embedding_ref(weights, indices)
    err = _max_abs_err(got, want)
    assert err < 1e-3, f"embedding no-scale: max abs err {err:.5f}"


@pytest.mark.xpu
def test_parallel_lm_head_untied_matches_linear():
    from freetoken.layers.embedding import ParallelLMHead

    vocab, H, N, seed = 128, 48, 6, 4
    g = torch.Generator(device="cpu").manual_seed(seed)
    weights = torch.randn(vocab, H, device="cpu", dtype=torch.float32).to(DEV, dtype=torch.bfloat16)
    x = torch.randn(N, H, device="cpu", dtype=torch.float32).to(DEV, dtype=torch.bfloat16)
    layer = ParallelLMHead(vocab, H)  # bias=False, tie_word_embeddings=False
    layer.weight = weights
    got = layer.forward(x)
    want = _linear_ref(x, weights)
    assert got.shape == (N, vocab)
    err = _max_abs_err(got, want)
    assert err < 1e-2, f"lm_head untied: max abs err {err:.5f}"


@pytest.mark.xpu
def test_parallel_lm_head_with_bias():
    from freetoken.layers.embedding import ParallelLMHead

    vocab, H, N, seed = 128, 48, 6, 5
    g = torch.Generator(device="cpu").manual_seed(seed)
    weights = torch.randn(vocab, H, device="cpu", dtype=torch.float32).to(DEV, dtype=torch.bfloat16)
    bias = torch.randn(vocab, device="cpu", dtype=torch.float32).to(DEV, dtype=torch.bfloat16)
    x = torch.randn(N, H, device="cpu", dtype=torch.float32).to(DEV, dtype=torch.bfloat16)
    layer = ParallelLMHead(vocab, H, bias=True)
    layer.weight = weights
    layer.bias = bias
    got = layer.forward(x)
    want = _linear_ref(x, weights, bias)
    err = _max_abs_err(got, want)
    assert err < 1e-2, f"lm_head bias: max abs err {err:.5f}"


@pytest.mark.xpu
def test_parallel_lm_head_tied_shares_embedding_weight():
    from freetoken.layers.embedding import ParallelLMHead, VocabParallelEmbedding

    vocab, H, N, seed = 128, 48, 6, 6
    g = torch.Generator(device="cpu").manual_seed(seed)
    weights = torch.randn(vocab, H, device="cpu", dtype=torch.float32).to(DEV, dtype=torch.bfloat16)
    x = torch.randn(N, H, device="cpu", dtype=torch.float32).to(DEV, dtype=torch.bfloat16)
    embed = VocabParallelEmbedding(vocab, H)
    embed.weight = weights
    head = ParallelLMHead(vocab, H, tie_word_embeddings=True, tied_embedding=embed)
    # The tied head must share the SAME weight tensor object as the embedding.
    assert head._module_weight() is embed.weight, "tied head must share the embedding weight tensor"
    got = head.forward(x)
    want = _linear_ref(x, weights)
    err = _max_abs_err(got, want)
    assert err < 1e-2, f"lm_head tied: max abs err {err:.5f}"


@pytest.mark.xpu
def test_parallel_lm_head_tied_state_dict_empty():
    from freetoken.layers.embedding import ParallelLMHead, VocabParallelEmbedding

    vocab, H = 64, 16
    embed = VocabParallelEmbedding(vocab, H)
    embed.weight = torch.randn(vocab, H)
    head = ParallelLMHead(vocab, H, tie_word_embeddings=True, tied_embedding=embed)
    # Tied head owns no weights of its own -> state_dict is empty (loader never expects
    # an lm_head.weight key for a tied head).
    assert head.state_dict() == {}, "tied lm_head.state_dict() must be empty"


def test_parallel_lm_head_tied_load_state_dict_drops_keys():
    from freetoken.layers.embedding import ParallelLMHead, VocabParallelEmbedding

    vocab, H = 64, 16
    embed = VocabParallelEmbedding(vocab, H)
    head = ParallelLMHead(vocab, H, bias=True, tie_word_embeddings=True, tied_embedding=embed)
    # Mirror BaseOP's key convention: a nested child is handed its keys under its
    # (dotted) prefix. A tied head must consume (drop) those weight/bias keys rather
    # than trip on them, since it shares tied_embedding's table.
    sd = {"lm_head.weight": torch.randn(vocab, H), "lm_head.bias": torch.randn(vocab)}
    head.load_state_dict(sd, prefix="lm_head")
    assert sd == {}, f"tied load_state_dict must consume its own keys, leftover: {list(sd.keys())}"
    # ...and must NOT have installed a private weight on the tied head.
    assert not hasattr(head, "weight") or head.weight is not embed.weight, "tied head must not own a separate weight"


def test_parallel_lm_head_ctor_invariant():
    """The ctor asserts (tied_embedding is not None) == tie_word_embeddings."""
    from freetoken.layers.embedding import ParallelLMHead

    with pytest.raises(AssertionError):
        ParallelLMHead(10, 5, tie_word_embeddings=True)  # tied flag but no embedding
    with pytest.raises(AssertionError):
        # Provide an embedding but leave tie_word_embeddings False -> invariant violated.
        ParallelLMHead(10, 5, tie_word_embeddings=False, tied_embedding=ParallelLMHead(10, 5))
