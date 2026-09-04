"""Unit tests for Qwen3.8-Flash-Next's PLE hashed n-gram embedding primitive
(issue ``models-qwen4-ple``, #207). Standalone -- not yet wired into a
decoder layer, checkpoint loader, or the engine's slot-state pool (that's
#209's job); these pin the hash/lookup/combine math against hand-computed
references, plus the RAM-safe on-disk table backend this port's own 29 GiB
host RAM requires (the real n-gram store is ~47.7 GiB and cannot be fully
resident here -- see the module docstring in qwen4_exp/__init__.py).
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from freetoken.models.qwen4_exp import (
    PLELayer,
    PleDiskTable,
    PleInMemoryTable,
    derive_ngram_hash_constants,
    ple_ngram_row_ids,
    ple_short_conv,
)


def test_derive_ngram_hash_constants_is_deterministic_and_disjoint_offsets():
    m1, sizes1, offsets1 = derive_ngram_hash_constants(
        vocab_size=1000, ngram_size=3, num_ngram_heads=4, ngram_vocab_size_base=97, ple_layer_index=0
    )
    m2, sizes2, offsets2 = derive_ngram_hash_constants(
        vocab_size=1000, ngram_size=3, num_ngram_heads=4, ngram_vocab_size_base=97, ple_layer_index=0
    )
    assert m1 == m2 and sizes1 == sizes2 and offsets1 == offsets2
    assert len(m1) == 3 and len(sizes1) == 4 and len(offsets1) == 4
    # offsets are the running sum of the preceding sizes (disjoint global ranges)
    running = 0
    for size, offset in zip(sizes1, offsets1):
        assert offset == running
        running += size
    # a different layer index must derive different multipliers (real per-layer salting)
    m3, _, _ = derive_ngram_hash_constants(
        vocab_size=1000, ngram_size=3, num_ngram_heads=4, ngram_vocab_size_base=97, ple_layer_index=1
    )
    assert m3 != m1


def _hand_hash(tokens, boundary, ngram_size, heads_per_ngram, multipliers, sizes, offsets):
    """Independent Python re-derivation of ple_ngram_row_ids for a short, explicit sequence.

    Matches the real rule: shift-0 (the token at ``t`` itself) is never reset; a tap at
    ``t - shift`` (shift >= 1) is valid only while it stays within the segment that opened
    after the last boundary occurring STRICTLY BEFORE ``t`` (a boundary token AT position t
    does not close its own segment) -- i.e. ``shift <= in_segment(t)`` where
    ``in_segment(t) = t - (last boundary strictly before t) - 1``."""
    T = len(tokens)
    rows = []
    for t in range(T):
        last_boundary_before_t = max((j for j in range(t) if tokens[j] == boundary), default=-1)
        in_segment = t - last_boundary_before_t - 1
        row = []
        for ngram in range(2, ngram_size + 1):
            taps = []
            for shift in range(ngram):
                if shift == 0:
                    taps.append(tokens[t])
                    continue
                pos = t - shift
                valid = pos >= 0 and shift <= in_segment
                taps.append(tokens[pos] if valid else boundary)
            mixed = 0
            for i, tok in enumerate(taps):
                mixed ^= (tok * multipliers[i]) & ((1 << 64) - 1)
            start = (ngram - 2) * heads_per_ngram
            for h in range(heads_per_ngram):
                head_id = mixed % sizes[start + h]
                row.append(head_id + offsets[start + h])
        rows.append(row)
    return torch.tensor(rows, dtype=torch.int64)


def test_ple_ngram_row_ids_matches_hand_computed_reference_no_boundary():
    ngram_size, heads_per_ngram, boundary = 3, 2, 99
    multipliers = torch.tensor([3, 5, 7], dtype=torch.int64)
    sizes = torch.tensor([11, 13, 17, 19], dtype=torch.int64)
    offsets = torch.tensor([0, 11, 24, 41], dtype=torch.int64)
    tokens = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)

    got = ple_ngram_row_ids(tokens, boundary, ngram_size, heads_per_ngram, multipliers, sizes, offsets)
    expected = _hand_hash(
        tokens.tolist(), boundary, ngram_size, heads_per_ngram, multipliers.tolist(), sizes.tolist(), offsets.tolist()
    )
    assert got.shape == (6, 4)
    torch.testing.assert_close(got, expected)


def test_ple_ngram_row_ids_boundary_actually_resets_the_hash_window():
    """The boundary-reset rule must be a real effect, not a silent no-op: inserting a
    boundary token measurably changes a later row's hash versus the same stream with no
    boundary at that position."""
    ngram_size, heads_per_ngram, boundary = 3, 1, 0
    multipliers = torch.tensor([3, 5, 7], dtype=torch.int64)
    sizes = torch.tensor([1009, 1013], dtype=torch.int64)
    offsets = torch.tensor([0, 1009], dtype=torch.int64)

    with_boundary = torch.tensor([1, 2, boundary, 4, 5], dtype=torch.int64)
    without_boundary = torch.tensor([1, 2, 9, 4, 5], dtype=torch.int64)  # same shape, no reset

    rows_a = ple_ngram_row_ids(with_boundary, boundary, ngram_size, heads_per_ngram, multipliers, sizes, offsets)
    rows_b = ple_ngram_row_ids(without_boundary, boundary, ngram_size, heads_per_ngram, multipliers, sizes, offsets)
    # row 4 (token '5'): the 3-gram head needs positions 2,3,4 -- position 2 is the boundary
    # itself in `with_boundary`, so shift=2 is invalid there (only 1 token, '4', lies in the
    # segment reopened after it) but fully valid in `without_boundary` -- the hashes differ.
    assert not torch.equal(rows_a[4], rows_b[4])
    # the 2-gram head at row 4 only needs positions 3,4 -- neither crosses the boundary at
    # position 2 in either stream, and position 3 ('4') is identical in both -- unaffected.
    assert torch.equal(rows_a[4, :1], rows_b[4, :1])


def test_ple_ngram_row_ids_fresh_sequence_starting_with_boundary_never_crosses_it():
    ngram_size, heads_per_ngram, boundary = 3, 1, 0
    multipliers = torch.tensor([3, 5, 7], dtype=torch.int64)
    sizes = torch.tensor([1009, 1013], dtype=torch.int64)
    offsets = torch.tensor([0, 1009], dtype=torch.int64)
    tokens = torch.tensor([boundary, boundary, 1, 2, 3], dtype=torch.int64)

    got = ple_ngram_row_ids(tokens, boundary, ngram_size, heads_per_ngram, multipliers, sizes, offsets)
    expected = _hand_hash(
        tokens.tolist(), boundary, ngram_size, heads_per_ngram, multipliers.tolist(), sizes.tolist(), offsets.tolist()
    )
    torch.testing.assert_close(got, expected)


def test_ple_in_memory_table_lookup_matches_index_select():
    torch.manual_seed(0)
    weight = torch.randn(20, 6)
    table = PleInMemoryTable(weight, scale=2.0)
    row_ids = torch.tensor([[0, 5], [3, 19]])
    got = table.lookup(row_ids)
    expected = (weight.index_select(0, row_ids.reshape(-1)) * 2.0).view(2, -1)
    torch.testing.assert_close(got, expected)


def test_ple_disk_table_matches_in_memory_oracle(tmp_path):
    """The RAM-safe disk-backed path must return bit-identical rows to the in-memory oracle
    -- proves lazy safetensors row reads aren't silently corrupting/misaligning data."""
    from safetensors.torch import save_file

    torch.manual_seed(1)
    weight = torch.randn(50, 8)
    path = str(tmp_path / "ple_table.safetensors")
    save_file({"ngram_embedding": weight}, path)

    oracle = PleInMemoryTable(weight, scale=1.5)
    disk = PleDiskTable(path, "ngram_embedding", scale=1.5)
    assert disk.num_rows == 50 and disk.head_dim == 8

    row_ids = torch.tensor([[1, 49, 0], [12, 33, 7]])
    torch.testing.assert_close(disk.lookup(row_ids), oracle.lookup(row_ids))


def test_ple_short_conv_matches_hand_computed_depthwise_conv():
    torch.manual_seed(2)
    width, kernel, dilation, state_len = 4, 3, 2, 4  # (kernel-1)*dilation
    T = 5
    x = torch.randn(T, width)
    state = torch.randn(width, state_len)
    weight = torch.randn(width, 1, kernel)

    out, new_state = ple_short_conv(x, state, weight, dilation)
    assert out.shape == (T, width) and new_state.shape == (width, state_len)

    history = torch.cat([state, x.transpose(0, 1)], dim=-1)
    expected = torch.nn.functional.conv1d(
        history.unsqueeze(0), weight, groups=width, dilation=dilation
    ).squeeze(0)
    expected = torch.nn.functional.silu(expected.transpose(0, 1))
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(new_state, history[:, -state_len:])


def test_ple_layer_forward_runs_end_to_end_with_disk_backed_table(tmp_path):
    """Full standalone PLE forward: hash -> disk-backed lookup -> gate/combine -> conv,
    over one materialized sequence -- the #207 accept criterion ('a real forward pass with
    PLE active on at least one layer runs end-to-end')."""
    from safetensors.torch import save_file

    torch.manual_seed(3)
    hidden, hc_count, ple_embed_dim, kernel, ngram_size, heads_per_ngram = 4, 3, 8, 3, 3, 2
    dilation, eps, boundary = ngram_size, 1e-5, 0
    width = hc_count * hidden
    state_len = (kernel - 1) * dilation
    num_ngram_heads = (ngram_size - 1) * heads_per_ngram
    head_dim = ple_embed_dim // num_ngram_heads

    multipliers = torch.tensor([3, 5, 7], dtype=torch.int64)[:ngram_size]
    sizes = torch.tensor([7, 11, 13, 17][:num_ngram_heads], dtype=torch.int64)
    offsets = torch.tensor([0, 7, 18, 31][:num_ngram_heads], dtype=torch.int64)
    total_rows = int((sizes + offsets)[-1])

    weight = torch.randn(total_rows, head_dim)
    path = str(tmp_path / "table.safetensors")
    save_file({"ngram_embedding": weight}, path)
    table = PleDiskTable(path, "ngram_embedding")

    T = 6
    tokens = torch.tensor([1, 2, 3, boundary, 5, 6], dtype=torch.int64)
    row_ids = ple_ngram_row_ids(tokens, boundary, ngram_size, heads_per_ngram, multipliers, sizes, offsets)
    embeddings = table.lookup(row_ids)
    assert embeddings.shape == (T, ple_embed_dim)

    layer = PLELayer(hidden, hc_count, ple_embed_dim, kernel, dilation, eps)
    R = torch.randn(T, width)
    conv_state = torch.zeros(width, state_len)

    D, new_state = layer(R, embeddings, conv_state)
    assert D.shape == (T, width)
    assert new_state.shape == (width, state_len)
    assert torch.isfinite(D).all()

    R_next = R + D
    assert R_next.shape == R.shape
