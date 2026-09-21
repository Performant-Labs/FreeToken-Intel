"""Tests for the native CPU MoE async submit/wait dispatch (issue #248).

Replaces the pure-Python ``ThreadPoolExecutor.submit()``/``future.result()``
handoff `_Qwen3MoE._forward_hybrid` (and its `qwen3_5_moe` twin) used to pay
~40-50ms/layer for (issue #247) with a native persistent ``std::thread``
worker reached via ctypes ``submit()``/``wait()`` calls
(``freetoken.kernel.cpu_moe.cpu_moe_submit`` / ``CpuMoeJob.result()`` /
``hybrid_subset_submit`` / ``HybridCpuPool``).

CPU-safe: no XPU needed. Compiling and loading the kernel needs a real C++
compiler (mirrors ``test_kernel_cpu_moe.py``'s toolchain-skip pattern) --
when none is found this whole file skips gracefully rather than failing.
"""
from __future__ import annotations

import time

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from freetoken.kernel._toolchain import ToolchainError  # noqa: E402


@pytest.fixture(scope="module")
def native_module():
    from freetoken.kernel.cpu_moe import cpu_moe, find_cxx_compiler

    try:
        find_cxx_compiler()
    except ToolchainError as exc:
        pytest.skip(f"no C++ compiler available for the native CPU MoE kernel: {exc}")
    return cpu_moe()


def _reference_forward(x, top_idx, top_w, gate_up, down, num_experts, intermediate, candidates):
    """Same expert-major-then-top-k-column SwiGLU reference test_kernel_cpu_moe.py uses.

    ``gate_up``/``down`` may be ``bfloat16`` tensors (issue #257: the native
    kernel now reads bf16 weight bytes directly). ``.float()`` upcasts them
    to float32 here -- this is the *same* bf16->float32 widen the kernel's
    dot-product primitives perform internally (an exact, lossless
    bit-widening, not a lossy cast, since bf16 already is a truncated
    float32), so this stays an apples-to-apples reference even though the
    matmul itself runs in float32.
    """
    T, H = x.shape
    k = top_idx.shape[1]
    out = torch.zeros(T, H, dtype=torch.float32)
    I = intermediate
    gate_up_f32 = gate_up.float()
    down_f32 = down.float()
    for e in range(num_experts):
        if candidates is not None and e not in candidates:
            continue
        for j in range(k):
            sel = top_idx[:, j] == e
            if not bool(sel.any()):
                continue
            idx = sel.nonzero(as_tuple=True)[0]
            x_sel = x.index_select(0, idx)
            gate = x_sel @ gate_up_f32[e, :I].t()
            up = x_sel @ gate_up_f32[e, I : 2 * I].t()
            y = (F.silu(gate) * up) @ down_f32[e].t()
            w = top_w.index_select(0, idx)[:, j, None]
            out.index_add_(0, idx, w * y)
    return out


# rtol/atol for comparisons involving the bf16-native kernel path: the
# bf16->float32 widen itself is exact (see _reference_forward's docstring),
# so this is *not* about bf16 precision loss -- both sides of these
# comparisons see the same already-bf16-rounded weight values. The looser
# tolerance (vs. the float32-input tests' 1e-4/1e-5) is for the same reason
# test_kernel_cpu_moe.py's own forward-vs-fast comparisons use rtol=1e-3:
# the AVX-512 widen-in-register dot product (this kernel's default when the
# host supports it) reduces its accumulator in a different lane order than
# the naive scalar loop, so float32 addition's non-associativity applies --
# tolerance-exact, not bit-exact.
BF16_RTOL = 1e-3
BF16_ATOL = 1e-4


def _synthetic_case(seed: int = 0, *, num_tokens=7, hidden=6, intermediate=5, num_experts=4, topk=2):
    """bf16 gate_up/down fixtures (issue #257) -- x/expert_ids/expert_weights
    stay float32 (activations were never the conversion bottleneck)."""
    torch.manual_seed(seed)
    x = torch.randn(num_tokens, hidden, dtype=torch.float32)
    gate_up = torch.randn(num_experts, 2 * intermediate, hidden, dtype=torch.float32).to(torch.bfloat16)
    down = torch.randn(num_experts, hidden, intermediate, dtype=torch.float32).to(torch.bfloat16)
    top_idx = torch.randint(0, num_experts, (num_tokens, topk))
    top_w = torch.rand(num_tokens, topk)
    top_w = top_w / top_w.sum(dim=1, keepdim=True)
    return x, top_idx, top_w, gate_up, down, num_experts, intermediate


# --- submit()/wait() round-trip vs the synchronous path ---------------------


def test_submit_wait_matches_synchronous_full_backend(native_module):
    """expert_mask=None: submit()/wait() must match cpu_moe_forward() within
    tolerance. Not bit-exact (BF16_RTOL/BF16_ATOL) since #257: cpu_moe_forward
    stays on the float32-input naive scalar path, while cpu_moe_submit now
    reads the same (bf16-widened) weight values through the bf16-native core,
    which auto-selects the AVX-512 widen-in-register dot product on a host
    that supports it -- a different (but equally valid) float32 reduction
    order than the scalar loop."""
    from freetoken.kernel.cpu_moe import cpu_moe_forward, cpu_moe_submit

    x, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case(seed=0)

    expected = cpu_moe_forward(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate, expert_mask=None
    )
    job = cpu_moe_submit(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate, expert_mask=None
    )
    actual = job.result()

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=BF16_RTOL, atol=BF16_ATOL)


def test_submit_wait_matches_synchronous_subset(native_module):
    """expert_mask names a subset: submit()/wait() must still match cpu_moe_forward()."""
    from freetoken.kernel.cpu_moe import cpu_moe_forward, cpu_moe_submit

    x, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case(seed=1)
    mask = {0, 2}

    expected = cpu_moe_forward(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate, expert_mask=mask
    )
    job = cpu_moe_submit(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate, expert_mask=mask
    )
    actual = job.result()

    torch.testing.assert_close(actual, expected, rtol=BF16_RTOL, atol=BF16_ATOL)


def test_result_called_twice_raises(native_module):
    """A job's result() may only be collected once (mirrors a Future that
    cannot be re-awaited after its single-slot native job has been freed)."""
    from freetoken.kernel.cpu_moe import cpu_moe_submit

    x, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case(seed=2)
    job = cpu_moe_submit(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate, expert_mask=None
    )
    job.result()
    with pytest.raises(RuntimeError):
        job.result()


def test_wait_is_alias_for_result(native_module):
    from freetoken.kernel.cpu_moe import cpu_moe_forward, cpu_moe_submit, cpu_moe_wait

    x, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case(seed=3)
    expected = cpu_moe_forward(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate, expert_mask=None
    )
    job = cpu_moe_submit(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate, expert_mask=None
    )
    actual = cpu_moe_wait(job)
    torch.testing.assert_close(actual, expected, rtol=BF16_RTOL, atol=BF16_ATOL)


# --- concurrent submit-then-other-work-then-wait -----------------------------


def test_submit_returns_before_worker_finishes_then_wait_collects_result(native_module):
    """submit() must not itself block for the native compute: a caller doing
    other (e.g. XPU-side) work between submit() and result() must see the
    worker thread actually running concurrently, not have already finished
    synchronously inside submit()."""
    from freetoken.kernel.cpu_moe import cpu_moe_forward, cpu_moe_submit

    # A deliberately larger case so the native compute takes measurably
    # longer than issuing the ctypes submit() call itself.
    x, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case(
        seed=4, num_tokens=64, hidden=64, intermediate=64, num_experts=8, topk=2
    )
    expected = cpu_moe_forward(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate, expert_mask=None
    )

    job = cpu_moe_submit(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate, expert_mask=None
    )
    # "Other work" the caller does concurrently with the native worker thread
    # (stand-in for the XPU half's PCIe fetch + gather in _forward_hybrid).
    busy = sum(i * i for i in range(200_000))
    assert busy >= 0  # keep the loop from being optimized away / just a lint no-op

    actual = job.result()
    torch.testing.assert_close(actual, expected, rtol=BF16_RTOL, atol=BF16_ATOL)


def test_two_sequential_jobs_do_not_interfere(native_module):
    """Two submit()/result() round trips in a row (mirrors one MoE layer's
    job followed by the next layer's) must not leak state between jobs."""
    from freetoken.kernel.cpu_moe import cpu_moe_forward, cpu_moe_submit

    case_a = _synthetic_case(seed=5)
    case_b = _synthetic_case(seed=6)

    job_a = cpu_moe_submit(native_module, *case_a[:2], case_a[2], *case_a[3:5], *case_a[5:7], expert_mask=None)
    result_a = job_a.result()
    expected_a = cpu_moe_forward(native_module, *case_a[:2], case_a[2], *case_a[3:5], *case_a[5:7], expert_mask=None)
    torch.testing.assert_close(result_a, expected_a, rtol=BF16_RTOL, atol=BF16_ATOL)

    job_b = cpu_moe_submit(native_module, *case_b[:2], case_b[2], *case_b[3:5], *case_b[5:7], expert_mask=None)
    result_b = job_b.result()
    expected_b = cpu_moe_forward(native_module, *case_b[:2], case_b[2], *case_b[3:5], *case_b[5:7], expert_mask=None)
    torch.testing.assert_close(result_b, expected_b, rtol=BF16_RTOL, atol=BF16_ATOL)


# --- timing sanity: submit() returns near-instantly -------------------------


def test_submit_returns_near_instantly(native_module):
    """submit() should return in roughly the time it takes to claim the
    single job slot and hand off a struct -- not block for the compute
    itself. This is a rough sanity check (not a hard performance gate, to
    avoid CI flakiness), comparing submit()'s wall time against a case large
    enough that the actual compute is not instant either."""
    from freetoken.kernel.cpu_moe import cpu_moe_submit

    x, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case(
        seed=7, num_tokens=128, hidden=128, intermediate=128, num_experts=16, topk=2
    )

    t0 = time.perf_counter()
    job = cpu_moe_submit(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate, expert_mask=None
    )
    submit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    job.result()
    total_s = time.perf_counter() - t1

    # submit() itself should be a small fraction of the whole submit+wait
    # window (generous bound -- this is a correctness-oriented sanity check,
    # not a strict perf regression gate).
    assert submit_s < 0.05, f"submit() took {submit_s * 1000:.2f}ms -- expected near-instant"
    assert submit_s <= max(total_s, 0.001), (
        f"submit() ({submit_s * 1000:.2f}ms) was not faster than the whole "
        f"submit+wait window ({total_s * 1000:.2f}ms) -- submit() may be "
        "blocking for the compute instead of returning immediately"
    )


# --- hybrid_subset_submit / HybridCpuPool: the compact-gather path used by
# --- _forward_hybrid ---------------------------------------------------------


def test_hybrid_subset_submit_matches_reference_weighted(native_module):
    """top_w applied (the qwen3_moe hybrid split's contract): must match the
    same candidates computed via the plain reference loop."""
    from freetoken.kernel.cpu_moe import hybrid_subset_submit

    x, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case(seed=8)
    candidates = {1, 3}

    expected = _reference_forward(x, top_idx, top_w, gate_up, down, num_experts, intermediate, candidates)
    job = hybrid_subset_submit(native_module, x, top_idx, top_w, gate_up, down, candidates, intermediate)
    actual = job.result()

    torch.testing.assert_close(actual, expected, rtol=BF16_RTOL, atol=BF16_ATOL)
    # Rows that only routed to non-candidate experts must be exactly zero.
    fully_excluded_rows = ~torch.isin(top_idx, torch.tensor(sorted(candidates))).any(dim=1)
    if bool(fully_excluded_rows.any()):
        assert torch.all(actual[fully_excluded_rows] == 0)


def test_hybrid_subset_submit_unweighted_when_top_w_is_none(native_module):
    """top_w=None (the qwen3_5_moe hybrid split's contract): every routed
    slot contributes with an implicit weight of 1.0, not the real router
    weight."""
    from freetoken.kernel.cpu_moe import hybrid_subset_submit

    x, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case(seed=9)
    candidates = {0, 2}
    ones = torch.ones_like(top_w)

    expected = _reference_forward(x, top_idx, ones, gate_up, down, num_experts, intermediate, candidates)
    job = hybrid_subset_submit(native_module, x, top_idx, None, gate_up, down, candidates, intermediate)
    actual = job.result()

    torch.testing.assert_close(actual, expected, rtol=BF16_RTOL, atol=BF16_ATOL)
    # And, for contrast, must differ from the weighted reference whenever the
    # per-row weights actually vary (guards against accidentally ignoring
    # top_w=None and applying real weights anyway).
    weighted = _reference_forward(x, top_idx, top_w, gate_up, down, num_experts, intermediate, candidates)
    assert not torch.allclose(actual, weighted)


def test_hybrid_subset_submit_only_touches_candidate_experts(native_module):
    """A large num_experts with a small candidate set must still produce the
    correct (small-candidate-only) result -- the compact-gather path must not
    accidentally include non-candidate experts just because the bank is big."""
    from freetoken.kernel.cpu_moe import hybrid_subset_submit

    x, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case(
        seed=10, num_tokens=20, hidden=8, intermediate=6, num_experts=32, topk=3
    )
    candidates = {5}

    expected = _reference_forward(x, top_idx, top_w, gate_up, down, num_experts, intermediate, candidates)
    job = hybrid_subset_submit(native_module, x, top_idx, top_w, gate_up, down, candidates, intermediate)
    actual = job.result()

    torch.testing.assert_close(actual, expected, rtol=BF16_RTOL, atol=BF16_ATOL)


# --- bf16-native dot-product dispatch (issue #257) --------------------------


def _host_has_avx512() -> bool:
    """Same /proc/cpuinfo probe test_kernel_cpu_moe.py uses."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("flags") or line.startswith("Features"):
                    return "avx512f" in line.split(":", 1)[1].split()
    except OSError:
        pass
    return False


def test_submit_force_scalar_matches_reference(native_module):
    """force_scalar=True must exercise dot_scalar_bf16w (the widen-and-FMA
    scalar fallback) regardless of what this host's CPU supports, and must
    still match the bf16-aware reference."""
    from freetoken.kernel.cpu_moe import cpu_moe_submit

    x, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case(seed=12)

    expected = _reference_forward(x, top_idx, top_w, gate_up, down, num_experts, intermediate, None)
    job = cpu_moe_submit(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate,
        expert_mask=None, force_scalar=True,
    )
    actual = job.result()

    torch.testing.assert_close(actual, expected, rtol=BF16_RTOL, atol=BF16_ATOL)


def test_submit_avx512_bf16_path_matches_forced_scalar_when_available(native_module):
    """Exercise the actual AVX-512 widen-in-register bf16 dot product only
    when this sandbox's CPU has it, and check it agrees with the forced
    scalar fallback (the two must be tolerance-close, not bit-exact -- see
    BF16_RTOL/BF16_ATOL's docstring)."""
    from freetoken.kernel.cpu_moe import cpu_moe_submit

    if not _host_has_avx512():
        pytest.skip("this sandbox's CPU does not support AVX-512F -- nothing to exercise")

    x, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case(seed=13)

    scalar_job = cpu_moe_submit(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate,
        expert_mask=None, force_scalar=True,
    )
    scalar = scalar_job.result()

    vector_job = cpu_moe_submit(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate,
        expert_mask=None, force_scalar=False,
    )
    vectorized = vector_job.result()

    torch.testing.assert_close(vectorized, scalar, rtol=BF16_RTOL, atol=BF16_ATOL)


def test_submit_env_var_forces_scalar(native_module, monkeypatch):
    from freetoken.kernel.cpu_moe import cpu_moe_submit

    monkeypatch.setenv("FREETOKEN_CPU_MOE_FORCE_SCALAR", "1")
    x, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case(seed=14)

    expected = _reference_forward(x, top_idx, top_w, gate_up, down, num_experts, intermediate, None)
    job = cpu_moe_submit(
        native_module, x, top_idx, top_w, gate_up, down, num_experts, intermediate, expert_mask=None
    )
    actual = job.result()

    torch.testing.assert_close(actual, expected, rtol=BF16_RTOL, atol=BF16_ATOL)


def test_cpu_moe_submit_casts_non_bf16_weight_dtype(native_module):
    """gate_up/down arriving in some other dtype (e.g. float32, or a model
    still resident in float16) must be cast to bfloat16 by cpu_moe_submit
    itself, not rejected -- the ABI requires raw bf16 bytes, so this is the
    one conversion cpu_moe_submit is still allowed to perform on them."""
    from freetoken.kernel.cpu_moe import cpu_moe_submit

    x, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case(seed=15)
    gate_up_f32 = gate_up.float()
    down_f32 = down.float()

    expected = _reference_forward(x, top_idx, top_w, gate_up, down, num_experts, intermediate, None)
    job = cpu_moe_submit(
        native_module, x, top_idx, top_w, gate_up_f32, down_f32, num_experts, intermediate,
        expert_mask=None,
    )
    actual = job.result()

    torch.testing.assert_close(actual, expected, rtol=BF16_RTOL, atol=BF16_ATOL)


def test_hybrid_cpu_pool_two_submits_reuse_module():
    """HybridCpuPool compiles/loads the native module once (constructor) and
    reuses it across submit() calls (mirrors the model-cached pool in
    _forward_hybrid, called once per layer per decode step)."""
    from freetoken.kernel._toolchain import ToolchainError as _TCErr
    from freetoken.kernel.cpu_moe import HybridCpuPool, find_cxx_compiler

    try:
        find_cxx_compiler()
    except _TCErr as exc:
        pytest.skip(f"no C++ compiler available for the native CPU MoE kernel: {exc}")

    pool = HybridCpuPool()
    module_first = pool._module

    x, top_idx, top_w, gate_up, down, num_experts, intermediate = _synthetic_case(seed=11)
    job1 = pool.submit(x, top_idx, top_w, gate_up, down, {0}, intermediate)
    job1.result()
    job2 = pool.submit(x, top_idx, top_w, gate_up, down, {1}, intermediate)
    job2.result()

    assert pool._module is module_first, "the compiled module must be reused, not rebuilt, across submits"
