// CPU MoE expert GEMM -- naive, single-threaded, correctness-first port.
//
// Issue #250 (child of the native-CPU-MoE epic #249, parent #248). Replaces
// the one-line `placeholder.cpp` stub with a real C++ port of the two
// reference implementations that must stay numerically consistent:
//
//   - python/freetoken/moe/cpu_executor.py:CpuMoeExecutor.forward  (the
//     `cpu` backend: every expert is a candidate).
//   - python/freetoken/models/qwen3_moe/__init__.py:_Qwen3MoE._cpu_subset_math
//     (the `hybrid` split's CPU half: only a *subset* of experts, named by a
//     caller-supplied mask, is a candidate -- the rest are the XPU half's
//     job).
//
// Both call sites share one accumulation rule this port preserves exactly:
// for each expert e in ascending id order, for each top-k column j in
// ascending order, every token routed to (e, j) is processed and its
// SwiGLU output down(silu(gate(x)) * up(x)) is scattered (added) into that
// token's output row. This "expert-major then top-k-column" order is
// load-bearing: it is what keeps this path's float32 rounding identical to
// the in-VRAM / offload reference paths that use the same nested order (see
// the comments in both Python sources above). A different loop order (e.g.
// token-major) would still be "correct" in the mathematical sense but would
// not reproduce the same float32 rounding, which is the actual acceptance
// bar here.
//
// This file is deliberately naive: no threading (that is issue #248's job,
// built on top of this), no SIMD (issue #252's job). One core routine
// (`moe_forward_core`) serves both use cases via an optional `expert_mask`:
// null means "every expert is a candidate" (the plain `cpu` backend),
// non-null restricts the loop to the experts flagged as candidates (the
// hybrid split's CPU half) -- so both Python call sites can eventually be
// backed by this same function without two near-duplicate C++ routines.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define FREETOKEN_CPU_MOE_X86 1
#endif

namespace {

inline float silu(float v) {
    return v / (1.0f + std::exp(-v));
}

// Shared core: computes, for every token t and top-k column j where
// expert_ids[t, j] is a candidate expert (per expert_mask), the SwiGLU
// expert output and accumulates expert_weights[t, j] * y into out[t, :].
//
// Buffer layouts (all row-major, float32):
//   x            [num_tokens, hidden]
//   expert_ids   [num_tokens, topk]       (int32 expert id per routed slot)
//   expert_weights [num_tokens, topk]     (router weight per routed slot)
//   gate_up      [num_experts, 2*intermediate, hidden]
//                    row e packs the gate projection [intermediate, hidden]
//                    (first `intermediate` rows) then the up projection
//                    [intermediate, hidden] (next `intermediate` rows) --
//                    both in [out, in] (weight) orientation, matching the
//                    pinned host bank layout the Python executors read.
//   down         [num_experts, hidden, intermediate]  ([out, in] orientation)
//   expert_mask  [num_experts] or nullptr -- nonzero means "this expert is a
//                    candidate here" (nullptr == every expert is)
//   out          [num_tokens, hidden] -- zeroed by this function, then
//                    accumulated into (never assumes the caller pre-zeroed it)
void moe_forward_core(
    const float* x, int num_tokens, int hidden,
    const int32_t* expert_ids, const float* expert_weights, int topk,
    const float* gate_up, const float* down,
    int num_experts, int intermediate,
    const uint8_t* expert_mask,
    float* out) {
    const size_t H = static_cast<size_t>(hidden);
    const size_t I = static_cast<size_t>(intermediate);
    const size_t gate_up_expert_stride = 2 * I * H;
    const size_t down_expert_stride = H * I;

    std::memset(out, 0, sizeof(float) * static_cast<size_t>(num_tokens) * H);

    // Scratch reused across (expert, column, token) iterations -- naive and
    // single-threaded, so one buffer is safe and avoids an allocation per
    // token.
    std::vector<float> gate(I);
    std::vector<float> up(I);
    std::vector<float> act(I);  // silu(gate) * up, hoisted out of the H-wide output loop

    for (int e = 0; e < num_experts; ++e) {
        if (expert_mask != nullptr && expert_mask[e] == 0) {
            continue;
        }
        const float* gu_e = gate_up + static_cast<size_t>(e) * gate_up_expert_stride;
        const float* gate_w = gu_e;               // [intermediate, hidden]
        const float* up_w = gu_e + I * H;          // [intermediate, hidden]
        const float* down_e = down + static_cast<size_t>(e) * down_expert_stride; // [hidden, intermediate]

        for (int j = 0; j < topk; ++j) {
            for (int t = 0; t < num_tokens; ++t) {
                if (expert_ids[static_cast<size_t>(t) * topk + j] != e) {
                    continue;
                }
                const float* x_row = x + static_cast<size_t>(t) * H;

                // gate = x_row @ gate_w.T ; up = x_row @ up_w.T
                for (size_t i = 0; i < I; ++i) {
                    const float* gw_row = gate_w + i * H;
                    float acc = 0.0f;
                    for (size_t h = 0; h < H; ++h) {
                        acc += x_row[h] * gw_row[h];
                    }
                    gate[i] = acc;
                }
                for (size_t i = 0; i < I; ++i) {
                    const float* uw_row = up_w + i * H;
                    float acc = 0.0f;
                    for (size_t h = 0; h < H; ++h) {
                        acc += x_row[h] * uw_row[h];
                    }
                    up[i] = acc;
                }

                // silu(gate[i]) * up[i] does not depend on the output index o,
                // so it must be computed once per i, not recomputed inside the
                // H-wide output loop below (that recomputation -- including a
                // std::exp() per iteration -- was PR #254's own review finding:
                // O(I*H) exp() calls instead of O(I), dominating this already
                // scalar/unvectorized GEMM's cost).
                for (size_t i = 0; i < I; ++i) {
                    act[i] = silu(gate[i]) * up[i];
                }

                const float w = expert_weights[static_cast<size_t>(t) * topk + j];
                float* out_row = out + static_cast<size_t>(t) * H;
                // y = act @ down_e.T ; out_row += w * y
                for (size_t o = 0; o < H; ++o) {
                    const float* down_row = down_e + o * I;
                    float acc = 0.0f;
                    for (size_t i = 0; i < I; ++i) {
                        acc += act[i] * down_row[i];
                    }
                    out_row[o] += w * acc;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Issue #252: runtime-dispatched vectorized (AVX-512) + thread-pooled fast
// path, additive alongside moe_forward_core above (which stays untouched as
// the guaranteed-correct, single-threaded, no-SIMD reference/fallback --
// freetoken_cpu_moe_forward, and the tests that exercise it, are unchanged).
//
// Design:
//   - Runtime CPU-feature dispatch (__builtin_cpu_supports("avx512f")), not a
//     compile-time #ifdef -- this .so must still load and run correctly on a
//     box without AVX-512 (dev boxes / CI). The AVX-512 row-compute function
//     is marked with __attribute__((target("avx512f,fma"))) so only *that*
//     function requires the ISA extension to be *emitted*; every other
//     function in this translation unit (including the scalar fallback) has
//     no such requirement and loads fine on any x86_64 host regardless of
//     what -m flags the file was compiled with. This means cxx_flags() in
//     cpu_moe.py needs no new global -mavx512f -- the target attribute is
//     the mechanism, per this issue's own suggested approach.
//   - Threading: work is partitioned across experts. Each worker thread owns
//     a contiguous slice of the expert range and accumulates into its own
//     private [num_tokens, hidden] buffer (no cross-thread writes, so no
//     locking/atomics needed); the buffers are summed into the caller's
//     `out` once all threads join. A single token's per-expert contributions
//     within one thread's slice are still accumulated in ascending (e, j)
//     order (matching moe_forward_core's contract for that thread's share of
//     experts), but the *global* order tokens' contributions land in `out`
//     is no longer strictly ascending-expert-then-ascending-column once more
//     than one thread is used (thread A's expert-7 contribution may add to a
//     row before thread B's expert-3 contribution does, if 3 and 7 landed in
//     different thread slices). float32 addition is commutative but not
//     associative, so multi-threaded runs are only guaranteed to match the
//     naive scalar path within float tolerance (rtol/atol), not bit-exact --
//     this matches the SIMD reduction's own tolerance-based guarantee (a
//     512-bit FMA accumulator also reduces in a different order than the
//     naive scalar loop). Single-threaded fast-path runs (num_threads <= 1)
//     preserve the exact ascending (e, j, t) order.
struct RowBuffers {
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> act;
    explicit RowBuffers(size_t intermediate) : gate(intermediate), up(intermediate), act(intermediate) {}
};

inline void row_compute_scalar(
    const float* x_row, const float* gate_w, const float* up_w, const float* down_e,
    size_t H, size_t I, float w, float* out_row, RowBuffers& buf) {
    for (size_t i = 0; i < I; ++i) {
        const float* gw_row = gate_w + i * H;
        float acc = 0.0f;
        for (size_t h = 0; h < H; ++h) acc += x_row[h] * gw_row[h];
        buf.gate[i] = acc;
    }
    for (size_t i = 0; i < I; ++i) {
        const float* uw_row = up_w + i * H;
        float acc = 0.0f;
        for (size_t h = 0; h < H; ++h) acc += x_row[h] * uw_row[h];
        buf.up[i] = acc;
    }
    for (size_t i = 0; i < I; ++i) buf.act[i] = silu(buf.gate[i]) * buf.up[i];
    for (size_t o = 0; o < H; ++o) {
        const float* down_row = down_e + o * I;
        float acc = 0.0f;
        for (size_t i = 0; i < I; ++i) acc += buf.act[i] * down_row[i];
        out_row[o] += w * acc;
    }
}

#if defined(FREETOKEN_CPU_MOE_X86)

bool avx512_available() {
    static const bool available = __builtin_cpu_supports("avx512f") != 0;
    return available;
}

__attribute__((target("avx512f,fma")))
inline float dot_avx512(const float* a, const float* b, size_t n) {
    __m512 acc = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

__attribute__((target("avx512f,fma")))
void row_compute_avx512(
    const float* x_row, const float* gate_w, const float* up_w, const float* down_e,
    size_t H, size_t I, float w, float* out_row, RowBuffers& buf) {
    for (size_t i = 0; i < I; ++i) buf.gate[i] = dot_avx512(x_row, gate_w + i * H, H);
    for (size_t i = 0; i < I; ++i) buf.up[i] = dot_avx512(x_row, up_w + i * H, H);
    for (size_t i = 0; i < I; ++i) buf.act[i] = silu(buf.gate[i]) * buf.up[i];
    for (size_t o = 0; o < H; ++o) out_row[o] += w * dot_avx512(buf.act.data(), down_e + o * I, I);
}

#else

bool avx512_available() { return false; }

#endif

using RowComputeFn = void (*)(const float*, const float*, const float*, const float*,
                               size_t, size_t, float, float*, RowBuffers&);

RowComputeFn select_row_compute(bool force_scalar) {
#if defined(FREETOKEN_CPU_MOE_X86)
    if (!force_scalar && avx512_available()) return row_compute_avx512;
#else
    (void)force_scalar;
#endif
    return row_compute_scalar;
}

// Runs the expert-major/top-k-column loop for experts in [expert_begin, expert_end)
// against `out` (which must already be zeroed / privately owned by the caller
// when running multi-threaded -- see moe_forward_core_fast below).
void moe_forward_expert_range(
    const float* x, int num_tokens, int hidden,
    const int32_t* expert_ids, const float* expert_weights, int topk,
    const float* gate_up, const float* down,
    int intermediate,
    const uint8_t* expert_mask,
    int expert_begin, int expert_end,
    RowComputeFn row_fn,
    float* out) {
    const size_t H = static_cast<size_t>(hidden);
    const size_t I = static_cast<size_t>(intermediate);
    const size_t gate_up_expert_stride = 2 * I * H;
    const size_t down_expert_stride = H * I;
    RowBuffers buf(I);

    for (int e = expert_begin; e < expert_end; ++e) {
        if (expert_mask != nullptr && expert_mask[e] == 0) continue;
        const float* gu_e = gate_up + static_cast<size_t>(e) * gate_up_expert_stride;
        const float* gate_w = gu_e;
        const float* up_w = gu_e + I * H;
        const float* down_e = down + static_cast<size_t>(e) * down_expert_stride;

        for (int j = 0; j < topk; ++j) {
            for (int t = 0; t < num_tokens; ++t) {
                if (expert_ids[static_cast<size_t>(t) * topk + j] != e) continue;
                const float* x_row = x + static_cast<size_t>(t) * H;
                const float w = expert_weights[static_cast<size_t>(t) * topk + j];
                float* out_row = out + static_cast<size_t>(t) * H;
                row_fn(x_row, gate_w, up_w, down_e, H, I, w, out_row, buf);
            }
        }
    }
}

// Vectorized (when available) + thread-pooled fast path. `num_threads <= 1`
// runs single-threaded with the exact same (e, j, t) ordering as
// moe_forward_core (only the row compute may be vectorized); `num_threads >
// 1` partitions the expert range across worker threads, each accumulating
// into a private buffer, summed into `out` at the end (see the design note
// above for why that is only tolerance-exact, not bit-exact).
void moe_forward_core_fast(
    const float* x, int num_tokens, int hidden,
    const int32_t* expert_ids, const float* expert_weights, int topk,
    const float* gate_up, const float* down,
    int num_experts, int intermediate,
    const uint8_t* expert_mask,
    int num_threads, int force_scalar,
    float* out) {
    const size_t H = static_cast<size_t>(hidden);
    const size_t out_elems = static_cast<size_t>(num_tokens) * H;
    std::memset(out, 0, sizeof(float) * out_elems);

    RowComputeFn row_fn = select_row_compute(force_scalar != 0);

    int threads = num_threads;
    if (threads <= 0) {
        unsigned hw = std::thread::hardware_concurrency();
        threads = hw > 0 ? static_cast<int>(hw) : 1;
    }
    if (threads > num_experts) threads = num_experts;
    if (threads < 1) threads = 1;

    if (threads <= 1) {
        moe_forward_expert_range(x, num_tokens, hidden, expert_ids, expert_weights, topk,
                                  gate_up, down, intermediate, expert_mask,
                                  0, num_experts, row_fn, out);
        return;
    }

    // Contiguous expert-range partition across threads; each thread gets its
    // own private accumulation buffer to avoid any cross-thread write to the
    // same output row.
    std::vector<std::vector<float>> partials(static_cast<size_t>(threads));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(threads));

    const int base = num_experts / threads;
    const int rem = num_experts % threads;
    int begin = 0;
    for (int p = 0; p < threads; ++p) {
        int count = base + (p < rem ? 1 : 0);
        int end = begin + count;
        partials[static_cast<size_t>(p)].assign(out_elems, 0.0f);
        float* partial_out = partials[static_cast<size_t>(p)].data();
        workers.emplace_back([=]() {
            if (begin >= end) return;
            moe_forward_expert_range(x, num_tokens, hidden, expert_ids, expert_weights, topk,
                                      gate_up, down, intermediate, expert_mask,
                                      begin, end, row_fn, partial_out);
        });
        begin = end;
    }
    for (auto& t : workers) t.join();

    for (int p = 0; p < threads; ++p) {
        const float* partial_out = partials[static_cast<size_t>(p)].data();
        for (size_t idx = 0; idx < out_elems; ++idx) out[idx] += partial_out[idx];
    }
}

}  // namespace

extern "C" {

// Synchronous entry point -- no threading/async here (issue #248's job on
// top of this). Returns 0 on success, a negative error code on a bad
// argument (null required pointer, or a non-positive dimension where one is
// not sensible).
//
//   x               [num_tokens, hidden] row-major float32
//   expert_ids      [num_tokens, topk] row-major int32 (routed expert id per slot)
//   expert_weights  [num_tokens, topk] row-major float32 (router weight per slot)
//   gate_up         [num_experts, 2*intermediate, hidden] row-major float32
//   down            [num_experts, hidden, intermediate] row-major float32
//   expert_mask     [num_experts] uint8, nonzero = candidate; NULL = all experts
//                       are candidates (the plain `cpu` backend's use case)
//   out             caller-allocated [num_tokens, hidden] row-major float32;
//                       zeroed and filled by this call
int freetoken_cpu_moe_forward(
    const float* x, int num_tokens, int hidden,
    const int32_t* expert_ids, const float* expert_weights, int topk,
    const float* gate_up, const float* down,
    int num_experts, int intermediate,
    const uint8_t* expert_mask,
    float* out) {
    if (x == nullptr || expert_ids == nullptr || expert_weights == nullptr ||
        gate_up == nullptr || down == nullptr || out == nullptr) {
        return -1;
    }
    if (num_tokens < 0 || hidden <= 0 || topk < 0 || num_experts < 0 || intermediate <= 0) {
        return -2;
    }
    if (num_tokens == 0 || topk == 0 || num_experts == 0) {
        // Nothing to route; still a well-defined "zero out and return" call.
        std::memset(out, 0, sizeof(float) * static_cast<size_t>(num_tokens) * static_cast<size_t>(hidden));
        return 0;
    }
    moe_forward_core(x, num_tokens, hidden, expert_ids, expert_weights, topk,
                      gate_up, down, num_experts, intermediate, expert_mask, out);
    return 0;
}

// Issue #252's fast path: runtime-dispatched AVX-512 (falls back to the same
// scalar math as freetoken_cpu_moe_forward above when the host lacks
// AVX-512, or when force_scalar is nonzero -- the test hook this project's
// "skip/force what the host can't exercise" testing philosophy needs) plus
// thread-pool parallelism across the expert range.
//
//   num_threads   worker thread count; <= 0 means "auto" (hardware_concurrency,
//                     clamped to num_experts so a thread never gets zero work)
//   force_scalar  nonzero forces the scalar row-compute even when AVX-512 is
//                     available (deterministic test coverage of the fallback
//                     branch on a host that does have AVX-512)
//
// Same argument/error-code contract as freetoken_cpu_moe_forward otherwise.
int freetoken_cpu_moe_forward_fast(
    const float* x, int num_tokens, int hidden,
    const int32_t* expert_ids, const float* expert_weights, int topk,
    const float* gate_up, const float* down,
    int num_experts, int intermediate,
    const uint8_t* expert_mask,
    int num_threads, int force_scalar,
    float* out) {
    if (x == nullptr || expert_ids == nullptr || expert_weights == nullptr ||
        gate_up == nullptr || down == nullptr || out == nullptr) {
        return -1;
    }
    if (num_tokens < 0 || hidden <= 0 || topk < 0 || num_experts < 0 || intermediate <= 0) {
        return -2;
    }
    if (num_tokens == 0 || topk == 0 || num_experts == 0) {
        std::memset(out, 0, sizeof(float) * static_cast<size_t>(num_tokens) * static_cast<size_t>(hidden));
        return 0;
    }
    moe_forward_core_fast(x, num_tokens, hidden, expert_ids, expert_weights, topk,
                           gate_up, down, num_experts, intermediate, expert_mask,
                           num_threads, force_scalar, out);
    return 0;
}

// Runtime capability probe for callers/tests: nonzero if this process's CPU
// supports AVX-512F (the ISA extension freetoken_cpu_moe_forward_fast's
// vectorized row-compute is gated on).
int freetoken_cpu_moe_avx512_available() {
    return avx512_available() ? 1 : 0;
}

}  // extern "C"
