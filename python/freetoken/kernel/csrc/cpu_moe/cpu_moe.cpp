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
// This file started deliberately naive (no threading, no SIMD); issue #252
// added a runtime-dispatched vectorized (AVX-512) + thread-pooled fast path
// alongside the original single-threaded scalar core, and issue #248 added a
// persistent native worker thread + ctypes submit()/wait() dispatch on top of
// that scalar core (see each section below for its own design notes). One
// core routine (`moe_forward_core`) serves both the plain `cpu` backend and
// the `hybrid` split's CPU half via an optional `expert_mask`: null means
// "every expert is a candidate", non-null restricts the loop to the experts
// flagged as candidates -- so both Python call sites are backed by this same
// function without two near-duplicate C++ routines.

#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
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

// --- Async dispatch (issue #248) -------------------------------------------
//
// Replaces the Python-side ThreadPoolExecutor.submit()/future.result()
// handoff (measured at 40-50ms/layer of pure Python-level GIL-contention /
// synchronization overhead, #247) with a persistent *native* worker thread
// that the Python side reaches via ctypes submit()/wait() calls. ctypes
// releases the GIL for the duration of a foreign function call, and this
// std::thread is spawned and owned entirely by this translation unit (never
// a Python thread) -- so, unlike ThreadPoolExecutor, it never needs to
// re-acquire the GIL to hand work off or collect a result.
//
// Design: one persistent worker thread (created lazily on the first submit,
// reused for the process's lifetime -- ~28+ MoE layers per decode step, so
// paying thread-creation cost per call would erase the benefit, exactly the
// reasoning the old ThreadPoolExecutor cache already documented) and a
// single in-flight "job slot". The call sites this serves (both
// _forward_hybrid ports) never submit a second job before waiting on the
// first -- one CPU-half job per MoE layer, submitted then waited on before
// the next layer's submit -- so a single-slot design is sufficient and
// avoids a general job-queue's extra bookkeeping. A submit() that arrives
// before the previous job's result has been collected simply blocks until
// it has (the same backpressure a max_workers=1 executor would apply), so
// the slot can never be silently double-booked or a result dropped.
//
// This dispatches to moe_forward_core (the naive scalar path from #250), not
// #252's vectorized moe_forward_core_fast above -- switching the async
// worker onto the fast path is a natural follow-up (both are additive and
// compatible; AsyncJob's fields already line up with moe_forward_core_fast's
// argument list minus num_threads/force_scalar), left for a later issue
// rather than folded into this one.
struct AsyncJob {
    const float* x = nullptr; int num_tokens = 0; int hidden = 0;
    const int32_t* expert_ids = nullptr; const float* expert_weights = nullptr; int topk = 0;
    const float* gate_up = nullptr; const float* down = nullptr;
    int num_experts = 0; int intermediate = 0;
    const uint8_t* expert_mask = nullptr;
    float* out = nullptr;
    int rc = 0;
};

int run_job(const AsyncJob& job) {
    if (job.x == nullptr || job.expert_ids == nullptr || job.expert_weights == nullptr ||
        job.gate_up == nullptr || job.down == nullptr || job.out == nullptr) {
        return -1;
    }
    if (job.num_tokens < 0 || job.hidden <= 0 || job.topk < 0 || job.num_experts < 0 ||
        job.intermediate <= 0) {
        return -2;
    }
    if (job.num_tokens == 0 || job.topk == 0 || job.num_experts == 0) {
        std::memset(job.out, 0, sizeof(float) * static_cast<size_t>(job.num_tokens) * static_cast<size_t>(job.hidden));
        return 0;
    }
    moe_forward_core(job.x, job.num_tokens, job.hidden, job.expert_ids, job.expert_weights,
                      job.topk, job.gate_up, job.down, job.num_experts, job.intermediate,
                      job.expert_mask, job.out);
    return 0;
}

class AsyncWorker {
public:
    AsyncWorker() : thread_(&AsyncWorker::run, this) {}

    // Not copyable/movable -- owns a thread and the synchronization state it
    // closes over.
    AsyncWorker(const AsyncWorker&) = delete;
    AsyncWorker& operator=(const AsyncWorker&) = delete;

    ~AsyncWorker() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            shutdown_ = true;
        }
        cv_job_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    // Claims the single job slot (blocking only if a previous job's result
    // has not yet been collected via wait() -- see class docstring) and
    // hands the job to the worker thread. Returns immediately after that --
    // the actual compute happens asynchronously on the worker thread.
    // Returns a handle >= 1.
    int64_t submit(const AsyncJob& job) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_slot_free_.wait(lk, [this] { return !occupied_; });
        job_ = job;
        job_.rc = 0;
        occupied_ = true;
        done_ = false;
        const int64_t handle = ++next_handle_;
        handle_ = handle;
        lk.unlock();
        cv_job_.notify_one();
        return handle;
    }

    // Blocks (condition_variable wait, not a busy-poll) until the job with
    // this handle has completed, then frees the slot for the next submit().
    // Returns the job's own rc (0 success, the same negative codes
    // freetoken_cpu_moe_forward uses for bad arguments), or -100 for an
    // unknown/already-collected handle.
    int wait(int64_t handle) {
        std::unique_lock<std::mutex> lk(mu_);
        if (!occupied_ || handle != handle_) {
            return -100;
        }
        cv_done_.wait(lk, [this, handle] { return done_ && handle_ == handle; });
        const int rc = job_.rc;
        occupied_ = false;
        lk.unlock();
        cv_slot_free_.notify_one();
        return rc;
    }

private:
    void run() {
        std::unique_lock<std::mutex> lk(mu_);
        for (;;) {
            cv_job_.wait(lk, [this] { return (occupied_ && !done_) || shutdown_; });
            if (shutdown_ && !(occupied_ && !done_)) {
                // Nothing pending -- safe to exit. (A shutdown that races a
                // still-pending job never happens in this module's own
                // usage: the process only tears down after every submitted
                // job has already been waited on.)
                return;
            }
            AsyncJob local = job_;
            lk.unlock();
            const int rc = run_job(local);
            lk.lock();
            job_.rc = rc;
            done_ = true;
            cv_done_.notify_all();
            // Loop back to cv_job_.wait() with lk held.
        }
    }

    // Declaration order matters here, not just initializer-list order: C++
    // constructs members in declaration order regardless of how the
    // constructor's initializer list writes them, and thread_'s constructor
    // (below) starts run() executing concurrently the instant it runs. If
    // thread_ were declared first, run() could observe mu_ (and the
    // condition variables, and job_'s own fields) before their constructors
    // have completed -- undefined behavior, a real data race, and a
    // (rare, scheduler-dependent, so easy to miss in testing) hang/crash.
    // Every piece of state run() touches must be declared -- and therefore
    // constructed -- before thread_ so it is already valid when the new
    // thread starts.
    std::mutex mu_;
    std::condition_variable cv_job_;
    std::condition_variable cv_done_;
    std::condition_variable cv_slot_free_;
    AsyncJob job_;
    bool occupied_ = false;
    bool done_ = false;
    bool shutdown_ = false;
    int64_t next_handle_ = 0;
    int64_t handle_ = 0;
    std::thread thread_;
};

// Function-local static: constructed on the first submit() (lazily, so a
// process that only ever calls the synchronous freetoken_cpu_moe_forward
// never pays for a worker thread it doesn't use), destroyed at process exit.
AsyncWorker& worker() {
    static AsyncWorker instance;
    return instance;
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

// Async entry points (issue #248) -- see the AsyncWorker class docstring
// above for the design. The caller (cpu_moe.py's cpu_moe_submit /
// CpuMoeJob.result()) must keep every buffer passed here alive until the
// matching freetoken_cpu_moe_wait() call returns: the worker thread reads
// and writes them by raw pointer, off the calling thread, for the whole
// submit-to-wait window.
//
//   Same buffer contract as freetoken_cpu_moe_forward (see above), plus:
//   returns a job handle >= 1 on success. This call never itself validates
//   the buffers/dimensions (that check runs on the worker thread and its
//   result surfaces through freetoken_cpu_moe_wait()'s return code) --
//   submit() is meant to return in the time it takes to claim the single job
//   slot and hand off a struct, not to do any real work.
int64_t freetoken_cpu_moe_submit(
    const float* x, int num_tokens, int hidden,
    const int32_t* expert_ids, const float* expert_weights, int topk,
    const float* gate_up, const float* down,
    int num_experts, int intermediate,
    const uint8_t* expert_mask,
    float* out) {
    AsyncJob job;
    job.x = x;
    job.num_tokens = num_tokens;
    job.hidden = hidden;
    job.expert_ids = expert_ids;
    job.expert_weights = expert_weights;
    job.topk = topk;
    job.gate_up = gate_up;
    job.down = down;
    job.num_experts = num_experts;
    job.intermediate = intermediate;
    job.expert_mask = expert_mask;
    job.out = out;
    return worker().submit(job);
}

// Blocks (condition_variable, not a busy-poll) until the job named by
// `handle` (from freetoken_cpu_moe_submit) has been computed by the worker
// thread, then returns its result code (0 success, the same negative codes
// freetoken_cpu_moe_forward uses for a bad argument, or -100 for an unknown
// / already-collected handle). The job's `out` buffer (passed to submit) is
// only valid to read once this call has returned 0.
int freetoken_cpu_moe_wait(int64_t handle) {
    return worker().wait(handle);
}

}  // extern "C"
