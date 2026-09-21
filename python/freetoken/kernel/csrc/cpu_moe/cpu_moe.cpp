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
#include <vector>

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

}  // extern "C"
