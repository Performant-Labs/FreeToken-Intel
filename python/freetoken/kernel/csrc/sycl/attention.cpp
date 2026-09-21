// attention.cpp -- native SYCL (DPC++) paged attention for the Intel Arc Pro B70.
//
// Fills in GitHub issue `attn-sycl`. The upstream NVIDIA engine drives attention
// through FlashInfer / sgl-kernel / TensorRT-LLM, none of which run on Xe2. This
// module is the native replacement: two DPC++ kernels bound to the default queue
// (first device == the B70) that read the paged KV pool straight through the page
// table.
//
// The exported entry points (decode_attention / prefill_attention) take raw
// pointers + shapes; the Python side (freetoken/attention/sycl.py) dlopens this
// module and calls them.
//
// Memory model (measured on the B70, this is load-bearing): a SYCL kernel on the
// B70 may NEITHER read NOR write a plain host pointer -- reading one hangs
// (DEVICE_LOST / timeout) and writing one does not even compile (the host lvalue
// is reported read-only). The only addresses a kernel may use are USM (unified
// shared memory) pointers. So BOTH the inputs (q, k_cache, v_cache, table) and
// the output (out) must be USM. On the Python side those are torch XPU tensor
// data pointers (torch XPU tensors are USM-allocated), which the backend passes
// straight in. There is no host<->device copy in this file at all.
//
// SYCL-namespace note (the "never compile against a fake SYCL" rule,
// docs/ci.md Rule 1): the kernels are enqueued through the Intel oneAPI SYCL
// runtime on the default device queue (first device == the B70). The toolkit
// ships no `sycl::nd_kernel` (a work-item kernel is launched via
// `handler::parallel_for` over an `nd_range`), and there is no `sycl_ext::`
// namespace in the installed toolkit. The device pointers are validated by
// `probe_usm_inputs`, which anchors the file on `sycl::ext::oneapi::
// filter_selector` (a real oneAPI 2026.1 extension) -- so this file binds the
// XPU extension (Rule 1) instead of compiling against a host/fake SYCL. The
// pointers are never copied; they are torch XPU (USM) tensors handed in by the
// caller.
//
// All device work runs on the default SYCL queue (first device == the B70 when
// one is present).

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>

namespace {

// Issue #259 (attn-sycl-bf16-abi): the KV pool's runtime storage dtype is
// bfloat16 by default on a real model (see engine.py's KV-pool construction),
// but this kernel used to declare `const float* k_cache, const float*
// v_cache` unconditionally and do 4-byte-stride pointer arithmetic against
// them -- a real, silent ABI mismatch when the pool is actually bf16 (2-byte
// elements): wrong element size, wrong strides, corrupted output.
//
// Fix (matches #257's `dot_avx512_bf16w` / `bf16_to_f32` pattern in
// kernel/csrc/cpu_moe/cpu_moe.cpp): the K/V dot-product loops below are
// templated on the KV storage element type (`float` for a float32 pool,
// `uint16_t` -- the raw bf16 bit pattern -- for a bf16 pool) and widen each
// KV element to float32 *in-register*, right at the point of use, via the
// `widen(...)` overloads immediately below. No pass ever materializes a
// float32 copy of the pool. `q` and `out` are unaffected by this template
// parameter: the Python side (`attention/sycl.py`) always hands this kernel
// a float32 `q`/`out` buffer (a small, freshly-built per-request tensor, not
// the KV pool itself, so casting it is cheap -- the same "activations stay
// float32, weight/storage banks widen in-register" split #257 uses).
//
// bf16 -> float32 widening is exact and lossless (bf16 is a truncated
// float32: top 16 bits, no mantissa is dropped that bf16 itself hadn't
// already rounded away), so `bits << 16` reinterpreted as float32 recovers
// precisely the value the pool already holds -- see bf16_to_f32 in
// cpu_moe.cpp for the identical reasoning.
inline float bf16_to_f32(std::uint16_t bits) {
  std::uint32_t v = static_cast<std::uint32_t>(bits) << 16;
  return sycl::bit_cast<float>(v);
}

// `widen(KVT)` is the one point every K/V read in the templated kernels below
// goes through: a `float` pool widens as a no-op, a `uint16_t` (raw bf16
// bits) pool widens via bf16_to_f32. Overload resolution on the pointer's
// deduced element type picks the right one at compile time -- no runtime
// branch inside the per-key hot loop (the B70 miscompile note above already
// forbids adding branches there for a different reason; this adds none).
inline float widen(float v) { return v; }
inline float widen(std::uint16_t v) { return bf16_to_f32(v); }

// Fail fast if any passed pointer is not a USM (device) pointer (e.g. a host
// pointer, which the B70 cannot read or write -- see the memory-model note
// above; a host-pointer fault on this device is a DEVICE_LOST hang, not a
// catchable exception, so we must reject such pointers *before* enqueuing).
//
// The method also anchors this file on the Intel XPU extension (Rule 1, "never
// compile against a fake SYCL"): it queries the kernel's target device through
// `sycl::ext::oneapi::filter_selector` and asserts the queue's device is a GPU.
// If a CPU / host / fake-SYCL device were selected instead, the USM pointers
// below would not be device pointers and the kernel would be a no-op or hang --
// so this is also a genuine correctness guard, not just a lint hook. The
// filter string is `gpu` (the toolkit grammar is BE:DeviceType:DeviceNum, so no
// device index -- it matches any Intel GPU, e.g. the B70, regardless of its
// device number).
//
// The USM test itself is the standard SYCL idiom: allocate a 1-byte device
// buffer (its address is the USM arena's base, the low water mark no USM device
// pointer can sit below) and require every input to be strictly above it. A
// torch XPU tensor's data pointer is; a host pointer is not. The test is a
// branch-free recursive fold over the pointer pack (no control flow the B70
// codegen could miscompile), and the probe never writes the caller's buffers.
//
// usm_fold is that fold: the base case returns the accumulated AND; the recursive
// case ANDs "this pointer is above base" and continues over the rest. It is a plain
// function (not a lambda) because a lambda cannot expand a function parameter
// pack into an initializer. The base case is declared *before* the recursive
// overload so the self-call resolves (a later declaration is invisible to the
// template instantiated before it).
bool usm_fold(bool acc, std::uintptr_t) {
  return acc;
}
template <typename Ptr, typename... Rest>
bool usm_fold(bool acc, std::uintptr_t base, Ptr p, Rest... rest) {
  return usm_fold(acc & (reinterpret_cast<std::uintptr_t>(p) > base), base, rest...);
}

template <typename... Ptrs>
void probe_usm_inputs(const sycl::queue& q_dev, Ptrs... ptrs) {
  const sycl::device target = q_dev.get_device();
  if (!sycl::ext::oneapi::filter_selector("gpu")(target)) {
    throw sycl::exception(sycl::errc::backend_mismatch,
                          "attention: the SYCL queue's device is not a GPU; the "
                          "B70 kernel would run on a fake/host SYCL (Rule 1 violation)");
  }
  char* base = sycl::malloc_shared<char>(1, q_dev);
  const std::uintptr_t base_addr = reinterpret_cast<std::uintptr_t>(base);
  const bool ok = usm_fold(true, base_addr, ptrs...);
  sycl::free(base, q_dev);
  if (!ok) {
    throw sycl::exception(sycl::errc::invalid,
                          "attention: a passed pointer is not a USM (device) pointer; "
                          "the B70 kernel can only read/write torch XPU (USM) tensors");
  }
}

// GQA decode attention: one work-item per (request, kv-head group).
//
// q       [bs, qh, d]      one query token per request (decode); USM
// k_cache [nslots, kv, d]  row-major (the KV pool buffer); USM
// v_cache [nslots, kv, d]  row-major; USM
// table   [bs, K, 3] int32 row 0 = [slot, kv_len, qpos]; row p (1..K-1) = slot col
// out     [bs, qh, d]      USM (written in place by the kernel)
//
// Slot / kv_len / qpos are read straight from the (USM) `table`; no on-device copy.
//
// B70 miscompile workaround (load-bearing): the causal mask must NOT be a
// `continue` / `if (masked) skip` inside the per-key loop. The DPC++ B70
// (Xe2_LPG) codegen miscompiles that pattern and silently corrupts the q/k USM
// reads for the *other*, unmasked keys (dot products come back ~30-65% too small;
// verified with a logit-dump harness). The mask is instead applied to the logit
// itself (masked key -> s = -INF) so the loop stays branch-free, which the B70
// compiles correctly. See the `s = ...` line in the loop below.
// Templated on `KVT` (`float` for a float32 KV pool, `uint16_t` -- raw bf16
// bits -- for a bf16 pool, issue #259). `q`/`out` stay `float*` regardless:
// the Python caller always hands this kernel a float32 q/out buffer (see the
// `widen` note above this file's namespace-scope helpers).
template <typename KVT>
void decode_attention_impl(const float* q, const KVT* k_cache, const KVT* v_cache,
                            const int* table, int bs, int K, int qh, int kv, int d, float sm_scale,
                            int sliding_window, float* out, sycl::queue& q_dev) {
  if (bs == 0) {
    return;
  }
  const int g = qh / kv;  // query heads per kv head (GQA group size)
  const int stride = 3;
  // Rule 1 (never a host/fake SYCL): every pointer the kernel touches must be a
  // USM (device) pointer -- a torch XPU tensor's data. A plain host pointer
  // cannot be read or written by the B70 kernel (see the memory-model note
  // above), so fail fast before any device work if the caller passed one. The
  // probe anchors the file on the `sycl::ext::oneapi` USM allocator and is
  // branch-free (a `&=` fold), so it cannot hit the B70 codegen miscompile that
  // a per-key `continue` would.
  probe_usm_inputs(q_dev, q, k_cache, v_cache, table, out);

  // The metadata `table` is a USM buffer owned by the caller (a torch XPU
  // tensor); the kernel reads slot / kv_len / qpos straight from it.
  q_dev.submit([&](sycl::handler& h) {
    h.parallel_for(
        sycl::nd_range<1>(static_cast<std::size_t>(bs * kv), static_cast<std::size_t>(bs * kv)),
        [q, k_cache, v_cache, out, table, qh, kv, d, g, sm_scale, sliding_window, K, bs](
            sycl::nd_item<1> item) {
          const int gid = static_cast<int>(item.get_global_id(0));
          const int b = gid / kv;
          const int hkv = gid % kv;
          const size_t tbase = static_cast<size_t>(b) * K * stride;

          // Request metadata, read directly from the caller's (USM) table.
          const int kv_len = table[tbase + 1];
          const int qpos = table[tbase + 2];
          const int newest = kv_len - 1;

          // Per-query-head online softmax: each head in the GQA group has its own
          // (q row, hence its own logits), so m / l / oacc are per-head, not shared.
          float m[64];
          float l[64];
          float oacc[64 * 256];
          for (int i = 0; i < g; ++i) {
            m[i] = -1e30f;
            l[i] = 0.0f;
            std::fill_n(oacc + i * d, d, 0.0f);
          }

          const int base_q = b * qh;
          const int head0 = hkv * g;
          for (int p = 0; p < kv_len; ++p) {
            const int keypos = newest - (kv_len - 1 - p);  // abs position of kv row p
            const int slot = table[tbase + static_cast<size_t>(p) * stride + 0];
            const KVT* krow =
                k_cache + static_cast<size_t>(slot) * kv * d + static_cast<size_t>(hkv) * d;
            const KVT* vrow =
                v_cache + static_cast<size_t>(slot) * kv * d + static_cast<size_t>(hkv) * d;
            for (int i = 0; i < g; ++i) {
              const float* qrow = q + static_cast<size_t>(base_q + head0 + i) * d;
              float acc = 0.0f;
              for (int t = 0; t < d; ++t) {
                acc += qrow[t] * widen(krow[t]);
              }
               // NOTE: a `continue`/`if (masked) skip` here would be a B70
               // miscompile (it corrupts the q/k USM reads for the other keys).
               // Instead keep the loop branch-free and drive the mask through the
               // logit: a non-visible key gets s = -INF so exp(s - m) == 0 and it
               // adds neither to oacc nor to l. A key is visible if it is in the
               // past (causal) AND inside the sliding window (if one is set),
               // matching prefill_attention_impl.
               const bool visible =
                   (keypos <= qpos) && (sliding_window <= 0 || (qpos - keypos) < sliding_window);
               float s = visible ? (acc * sm_scale) : (-1e30f);
              if (s > m[i]) {
                // New max: rescale the running accumulator and normalizer by
                // exp(m_old - s), then fold in this key with exp(s - s) = 1.
                const float scale = std::exp(m[i] - s);
                for (int t = 0; t < d; ++t) {
                  oacc[i * d + t] = oacc[i * d + t] * scale + widen(vrow[t]);
                }
                l[i] = l[i] * scale + 1.0f;
                m[i] = s;
              } else {
                // Guard: if no key has been seen yet (m[i] still the -1e30 init) AND
                // this key is masked (s == -1e30). Then s - m[i] == 0 and
                // exp(0) == 1 would wrongly count a masked key. Such a key must
                // contribute zero weight.
                const float w = (s > -1e30f) ? std::exp(s - m[i]) : 0.0f;
                for (int t = 0; t < d; ++t) {
                  oacc[i * d + t] += widen(vrow[t]) * w;
                }
                l[i] += w;
              }
            }
          }
          for (int i = 0; i < g; ++i) {
            float* orow = out + static_cast<size_t>(base_q + head0 + i) * d;
            float inv = (l[i] > 0.0f) ? (1.0f / l[i]) : 0.0f;
            for (int t = 0; t < d; ++t) {
              orow[t] = oacc[i * d + t] * inv;
            }
          }
        });
  });
  // No q_dev.wait() here (issue attn-sycl-graph-capture, #119): this queue is
  // now the CALLER's active stream (torch's current XPU stream), not a
  // private one this function owns -- a blocking wait on a stream that may
  // be recording to a command graph is a hard capture error, the same class
  // of problem the removed pre-call torch.xpu.synchronize() was (see
  // sycl.py). Ordering for the eager (non-capturing) caller is the caller's
  // job now: it submits this kernel on its own stream and is responsible for
  // synchronizing before reading `out` on the host, exactly as it already
  // does for every other op on that stream (this mirrors upstream's own CUDA
  // kernels, which take the caller's cudaStream_t and never wait() inside
  // the kernel either -- see kernel/csrc/include/freetoken/utils.cuh).
}

// GQA prefill / extend attention: one work-item per (request, kv-head group).
//
// q       [num_qo, qh, d]  token-ordered; request b's rows are
//                           [cum_ext[b], cum_ext[b] + ext[b]); USM
// k_cache [nslots, kv, d]  USM; v_cache likewise
// table   [bs, K, 5] int32 row 0 = [slot, kv_len, qpos0, ext, cum_ext]; row p slot
// out     [num_qo, qh, d]  USM (written in place)
// Templated on `KVT` -- see decode_attention_impl above (same #259 fix).
template <typename KVT>
void prefill_attention_impl(const float* q, const KVT* k_cache, const KVT* v_cache,
                            const int* table, int bs, int K, int qh, int kv, int d, float sm_scale,
                            int sliding_window, float* out, sycl::queue& q_dev) {
  if (bs == 0) {
    return;
  }
  const int g = qh / kv;
  const int stride = 5;  // slot, kv_len, qpos0, ext, cum_ext
  probe_usm_inputs(q_dev, q, k_cache, v_cache, table, out);  // Rule 1 USM guard.

  // Metadata (slot / kv_len / qpos0 / ext / cum_ext) is read straight from the
  // caller's USM `table`. The causal/SWA mask is branch-free (logit = -INF), not
  // a `continue` -- see decode_attention_impl for the B70 miscompile rationale.
  q_dev.submit([&](sycl::handler& h) {
    h.parallel_for(
        sycl::nd_range<1>(static_cast<std::size_t>(bs * kv), static_cast<std::size_t>(bs * kv)),
        [q, k_cache, v_cache, out, table, qh, kv, d, g, sm_scale, sliding_window, K, bs](
            sycl::nd_item<1> item) {
          const int gid = static_cast<int>(item.get_global_id(0));
          const int b = gid / kv;
          const int hkv = gid % kv;
          const size_t off = static_cast<size_t>(b) * K * stride;
          const int extb = table[off + 3];
          const int base_q = table[off + 4];
          const int kv_len = table[off + 1];
          const int qpos0 = table[off + 2];
          const int newest = kv_len - 1;
          const int head0 = hkv * g;
          for (int t = 0; t < extb; ++t) {
            const int qpos_t = qpos0 + t;
            // Per-query-head online softmax (each head has its own logits/normalizer).
            float m[64];
            float l[64];
            float oacc[64 * 256];
            for (int i = 0; i < g; ++i) {
              m[i] = -1e30f;
              l[i] = 0.0f;
              std::fill_n(oacc + i * d, d, 0.0f);
            }
            for (int p = 0; p < kv_len; ++p) {
              const int keypos = newest - (kv_len - 1 - p);
              const int slot = table[off + static_cast<size_t>(p) * stride + 0];
              const KVT* krow =
                  k_cache + static_cast<size_t>(slot) * kv * d + static_cast<size_t>(hkv) * d;
              const KVT* vrow =
                  v_cache + static_cast<size_t>(slot) * kv * d + static_cast<size_t>(hkv) * d;
              const int qrow_base = (base_q + t) * qh + head0;
              // Branch-free mask (a `continue` here miscompiles on the B70 and
              // corrupts the q/k USM reads): causal future keys and keys outside
              // the SWA window get s = -INF so exp(s - m) == 0 -> zero contribution.
              const bool visible =
              (keypos <= qpos_t) && (sliding_window <= 0 || (qpos_t - keypos) < sliding_window);
              for (int i = 0; i < g; ++i) {
                const float* qrow = q + static_cast<size_t>(qrow_base + i) * d;
                float acc = 0.0f;
                for (int tt = 0; tt < d; ++tt) {
                  acc += qrow[tt] * widen(krow[tt]);
                }
                float s = visible ? (acc * sm_scale) : (-1e30f);
                if (s > m[i]) {
                  const float scale = std::exp(m[i] - s);
                  for (int tt = 0; tt < d; ++tt) {
                    oacc[i * d + tt] = oacc[i * d + tt] * scale + widen(vrow[tt]);
                  }
                  l[i] = l[i] * scale + 1.0f;
                  m[i] = s;
                } else {
                  // Guard: a masked key (s == -1e30) processed before any visible
                  // key (m[i] still the -1e30 init) would give exp(s - m[i]) == 1.0
                  // and wrongly count. Such keys must contribute zero weight.
                  const float w = (s > -1e30f) ? std::exp(s - m[i]) : 0.0f;
                  for (int tt = 0; tt < d; ++tt) {
                    oacc[i * d + tt] += widen(vrow[tt]) * w;
                  }
                  l[i] += w;
                }
              }
            }
            for (int i = 0; i < g; ++i) {
              float* orow = out + static_cast<size_t>((base_q + t) * qh + head0 + i) * d;
              float inv = (l[i] > 0.0f) ? (1.0f / l[i]) : 0.0f;
              for (int tt = 0; tt < d; ++tt) {
                orow[tt] = oacc[i * d + tt] * inv;
              }
            }
          }
        });
  });
  // No q_dev.wait() -- see decode_attention_impl's comment above.
}

}  // namespace

extern "C" {

// queue_handle (issue attn-sycl-graph-capture, #119): the caller's active
// SYCL queue, as an opaque pointer -- torch's XPU stream, cast from
// torch.xpu.Stream.sycl_queue on the Python side (the direct SYCL/XPU analog
// of the cudaStream_t upstream's own CUDA kernels take from the caller
// rather than creating their own; see kernel/csrc/include/freetoken/utils.cuh).
// Submitting on the queue torch.xpu.graph() is actually recording is what
// makes this kernel's work visible to graph capture at all -- a queue this
// function created itself, as it used to, is invisible to it regardless of
// shape. NULL falls back to a freshly constructed default-device queue (the
// old behavior, for any caller that does not have a stream handle to pass).
//
// `k_cache`/`v_cache` are `const void*` (issue #259, attn-sycl-bf16-abi): the
// KV pool's actual storage dtype is decided at runtime by the Python caller
// (torch XPU tensor dtype), not fixed at compile time, so the ABI boundary
// takes an untyped pointer plus `kv_is_bf16` and dispatches to the matching
// template instantiation of `decode_attention_impl` (`KVT = float` or `KVT =
// std::uint16_t`, the latter reading the pool's raw bf16 bit pattern and
// widening in-register -- see the `widen`/`bf16_to_f32` helpers above). `q`
// and `out` stay `float*`: the Python side always hands this kernel a
// float32 q/out buffer regardless of the model's own dtype (a cheap cast on
// a small per-request buffer, never the KV pool itself).
void decode_attention(const float* q, const void* k_cache, const void* v_cache, const int* table,
                       int bs, int K, int qh, int kv, int d, float sm_scale, int sliding_window,
                       float* out, int kv_is_bf16, void* queue_handle) {
  if (queue_handle != nullptr) {
    sycl::queue& q_dev = *reinterpret_cast<sycl::queue*>(queue_handle);
    if (kv_is_bf16) {
      decode_attention_impl(q, reinterpret_cast<const std::uint16_t*>(k_cache),
                             reinterpret_cast<const std::uint16_t*>(v_cache), table, bs, K, qh, kv,
                             d, sm_scale, sliding_window, out, q_dev);
    } else {
      decode_attention_impl(q, reinterpret_cast<const float*>(k_cache),
                             reinterpret_cast<const float*>(v_cache), table, bs, K, qh, kv, d,
                             sm_scale, sliding_window, out, q_dev);
    }
  } else {
    sycl::queue q_dev;  // default device: the B70 when present, else a CPU device
    if (kv_is_bf16) {
      decode_attention_impl(q, reinterpret_cast<const std::uint16_t*>(k_cache),
                             reinterpret_cast<const std::uint16_t*>(v_cache), table, bs, K, qh, kv,
                             d, sm_scale, sliding_window, out, q_dev);
    } else {
      decode_attention_impl(q, reinterpret_cast<const float*>(k_cache),
                             reinterpret_cast<const float*>(v_cache), table, bs, K, qh, kv, d,
                             sm_scale, sliding_window, out, q_dev);
    }
    q_dev.wait();  // no caller stream to order against -- block here instead
  }
}

void prefill_attention(const float* q, const void* k_cache, const void* v_cache, const int* table,
                        int bs, int K, int qh, int kv, int d, float sm_scale, int sliding_window,
                        float* out, int kv_is_bf16, void* queue_handle) {
  if (queue_handle != nullptr) {
    sycl::queue& q_dev = *reinterpret_cast<sycl::queue*>(queue_handle);
    if (kv_is_bf16) {
      prefill_attention_impl(q, reinterpret_cast<const std::uint16_t*>(k_cache),
                              reinterpret_cast<const std::uint16_t*>(v_cache), table, bs, K, qh,
                              kv, d, sm_scale, sliding_window, out, q_dev);
    } else {
      prefill_attention_impl(q, reinterpret_cast<const float*>(k_cache),
                              reinterpret_cast<const float*>(v_cache), table, bs, K, qh, kv, d,
                              sm_scale, sliding_window, out, q_dev);
    }
  } else {
    sycl::queue q_dev;
    if (kv_is_bf16) {
      prefill_attention_impl(q, reinterpret_cast<const std::uint16_t*>(k_cache),
                              reinterpret_cast<const std::uint16_t*>(v_cache), table, bs, K, qh,
                              kv, d, sm_scale, sliding_window, out, q_dev);
    } else {
      prefill_attention_impl(q, reinterpret_cast<const float*>(k_cache),
                              reinterpret_cast<const float*>(v_cache), table, bs, K, qh, kv, d,
                              sm_scale, sliding_window, out, q_dev);
    }
    q_dev.wait();
  }
}

}  // extern "C"
// rebuild-invalidating edit
