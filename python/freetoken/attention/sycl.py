"""Native SYCL attention on Xe2 (replaces FlashInfer / sgl-kernel CUDA).

Upstream NVIDIA path: python/freetoken/attention/fi.py + fa.py
Fill in: GitHub issue `attn-sycl` (see docs/architecture.md).

This is the *fast* attention backend for the Intel Arc Pro B70. Where the
reference ``triton`` backend computes GQA attention in pure torch (correct but
a Python loop over requests, with a gather per layer), this backend hands the
whole batch to a hand-written oneAPI DPC++ (SYCL) kernel -- one ``nd_range``
launch per phase that runs the full paged, grouped-query, causal (or sliding-
window) attention on the B70. It implements the same ``BaseAttnBackend``
contract, so the model's ``forward`` is unchanged.

The kernel (``csrc/sycl/attention.cpp``) is compiled with ``icpx -fsycl`` and
loaded through ``freetoken.kernel.utils`` (AOT cache + JIT fallback). All five
pointers it touches -- ``q`` / ``k_cache`` / ``v_cache`` / ``table`` / ``out`` --
are torch XPU (USM) tensors, and the kernel reads the layout note in that file
for the exact per-phase ``table`` this module must build.

Only an XPU can run it: on a CPU-only box the backend raises on first use (the
engine's ``"auto"`` resolution never picks ``sycl`` -- ``torch`` is the default).
"""
from __future__ import annotations

import ctypes
import pathlib

import torch

from freetoken.kernel import _toolchain, aot
from freetoken.kernel import utils as kernel_utils

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata

# The compiled kernel's entry points (see csrc/sycl/attention.cpp).
_KERNEL_NAME = "attention"
_KERNEL_SRC = pathlib.Path(__file__).parent.parent / "kernel" / "csrc" / "sycl" / "attention.cpp"

# (const float* q, const float* kc, const float* vc, const int* table,
#  int bs, int K, int qh, int kv, int d, float sm_scale, int sliding_window,
#  float* out, void* queue_handle)
# queue_handle (issue attn-sycl-graph-capture, #119): the caller's active SYCL
# queue (torch.xpu.Stream.sycl_queue), or NULL to fall back to a fresh
# default-device queue -- see attention.cpp's decode_attention docstring.
_FN_TYPES = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


class SyclMetadata(BaseAttnMetadata):
    """Per-phase USM ``table`` tensors the SYCL kernel reads (built on the CPU).

    A ``table`` is a torch XPU int32 tensor (a USM pointer the kernel can read --
    see the memory-model note in ``attention.cpp``). Each request ``b`` owns a
    block of ``K`` rows; the key slot for attended position ``p`` (``0 <= p <
    kv_len``) is read from row ``b*K + p``, column 0, so the ``kv_len`` key slots
    occupy rows ``[0, kv_len)`` -- *including row 0*, whose other columns carry
    the per-request metadata (``kv_len``, ``qpos``[/``ext``]). The stride differs
    by phase (decode ``[bs, K, 3]`` vs prefill ``[bs, K, 5]``), so the two phases
    carry *separate* tables and are launched separately -- a batch may mix
    prefills and decodes (the engine sets ``batch.phase`` per the "any prefill?"
    rule), so each request is routed to the kernel that matches its own phase.
    """

    def __init__(self, decode: torch.Tensor | None, prefill: torch.Tensor | None) -> None:
        self.decode = decode  # [bs_dec, K, 3] USM int32 (or None if no decode reqs)
        self.prefill = prefill  # [bs_pre, K, 5] USM int32 (or None if no prefill reqs)

    def get_last_indices(self, bs: int) -> torch.Tensor:
        # Vestigial interface method: the model's per-layer forward consumes the
        # tables via the global context, never via this accessor. For the decode
        # table, row 0 col 0 is the *first* key slot and row 0 col 2 is the query
        # position (the row-0 layout is [slot, kv_len, qpos]); the "last index" a
        # decode request attends up to is its query position, so return col 2.
        if self.decode is None:
            return torch.empty((bs,), dtype=torch.int32)
        return self.decode[:, 0, 2][:bs]


def _xpu_available() -> bool:
    try:
        return bool(getattr(torch, "xpu", None) and torch.xpu.is_available())
    except Exception:
        return False


def _get_ctx():
    from freetoken.core import get_global_ctx

    return get_global_ctx()


class SyclAttentionBackend(BaseAttnBackend):
    """Paged grouped-query attention on the B70, driven by a native SYCL kernel.

    ``forward(q, k, v, layer_id, batch, attn_spec)`` takes ``q`` head-major
    ``[num_q_heads, num_tokens, head_dim]`` and ``k`` / ``v`` head-major
    ``[num_kv_heads, num_tokens, head_dim]`` (the *new* tokens this step; one per
    request in decode, the whole prompt in prefill). It builds the USM metadata
    tables from the batch + page table, transposes the queries to the token-major
    layout the kernel expects, routes each request to the kernel matching its own
    phase (a batch may mix
    prefills and decodes), and calls the compiled ``decode_attention`` /
    ``prefill_attention`` entry point through ctypes. The K/V history is read
    from the pool via the identity page table (slot ``pos`` holds the token at
    position ``pos``), exactly as the reference backend does -- the kernel just
    does the attention math on-device instead of in torch.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device("xpu" if _xpu_available() else "cpu")
        self.capture = None
        self.capture_bs: list[int] = []
        self.max_graph_bs = 0
        # Issue attn-sycl-graph-capture (#119): forward() now submits the
        # kernel on the CALLER's active SYCL queue (torch's current XPU
        # stream) rather than one this backend creates itself, so
        # torch.xpu.graph() can actually see and record the launch. While
        # _capturing is armed, forward() also skips the pre/post
        # torch.xpu.synchronize() calls it otherwise does for host-visible
        # correctness -- a host sync during capture is a hard error (the
        # error this issue exists to fix: "wait cannot be called for a queue
        # which is recording to a command graph").
        self._capturing = False
        self._decode = None
        self._prefill = None
        self._kv_num_slots = self._resolve_num_slots(config)

    # -- lazy kernel load ----------------------------------------------------

    def _resolve_num_slots(self, config) -> int:
        """The KV pool's slot count (``num_pages * page_size``).

        The engine attaches the pool to the global context ahead of building the
        backend, so read it from there; fall back to deriving it from the engine
        config the way the pool is sized (``num_page_override`` or
        ``max_running_req * max_seq_len``).

        Both lookups are best-effort: the constructor must stay lazy-safe (see
        ``_ensure_loaded``'s docstring) even when ``config`` is a minimal / mock
        object with none of the expected fields (a real ``EngineConfig`` always
        has them; only a deliberately-bare object, as the "constructor is safe"
        test uses, does not). 0 here just means "not yet known" -- real usage
        always has a real config or an attached pool by the time a kernel
        actually runs, in ``_ensure_loaded`` / the forward methods below.
        """
        try:
            pool = getattr(_get_ctx(), "kv_cache", None)
            if pool is not None and hasattr(pool, "num_slots"):
                return int(pool.num_slots)
        except Exception:
            pass
        try:
            num_pages = config.num_page_override or (config.max_running_req * config.max_seq_len)
            return int(num_pages) * int(config.page_size)
        except AttributeError:
            return 0

    def _ensure_loaded(self) -> None:
        """Compile (or load from the AOT cache) attention.cpp and bind entry points.

        Done lazily on first use: the module imports cleanly on a CPU-only box
        (it only touches torch.xpu when actually running a kernel), but the
        compile needs the oneAPI toolchain + an XPU, so it must not run in the
        constructor on a box that has neither.
        """
        if self._decode is not None:
            return
        if not _xpu_available():
            raise RuntimeError(
                "the SYCL attention backend requires a torch XPU device; "
                "use the 'torch' (reference) backend on a CPU-only box"
            )
        if not _KERNEL_SRC.is_file():
            raise _toolchain.ToolchainError(f"attention kernel source not found at {_KERNEL_SRC}")
        # The existence check and the build must use the SAME cache key, else a
        # miss-check against one key would dlopen a path the build never wrote
        # (two historically-diverged keys: the JIT _build_key folds in the source
        # bytes, the AOT cache_key did not -- a source edit then left the
        # miss-check pointing at an empty dir while the build wrote elsewhere).
        # Both now use aot.cache_key(name, source), which folds in the source.
        cache_dir = kernel_utils._jit_cache_dir()
        so_path = cache_dir / aot.cache_key(_KERNEL_NAME, str(_KERNEL_SRC)) / f"{_KERNEL_NAME}.so"
        if not so_path.is_file():
            aot.build_aot_cache(_KERNEL_NAME, str(_KERNEL_SRC), str(cache_dir))
        self._module = kernel_utils.KernelModule(path=so_path, loaded=kernel_utils._load(so_path), from_cache=False)
        decode = self._module.loaded.decode_attention
        decode.argtypes = _FN_TYPES
        decode.restype = None
        prefill = self._module.loaded.prefill_attention
        prefill.argtypes = _FN_TYPES
        prefill.restype = None
        self._decode = decode
        self._prefill = prefill

    def _ptr(self, tensor: torch.Tensor) -> ctypes.c_void_p:
        return ctypes.c_void_p(tensor.data_ptr())

    # -- BaseAttnBackend interface -------------------------------------------

    def prepare_metadata(self, batch) -> None:
        self.metadata = self._build_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list) -> None:
        self.max_graph_bs = max(bs_list) if bs_list else 0

    def prepare_for_capture(self, batch) -> None:
        self._capturing = True

    def reset_capture(self) -> None:
        super().reset_capture()
        self._capturing = False

    def prepare_for_replay(self, batch) -> None:
        # Nothing to rebind: forward() rebuilds the tables from the live batch.
        return None

    # -- metadata + forward ---------------------------------------------------

    def _build_metadata(self, batch) -> SyclMetadata:
        """Build the USM metadata tables from the batch + page table.

        The model drives this backend **per request** (the decoder layer runs
        each request's layers on its own hidden slice, so ``forward`` receives
        one request's q/k/v at a time and passes that request's ``table_idx``).
        The tables carry **one row per request, in batch order** (row ``b`` ==
        ``batch.reqs[b]``); ``forward`` selects the row matching the current
        ``table_idx`` and launches the kernel with ``bs=1`` over that row.
        Requests are classified by their *own* phase (extend_len 1 -> decode,
        else prefill) rather than the batch-level flag (a step may mix phases).
        Row 0 of each request carries its slot / kv_len / qpos (and ext / cum_ext
        for prefill); the remaining rows carry the key slot index for each
        attended position from the pool's identity page table.
        """
        device = self.device
        pool = _get_ctx().kv_cache
        K = self._kv_num_slots

        # Per-request phase: a request is in *decode* once a token has been
        # generated (device_len > len(input_ids)); its first step is *prefill*.
        # This is the SAME signal the engine's step() and the model's forward use
        # (NOT ``req.extend_len == 1`` -- extend_len is device_len - cached_len
        # and grows every step, so it is 1 only on a degenerate first decode).
        # The per-request new-token count comes from ``batch.extend_lens`` (the
        # engine's authoritative vector: prompt_len in prefill, 1 in decode).
        # Per-request phase: ``batch.phase`` (the scheduler's authoritative,
        # always-correct flag -- every other backend in this codebase trusts it
        # uniformly, e.g. TritonAttentionBackend / _forward_offload). The
        # previous per-request re-derivation (``req.device_len !=
        # len(req.input_ids)``) miscompares equal on the FIRST decode step
        # following a full prompt prefill: the engine's post-step bookkeeping
        # appends the sampled token to input_ids on a prefill step (input_ids
        # grows to prompt_len+1) and bumps device_len to prompt_len+1 in the
        # same step, so the two coincidentally match going into the very next
        # step and that step is misclassified as ANOTHER prefill (kv_len=1
        # instead of the real history length) -- corrupting decode attention
        # from the second generated token onward. Confirmed by tracing the
        # real per-step tables: this file's own docstrings elsewhere in this
        # module warn against exactly this class of shape-based phase check.
        is_decode_batch = batch.phase == "decode"
        dec_idx: list[int] = []
        pre_idx: list[int] = []
        for b, req in enumerate(batch.reqs):
            (dec_idx if is_decode_batch else pre_idx).append(b)
        bs_dec, bs_pre = len(dec_idx), len(pre_idx)

        # ``positions`` / ``out_loc`` are token-indexed: one entry per *new* token,
        # flattened across requests in request order. Request ``b``'s new tokens
        # occupy the *global* token range [gbase[b], gbase[b] + ext_b). Under the
        # identity page table the new token's out_loc slot == its absolute position
        # == gbase[b] + j, so deriving the query position / history length from
        # gbase (NOT from out_loc[i] indexed by the request's position in its own
        # phase list -- that only coincides with the token offset in a pure
        # single-phase batch) keeps the metadata correct for mixed-phase batches.
        # Per-request new-token counts: the engine's authoritative vector
        # (prompt_len in prefill, 1 in decode), else derive from request state.
        # Flatten to plain ints (extend_lens may be a tensor) so downstream
        # indexing / arithmetic is dtype-independent.
        exts = batch.extend_lens
        if exts is not None:
            exts = [int(e) for e in exts]
        else:
            exts = [
                (r.device_len - r.cached_len) if r.device_len == len(r.input_ids) else 1
                for r in batch.reqs
            ]
        # gbase[b] = the global *token* offset of request b's first new token
        # (cumulative sum of the preceding requests' extend lengths) -- the index
        # into the flattened batch.positions / batch.out_loc tensors.
        gbase = [0]
        for e in exts[:-1]:
            gbase.append(gbase[-1] + e)

        # The causal mask compares a query's *absolute* position (its index in the
        # sequence) against the key positions (0 .. kv_len-1). batch.positions is
        # the flatten of absolute positions in token order, so this request's
        # first new token's absolute position is batch.positions[gbase[b]]. (The
        # token offset gbase[b] itself is NOT the absolute position once a request
        # has generated tokens -- a decode request at token offset b sits at
        # absolute position device_len-1.)
        positions = batch.positions
        if positions is None:
            # No token tensors (defensive): fall back to the token offsets, which
            # equal the absolute positions only in a pure first-step prefill batch.
            positions = torch.tensor(gbase, dtype=torch.int64)

        # The kernel's table ABI (attention.cpp): row 0 of a request's block is
        # [slot, kv_len, qpos] (decode, stride 3) or [slot, kv_len, qpos0, ext,
        # cum_ext] (prefill, stride 5), and the key-slot for position p is read
        # from row p col 0 for p in 0..kv_len-1. So the key slots occupy rows
        # [0, kv_len) -- row 0 col 0 is the *first* slot, and the last slot (row
        # kv_len-1) is the newest. (Storing qpos at row 0 col 0 and the slots at
        # rows [1, kv_len) -- the obvious-looking layout -- is an off-by-one that
        # reads qpos as a slot and drops the newest key; it is invisible with
        # zeroed weights but corrupts attention on real weights.)
        dec_table = torch.zeros((max(bs_dec, 1), K, 3), dtype=torch.int32)
        for i, b in enumerate(dec_idx):
            req = batch.reqs[b]
            kv_len = req.device_len  # full history incl. the just-appended token
            qpos = int(positions[gbase[b]].item())  # absolute position of the new token
            dec_table[i, 0, 1] = kv_len
            dec_table[i, 0, 2] = qpos
            # Key slots for positions 0..kv_len-1 -> rows 0..kv_len-1 col 0
            # (row 0 col 0 is the first slot; it shares row 0 with kv_len/qpos).
            dec_table[i, 0 : kv_len, 0] = pool.page_table[req.table_idx, torch.arange(kv_len)]

        pre_table = torch.zeros((max(bs_pre, 1), K, 5), dtype=torch.int32)
        for i, b in enumerate(pre_idx):
            req = batch.reqs[b]
            ext = exts[b]
            qpos0 = int(positions[gbase[b]].item())  # absolute pos of first new token
            kv_len = req.cached_len + ext  # all history through this step
            pre_table[i, 0, 1] = kv_len
            pre_table[i, 0, 2] = qpos0
            pre_table[i, 0, 3] = ext
            pre_table[i, 0, 4] = 0  # per-request launch: this slice starts at token 0
            # Key slots for positions 0..kv_len-1 -> rows 0..kv_len-1 col 0.
            pre_table[i, 0 : kv_len, 0] = pool.page_table[req.table_idx, torch.arange(kv_len)]

        # The kernel reads the tables as USM pointers; they must be torch XPU
        # tensors (a host tensor's data_ptr is not a USM pointer and the kernel
        # could not read it -- see the memory-model note in attention.cpp).
        # .to(device) + synchronize makes the host-built tables visible to the
        # device before any kernel reads them.
        dec_usm = dec_table.to(device) if bs_dec else None
        pre_usm = pre_table.to(device) if bs_pre else None
        torch.xpu.synchronize()
        return SyclMetadata(dec_usm, pre_usm)

    def forward(
        self,
        q,
        k,
        v,
        layer_id: int,
        batch,
        attn_spec: AttentionSpec | None = None,
        table_idx: int | None = None,
    ) -> torch.Tensor:
        # The model drives this backend **per request**: the decoder layer runs
        # each request's layers on its own hidden slice, so q/k/v hold ONLY this
        # request's new tokens (head-major [heads, ext, d]) and ``table_idx``
        # names the request. We therefore launch the kernel with ``bs=1`` over
        # this request's table row -- row ``b`` of the (one-row-per-request,
        # batch-ordered) table is ``batch.reqs[b]``, so the row to launch is the
        # one whose table_idx matches. The output is written back to the same
        # [heads, ext, d] token-major block the model reads via out.transpose.
        self._ensure_loaded()
        window = int(attn_spec.sliding_window) if (attn_spec is not None and attn_spec.sliding_window) else 0
        metadata = getattr(self, "metadata", None)
        if metadata is None:
            metadata = self._build_metadata(batch)
            self.metadata = metadata
        qh = q.shape[0]
        kv = k.shape[0]
        d = q.shape[-1]
        ext = q.shape[1]
        sm_scale = (
            float(attn_spec.sm_scale)
            if (attn_spec is not None and attn_spec.sm_scale is not None)
            else (1.0 / (d ** 0.5))
        )
        pool = _get_ctx().kv_cache
        k_cache, v_cache = pool.k_buffer, pool.v_buffer
        table_dim = self._kv_num_slots  # the pool's slot count -- the kernel's slot bound

        # The kernel's table ABI indexes request ``i`` at table row ``i`` and
        # reads rows [i, i+K). A ``bs=1`` launch of *this* request must therefore
        # sit at table row 0. The stored table has one row per request in batch
        # order, so find this request's row and make a 1-row USM view of it.
        req = next((r for r in batch.reqs if table_idx is None or r.table_idx == table_idx), batch.reqs[0])
        b = batch.reqs.index(req)
        # Same phase signal as _build_metadata: batch.phase (see that method's
        # docstring for why the previous per-request device_len/input_ids
        # comparison was wrong -- it miscompared equal on the first decode step
        # after a full prefill).
        is_decode = batch.phase == "decode"
        table = metadata.decode if is_decode else metadata.prefill
        if table is None:
            # No rows of this phase were built (should not happen when the
            # request is of this phase); fall back to the other phase's table.
            table = metadata.decode if metadata.prefill is None else metadata.prefill
        # The stored table has one row per request *of this phase*, in batch
        # order: row i == the i-th request (in batch order) of this phase --
        # exactly the dec_idx / pre_idx order _build_metadata used. The kernel's
        # bs=1 launch reads rows [0, 0+K), so point it at this request's row by
        # passing a 1-row USM slice at the phase-list index (a torch slice of an
        # XPU tensor stays USM-backed: same storage, a row offset). Every
        # request in the batch shares the same phase (batch.phase is uniform),
        # so the phase-list index is just this request's position in the batch.
        phase_idx = b
        one_row = table[phase_idx : phase_idx + 1]
        # q is head-major [qh, ext, d]; the kernel wants token-major [ext, qh, d].
        q_tok = q.transpose(0, 1).contiguous()
        out = torch.zeros((ext, qh, d), device=q.device, dtype=torch.float32)

        # The caller's active SYCL queue (torch's current XPU stream), passed
        # through so the kernel submits onto it instead of a queue this
        # backend creates itself (issue attn-sycl-graph-capture, #119) -- the
        # SYCL/XPU analog of upstream's CUDA kernels taking the caller's
        # cudaStream_t. Making that queue's work visible to torch.xpu.graph()
        # is what lets capture see this kernel launch at all.
        queue_handle = ctypes.c_void_p(torch.xpu.current_stream().sycl_queue)
        # A host sync during capture is a hard error (see __init__'s
        # docstring); ordinary (non-capturing) callers still get the same
        # host-visible-before-return guarantee these always provided.
        if not self._capturing:
            torch.xpu.synchronize()
        if is_decode:
            self._decode(
                self._ptr(q_tok),
                self._ptr(k_cache),
                self._ptr(v_cache),
                self._ptr(one_row),
                1,
                table_dim,
                qh,
                kv,
                d,
                ctypes.c_float(sm_scale),
                window,
                self._ptr(out),
                queue_handle,
            )
        else:
            self._prefill(
                self._ptr(q_tok),
                self._ptr(k_cache),
                self._ptr(v_cache),
                self._ptr(one_row),
                1,
                table_dim,
                qh,
                kv,
                d,
                ctypes.c_float(sm_scale),
                window,
                self._ptr(out),
                queue_handle,
            )
        if not self._capturing:
            torch.xpu.synchronize()
        # The kernel wrote token-major [ext, qh, d]; the model wants head-major
        # [qh, ext, d] (it does o_proj on out.transpose(1, 2)).
        return out.transpose(0, 1).contiguous()


__all__ = ["SyclAttentionBackend", "SyclMetadata"]
