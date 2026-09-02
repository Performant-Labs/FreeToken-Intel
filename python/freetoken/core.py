"""Request / batch / global context. Device tensors are XPU, not CUDA."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Literal, Tuple


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024
    stop_strs: list[str] = field(default_factory=list)

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Req:
    input_ids: object
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: object
    mm_embeds: object | None = None
    linear_slot_idx: int | None = None
    aborted: bool = False
    toolcall_anchor_len: int | None = None

    def __post_init__(self) -> None:
        self.device_len = len(self.input_ids) if hasattr(self.input_ids, "__len__") else 0
        # ``output_len`` is the number of tokens to *generate*. The prefill step
        # already emits the first one, so the history length at which the request
        # is complete is prompt_len + output_len (device_len includes the prompt
        # plus every generated token).
        self.max_device_len = self.device_len + max(0, self.output_len)

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    @property
    def can_decode(self) -> bool:
        return self.remain_len > 0 and not self.aborted


@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    padded_reqs: List[Req] = field(default_factory=list)
    log_new_tokens: int = 0
    log_cached_tokens: int = 0
    prompt_admissions: List[Tuple[int, int, int]] = field(default_factory=list)
    # Device tensors the model's forward() consumes (set by the engine each step):
    # input_ids [num_tokens] (prefill: flattened prompts; decode: one per req),
    # positions [num_tokens] (absolute token positions), out_loc [num_tokens]
    # (KV-cache slot to write each token's key/value into).
    input_ids: object | None = None
    positions: object | None = None
    out_loc: object | None = None
    # Per-request new-token count (== extend_len in prefill, 1 in decode), in
    # request order. The model's per-request forward slices the token tensors by
    # these; the engine fills it in step(). A batch may mix phases (one request
    # still prefilling its prompt, another already decoding), so a single
    # batch-level phase flag is NOT enough to size each request's slice -- this
    # per-request vector is the authoritative count.
    extend_lens: object | None = None

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)


@dataclass
class Context:
    page_size: int
    moe_offload_cache: object | None = None
    linear_state_pool: object | None = None
    # Runtime tensors the model's forward() reads (owned by the engine):
    #   kv_cache    -- the paged K/V pool (see kvcache.create_kv_pool)
    #   attn_backend-- the attention backend (prepare_metadata + forward)
    #   page_table  -- [max_running_req+1, max_seq_len] int32 slot indices
    kv_cache: object | None = None
    attn_backend: object | None = None
    page_table: object | None = None
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    """Install ``ctx`` as the process-global context (replaces any existing one).

    The reference engine (and the tests) may construct more than one engine in
    a single process, each with its own context, so this is a plain
    assignment rather than an assert-None guard.
    """
    global _GLOBAL_CTX
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX


def reset_global_ctx() -> None:
    """Clear the process-global context (used by tests to restore a clean state)."""
    global _GLOBAL_CTX
    _GLOBAL_CTX = None
