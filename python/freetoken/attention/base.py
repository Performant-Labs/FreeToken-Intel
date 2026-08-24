from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List


class AttnType(str, Enum):
    FULL = "full"
    SWA = "swa"
    MLA = "mla"
    DSA = "dsa"
    DSV4 = "dsv4"
    LINEAR = "linear"
    BSA = "bsa"

    @property
    def backend_driven(self) -> bool:
        return self is not AttnType.LINEAR


@dataclass
class AttentionSpec:
    sliding_window: int | None = None
    sm_scale: float | None = None
    sinks: object | None = None


@dataclass
class BaseAttnMetadata(ABC):
    @abstractmethod
    def get_last_indices(self, bs: int): ...


class BaseAttnBackend(ABC):
    @abstractmethod
    def forward(self, q, k, v, layer_id: int, batch, attn_spec: AttentionSpec | None = None): ...

    @abstractmethod
    def prepare_metadata(self, batch) -> None: ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...

    @abstractmethod
    def prepare_for_capture(self, batch) -> None: ...

    @abstractmethod
    def prepare_for_replay(self, batch) -> None: ...

    def reset_capture(self) -> None:
        if hasattr(self, "capture"):
            self.capture = None
        if hasattr(self, "capture_bs"):
            self.capture_bs = []
        if hasattr(self, "max_graph_bs"):
            self.max_graph_bs = 0


class HybridBackend(BaseAttnBackend):
    def __init__(self, prefill_backend: BaseAttnBackend, decode_backend: BaseAttnBackend) -> None:
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(self, q, k, v, layer_id: int, batch, attn_spec: AttentionSpec | None = None):
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch, attn_spec=attn_spec)

    def prepare_metadata(self, batch) -> None:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch) -> None:
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch) -> None:
        self.decode_backend.prepare_for_replay(batch)

    def reset_capture(self) -> None:
        self.decode_backend.reset_capture()
