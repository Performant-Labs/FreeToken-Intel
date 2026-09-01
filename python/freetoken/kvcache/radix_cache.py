"""Radix prefix cache over a paged KV pool.

Upstream NVIDIA path: python/freetoken/kvcache/radix_cache.py
Fill in: GitHub issue `kvcache` (see docs/architecture.md).

A radix tree whose nodes hold (token-key, pool-slot-index) spans. ``match_prefix``
walks the tree to recover the length of an incoming prompt already resident in the
pool (and the slot indices of that prefix, via the handle); ``insert_prefix`` adds
a new span (splitting an existing node at the divergence point); ``evict`` frees the
least-recently-used, unref'd leaf spans to reclaim pool slots.

This is the pure-Python, page-aligned tree upstream ships. The only Intel-specific
adaptation is the key comparator: upstream's ``fast_compare_key`` is an AOT-compiled
C++ kernel; the Intel port uses the pure-torch fallback in ``_compare.py`` (an AOT
port is a perf follow-up).

The tree stores *page-aligned* spans (``page_size`` token granularity), matching the
paged pool's allocation unit. ``page_size`` is read from the global context unless an
explicit value is passed (e.g. a test that wants a granularity other than the
engine's). Handles returned by ``match_prefix`` / ``insert_prefix`` must be
``lock_handle``d before their indices are consumed, or ``evict`` may free them first.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import torch
from freetoken.core import get_global_ctx
from freetoken.utils import align_down

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo
from ._compare import fast_compare_key

KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]


class RadixTreeNode:
    counter: int = 0

    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: Dict[Any, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()

        # these fields should be updated later
        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)

    def set_parent(self, parent: RadixTreeNode) -> None:
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        return self._value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        return fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> RadixTreeNode:
        assert 0 < pos < self.length
        parent = self.parent

        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count

        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: "RadixTreeNode") -> bool:
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    # ``cached_len`` is declared here (not inherited) so the frozen-dataclass
    # __init__ is generated from a clean field set; it sorts first (no default).
    # Constructed as RadixCacheHandle(cached_len, node).
    cached_len: int
    node: RadixTreeNode

    def get_matched_indices(self) -> torch.Tensor:
        node = self.node
        value_list: List[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()
        if not value_list:
            # Zero-length match: the handle points at root, which has no _key
            # (set_key_value is only ever called for non-root nodes). Build the
            # empty index tensor from the key when available, else just the dtype.
            key = self.node.__dict__.get("_key")
            if key is not None:
                return key.new_empty(0, dtype=torch.int32)
            return torch.empty(0, dtype=torch.int32)
        return torch.cat(value_list)


class RadixPrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device, page_size: int | None = None):
        super().__init__()
        self.device = device
        # Explicit page_size beats the global ctx: a manager constructed with
        # page_size != ctx (tests, future multi-granularity) must not silently
        # split the two (the tree would align by one granularity while the
        # manager frees by the other -> orphaned pages).
        self.page_size = get_global_ctx().page_size if page_size is None else int(page_size)
        self.key_fn = _get_key_fn(self.page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # root is always protected

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        node, prefix_len = self._tree_walk(input_ids)
        return MatchResult(prefix_len, RadixCacheHandle(prefix_len, node))

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len != insert_len:  # NOTE: prefix_len < insert_len
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert (
                leave_nodes
            ), f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[self.key_fn(node._key)]
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)

    def reset(self) -> None:
        self.root_node.children.clear()
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1
        self.evictable_size = 0
        self.protected_size = 0

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def check_integrity(self) -> None:
        # Walk the tree and confirm every node's (key, value) spans are page-aligned
        # and consistent with the parent's child pointer.
        stack: List[RadixTreeNode] = [self.root_node]
        while stack:
            node = stack.pop()
            if node is self.root_node:
                for child in node.children.values():
                    stack.append(child)
            else:
                assert node._key.numel() % self.page_size == 0
                assert node._value.numel() % self.page_size == 0
                parent_child = self.key_fn(node._key)
                assert self.root_node.children or node._parent is not None
                if node._parent is not None:
                    assert parent_child in node._parent.children
                    assert node._parent.children[parent_child] is node
                for child in node.children.values():
                    stack.append(child)

    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        nodes: List[RadixTreeNode] = [self.root_node]
        leave_nodes: List[RadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes

    def _tree_walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child_node is None:
                return node, prefix_len
            node = child_node  # walk to child node

            # NOTE: at least 1 page is matched, so match_len >= page_size
            match_len = node.get_match_len(input_ids[prefix_len:])
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len

            # need to split the node if not fully matched
            if match_len != node.length:
                node = node.split_at(match_len)
                node.timestamp = tic
                return node, prefix_len

            # update timestamp for accessed node
            node.timestamp = tic

        return node, prefix_len


def _get_key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())
