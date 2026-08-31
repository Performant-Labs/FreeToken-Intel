"""Vocabulary embedding and LM-head layers, ported from upstream CUDA to pure torch.

Upstream path: ``python/freetoken/layers/embedding.py``.

Upstream implements :class:`VocabParallelEmbedding` / :class:`ParallelLMHead` as
``BaseOP``s whose forward gathers rows with a CUDA JIT ``indexing`` kernel and
folds in an NCCL all-reduce for tensor-parallel shards. The XPU port replaces the
custom kernels with the equivalent PyTorch ops (``F.embedding`` row-gather and
``F.linear``) while keeping the *exact* upstream Python API -- constructor signature,
``forward`` signature, weight-tying, and the ``embed_scale`` -- so the model call
sites (``embed_tokens.forward(input_ids)`` / ``lm_head(...)``) are unchanged.

Two deliberate differences from upstream, both safe on this single-device build:

* The tensor-parallel shard bookkeeping (``vocab_range`` + ``all_reduce`` /
  ``all_gather``) is present in the signatures for API parity but is a no-op when
  ``tp_size == 1`` (this repo is single-GPU); a real multi-GPU shard would need the
  NCCL path, which is out of scope here.
* The cached ``embed_scale`` scalar is materialised lazily in the input's dtype and
  stored under an underscore-prefixed attribute (``_embed_scale_t``) so ``BaseOP``'s
  ``state_dict``/``load_state_dict`` machinery skips it -- a runtime artefact, not a
  checkpoint weight.

``VocabParallelEmbedding`` is a ``BaseOP`` (not ``nn.Embedding``): it owns a plain
``weight`` tensor whose shape/dtype are installed by the checkpoint loader
(``BaseOP.load_state_dict`` copies the HF ``model.embed_tokens.weight`` entry in
place). The model's ``self.to(device)`` then moves it to the XPU before any forward.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

from freetoken.layers.base import BaseOP


def _tp_info() -> Tuple[int, int]:
    import torch

    # Upstream resolves tensor-parallel rank/size from the process group. This
    # repo is single-device: rank 0 / size 1, so the TP branches are inert no-ops.
    return 0, 1


def _div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


class VocabParallelEmbedding(BaseOP):
    """Gather token embeddings by row (upstream ``VocabParallelEmbedding``).

    Mirrors the upstream API: ``__init__(num_embeddings, embedding_dim,
    embed_scale=None)`` and ``forward(x) -> [len(x), embedding_dim]``. Upstream
    gathers rows with a CUDA ``indexing`` kernel and all-reduces across TP shards;
    here the row-gather is ``torch.index`` (pure torch) and the TP fold is a no-op
    for ``tp_size == 1``. ``embed_scale (Gemma's ``sqrt(hidden)``) is applied lazily in the
    input dtype so it stays checkpoint- and capture-safe.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, embed_scale: Optional[float] = None) -> None:
        import torch

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embed_scale = embed_scale
        tp_rank, tp_size = _tp_info()
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        num_embeddings_tp = _div_ceil(num_embeddings, tp_size)
        # Vocab shard for this rank: [start, end) over the full vocab.
        self.vocab_range: Tuple[int, int] = (num_embeddings_tp * tp_rank, min(num_embeddings_tp * tp_rank + num_embeddings_tp, num_embeddings))
        # CPU placeholder; the checkpoint loader installs the real (device/dtype)
        # weight in place, then the model moves it. Matches upstream's torch.empty.
        self.weight = torch.empty(num_embeddings_tp, embedding_dim)
        # Lazy, dtype-matched scale buffer (underscore-prefixed -> not in state_dict).
        self._embed_scale_t: Optional[Any] = None

    def _get_embed_scale(self, x) -> "torch.Tensor":
        import torch

        if self.embed_scale is None:
            raise RuntimeError("VocabParallelEmbedding called with embed_scale but it is None")
        if self._embed_scale_t is None or self._embed_scale_t.device != x.device:
            self._embed_scale_t = torch.full((), float(self.embed_scale), device=x.device, dtype=x.dtype)
        return self._embed_scale_t

    def forward(self, x: Any) -> Any:
        import torch
        import torch.nn.functional as F

        y = F.embedding(x, self.weight)
        # TP shard fold (no-op on this single-GPU build; kept for API parity).
        if self.tp_size > 1:
            from freetoken.comm import get_comm  # placeholder: no comm on this build
            y = get_comm(self.tp_rank, self.tp_size).all_reduce(y)
        if self.embed_scale is not None:
            y = y * self._get_embed_scale(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """LM head that optionally shares the embedding table (upstream parity).

    Same constructor as upstream:
    ``__init__(num_embeddings, embedding_dim, bias=False,
    tie_word_embeddings=False, tied_embedding=None)`` and asserts the
    ``tied_embedding``/``tie_word_embeddings`` invariant. When tied, the head exposes
    no weights of its own (``state_dict`` -> ``{}``) and its forward computes
    ``F.linear(x, tied_embedding.weight, bias)`` -- the shared table, not a private
    copy. During prefill it first keeps only each sequence's last-position hidden
    state, then projects to the vocab. On this single-GPU build the TP all-gather /
    vocab-trim is a no-op.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: Optional[VocabParallelEmbedding] = None,
    ) -> None:
        assert (tied_embedding is not None) == tie_word_embeddings, (
            "tied_embedding must be provided iff tie_word_embeddings is True"
        )
        super().__init__(num_embeddings, embedding_dim, embed_scale=None)
        self.bias_enabled = bias
        self.tie_word_embeddings = tie_word_embeddings
        self.tied_embedding = tied_embedding
        if bias:
            import torch

            self.bias = torch.empty(num_embeddings)
        else:
            self.bias = None

    def state_dict(self, *, prefix: str = "", result: Optional[dict] = None) -> dict:
        # Tied head has no own weights (shares tied_embedding's table). Upstream
        # returns {} when tied so the loader never expects an lm_head.weight key.
        if self.tie_word_embeddings:
            return result if result is not None else {}
        return super().state_dict(prefix=prefix, result=result)

    def load_state_dict(self, state_dict: dict, *, prefix: str = "", _internal: bool = False) -> None:
        if self.tie_word_embeddings:
            # A tied head owns no weights (it shares tied_embedding's table), so any
            # <prefix>.weight / <prefix>.bias keys the loader hands us are consumed
            # (dropped) rather than installed. Match a key at this prefix whether it
            # is nested ("prefix.weight") or top-level ("prefix"), mirroring how
            # BaseOP.load_state_dict builds keys via _concat_prefix.
            if not _internal:
                for k in list(state_dict.keys()):
                    # Nested call: drop keys at this prefix (nested "prefix.weight" or
                    # the prefix itself). A top-level tied call (prefix="") has no
                    # keys of its own to drop -- its inherited weight/bias are its own
                    # params and are left for the normal walk (harmless: a tied head is
                    # never state_dict()ed, so the loader never feeds it weight keys).
                    if prefix and (k == prefix or k.startswith(prefix + ".")):
                        state_dict.pop(k, None)
                if state_dict:
                    raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
            return
        super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)

    def _module_weight(self):
        if self.tie_word_embeddings:
            return self.tied_embedding.weight if self.tied_embedding is not None else self.weight
        return self.weight

    def forward(self, x: Any, attn_metadata: Any = None) -> Any:
        import torch
        import torch.nn.functional as F

        weight = self._module_weight()
        # Prefill: keep only the last position of each sequence before projecting
        # (the big vocab matmul over redundant positions is wasted compute). Upstream
        # gates this on ``attn_metadata.is_prefill`` and pulls the per-sequence last
        # position via ``attn_metadata.get_last_indices(bs)``. Decode passes no
        # metadata (or a decode metadata), so this is a no-op and every row projects.
        if attn_metadata is not None and getattr(attn_metadata, "is_prefill", False):
            get_last = getattr(attn_metadata, "get_last_indices", None)
            bs = getattr(attn_metadata, "bs", None) or getattr(attn_metadata, "batch_size", None)
            if get_last is not None and bs is not None:
                try:
                    last_idx = get_last(bs)
                    if last_idx is not None:
                        x = x[last_idx]
                except Exception:
                    pass
        logits = F.linear(x, weight, self.bias if self.bias_enabled else None)
        if self.tp_size > 1:
            from freetoken.comm import get_comm  # placeholder: no comm on this build
            logits = get_comm(self.tp_rank, self.tp_size).all_gather(logits, dim=1)
            logits = logits[:, : self.num_embeddings]
        return logits
