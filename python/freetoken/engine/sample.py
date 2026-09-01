"""Token sampling.

Upstream NVIDIA path: python/freetoken/engine/sample.py
Fill in: GitHub issue `engine-loop` (see docs/architecture.md).

Pure-torch sampler: the engine passes the per-request last-position logits plus
one ``BatchSamplingArgs`` (aligned to the batch) and gets back the next-token
ids (one per request, same order). Greedy is ``argmax``; otherwise a
temperature-scaled, top-k / top-p truncated distribution is sampled (standard HuggingFace semantics). All
ops run on the XPU (and CPU).
"""
from __future__ import annotations

import torch

from freetoken.core import SamplingParams


class BatchSamplingArgs:
    """Per-request sampling settings for one engine step.

    ``temperature[i]`` == 0 (greedy) selects the argmax path; otherwise a
    temperature-scaled, top-k / top-p truncated distribution is sampled.
    """

    def __init__(self, sampling_params_list) -> None:
        self.temperature = torch.tensor(
            [p.temperature for p in sampling_params_list], dtype=torch.float32
        )
        self.top_k = torch.tensor([p.top_k for p in sampling_params_list], dtype=torch.int64)
        self.top_p = torch.tensor([p.top_p for p in sampling_params_list], dtype=torch.float32)
        self.ignore_eos = torch.tensor([p.ignore_eos for p in sampling_params_list], dtype=torch.bool)
        self.stop_strs = [p.stop_strs for p in sampling_params_list]


class Sampler:
    """Turn per-request logits into next-token ids honoring sampling settings."""

    def __init__(self, eos_token_id: int, device, *, repeat_sampling: bool = False) -> None:
        self.eos_token_id = eos_token_id
        self.device = device
        self.repeat_sampling = repeat_sampling

    def sample(
        self,
        logits: torch.Tensor,
        sampling_args: BatchSamplingArgs,
        input_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return next-token ids ``[bs]`` (int64), one per request.

        ``logits`` is ``[bs, vocab_size]`` (the last-position logits per
        request); ``sampling_args`` is aligned to the same batch order.
        """
        bs = logits.shape[0]
        if bs == 0:
            return torch.empty((0,), device=logits.device, dtype=torch.int64)

        # BatchSamplingArgs builds its tensors device-less (plain torch.tensor(...),
        # which lands on CPU); the greedy-only path never mixed them with logits in
        # an op that requires matching devices (mask-indexing tolerates it), so this
        # went unnoticed until a real temperature > 0 request engaged the stochastic
        # branch below (`stoch_logits / temp[:, None]`), which does elementwise math
        # and hard-errors on a device mismatch. Move onto the logits' device once,
        # up front, so every path (greedy or stochastic) is consistent.
        device = logits.device
        sampling_args.temperature = sampling_args.temperature.to(device)
        sampling_args.top_k = sampling_args.top_k.to(device)
        sampling_args.top_p = sampling_args.top_p.to(device)

        greedy_mask = sampling_args.temperature == 0
        tokens = torch.empty((bs,), device=logits.device, dtype=torch.int64)

        if greedy_mask.any():
            tokens[greedy_mask] = logits[greedy_mask].argmax(dim=-1)

        stochastic_mask = ~greedy_mask
        if stochastic_mask.any():
            rows = torch.nonzero(stochastic_mask, as_tuple=False).squeeze(-1)
            stoch_logits = logits[rows]  # [m, V]
            temp = sampling_args.temperature[stochastic_mask].clamp_min(1e-5).to(stoch_logits.dtype)
            top_k = sampling_args.top_k[stochastic_mask].long()
            top_p = sampling_args.top_p[stochastic_mask].clamp(min=0.0, max=1.0)

            log_probs = torch.log_softmax(stoch_logits / temp[:, None], dim=-1)
            vocab = log_probs.shape[-1]

            # top-k: drop everything outside the k-largest logits. k==0 disables.
            k = top_k
            if (k > 0).any() and vocab:
                k_min = int(k.min().item())
                if 0 < k_min < vocab:
                    k_thresh, _ = torch.topk(log_probs, k_min, dim=-1)
                    k_thresh = k_thresh[:, -1:]
                    log_probs = torch.where(log_probs < k_thresh, float("-inf"), log_probs)

            # top-p (nucleus): drop tokens beyond the smallest prefix with mass >= p.
            if (top_p < 1.0).any() and vocab:
                p_min = float(top_p.min().item())
                if 0.0 < p_min < 1.0:
                    sorted_log, sorted_idx = torch.sort(log_probs, dim=-1, descending=True)
                    cum = torch.cumsum(torch.exp(sorted_log), dim=-1)
                    # A token at sorted position j is dropped once the mass of the
                    # tokens *before* it already reaches p.
                    prefix = torch.cumsum(torch.exp(sorted_log).roll(1, dims=-1), dim=-1)
                    prefix[..., 0] = 0.0
                    drop_sorted = prefix > p_min
                    keep_sorted = torch.ones_like(drop_sorted, dtype=torch.bool)
                    keep_sorted = torch.where(drop_sorted, torch.zeros_like(drop_sorted, dtype=torch.bool), keep_sorted)
                    # Scatter the keep-mask back to the original vocab order.
                    keep = torch.zeros_like(keep_sorted)
                    keep.scatter_(dim=-1, index=sorted_idx, src=keep_sorted)
                    log_probs = torch.where(keep, log_probs, float("-inf"))

            # Any row fully masked (e.g. p=0) falls back to argmax.
            probs = torch.softmax(log_probs, dim=-1)
            row_all_masked = torch.isnan(probs).any(dim=-1)
            if row_all_masked.any():
                probs = torch.where(
                    row_all_masked[:, None].expand_as(probs),
                    torch.zeros_like(probs),
                    probs,
                )
            drawn = torch.multinomial(probs, num_samples=1).squeeze(-1)
            tokens[rows] = drawn
        return tokens


def sample(
    logits: torch.Tensor,
    sampling_args: BatchSamplingArgs,
    sampler: Sampler | None = None,
) -> torch.Tensor:
    """Module-level entry point (delegates to a default Sampler when omitted)."""
    if sampler is None:
        sampler = Sampler(eos_token_id=-1, device=logits.device)
    return sampler.sample(logits, sampling_args)


__all__ = ["BatchSamplingArgs", "Sampler", "sample"]
