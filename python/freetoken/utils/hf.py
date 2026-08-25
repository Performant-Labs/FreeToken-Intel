"""Hugging Face config / tokenizer / weight helpers.

Upstream NVIDIA path: python/freetoken/utils/hf.py
Fill in: GitHub issue `models-loader` (see docs/architecture.md).

Only the subset FreeToken-Intel needs for the B70 port is implemented here.
Network access (``snapshot_download``) is deferred behind the functions that use
it, so importing this module -- and running the CPU-only test suite -- needs no
network. Checkpoint-local helpers (``RawConfigShim``, config/tokenizer readers
over a local directory) work fully offline.
"""
from __future__ import annotations

import functools
import json
import os
from typing import Any, FrozenSet


class RawConfigShim:
    """Attribute view over a checkpoint's raw ``config.json``.

    Fallback for model types newer than the installed ``transformers`` (which
    raises on an unknown ``model_type`` when the checkpoint ships no
    ``auto_map``). FreeToken only *reads* config fields (``parse_config``) and
    never instantiates modeling code, so the raw JSON is enough. ``*_config``
    sub-dicts are wrapped for attribute access, matching ``PretrainedConfig``'s
    nested-config behavior; every other value is served raw.
    """

    def __init__(self, data: dict | None = None, **kwargs: Any) -> None:
        self._data = {**(data or {}), **kwargs}

    def __getattr__(self, name: str) -> Any:
        if name == "_name_or_path":
            # PretrainedConfig carries the checkpoint path under this name; some
            # parse_config implementations read it, so it must survive the
            # underscore guard below.
            return self.__dict__.get("_data", {}).get("_name_or_path", "")
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(name) from None
        if name.endswith("_config") and isinstance(value, dict):
            return RawConfigShim(value)
        return value

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self._data))  # deep copy; callers may mutate


def _raw_config_json(model_path: str) -> dict:
    path = os.path.join(model_path, "config.json")
    if not os.path.isfile(path):
        raise ValueError(f"no config.json in local checkpoint '{model_path}'")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_hf_config(model_path: str) -> Any:
    from transformers import AutoConfig

    # trust_remote_code: checkpoints that ship a custom config class via
    # ``auto_map`` (e.g. MiniMax-M2) refuse to load without it. We only read
    # config fields, never the checkpoint's modeling code.
    try:
        return AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except ValueError as exc:
        # Unknown model_type on this transformers version: serve off the raw
        # JSON instead of failing. Anything else (bad path, malformed JSON) stays
        # fatal.
        if "model type" not in str(exc):
            raise
        return RawConfigShim(_raw_config_json(model_path), _name_or_path=model_path)


def _cached_hub_dir(model_path: str) -> str | None:
    """The local snapshot dir for a hub id already present in the HF cache.

    Returns None when the id is not cached, so callers can decide whether to
    proceed (cached) or attempt a download (uncached). This is what lets the
    ``ft serve`` spine read a config offline: it only needs the config, which
    lives in the cache, and must not trigger a multi-GB weight download.
    """
    import os

    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001 -- no hub lib -> nothing cached
        return None
    for filename in ("config.json", "model.safetensors.index.json"):
        path = try_to_load_from_cache(model_path, filename)
        if path is not None:
            return os.path.dirname(path)
    return None


@functools.lru_cache(maxsize=None)
def cached_load_hf_config(model_path: str) -> Any:
    """Load the checkpoint's config, cached per path.

    A local directory is read directly. A hub id already present in the HF cache
    is read from the cache (offline). Only an uncached hub id triggers a
    download (see :func:`download_hf_weight`).
    """
    if os.path.isdir(model_path):
        local = model_path
    else:
        cached = _cached_hub_dir(model_path)
        local = cached if cached is not None else download_hf_weight(model_path)
    config = _load_hf_config(local)
    if isinstance(config, RawConfigShim):
        return RawConfigShim(config.to_dict(), _name_or_path=model_path)
    return config


def download_hf_weight(model_path: str) -> str:
    """Resolve ``model_path`` to a local directory holding the weight shards.

    A local directory is returned as-is (offline). A HF hub id is resolved via
    ``snapshot_download`` (restricted to safetensors) -- network is only touched
    here, and only for non-local paths.
    """
    if os.path.isdir(model_path):
        return model_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_path, allow_patterns=["*.safetensors"])
    except Exception as e:  # noqa: BLE001 -- re-wrap with the path context
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a resolvable "
            f"model ID: {e}"
        ) from e


def load_tokenizer(model_path: str):
    """Load the checkpoint's tokenizer (offline for a local directory)."""
    from transformers import AutoTokenizer

    local = download_hf_weight(model_path)
    return AutoTokenizer.from_pretrained(local)


def load_eos_token_ids(model_path: str, tokenizer) -> FrozenSet[int]:
    """Full set of stop-token ids: the tokenizer's eos unioned with the ids listed
    in ``generation_config.json`` (chat models often end a turn on a non-eos token).
    """
    from transformers import GenerationConfig

    local = download_hf_weight(model_path)
    ids: set[int] = set()
    if getattr(tokenizer, "eos_token_id", None) is not None:
        ids.add(int(tokenizer.eos_token_id))
    try:
        gen_eos = GenerationConfig.from_pretrained(local).eos_token_id
    except Exception:  # noqa: BLE001 -- no generation config is fine
        gen_eos = None
    if isinstance(gen_eos, int):
        ids.add(gen_eos)
    elif isinstance(gen_eos, (list, tuple)):
        ids.update(int(x) for x in gen_eos)
    return frozenset(ids)


def load_generation_sampling(model_path: str) -> dict[str, Any]:
    """Recommended sampling defaults from ``generation_config.json`` (sglang's
    ``sampling_defaults='model'``). Returns ``{temperature, top_k, top_p}`` for the
    keys present; ``{"temperature": 0.0}`` when the model recommends greedy
    (``do_sample=false``); empty dict when there is nothing to read.
    """
    from transformers import GenerationConfig

    local = download_hf_weight(model_path)
    try:
        gc = GenerationConfig.from_pretrained(local)
    except Exception:  # noqa: BLE001 -- no generation config is fine
        return {}
    if getattr(gc, "do_sample", None) is False:
        return {"temperature": 0.0}
    out: dict[str, Any] = {}
    for key in ("temperature", "top_k", "top_p"):
        val = getattr(gc, key, None)
        if val is not None:
            out[key] = val
    return out


def load_toolcall_anchor_id(tokenizer, opener: str | None) -> int | None:
    """The single token id of ``opener`` -- the wire format's unique tool-call
    opening marker. None when there is no opener or the tokenizer spells it with
    more than one token (the scheduler matches sampled ids one at a time)."""
    if not opener:
        return None
    try:
        ids = tokenizer.encode(opener, add_special_tokens=False)
    except Exception:  # noqa: BLE001 -- unspellable opener -> no anchor
        return None
    return int(ids[0]) if len(ids) == 1 else None


__all__ = [
    "RawConfigShim",
    "cached_load_hf_config",
    "download_hf_weight",
    "load_tokenizer",
    "load_eos_token_ids",
    "load_generation_sampling",
    "load_toolcall_anchor_id",
]
