"""Hugging Face config / tokenizer / weight helpers.

Upstream NVIDIA path: python/freetoken/utils/hf.py
Fill in: GitHub issue `models-loader` (see docs/architecture.md).
"""
from __future__ import annotations

from freetoken._stub import unimplemented


def cached_load_hf_config(*args, **kwargs):
    unimplemented("cached_load_hf_config", "models-loader")
def download_hf_weight(*args, **kwargs):
    unimplemented("download_hf_weight", "models-loader")
def load_eos_token_ids(*args, **kwargs):
    unimplemented("load_eos_token_ids", "models-loader")
def load_generation_sampling(*args, **kwargs):
    unimplemented("load_generation_sampling", "models-loader")
def load_tokenizer(*args, **kwargs):
    unimplemented("load_tokenizer", "models-loader")
def load_toolcall_anchor_id(*args, **kwargs):
    unimplemented("load_toolcall_anchor_id", "models-loader")

