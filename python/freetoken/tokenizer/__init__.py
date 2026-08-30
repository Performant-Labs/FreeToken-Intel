"""Tokenizer: chat-template encode and incremental decode.

Upstream NVIDIA path: python/freetoken/tokenizer/__init__.py

On the Intel port the tokenizer is in-process, not a separate ZMQ worker
process (upstream's ``start_tokenizer`` / ``tokenize_worker`` /
``detokenize_worker`` are deliberately not ported): the B70 serve seam owns
termination at the engine and needs only a synchronous encode/decode pair, so
the managers here are constructed against a transformers ``AutoTokenizer`` and
called directly from the request path.
"""

from __future__ import annotations

from freetoken.tokenizer.detokenize import DetokenizeManager
from freetoken.tokenizer.tokenize import TokenizeManager

__all__ = ["DetokenizeManager", "TokenizeManager"]
