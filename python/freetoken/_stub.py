"""Shared helper for Intel stubs.

Later issues replace ``unimplemented`` call sites with real XPU / SYCL code.
The *issue* slug matches a GitHub issue title prefix so the backlog and the
tree stay aligned.
"""
from __future__ import annotations

from typing import NoReturn


class NotYetImplemented(NotImplementedError):
    """Raised by a scaffolded API that has no Intel implementation yet."""


def unimplemented(feature: str, issue: str) -> NoReturn:
    raise NotYetImplemented(
        f"{feature} is a stub for the Intel Arc Pro B70 port. "
        f"Implement it under GitHub issue `{issue}`. "
        "See docs/architecture.md for the NVIDIA -> Intel mapping."
    )
