import pytest

from freetoken._stub import NotYetImplemented
from freetoken.moe import create_moe_backend
from freetoken.utils.arch import B70_MEMORY_BANDWIDTH_GBS, is_xpu_available


def test_fused_moe_is_stub():
    backend = create_moe_backend("fused")
    with pytest.raises(NotYetImplemented, match="moe-fused"):
        backend.forward(None, None, None, None, 8, True, "silu", False)


def test_b70_constants():
    assert B70_MEMORY_BANDWIDTH_GBS == 608
    assert isinstance(is_xpu_available(), bool)
