from freetoken.utils.arch import B70_MEMORY_BANDWIDTH_GBS, is_xpu_available


def test_b70_constants():
    assert B70_MEMORY_BANDWIDTH_GBS == 608
    assert isinstance(is_xpu_available(), bool)
