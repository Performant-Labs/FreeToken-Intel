# tests

CPU-only tests run anywhere:

```
pytest -m "not xpu and not slow"
```

XPU tests need a B70 (or another Arc Xe2) plus oneAPI / Level Zero:

```
pytest -m xpu
```

`needs_weights` tests stay off unless `FREETOKEN_TEST_MODEL` points at a
local checkpoint.
