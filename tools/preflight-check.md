# Prompt: FreeToken-Intel pre-flight check

You are a development-environment auditor for **FreeToken-Intel** (Intel Arc
Pro B70 / SYCL / XPU). Execute this prompt by **running commands on the
machine**. Do not guess. Do not install, uninstall, write files, or modify
venvs unless the user explicitly asked you to fix failures after the report.

Source of truth (read if present; do not invent extra requirements):

- `docs/dev-requirements.md`
- `docs/dev-setup.md`
- `docs/intel-b70.md`

Repo root is the working tree that contains `pyproject.toml` and
`python/freetoken/`. If you are not there, `cd` to it first.

## Goal

Produce a pre-flight report with two tracks:

| Track | Meaning |
| --- | --- |
| **CPU** | Can edit Python and run unit tests in `.venv` |
| **XPU** | Can talk to the B70: driver, oneAPI, `.venv-xpu`, `torch.xpu` |

CPU failures are blocking for all development. XPU failures are blocking
only for kernel / device / serve work; mark them **SKIP** on a CPU-only
box (no Arc GPU), **FAIL** if a GPU is present but the stack is broken.

## Procedure

Run each check below with the shell. Record **PASS**, **FAIL**, or
**SKIP** plus the actual command output (trim to the relevant lines).

Use the **venv interpreters** when a check is about Python packages:

- CPU: `.venv/bin/python` and `.venv/bin/ft` if `.venv` exists
- XPU: `.venv-xpu/bin/python` and `.venv-xpu/bin/ft` if `.venv-xpu` exists

Do not use `/usr/bin/python3` for package or `ft` checks. Do not
`source .venv` on top of `.venv-xpu`.

If `.venv-xpu` exists, `source /opt/intel/oneapi/setvars.sh` in that
same command only when the file exists (path may differ; search
`setvars.sh` under `/opt/intel` if needed).

### A. System (both tracks)

1. `uname -s` and `uname -m` — need Linux + `x86_64`.
2. `python3 --version` — need 3.10+.
3. `git --version`
4. `g++ --version` (first line)
5. `cmake --version` (first line)
6. `ninja --version` (optional; FAIL only if missing when XPU track is in play)
7. `test -f pyproject.toml && test -d python/freetoken`

### B. CPU venv (`.venv`)

8. `.venv` exists and `.venv/bin/python` runs.
9. That python’s sys.prefix is inside the repo `.venv` (not system).
10. `import freetoken` works; print `freetoken.__version__`.
11. `.venv/bin/ft --version` prints `freetoken-intel version`.
12. `.venv/bin/python -c "import torch"` — **PASS if ImportError**
    (CPU venv must not contain torch). **FAIL if torch imports.**
13. `.venv/bin/pytest -m "not xpu and not slow" -q` from repo root —
    all tests pass.

### C. GPU presence (decides XPU vs SKIP)

14. `lspci | grep -iE 'VGA|Display|3D'` (or `lspci` if grep empty).
    - Arc / B70 / Battlemage / Xe → continue XPU checks.
    - No Intel GPU → remaining XPU checks **SKIP**, say CPU-only.

### D. XPU system (not in a venv)

15. Level Zero loader: `ls /usr/lib*/libze_loader.so* /usr/lib/*/libze_loader.so* 2>/dev/null`
16. `sycl-ls` after setvars if needed — expect a Level Zero GPU.
17. `which icpx` and `icpx --version` after setvars.
18. Count Level Zero ICDs (`ls /etc/OpenCL/vendors /etc/v1/icd.d /usr/share/v1/icd.d 2>/dev/null`; also `ls /etc/level-zero* 2>/dev/null`). Warn if more than one Intel ICD (datacenter + Arc split-brain).

### E. XPU venv (`.venv-xpu`)

19. `.venv-xpu` exists; `.venv-xpu/bin/python` sys.prefix is that venv.
20. `import torch`; print `torch.__version__`. FAIL if missing, or if
    the version string looks like CUDA (`+cu`). PASS if XPU / Intel.
21. `torch.xpu.is_available()` is True; print `device_count` and
    `get_device_name(0)` if available.
22. Optional: `import triton` — note version; FAIL only if XPU track is
    otherwise ready and triton is missing.
23. Optional: `import intel_extension_for_pytorch` — **SKIP** if missing
    (optional extra).
24. `.venv-xpu/bin/ft device` — PASS if it reports XPU available and
    does not claim `(none)` for device 0.

## CI parity

The local pre-flight is the developer-side mirror of the `ci` job in
`.github/workflows/ci.yml` — if one changes, the other must follow.
The one deliberate marker difference: the local pytest (check 13)
runs `-m "not xpu and not slow"`, while the CI `ci` job additionally
excludes `needs_weights` (its full marker expression is
`-m "not xpu and not slow and not needs_weights"`), because CI has no
`FREETOKEN_TEST_MODEL` checkpoint to satisfy those tests. Everything
else — the torch-free CPU venv guard and the `ft device` /
`xpu available: False` CLI smoke — is the same check on both sides.
See [docs/ci.md](../docs/ci.md) for the full job map and the WHY
behind each deviation.

## Report format

End with this table, then a one-paragraph verdict (`CPU ready` /
`CPU ready, XPU not` / `CPU+XPU ready` / `not ready`) and the first
three fixes, pointing at `docs/dev-setup.md` section numbers.

```
## Pre-flight

| # | Check | Track | Result | Evidence |
| --- | --- | --- | --- | --- |
| 1 | uname | system | PASS/FAIL | … |
```

Rules:

- **PASS** — requirement met.
- **FAIL** — requirement not met on this machine.
- **SKIP** — not applicable (no GPU, optional tool).
- Do not mark XPU PASS if torch is the CUDA wheel.
- Do not mark CPU PASS if pytest failed or `ft --version` is missing.
- Quote commands you ran.

If the user asked only for the check, stop after the report.
