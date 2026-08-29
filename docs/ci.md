# CI

FreeToken-Intel has five checks. Four gate every PR to `main`
(`ci.yml`); one runs nightly on the B70 fleet (`xpu.yml`). An optional
bot review (`pr-review.yml`) runs in parallel and is advisory — it
never gates.

This page is the repo-side complement to the playbook's
`infrastructure/ci-platform.md`. If the two disagree, trust the
playbook page for org mechanics (runner fleet, variable wiring) and
this page for the repo's checks and the WHY behind each deviation
from it.

## Job map

```mermaid
flowchart TD
    subgraph "ci.yml (every PR + push to main)"
        PF["pre-flight<br/>(actionlint)"] --> SS["secret-scan<br/>(gitleaks)"]
        PF --> CI["ci<br/>(tests + CLI smoke)"]
        PF --> CF["conformance<br/>(CUDA→SYCL + version)"]
    end
    subgraph "xpu.yml (nightly 02:30 UTC + dispatch)"
        CHK["check<br/>(hosted, cheap)"] -->|build=true| BLD["build<br/>(B70 fleet)"]
    end
    PR["PR-Agent review<br/>(advisory, bot)"]
```

| Check | Where | What it guards |
| --- | --- | --- |
| **pre-flight** | hosted | Workflow files parse and lint. Fails first, before any expensive job. |
| **secret-scan** | hosted | No secret enters the tree. **Hard-fails** on a finding. |
| **ci** | hosted | The CPU contract: torch-free venv, full CPU suite, live CLI smoke. |
| **conformance** | hosted | The port's invariants: SYCL uses `sycl::ext::oneapi`, no CUDA backdoors, one version source. |
| **xpu-nightly** | B70 fleet | The XPU half of the contract: torch present, `torch.xpu` alive, xpu-marked tests pass. |
| **PR-Agent review** | fleet | Advisory bot score + label. Not a gate. |

`pre-flight` has no `needs:` and runs first: a broken workflow file
fails fast and cheap instead of being discovered three jobs deep.
`secret-scan` and `ci` are independent of each other.

## Why these shapes

**Public repo ⇒ `ubuntu-latest` hardcoded.** A fork PR must never
execute on owned hardware. The org `CI_RUNNER` variable is a
private-org mechanism and does not apply to a public repo, so the
four `ci.yml` jobs pin `ubuntu-latest` directly. Consequently they
deliberately carry **no** `github.repository == ...` identity guard —
all their jobs run hosted, so there is nothing to guard. `xpu.yml` is
the opposite: its `build` job runs on the self-hosted B70 fleet, so
**every** job in that workflow carries the explicit
`github.repository == 'Performant-Labs/FreeToken-Intel'` guard. That
guard is the belt to the org-variable suspenders (a fork cannot see
the org variable, so its `runs-on` would be empty anyway); the pair
does not depend on the other, and the guard survives an org transfer.

**The XPU track is nightly-only.** Fork PRs get no per-PR access to
the fleet, so the torch/XPU path is verified on a `main`-only
schedule instead of on every push. The trade is explicit: per-PR CI
stays fork-safe and torch-free; the torch path is verified on a
trusted, scheduled runner. A broken torch path is therefore caught by
the nightly — up to 24 h after merge — not by the PR. That is the same
shape as upstream FreeToken's release-only CI (below).

**Gitleaks hard-fails here; the playbook warns.** The playbook runs
its scan warn-only because 21 repos carry untriaged historical
secrets. This repo has zero historical secret commits and is public,
so a warn-only period would be a standard nobody reads — the exact
failure mode the playbook's own doc warns about. Here, exit code 2
(leaks found) uploads the redacted report **and** fails the job. If
the playbook ever flips this repo onto warn-only, this paragraph
explains the trade.

**Python 3.11, not 3.12.** Both dev venvs (`.venv` and `.venv-xpu`)
are 3.11. The CPU CI venv must match the XPU venv's minor so their
wheel tags (cp311) stay aligned; a CPU venv on a different minor is a
silent divergence trap.

**No no-op fallback on the XPU runner.** `xpu.yml` uses
`runs-on: ${{ vars.CI_RUNNER }}` with **no** `|| 'ubuntu-latest'`.
The playbook's original `||` was a bug it has since retired: a repo
that couldn't see the org variable silently ran hosted and stayed
green, invisibly (15 of 21 repos were doing exactly that). Here, a
missing or stale variable yields an empty `runs-on` and the job fails
immediately — drift announces itself instead of being found months
later. The check job (below) also fails the build if the runner is
re-pointed at a box with no XPU.

## The dual-venv contract

The load-bearing separation in this repo: `.venv` (CPU) must **never**
contain torch; torch (the XPU build) lives **only** in `.venv-xpu`.
The whole CUDA→SYCL premise rests on the CPU side never importing a
CUDA/CPU torch. Two CI guards enforce the two halves, and a violation
of either is a **contract breach, not a flake**:

* `ci.yml` → `ci` job → *"venv contract: torch must NOT be
  importable"*: fails if `import torch` **succeeds** in the CPU venv.
* `ci.yml` → `ci` job → *CLI smoke*: asserts `ft device` reports
  `xpu available: False` on the GPU-less hosted runner. A hosted
  runner that reports an XPU as available is fleet misconfiguration —
  the assertion lets it fail loudly instead of `|| true`-ing away.
  (The assertion reads a `tee`'d file with a pipe-free `grep`, not the
  pipeline, to avoid the `pipefail`/`grep -q`/SIGPIPE false-negative —
  see `tools/validate.sh` notes.)
* `xpu.yml` → `build` job → *"venv contract: .venv-xpu …"*: the other
  half — the venv where torch **is** required and `torch.xpu.is_available()`
  must be True.

## The conformance gate

`ci.yml` → `conformance` job. The cheapest, most declarative gate in
the repo: pure `bash` + `git` + `grep`, no `pip install`, no Node, no
toolchain. It exists to make the port's non-negotiable invariants
*drift-announcing* — a violation fails the PR the moment it lands, in
a step whose name says what it is, instead of surfacing as a flaky
build or a wrong wheel a release later. Three checks, three
invariants (tracked in [#46]):

* **SYCL extension rule** — every file under
  `python/freetoken/kernel/csrc/` that `#include`s
  `<sycl/sycl.hpp>` must also reference the `sycl::ext::oneapi`
  extension namespace (e.g. a `sycl::ext::oneapi::accessor_property_list`
  accessor, or any `sycl::ext::oneapi::*` type). The "never compile
  against a fake SYCL" rule: a kernel written against the plain `sycl::`
  API binds none of the XPU extensions, so it would *look* portable and
  silently stop being GPU work. A file that legitimately uses only the
  standard SYCL API is listed in `.github/ci-conformance-allowlist`
  (one path per line, `#` comments); a new file that includes the
  header, uses no `sycl::ext::oneapi`, and is not allowlisted **fails**
  the job and names the file. (The namespace is `sycl::ext::oneapi`
  because that is where the installed oneAPI 2026.1 toolkit puts its
  XPU extensions — there is no `sycl_ext::` namespace in that toolkit.)
* **No CUDA backdoors** — a `#include <cuda*>`, `cublas`, `cub/`, or
  `nvcc` reference in any source file is a premise violation, not a
  style issue: the whole point of the port is CUDA→SYCL. The check is
  scoped to code extensions (`.c .cc .cpp .cxx .h .hpp .hxx .cu .cuh
  .py .toml .cfg`); docs may *reference* CUDA by name (the
  CUDA→Intel map in `docs/architecture.md`) without tripping the gate.
* **Version-source singularity** — `python/freetoken/version.py` is
  the *only* place a `__version__` literal for the freetoken package
  is declared; `pyproject.toml` keeps `dynamic = ["version"]` pointing
  at it. A second `__version__` anywhere in the tracked tree is
  two-sources-of-truth drift, exactly what the playbook's
  release-conformance gate exists to stop. Two exemptions: the
  separate `freetoken-kernel-cache` distributable owns its own
  independent version (allowlisted in the job), and the root
  `pyproject.toml`'s `version = {attr = ...}` is the *mechanism* that
  points at the source, not a literal.

**Why greps and not a compiler.** A full `icpx -fsycl` compile of the
kernel sources is the *strongest* form of this check but needs the
oneAPI toolchain, which lives on the B70 fleet (and is not a public
hosted image) — that belongs to the nightly, not to a fork-safe per-PR
gate. The greps are the part that must run on every PR *now*; they
catch the invariant at the text level where it can be checked for free,
and the nightly's real compile is the backstop that catches what a
grep can't. (See "XPU nightly operations" below.)

**To add a legitimately-standard-SYCL file:** add its path to
`.github/ci-conformance-allowlist` in the same PR, with a comment
saying why it needs no `sycl::ext::oneapi`. If the file later starts
using `kernel-sycl`'s extension API, remove it from the list in that PR.
Allowlisting a file *so it can dodge the check* is exactly the drift
the rule exists to catch — treat an allowlist addition as a decision
the conformance gate is there to police.

## CTRF artifacts

Both `ci.yml` (CPU) and `xpu.yml` (XPU) upload a CTRF JSON test
report as a workflow artifact (`ctrf-reports` / `xpu-ctrf-<run>-<attempt>`),
30-day retention. There is **no PR check-run summary** for CTRF
(playbook parity): reading it today means downloading the artifact
from the run. The CTRF upload is the **only** `continue-on-error` in
either workflow, and it runs `if: always()` — an upload failure
(network blip) must never turn a red or green test run into a
confusing third state, and a failing run is exactly when the report is
most needed.

## XPU nightly operations

* **Runner label:** `CI_RUNNER` (org variable, resolves to the
  self-hosted B70 fleet — e.g. `jupiter-b70`). No fallback.
* **Stamp-skip (idempotency):** a cheap hosted `check` job compares
  `GITHUB_SHA` against the last **green** `xpu-nightly` run on the
  branch (via the GitHub API). Equal → `build=false` and the fleet job
  is skipped with a visible "already green at this commit" notice. A
  no-op night must not touch the B70. A `workflow_dispatch` with
  `force: true` bypasses the check. Runs are serialized per-branch
  (`concurrency`, no cancel-in-progress) so two runs never share the
  one GPU runner.
* **Is the GPU actually there (2-min diagnostic):** the `build` job's
  first step primes oneAPI (`setvars-xpu.sh`) and then runs `sycl-ls`.
  If `sycl-ls` is missing → oneAPI absent from the image; if it runs
  but finds no Level Zero device → the GPU ICD is not loaded. Both
  fail loudly and early, in the step whose name says what happened.
* **When it goes red:** distinguish **toolchain drift** (oneAPI/venv
  drift — re-bake the runner image, re-run `docs/dev-setup.md`) from a
  **real regression** (a test that used to pass now fails on the B70).
  The CTRF artifact tells you which tests failed; the `sycl-ls` output
  in the run log tells you whether the GPU was even present.

## Upstream comparison (vs FlashML-org/FreeToken)

We are porting FlashML's FreeToken. Upstream has **no PR-gated CI at
all** — its workflows are two publishing pipelines (nightly wheels,
tag release) plus Copilot review. Merges to `main` run no tests, no
lint, no secret scan; a broken `main` is caught by the nightly, i.e.
up to 24 h after merge. Our shape gates **per-PR**; they gate
**per-release**. Four of upstream's release-ops practices are strictly
better, so we borrowed them:

| # | Upstream practice | Where it landed here |
| --- | --- | --- |
| 1 | SHA-stamp idempotency (skip a no-op nightly before touching the build node) | `xpu.yml` `check` job (above) |
| 2 | Build/publish split with credential quarantine (build on self-hosted, publish from a hosted runner behind an `environment:` reviewer gate; the cross-repo token is never present on the build node) | Governing principle recorded for #31's future wheel work: **Uranus builds, GitHub-hosted publishes, behind an `environment:` reviewer gate, with tools run via `pipx run` / a fresh venv** |
| 3 | Fork-proofing (repo-identity guard on every self-hosted job) | `xpu.yml` guards every job; `ci.yml` comments explain why it needs none |
| 4 | Pin everything (actions to full SHAs; publish tools via `pipx run`) | All `uses:` in both workflows are SHA-pinned with a `# vN` comment (below) |

## Rules

* **Pin actions deliberately.** Every `uses:` in `ci.yml`, `xpu.yml`,
  and the vendored `pr-review-reusable.yml` (invoked by ci.yml's
  PR-agent job) is pinned to a full commit SHA (not a floating tag)
  with a trailing `# vN` comment naming the release the SHA is. Bump
  pins **deliberately, one PR per action, never implicitly** — an
  unpinned tag is the same failure class as the playbook's retired
  `CI_RUNNER || 'ubuntu-latest'` bug: a silent, invisible behavior
  change (a new action release could change checkout or upload
  semantics under a green-looking run).
* **No mutable interpreters in publish-shaped steps.** Any step that
  touches artifacts or credentials installs its tooling into the job's
  own fresh venv (or runs it via `pipx run`); it never relies on a
  pre-installed interpreter on the runner image. This is upstream's
  `pipx run twine` rule, inherited by #31's wheel workflows so a
  compromised or drifted runner can't substitute a trojaned installed
  copy.

## Related

* Machine-side story (two venvs, driver, oneAPI, local pre-flight):
  [docs/dev-setup.md](dev-setup.md)
* Org-side story (runner fleet, `CI_RUNNER`, variable wiring):
  playbook `infrastructure/ci-platform.md`
* Local mirror of the CI gates (run `tools/validate.sh` before
  pushing): [tools/validate.sh](../tools/validate.sh)
