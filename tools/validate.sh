#!/usr/bin/env bash
# tools/validate.sh -- the single local command that mirrors the CI gates.
#
# Run it from anywhere in the repo:
#
#   bash tools/validate.sh                 # default scan range
#   bash tools/validate.sh <base-ref>      # e.g. bash tools/validate.sh origin/main
#
# Each gate prints exactly one line:
#
#   PASS  <gate>  <detail>     the gate ran and its command exited 0
#   FAIL  <gate>  <detail>     the gate ran and its command failed
#   SKIP  <gate>  <detail>     the gate could not run here (missing tool /
#                              needs the fleet); a SKIP is NOT a failure
#
# followed by a verdict line and a non-zero exit if anything FAILED. A SKIP
# counts as a pass (it is "not runnable in this environment", not "the code is
# wrong"), which is what lets a fresh clone -- or a box without actionlint --
# run the command and get a meaningful answer on the gates it CAN run.
#
# WHY THIS EXISTS (epic #42, issue #51): CI is the source of truth for whether
# a change is mergeable, but a red CI run is a slow way to find out. This is
# the local mirror: the SAME commands the workflow steps run, so a green local
# run means the corresponding CI job will be green too. The rule (playbook
# standard) is NO DRIFT -- if a gate here ever needs to diverge from the
# matching workflow step, say why in the comment next to it.
#
# GATE -> CI-JOB MAP (which workflow step each gate mirrors):
#   workflow-lint      ci.yml "Pre-flight (actionlint)"        #43
#   secret-scan        ci.yml "Secret scan (gitleaks)"        #44
#   ci:compileall      ci.yml "Byte-compile (syntax-class)"   #45
#   ci:venv-contract   ci.yml "venv contract: torch must NOT be importable"  #45
#   ci:cli-smoke       ci.yml "CLI smoke"                     #45
#   ci:tests           ci.yml "Unit tests (CPU)"             #45
#   conformance        (no CI job yet) -- stub, #46
#   xpu-nightly        xpu.yml (static checks only)           #47/#50
#
# Scaffold-first (this issue): the framework below is real; the gate BODIES
# for #43/#44/#45 are implemented against the current workflows, while
# `conformance` and the execution half of `xpu-nightly` remain stubs until
# their parent issues land the thing they mirror. A stub prints
#   SKIP  <gate>  (not yet implemented -- see #NN)
# and counts as a pass, so `bash tools/validate.sh` is green from the day this
# lands even where the underlying CI job does not exist yet.

set -uo pipefail

# --- resolve the repo root and cd there (gates use repo-relative paths) -----
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

# --- framework: result accounting --------------------------------------------
RESULTS=()     # one "VERDICT gate" entry per gate, in run order
PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

# record <PASS|FAIL|SKIP> <gate> <detail>
record() {
  local verdict="$1" gate="$2" detail="$3"
  case "$verdict" in
    PASS) PASS_COUNT=$((PASS_COUNT + 1)) ;;
    FAIL) FAIL_COUNT=$((FAIL_COUNT + 1)) ;;
    SKIP) SKIP_COUNT=$((SKIP_COUNT + 1)) ;;
    *) echo "validate.sh: internal error: unknown verdict '$verdict'" >&2; return 1 ;;
  esac
  RESULTS+=("$verdict $gate")
  # PASS lines stay quiet-ish; FAIL is the one you want to see loudly.
  if [ "$verdict" = "FAIL" ]; then
    printf 'FAIL  %-18s %s\n' "$gate" "$detail"
  else
    printf '%s  %-18s %s\n' "$verdict" "$gate" "$detail"
  fi
}

# fail <gate> <detail> -- record a FAIL and return non-zero.
fail() { record FAIL "$1" "$2"; return 1; }
# pass <gate> <detail> -- record a PASS and return 0.
pass() { record PASS "$1" "$2"; return 0; }

# --- argument parsing: optional PR base ref ----------------------------------
# Usage: validate.sh [base-ref]
# base-ref is the secret-scan base. Defaults to the PR base when this branch is
# a PR (matches the CI job's origin/main..HEAD), else the tip commit (HEAD^!),
# mirroring ci.yml's "Determine scan range" degradation rules.
PR_BASE=""
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      cat <<'USAGE'
Usage: bash tools/validate.sh [base-ref]

Local mirror of the CI gates (ci.yml + xpu-nightly). Each gate prints one
PASS/FAIL/SKIP line; the script exits non-zero if any gate FAILs.

Arguments:
  base-ref   base ref for the secret-scan range. Defaults to the PR base when
             this branch is a PR (origin/main..HEAD), else the tip commit.

A SKIP is not a failure: a gate that cannot run in this environment (missing
tool, or a check that needs the XPU fleet) skips rather than failing.
USAGE
      exit 0
      ;;
    -*)
      echo "validate.sh: unknown option '$arg' (see --help)" >&2
      exit 2
      ;;
    *)
      if [ -n "$PR_BASE" ]; then
        echo "validate.sh: only one positional argument (base-ref) is supported; got '$arg'" >&2
        exit 2
      fi
      PR_BASE="$arg"
      ;;
  esac
done

# Compute the secret-scan range (mirrors ci.yml's "Determine scan range"):
#   - explicit base-ref argument, if given and resolvable
#   - else the PR base (origin/HEAD or origin/main) .. HEAD, when on a PR branch
#   - else HEAD^! (the tip commit only -- never a bare HEAD, which would walk
#     full history; matches the CI degradation rule)
ZERO_SHA="0000000000000000000000000000000000000000"
resolves() { git cat-file -e "${1}^{commit}" 2>/dev/null; }
SCAN_RANGE=""
if [ -n "$PR_BASE" ]; then
  if resolves "$PR_BASE"; then
    SCAN_RANGE="${PR_BASE}..HEAD"
  else
    echo "validate.sh: base ref '$PR_BASE' does not resolve to a commit here -- falling back to the tip commit." >&2
    SCAN_RANGE="HEAD^!"
  fi
else
  # On a PR branch? CI uses the PR's base.sha; the local analog is the merge
  # base with the default branch when there is one to compare against.
  if git rev-parse --verify origin/HEAD >/dev/null 2>&1; then
    PR_BASE_RESOLVED="$(git merge-base origin/HEAD HEAD 2>/dev/null || true)"
    if [ -n "$PR_BASE_RESOLVED" ] && [ "$PR_BASE_RESOLVED" != "$ZERO_SHA" ] && [ "$PR_BASE_RESOLVED" != "$(git rev-parse HEAD)" ]; then
      SCAN_RANGE="${PR_BASE_RESOLVED}..HEAD"
    fi
  fi
  if [ -z "$SCAN_RANGE" ]; then
    SCAN_RANGE="HEAD^!"
  fi
fi

echo "tools/validate.sh -- local mirror of the CI gates"
echo "repo:   $ROOT"
echo "range:  $SCAN_RANGE"
echo

# --- gate 1: workflow-lint (mirrors ci.yml "Pre-flight (actionlint)") --------
# CI pins actionlint 1.7.12 via a pinned download script. Locally we use
# whatever actionlint is on PATH (no network) and SKIP -- with a notice -- if
# it is not installed, so a fresh clone still runs the rest. The file glob is
# the same as CI (.github/workflows/*.yml).
gate_workflow_lint() {
  if ! command -v actionlint >/dev/null 2>&1; then
    record SKIP workflow-lint "actionlint not installed (CI pins 1.7.12) -- install it to run this gate locally"
    return 0
  fi
  local files
  files=$(find .github/workflows -maxdepth 1 -name '*.yml' -type f 2>/dev/null | wc -l)
  if actionlint .github/workflows/*.yml >/dev/null 2>&1; then
    pass workflow-lint "actionlint: $files workflow file(s) clean"
  else
    local rc=$?
    fail workflow-lint "actionlint: $files workflow file(s) -- see actionlint output above (exit $rc)"
  fi
}

# --- gate 2: secret-scan (mirrors ci.yml "Secret scan (gitleaks)") ----------
# CI runs: gitleaks git --log-opts="$RANGE" --redact --no-banner --exit-code 2
# and treats exit 2 (leaks found) as a HARD FAIL, any other non-zero as a
# scanner/infra error (also fail). We reproduce that 0/2/other branching -- so
# this gate does NOT go through run_gate, which only knows 0-vs-nonzero. If
# gitleaks is not installed locally we SKIP (the CI job always has it; a box
# without it simply cannot run this gate).
gate_secret_scan() {
  if ! command -v gitleaks >/dev/null 2>&1; then
    record SKIP secret-scan "gitleaks not installed (CI pins 8.30.1) -- install it to run this gate locally"
    return 0
  fi
  local tmp_report
  tmp_report="$(mktemp "${TMPDIR:-/tmp}/gitleaks-validate.XXXXXX")"
  # --log-opts is intentionally unquoted: gitleaks expects the value as a
  # single --log-opts argument (it accepts the "base..head" range in that form),
  # and we deliberately do not want it split.
  # shellcheck disable=SC2086
  gitleaks git --log-opts="$SCAN_RANGE" --redact --no-banner --exit-code 2 \
    --report-format json --report-path "$tmp_report" >/dev/null 2>&1
  local code=$?
  if [ "$code" -eq 0 ]; then
    rm -f "$tmp_report"
    pass secret-scan "gitleaks: 0 findings in $SCAN_RANGE"
  elif [ "$code" -eq 2 ]; then
    local n
    n=$(grep -c '"Fingerprint"' "$tmp_report" 2>/dev/null || echo 0)
    rm -f "$tmp_report"
    fail secret-scan "gitleaks: $n leak(s) found in $SCAN_RANGE -- a finding is a hard fail here (no historical baseline)"
  else
    rm -f "$tmp_report"
    fail secret-scan "gitleaks exited $code (not 0=clean or 2=leaks) -- scanner/config problem, not a benign finding"
  fi
}

# --- gate 3: ci venv (the four #45 gates share the CPU .venv) ---------------
# ci.yml builds the CPU .venv (Python 3.11, NO torch) and runs the four steps
# in it. Locally we reuse the developer's .venv (docs/dev-setup.md) instead of
# building a throwaway one. If .venv is absent there is nowhere to run the
# python-based gates, so all four SKIP with a pointer to the setup doc.
VENV_PY=""
if [ -x .venv/bin/python ]; then
  VENV_PY=".venv/bin/python"
fi

# gate 3a: ci:compileall -- mirrors ci.yml "Byte-compile (syntax-class check)":
#   python -m compileall -q python/
gate_ci_compileall() {
  if [ -z "$VENV_PY" ]; then
    record SKIP ci:compileall "no .venv -- create it per docs/dev-setup.md (section 5) to run the ci gates"
    return 0
  fi
  if "$VENV_PY" -m compileall -q python/ >/dev/null 2>&1; then
    pass ci:compileall "byte-compiled python/ -- no syntax-class errors"
  else
    fail ci:compileall "python -m compileall -q python/ reported a syntax-class error"
  fi
}

# gate 3b: ci:venv-contract -- mirrors ci.yml "venv contract: torch must NOT be
# importable": if `import torch` SUCCEEDS in the CPU venv the dual-venv contract
# is broken. This is the load-bearing separation (CPU venv never has torch).
gate_ci_venv_contract() {
  if [ -z "$VENV_PY" ]; then
    record SKIP ci:venv-contract "no .venv -- create it per docs/dev-setup.md (section 5) to run the ci gates"
    return 0
  fi
  if "$VENV_PY" -c "import torch" 2>/dev/null; then
    fail ci:venv-contract "'import torch' SUCCEEDED in .venv -- the dual-venv contract requires torch ONLY in .venv-xpu (docs/dev-setup.md)"
  else
    pass ci:venv-contract "torch is not importable in .venv (contract holds)"
  fi
}

# gate 3c: ci:cli-smoke -- mirrors ci.yml "CLI smoke": ft --version, ft --help,
# and `ft device` must report 'xpu available: False' (a GPU-less box reporting
# True is the fleet drift this gate catches).
#
# Two deliberate adaptations vs the CI step (documented, not drift):
#   1. We invoke the entrypoint as `<venv>/bin/ft` (the installed console
#      script) instead of `ft`. On CI the venv's bin is on PATH, so bare `ft`
#      resolves; locally the venv is not activated, so we address the same
#      entrypoint by its absolute path. Same command, same binary.
#   2. CI runs `ft device | tee device.out` (no set -e) and decides on a
#      pipe-free grep of the tee'd file -- because with pipefail a grep -q that
#      matches early can let the upstream producer die on SIGPIPE and flip the
#      pipeline's exit status to the producer's non-zero. We reproduce that:
#      capture `ft device` to a file, run the decision as a separate pipe-free
#      grep. Note `print_device_report` intentionally returns 1 when no XPU is
#      present (a "no device" status), so we MUST NOT gate on ft's exit code --
#      the pass/fail signal is the grep, exactly as in CI.
gate_ci_cli_smoke() {
  if [ -z "$VENV_PY" ]; then
    record SKIP ci:cli-smoke "no .venv -- create it per docs/dev-setup.md (section 5) to run the ci gates"
    return 0
  fi
  local FT=".venv/bin/ft"
  local device_out
  device_out="$(mktemp "${TMPDIR:-/tmp}/ft-device-validate.XXXXXX")"
  # --version and --help must both exit 0 (a healthy CLI).
  if ! "$FT" --version >/dev/null 2>&1 || ! "$FT" --help >/dev/null 2>&1; then
    rm -f "$device_out"
    fail ci:cli-smoke "ft --version or ft --help failed in .venv"
    return 0
  fi
  # Capture the device report; its exit code is NOT the pass/fail signal (see
  # note 2 above) -- a healthy GPU-less box returns 1 here.
  "$FT" device >"$device_out" 2>/dev/null
  if grep -q "xpu available: False" "$device_out"; then
    rm -f "$device_out"
    pass ci:cli-smoke "ft --version/--help OK; ft device reports 'xpu available: False'"
  else
    # Either this box actually has an XPU (so the 'False' assertion is wrong
    # here, not a regression) or device-detection drifted. Show the report so
    # the operator can tell which.
    fail ci:cli-smoke "ft device did not report 'xpu available: False' -- either this box has an XPU (expected on XPU dev machines; run the xpu nightly there) or device-detection drifted. Report:"
    sed 's/^/    /' "$device_out"
    rm -f "$device_out"
  fi
}

# gate 3d: ci:tests -- mirrors ci.yml "Unit tests (CPU)":
#   pytest -m "not xpu and not slow and not needs_weights"
# The CI job installs pytest-json-ctrf and writes a CTRF artifact; locally we
# run the same selection without the CTRF reporter (it is a CI-artifact concern,
# not a pass/fail one) -- a documented, deliberate local/CI divergence.
gate_ci_tests() {
  if [ -z "$VENV_PY" ]; then
    record SKIP ci:tests "no .venv -- create it per docs/dev-setup.md (section 5) to run the ci gates"
    return 0
  fi
  if ! "$VENV_PY" -c "import pytest" >/dev/null 2>&1; then
    record SKIP ci:tests "pytest not importable in .venv -- 'pip install -e .[dev]' to add the dev extras"
    return 0
  fi
  if "$VENV_PY" -m pytest -m "not xpu and not slow and not needs_weights" -q >/dev/null 2>&1; then
    pass ci:tests "CPU suite green (xpu/slow/needs_weights deselected, as in CI)"
  else
    fail ci:tests "pytest -m 'not xpu and not slow and not needs_weights' reported failures"
  fi
}

# --- gate 4: conformance (STUB -- parent issue #46 has not landed) ----------
# When #46 adds the `conformance` CI job (sycl_ext rule + allowlist, no-CUDA,
# single version source) this stub is replaced by the same grep/git checks that
# job runs. Until then: a pass-by-skip.
gate_conformance() {
  record SKIP conformance "(not yet implemented -- see #46: sycl_ext rule, no-CUDA, single version source)"
}

# --- gate 5: xpu-nightly (#47/#50) ------------------------------------------
# The XPU nightly runs on the self-hosted B70 fleet, so its EXECUTION cannot be
# mirrored locally (a CPU box has no XPU and must never pretend to). We do the
# half that IS statically checkable on any box -- the workflow file exists, is
# actionlint-clean (when actionlint is available), and carries the two safety
# invariants (the repo-identity guard on the build job, and the no-fallback
# CI_RUNNER runs-on) -- then SKIP the execution with a pointer to the fleet.
gate_xpu_nightly() {
  local wf=".github/workflows/xpu.yml"
  if [ ! -f "$wf" ]; then
    record SKIP xpu-nightly "$wf not present in this tree"
    return 0
  fi
  local problems=()
  local lint_state="skipped (actionlint not installed)"
  # actionlint-clean (only assertable when the tool is present)
  if command -v actionlint >/dev/null 2>&1; then
    if actionlint "$wf" >/dev/null 2>&1; then
      lint_state="actionlint-clean"
    else
      problems+=("actionlint: $wf has lint findings")
    fi
  fi
  # repo-identity guard present (a fork must never reach the fleet build job)
  if ! grep -q "github.repository == 'Performant-Labs/FreeToken-Intel'" "$wf"; then
    problems+=("missing repo-identity guard (build job must gate on github.repository)")
  fi
  # no silent CPU fallback on the fleet label (the retired `|| 'ubuntu-latest'` bug)
  if grep -Eq "runs-on:.*\|\|" "$wf"; then
    problems+=("a runs-on uses a '||' fallback -- the XPU nightly must fail loudly, never fall back to a GPU-less runner")
  fi
  if [ "${#problems[@]}" -gt 0 ]; then
    fail xpu-nightly "static checks: ${problems[*]}"
  else
    record SKIP xpu-nightly "static checks OK (exists, $lint_state, repo-identity guard + no runs-on fallback present) -- execution needs an intel-xpu runner; run on the fleet"
  fi
}

# --- run every gate in CI order ----------------------------------------------
gate_workflow_lint
gate_secret_scan
gate_ci_compileall
gate_ci_venv_contract
gate_ci_cli_smoke
gate_ci_tests
gate_conformance
gate_xpu_nightly

# --- verdict -----------------------------------------------------------------
echo
if [ "$FAIL_COUNT" -gt 0 ]; then
  echo "VERDICT: $FAIL_COUNT FAILED ($PASS_COUNT passed, $SKIP_COUNT skipped) -- NOT mergeable until the FAILs are addressed."
  exit 1
else
  echo "VERDICT: OK ($PASS_COUNT passed, $SKIP_COUNT skipped, 0 failed)."
  if [ "$SKIP_COUNT" -gt 0 ]; then
    echo "         Note: $SKIP_COUNT gate(s) were skipped (missing local tool or fleet-only). They still run in CI."
  fi
  exit 0
fi
