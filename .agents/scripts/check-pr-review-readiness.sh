#!/usr/bin/env bash
# check-pr-review-readiness.sh — Verify all enabled PR review methods are operational.
#
# Reads .agents/pr-review.yml to determine which methods are expected, then
# checks each one. Always present in the project regardless of config.
#
# Usage: .agents/scripts/check-pr-review-readiness.sh
# Run from the repo root.

set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

PASS="✅"; FAIL="❌"; WARN="⚠️ "
errors=0

# ── Config reader (yq with grep fallback) ──────────────────────────────────
CONFIG_FILE=".agents/pr-review.yml"

cfg_get() {
  local key="$1" fallback="${2:-}"
  if [[ ! -f "$CONFIG_FILE" ]]; then echo "$fallback"; return; fi
  if command -v yq &>/dev/null; then
    local val
    val=$(yq e "${key} // \"\"" "$CONFIG_FILE" 2>/dev/null)
    echo "${val:-$fallback}"
  else
    # PATH-AWARE fallback. The previous version matched only the LEAF name and
    # took the first hit anywhere in the file:
    #     grep -E "^\s*${key##*.}:" … | head -1
    # `enabled` appears under pr_agent, local_review AND dual_review, so
    # `.dual_review.enabled` returned pr_agent's value. On a config with
    # pr_agent enabled and dual_review disabled — the shape this pipeline
    # installs by default — the readiness check ran the dual-review section and
    # reported a missing dual-review.sh as an ERROR for repos that had
    # deliberately turned it off. Same class of defect as #433: a checker
    # reporting on something it never actually read.
    #
    # Walks the two-level path instead: find `^section:` at column 0, then the
    # first `key:` indented beneath it, stopping at the next column-0 key.
    local section="${key#.}"; section="${section%%.*}"
    local leaf="${key##*.}"
    local val
    val=$(awk -v sec="$section" -v leaf="$leaf" '
      # column-0 key: entering a new section (and leaving any previous one)
      /^[A-Za-z_][A-Za-z0-9_]*:/ { in_sec = ($0 ~ "^" sec ":"); next }
      in_sec && $0 ~ "^[[:space:]]+" leaf ":" {
        sub(/^[^:]*:[[:space:]]*/, "")
        print; exit
      }
    ' "$CONFIG_FILE" 2>/dev/null)
    # Strip a trailing comment, surrounding quotes (BOTH kinds — the old sed
    # stripped only `"`, so single-quoted values such as shell_alias: '' came
    # back as the two-character string "''" and read as a configured alias),
    # any trailing CR, and surrounding whitespace.
    val="${val%%#*}"
    val="$(printf '%s' "$val" \
      | sed -e 's/[[:space:]]*$//' -e 's/\r//g' -e "s/^['\"]//" -e "s/['\"]$//")"
    echo "${val:-$fallback}"
  fi
}

PROJECT_NAME="$(cfg_get '.project.name' "$(basename "$PWD")")"

echo ""
echo "═══════════════════════════════════════════════"
echo " ${PROJECT_NAME} — PR Review Readiness Check"
echo "═══════════════════════════════════════════════"

# ── Check: config file itself ────────────────────────────────────────────────
echo ""
echo "── Config ──"
if [[ -f "$CONFIG_FILE" ]]; then
  echo "${PASS} .agents/pr-review.yml present"
else
  echo "${WARN} .agents/pr-review.yml missing — run setup-pr-review.sh to create it"
  echo "     (continuing with defaults: assuming all methods enabled)"
fi

PR_AGENT_ENABLED="$(cfg_get '.pr_agent.enabled' 'false')"
LOCAL_ENABLED="$(cfg_get '.local_review.enabled' 'false')"
DUAL_ENABLED="$(cfg_get '.dual_review.enabled' 'false')"
DCO_ENABLED="$(cfg_get '.dco.required' "$(cfg_get '.dco.enabled' 'false')")"
SHELL_ALIAS="$(cfg_get '.local_review.shell_alias' '')"

# ── Method 1: PR-Agent (cloud, GitHub Actions) ────────────────────────────────
if [[ "$PR_AGENT_ENABLED" == "true" ]]; then
  echo ""
  echo "── Method 1: PR-Agent CI (GitHub Actions) ──"

  if [[ -f ".pr_agent.toml" ]]; then
    echo "${PASS} .pr_agent.toml present"
  else
    echo "${FAIL} .pr_agent.toml missing — copy from playbook/pipelines/pr-reviewer/templates/"
    ((errors++))
  fi

  if [[ -f ".github/workflows/pr-review.yml" ]]; then
    echo "${PASS} .github/workflows/pr-review.yml present"
  else
    echo "${FAIL} .github/workflows/pr-review.yml missing — copy from playbook/pipelines/pr-reviewer/templates/"
    ((errors++))
  fi

  echo "${WARN} DEEPSEEK_API_KEY — cannot verify GitHub secrets locally."
  echo "     Confirm it is set: GitHub → repo → Settings → Secrets → Actions"
else
  echo ""
  echo "── Method 1: PR-Agent CI ── DISABLED (pr_agent.enabled: false)"
fi

# ── Method 2: Local review (claude -p) ─────────────────────────────────────
if [[ "$LOCAL_ENABLED" == "true" ]]; then
  echo ""
  echo "── Method 2: Local review (claude -p) ──"

  if [[ -x ".agents/scripts/pr-review.sh" ]]; then
    echo "${PASS} .agents/scripts/pr-review.sh present and executable"
  elif [[ -f ".agents/scripts/pr-review.sh" ]]; then
    echo "${FAIL} .agents/scripts/pr-review.sh exists but is not executable (run: chmod +x .agents/scripts/pr-review.sh)"
    ((errors++))
  else
    echo "${FAIL} .agents/scripts/pr-review.sh missing — copy from playbook/pipelines/pr-reviewer/scripts/"
    ((errors++))
  fi

  if command -v claude &>/dev/null; then
    echo "${PASS} claude CLI found: $(command -v claude)"
    if claude auth status &>/dev/null 2>&1; then
      echo "${PASS} claude is authenticated"
    else
      echo "${WARN} claude may not be authenticated — run: claude auth login"
    fi
  else
    echo "${FAIL} claude CLI not found — install Claude Code"
    ((errors++))
  fi

  if command -v gh &>/dev/null; then
    echo "${PASS} gh CLI found: $(command -v gh)"
    if gh auth status &>/dev/null 2>&1; then
      echo "${PASS} gh is authenticated"
    else
      echo "${WARN} gh not authenticated — run: gh auth login"
    fi
  else
    echo "${FAIL} gh CLI not found — install: brew install gh"
    ((errors++))
  fi

  if [[ -n "$SHELL_ALIAS" ]]; then
    if [[ -f ".agents/scripts/shell-aliases.sh" ]]; then
      echo "${PASS} shell-aliases.sh present (alias: ${SHELL_ALIAS})"
      echo "     Source it from ~/.zshrc if you haven't already."
    else
      echo "${WARN} shell-aliases.sh missing — alias '${SHELL_ALIAS}' not available"
      echo "     Copy from playbook/pipelines/pr-reviewer/scripts/"
    fi
  fi
else
  echo ""
  echo "── Method 2: Local review ── DISABLED (local_review.enabled: false)"
fi

# ── Method 3: O3 dual-review ────────────────────────────────────────────────
if [[ "$DUAL_ENABLED" == "true" ]]; then
  echo ""
  echo "── Method 3: O3 dual-review ──"

  if [[ -x ".agents/scripts/dual-review.sh" ]]; then
    echo "${PASS} .agents/scripts/dual-review.sh present and executable"
  elif [[ -f ".agents/scripts/dual-review.sh" ]]; then
    echo "${FAIL} .agents/scripts/dual-review.sh exists but is not executable"
    ((errors++))
  else
    echo "${FAIL} .agents/scripts/dual-review.sh missing — copy from playbook/workflow/dual-review.sh"
    ((errors++))
  fi

  # Source .env to check for key
  # shellcheck disable=SC1091
  if [[ -f ".env" ]]; then set -a; source .env 2>/dev/null || true; set +a; fi

  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    echo "${PASS} OPENAI_API_KEY found in environment"
  else
    echo "${FAIL} OPENAI_API_KEY not set — dual-review.sh will fail"
    echo "     Add it to your .env or shell profile."
    ((errors++))
  fi

  if ! command -v jq &>/dev/null; then
    echo "${FAIL} jq not found — required for dual-review.sh (brew install jq)"
    ((errors++))
  else
    echo "${PASS} jq found: $(command -v jq)"
  fi
else
  echo ""
  echo "── Method 3: O3 dual-review ── DISABLED (dual_review.enabled: false)"
fi

# ── DCO workflow ─────────────────────────────────────────────────────────────
if [[ "$DCO_ENABLED" == "true" ]]; then
  echo ""
  echo "── DCO enforcement ──"
  if [[ -f ".github/workflows/dco.yml" ]]; then
    echo "${PASS} .github/workflows/dco.yml present"
  else
    echo "${FAIL} .github/workflows/dco.yml missing — copy from playbook/pipelines/pr-reviewer/templates/"
    ((errors++))
  fi
fi

# ── Manual test hint ──────────────────────────────────────────────────────────
if [[ "$LOCAL_ENABLED" == "true" ]]; then
  echo ""
  echo "── Manual local test ──"
  echo "   .agents/scripts/pr-review.sh <PR-number>"
  echo "   .agents/scripts/pr-review.sh <PR-number> --post"
  if [[ -n "$SHELL_ALIAS" ]]; then
    echo "   ${SHELL_ALIAS} <PR-number>   (if shell-aliases.sh is sourced)"
  fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════"
if [[ $errors -eq 0 ]]; then
  echo " ${PASS} All hard checks passed."
else
  echo " ${FAIL} ${errors} hard check(s) failed — fix them before opening a PR."
fi
echo "═══════════════════════════════════════════════"
echo ""

exit $errors
