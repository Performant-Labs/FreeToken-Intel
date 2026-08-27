#!/usr/bin/env bash
# setup-pr-review.sh — Interactive wizard that generates .agents/pr-review.yml
#
# Usage:
#   .agents/scripts/setup-pr-review.sh [--update]
#
# Flags:
#   --update   Re-run over an existing config (prompts show current values as defaults)
#
# Prerequisites detected automatically:
#   gh CLI authenticated, claude CLI authenticated,
#   DEEPSEEK_API_KEY / OPENAI_API_KEY in env or .env,
#   GitHub remote present, Playwright in package.json / pyproject.toml
#
# After running: copy or symlink the canonical scripts and workflow templates
# from playbook/pipelines/pr-reviewer/ into this project.

set -euo pipefail

# ── Colours ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}▸${RESET} $*"; }
ok()      { echo -e "${GREEN}✓${RESET} $*"; }
warn()    { echo -e "${YELLOW}⚠${RESET}  $*"; }
err()     { echo -e "${RED}✗${RESET} $*"; }
heading() { echo -e "\n${BOLD}$*${RESET}"; }
divider() { echo -e "${CYAN}────────────────────────────────────────────${RESET}"; }

# ── Helpers ────────────────────────────────────────────────────────────────

# ask <prompt> <default>  →  prints answer to stdout
ask() {
  local prompt="$1" default="$2"
  local display_default=""
  [[ -n "$default" ]] && display_default=" [${default}]"
  echo -en "${BOLD}${prompt}${RESET}${display_default}: " >&2
  local answer
  read -r answer
  echo "${answer:-$default}"
}

# ask_yn <prompt> <default y|n>  →  exits 0 for yes, 1 for no
ask_yn() {
  local prompt="$1" default="${2:-n}"
  local choices="y/n"
  [[ "$default" == "y" ]] && choices="Y/n" || choices="y/N"
  echo -en "${BOLD}${prompt}${RESET} (${choices}): " >&2
  local answer
  read -r answer
  answer="${answer:-$default}"
  [[ "${answer,,}" == "y" ]]
}

# yq_get <file> <key> <fallback>
yq_get() {
  local file="$1" key="$2" fallback="${3:-}"
  if command -v yq &>/dev/null; then
    yq e "${key} // \"${fallback}\"" "$file" 2>/dev/null || echo "$fallback"
  else
    # plain-grep fallback — handles simple scalar values
    local stripped="${key##*.}"
    grep -E "^\s*${stripped}:" "$file" 2>/dev/null \
      | head -1 | sed 's/.*: *//;s/"//g;s/\r//' || echo "$fallback"
  fi
}

# ── Phase 1: Silent detection ───────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
CONFIG_FILE="${PROJECT_ROOT}/.agents/pr-review.yml"
UPDATE_MODE=false
[[ "${1:-}" == "--update" ]] && UPDATE_MODE=true

heading "PR Reviewer Setup Wizard"
divider
info "Detecting environment…"

# Existing config
EXISTING_CONFIG=false
[[ -f "$CONFIG_FILE" ]] && EXISTING_CONFIG=true

if $EXISTING_CONFIG && ! $UPDATE_MODE; then
  warn "Config already exists at .agents/pr-review.yml"
  if ! ask_yn "Re-run wizard to update it?" "n"; then
    info "Aborting. Run with --update to force."
    exit 0
  fi
  UPDATE_MODE=true
fi

# GitHub remote
HAS_REMOTE=false
GITHUB_REMOTE=""
if git remote get-url origin &>/dev/null 2>&1; then
  GITHUB_REMOTE="$(git remote get-url origin 2>/dev/null)"
  if [[ "$GITHUB_REMOTE" == *github.com* ]]; then
    HAS_REMOTE=true
    ok "GitHub remote: ${GITHUB_REMOTE}"
  else
    warn "Remote found but not GitHub: ${GITHUB_REMOTE}"
  fi
else
  warn "No git remote found — CI features will be skipped"
fi

# gh CLI
GH_AUTHED=false
if command -v gh &>/dev/null && gh auth status &>/dev/null 2>&1; then
  GH_AUTHED=true
  ok "gh CLI: authenticated"
else
  warn "gh CLI: not authenticated (run: gh auth login)"
fi

# claude CLI
CLAUDE_AUTHED=false
if command -v claude &>/dev/null; then
  CLAUDE_AUTHED=true
  ok "claude CLI: found"
else
  warn "claude CLI: not found (local review will be skipped)"
fi

# DeepSeek API key
DEEPSEEK_KEY=false
if [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then
  DEEPSEEK_KEY=true
  ok "DEEPSEEK_API_KEY: found in environment"
elif [[ -f "${PROJECT_ROOT}/.env" ]] && grep -q "DEEPSEEK_API_KEY" "${PROJECT_ROOT}/.env" 2>/dev/null; then
  DEEPSEEK_KEY=true
  ok "DEEPSEEK_API_KEY: found in .env"
else
  warn "DEEPSEEK_API_KEY: not found (PR-Agent CI will need it as a GitHub secret)"
fi

# OpenAI API key (for O3 dual-review)
OPENAI_KEY=false
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  OPENAI_KEY=true
  ok "OPENAI_API_KEY: found in environment"
elif [[ -f "${PROJECT_ROOT}/.env" ]] && grep -q "OPENAI_API_KEY" "${PROJECT_ROOT}/.env" 2>/dev/null; then
  OPENAI_KEY=true
  ok "OPENAI_API_KEY: found in .env"
else
  warn "OPENAI_API_KEY: not found (O3 dual-review will be skipped)"
fi

# Playwright
HAS_PLAYWRIGHT=false
if [[ -f "${PROJECT_ROOT}/package.json" ]] && grep -q "playwright" "${PROJECT_ROOT}/package.json" 2>/dev/null; then
  HAS_PLAYWRIGHT=true
  ok "Playwright: detected in package.json"
elif [[ -f "${PROJECT_ROOT}/pyproject.toml" ]] && grep -q "playwright" "${PROJECT_ROOT}/pyproject.toml" 2>/dev/null; then
  HAS_PLAYWRIGHT=true
  ok "Playwright: detected in pyproject.toml"
else
  warn "Playwright: not detected (test tiers T2+ will be skipped)"
fi

# Pull defaults from existing config when in update mode
cfg_get() {
  local key="$1" fallback="${2:-}"
  if $UPDATE_MODE && $EXISTING_CONFIG; then
    yq_get "$CONFIG_FILE" "$key" "$fallback"
  else
    echo "$fallback"
  fi
}

# ── Phase 2: Questions ──────────────────────────────────────────────────────

divider
heading "Project"

PROJECT_NAME="$(cfg_get '.project.name' "$(basename "$PROJECT_ROOT")")"
PROJECT_NAME="$(ask "Project name" "$PROJECT_NAME")"

PROJECT_TZ="$(cfg_get '.project.timezone' "America/Los_Angeles")"
PROJECT_TZ="$(ask "Timezone (for review footers)" "$PROJECT_TZ")"

# ── PR-Agent CI ──
# PR_AGENT_ENABLED drives what this wizard INSTALLS (the caller workflow, the
# secret hints, the next-steps block). It is deliberately not written into
# .agents/pr-review.yml: the presence of .github/workflows/pr-review.yml IS the
# on-switch, and a second switch in the config that can silently disagree with
# the first is worse than none (#490).
PR_AGENT_ENABLED=false
PR_AGENT_SKIP_DOCS=false
PR_AGENT_STATUS_CHECK=true
PR_AGENT_HIGH_STAKES="high-stakes"
PR_AGENT_MODEL="deepseek/deepseek-v4-pro"

if $HAS_REMOTE; then
  divider
  heading "PR-Agent CI (GitHub Actions)"
  info "Requires DEEPSEEK_API_KEY as a GitHub secret."
  _default_pra="$(cfg_get '.pr_agent.enabled' 'n')"; [[ "$_default_pra" == "true" ]] && _default_pra="y"
  if ask_yn "Enable PR-Agent automated review?" "$_default_pra"; then
    PR_AGENT_ENABLED=true

    if ! $DEEPSEEK_KEY; then
      warn "DEEPSEEK_API_KEY not found locally."
      info "Make sure the secret is set in GitHub: gh secret set DEEPSEEK_API_KEY"
    fi

    PR_AGENT_MODEL="$(cfg_get '.pr_agent.model' 'deepseek/deepseek-v4-pro')"
    PR_AGENT_MODEL="$(ask "PR-Agent model" "$PR_AGENT_MODEL")"

    PR_AGENT_HIGH_STAKES="$(cfg_get '.pr_agent.high_stakes_label' 'high-stakes')"
    PR_AGENT_HIGH_STAKES="$(ask "High-stakes label name" "$PR_AGENT_HIGH_STAKES")"

    _default_skip="$(cfg_get '.pr_agent.skip_docs_only' 'n')"; [[ "$_default_skip" == "true" ]] && _default_skip="y"
    ask_yn "Skip review for docs-only / planning-only PRs?" "$_default_skip" && PR_AGENT_SKIP_DOCS=true

    _default_sc="$(cfg_get '.pr_agent.status_check' 'y')"; [[ "$_default_sc" == "false" ]] && _default_sc="n"
    ask_yn "Post a commit status check (green/red) after review?" "${_default_sc:-y}" && PR_AGENT_STATUS_CHECK=true || PR_AGENT_STATUS_CHECK=false

    # No cancel_in_flight prompt. Cancellation is owned by the caller workflow's
    # `concurrency:` block, which is event-scoped on purpose:
    #   cancel-in-progress: ${{ github.event_name != 'issue_comment' }}
    # A push SHOULD cancel a superseded review; a PR comment must NEVER cancel a
    # running one, because GitHub evaluates concurrency at run CREATION, before
    # any job-level `if`. Reducing that to a config boolean would be a
    # regression, so the key was deleted rather than wired up (#490).
  fi
else
  warn "Skipping PR-Agent CI (no GitHub remote)"
fi

# ── Local review ──
LOCAL_ENABLED=false
LOCAL_MODEL="claude-sonnet-4-6"
LOCAL_ALIAS=""

if $CLAUDE_AUTHED; then
  divider
  heading "Local review (claude -p)"
  _default_local="$(cfg_get '.local_review.enabled' 'n')"; [[ "$_default_local" == "true" ]] && _default_local="y"
  if ask_yn "Enable local review script (pr-review.sh)?" "$_default_local"; then
    LOCAL_ENABLED=true

    LOCAL_MODEL="$(cfg_get '.local_review.default_model' 'claude-sonnet-4-6')"
    LOCAL_MODEL="$(ask "Default model for local review" "$LOCAL_MODEL")"

    LOCAL_ALIAS="$(cfg_get '.local_review.shell_alias' '')"
    LOCAL_ALIAS="$(ask "Shell alias (e.g. pr:review — leave blank to skip)" "$LOCAL_ALIAS")"
  fi
else
  warn "Skipping local review (claude CLI not found)"
fi

# ── O3 dual-review ──
DUAL_ENABLED=false
DUAL_MODEL="gpt-5.5"
DUAL_BRIEF=true
DUAL_IMPL=true

if $OPENAI_KEY; then
  divider
  heading "O3 dual-review (independent architecture gate)"
  info "Fires during the implement loop: before implementation (brief) and/or after (impl)."
  _default_dual="$(cfg_get '.dual_review.enabled' 'n')"; [[ "$_default_dual" == "true" ]] && _default_dual="y"
  if ask_yn "Enable O3 dual-review?" "$_default_dual"; then
    DUAL_ENABLED=true

    DUAL_MODEL="$(cfg_get '.dual_review.model' 'gpt-5.5')"
    DUAL_MODEL="$(ask "Dual-review model" "$DUAL_MODEL")"

    _default_brief="$(cfg_get '.dual_review.on_brief' 'y')"; [[ "$_default_brief" == "false" ]] && _default_brief="n"
    ask_yn "Review the task brief (Phase 4)?" "${_default_brief:-y}" || DUAL_BRIEF=false

    _default_impl="$(cfg_get '.dual_review.on_impl' 'y')"; [[ "$_default_impl" == "false" ]] && _default_impl="n"
    ask_yn "Review the implementation (Phase 7)?" "${_default_impl:-y}" || DUAL_IMPL=false
  fi
else
  warn "Skipping O3 dual-review (OPENAI_API_KEY not found)"
fi

# ── DCO ──
DCO_ENABLED=false

if $HAS_REMOTE; then
  divider
  heading "DCO sign-off enforcement"
  info "Adds a dco.yml GitHub Actions workflow requiring Signed-off-by on all commits."
  _default_dco="$(cfg_get '.dco.required' 'n')"; [[ "$_default_dco" == "true" ]] && _default_dco="y"
  ask_yn "Enable DCO enforcement?" "$_default_dco" && DCO_ENABLED=true
fi

# ── PR template ──
PR_TEMPLATE_ENABLED=false

divider
heading "PR body template"
info "Installs .github/pull_request_template.md with acceptance-criteria checklist."
_default_pt="$(cfg_get '.pr_template.enabled' 'n')"; [[ "$_default_pt" == "true" ]] && _default_pt="y"
ask_yn "Enable PR template?" "$_default_pt" && PR_TEMPLATE_ENABLED=true

# ── Test tiers ──
TIER_T1=true
TIER_T2=false
TIER_T25=false
TIER_T3=false

if $HAS_PLAYWRIGHT; then
  divider
  heading "Test tier hierarchy"
  info "Controls which tiers the test-writer agent is expected to produce."
  info "T1 Headless is always on. Higher tiers require Playwright."

  _default_t2="$(cfg_get '.test_tiers.t2_aria' 'n')"; [[ "$_default_t2" == "true" ]] && _default_t2="y"
  ask_yn "T2 ARIA — accessibility tree for unauthenticated routes?" "$_default_t2" && TIER_T2=true

  _default_t25="$(cfg_get '.test_tiers.t2_5_authenticated' 'n')"; [[ "$_default_t25" == "true" ]] && _default_t25="y"
  ask_yn "T2.5 Authenticated — ARIA tree for logged-in routes?" "$_default_t25" && TIER_T25=true

  _default_t3="$(cfg_get '.test_tiers.t3_visual' 'n')"; [[ "$_default_t3" == "true" ]] && _default_t3="y"
  ask_yn "T3 Visual — screenshot sign-off per design slice?" "$_default_t3" && TIER_T3=true
else
  warn "Skipping test tiers T2+ (Playwright not detected)"
fi

# ── Phase 3: Write config ────────────────────────────────────────────────────

divider
heading "Writing config"

mkdir -p "${PROJECT_ROOT}/.agents"

cat > "$CONFIG_FILE" <<YAML
# .agents/pr-review.yml
# Generated by setup-pr-review.sh on $(date '+%Y-%m-%d').
# Re-run .agents/scripts/setup-pr-review.sh --update to change settings.

project:
  name: "${PROJECT_NAME}"
  timezone: "${PROJECT_TZ}"

# Every key below is read by pr-review-reusable.yml. Anything it does not
# recognise now raises a ::warning:: annotation on the PR rather than being
# silently discarded — see pipelines/pr-reviewer/README.md for the full schema.
pr_agent:
  model: "${PR_AGENT_MODEL}"
  max_tokens: 131072
  output_tokens: 64000
  high_stakes_label: "${PR_AGENT_HIGH_STAKES}"
  skip_docs_only: ${PR_AGENT_SKIP_DOCS}
  score_labels: true
  score_banner: true
  status_check: ${PR_AGENT_STATUS_CHECK}

local_review:
  enabled: ${LOCAL_ENABLED}
  default_model: "${LOCAL_MODEL}"
  shell_alias: "${LOCAL_ALIAS}"

dual_review:
  enabled: ${DUAL_ENABLED}
  model: "${DUAL_MODEL}"
  on_brief: ${DUAL_BRIEF}
  on_impl: ${DUAL_IMPL}

dco:
  required: ${DCO_ENABLED}

pr_template:
  enabled: ${PR_TEMPLATE_ENABLED}

test_tiers:
  t1_headless: ${TIER_T1}
  t2_aria: ${TIER_T2}
  t2_5_authenticated: ${TIER_T25}
  t3_visual: ${TIER_T3}
YAML

ok "Written: .agents/pr-review.yml"

# ── Generate shell-aliases.sh if alias configured ────────────────────────────
if [[ -n "$LOCAL_ALIAS" ]]; then
  ALIASES_FILE="${PROJECT_ROOT}/.agents/scripts/shell-aliases.sh"
  TEMPLATE_SRC=""
  # Find the template relative to this wizard (works from playbook or copied location)
  _wizard_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "${_wizard_dir}/shell-aliases.sh" ]]; then
    TEMPLATE_SRC="${_wizard_dir}/shell-aliases.sh"
  fi

  if [[ -n "$TEMPLATE_SRC" ]]; then
    mkdir -p "${PROJECT_ROOT}/.agents/scripts"
    sed "s/@@ALIAS@@/${LOCAL_ALIAS}/g" "$TEMPLATE_SRC" > "$ALIASES_FILE"
    chmod +x "$ALIASES_FILE"
    ok "Generated: .agents/scripts/shell-aliases.sh  (alias: ${LOCAL_ALIAS})"
  else
    warn "shell-aliases.sh template not found alongside wizard — generate manually."
  fi
fi

# ── Summary ──────────────────────────────────────────────────────────────────

divider
heading "Summary"

print_feature() {
  local name="$1" enabled="$2" note="${3:-}"
  if [[ "$enabled" == "true" ]]; then
    ok "${name}${note:+  (${note})}"
  else
    err "${name}${note:+  — ${note}}"
  fi
}

print_feature "PR-Agent CI"       "$PR_AGENT_ENABLED"    "$($PR_AGENT_ENABLED && echo "model: ${PR_AGENT_MODEL}" || echo "no GitHub remote or not opted in")"
print_feature "Local review"      "$LOCAL_ENABLED"       "$($LOCAL_ENABLED && echo "model: ${LOCAL_MODEL}" || echo "claude CLI not found or not opted in")"
print_feature "Shell alias"       "$([[ -n "$LOCAL_ALIAS" ]] && echo true || echo false)"  "${LOCAL_ALIAS:-not configured}"
print_feature "O3 dual-review"    "$DUAL_ENABLED"        "$($DUAL_ENABLED && echo "model: ${DUAL_MODEL}" || echo "OPENAI_API_KEY not found or not opted in")"
print_feature "DCO enforcement"   "$DCO_ENABLED"
print_feature "PR template"       "$PR_TEMPLATE_ENABLED"
print_feature "T2 ARIA"           "$TIER_T2"             "$($TIER_T2 || $HAS_PLAYWRIGHT && echo "" || echo "Playwright not detected")"
print_feature "T2.5 Authenticated" "$TIER_T25"
print_feature "T3 Visual"         "$TIER_T3"

# ── Next steps ───────────────────────────────────────────────────────────────

divider
heading "Next steps"

NEXT=0

if $PR_AGENT_ENABLED && ! $DEEPSEEK_KEY; then
  NEXT=$((NEXT+1))
  echo -e "  ${NEXT}. Add DEEPSEEK_API_KEY to GitHub secrets:"
  echo    "       gh secret set DEEPSEEK_API_KEY"
fi

if $PR_AGENT_ENABLED; then
  NEXT=$((NEXT+1))
  echo -e "  ${NEXT}. Copy the canonical GitHub Actions workflow:"
  echo    "       cp \$AI_GUIDANCE/pipelines/pr-reviewer/templates/pr-review.yml .github/workflows/"
fi

if $DCO_ENABLED; then
  NEXT=$((NEXT+1))
  echo -e "  ${NEXT}. Copy the DCO workflow:"
  echo    "       cp \$AI_GUIDANCE/pipelines/pr-reviewer/templates/dco.yml .github/workflows/"
fi

if $PR_TEMPLATE_ENABLED; then
  NEXT=$((NEXT+1))
  echo -e "  ${NEXT}. Copy the PR body template:"
  echo    "       cp \$AI_GUIDANCE/pipelines/pr-reviewer/templates/pull_request_template.md .github/"
fi

if $LOCAL_ENABLED || $DUAL_ENABLED; then
  NEXT=$((NEXT+1))
  echo -e "  ${NEXT}. Copy the canonical scripts:"
  echo    "       cp \$AI_GUIDANCE/pipelines/pr-reviewer/scripts/*.sh .agents/scripts/"
  echo    "       chmod +x .agents/scripts/*.sh"
fi

if [[ -n "$LOCAL_ALIAS" ]]; then
  NEXT=$((NEXT+1))
  echo -e "  ${NEXT}. Source the shell alias in your shell config:"
  echo    "       echo 'source \$(git rev-parse --show-toplevel)/.agents/scripts/shell-aliases.sh' >> ~/.zshrc"
fi

if $PR_AGENT_ENABLED || $LOCAL_ENABLED; then
  NEXT=$((NEXT+1))
  echo -e "  ${NEXT}. Verify everything is ready:"
  echo    "       .agents/scripts/check-pr-review-readiness.sh"
fi

[[ $NEXT -eq 0 ]] && info "Nothing else to do — all features are self-contained."

echo ""
