#!/usr/bin/env bash
# pr-review.sh — Spec-enforcer PR review via `claude -p`
#
# Usage:
#   .agents/scripts/pr-review.sh <PR-number> [--post] [--model <model>]
#
#   --post            Post the review as a GitHub PR comment via `gh pr review`.
#   --model <model>   Override the model (default: $REVIEWER_MODEL, then
#                     local_review.default_model from .agents/pr-review.yml,
#                     then claude-sonnet-4-6).
#
# Prerequisites:
#   - `claude` CLI installed and authenticated
#   - `gh` CLI installed and authenticated
#   - Run from the repo root (CLAUDE.md must be in cwd)

set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

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
    local stripped="${key##*.}"
    local val
    val=$(grep -E "^\s*${stripped}:" "$CONFIG_FILE" 2>/dev/null | head -1 \
          | sed 's/.*: *//;s/"//g;s/\r//')
    echo "${val:-$fallback}"
  fi
}

# ── Argument parsing ────────────────────────────────────────────────────────
PR_NUMBER="${1:-}"
POST_TO_GH=false

DEFAULT_MODEL="${REVIEWER_MODEL:-}"
[[ -z "$DEFAULT_MODEL" ]] && DEFAULT_MODEL="$(cfg_get '.local_review.default_model' 'claude-sonnet-4-6')"
MODEL="$DEFAULT_MODEL"

if [[ -z "$PR_NUMBER" ]]; then
  echo "Usage: $0 <PR-number> [--post] [--model <model>]" >&2
  exit 1
fi

shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --post)  POST_TO_GH=true; shift ;;
    --model) MODEL="${2:?--model requires a value}"; shift 2 ;;
    *)       echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

# ── Preflight checks ────────────────────────────────────────────────────────
if [[ ! -f "CLAUDE.md" ]]; then
  echo "Error: run from the repo root (CLAUDE.md not found in cwd)" >&2
  exit 1
fi

for cmd in claude gh; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "Error: '$cmd' not found — install it first." >&2
    exit 1
  fi
done

# ── Fetch PR data ────────────────────────────────────────────────────────────
echo "→ Fetching PR #${PR_NUMBER} metadata..." >&2
PR_META=$(gh pr view "$PR_NUMBER" \
  --json title,body,author,headRefName,baseRefName \
  --template '## PR: {{.title}}
Branch: {{.headRefName}} -> {{.baseRefName}}
Author: {{.author.login}}

### Description
{{.body}}')

echo "→ Fetching diff..." >&2
PR_DIFF=$(gh pr diff "$PR_NUMBER")

# ── Build prompt (safe injection) ───────────────────────────────────────────
# Static template uses a QUOTED heredoc ('EOF') — no variable expansion.
# Untrusted PR content is appended with printf '%s' (literal, no interpretation).
PROJECT_NAME="$(cfg_get '.project.name' "$(basename "$PWD")")"

PROMPT_FILE=$(mktemp /tmp/pr-review-XXXXXX.md)
trap 'rm -f "$PROMPT_FILE"' EXIT

cat > "$PROMPT_FILE" <<'ENDSTATIC'
You are a Spec-enforcer. CLAUDE.md in this repo has been loaded automatically
as your project context. Perform a thorough PR review now.

## Scope gate — DO THIS FIRST

0. Classify the diff. If it changes ONLY non-code files — documentation and
   Markdown (`*.md`, `docs/**`), comments, or other prose/text with no
   executable code — then this is a **docs-only PR** and the code conventions
   below DO NOT apply. In that case:
   - Do NOT require tests, version bumps, or script/file-naming alignment.
   - Do NOT manufacture BLOCK findings about files that contain no code, and do
     NOT cite code conventions (e.g. "tests required", "versioning") against
     prose changes.
   - Emit verdict **N/A** (`> [!NOTE]`) with a one-line reason, plus any genuine
     prose nits as INFO, and STOP.
   Only proceed to the steps below when the diff contains actual code changes.

## What to do (code changes only)

1. Read every file in skills/ that is relevant to this diff.
2. Check the diff against every forbidden pattern listed in CLAUDE.md under
   "Forbidden patterns". Each is a BLOCK finding if present.
3. Check docs/planning/decisions.md (if it exists) — verify the diff does not
   contradict any locked decision.
4. Check for missing tests: new routes or public functions **in code** must have
   corresponding test files. Never block on missing tests for docs-only or
   comment-only changes.
5. Check for secrets committed: credentials, API keys, tokens → BLOCK immediately.
6. Apply any project-specific rules documented in CLAUDE.md.

## Output format (GitHub-rendered markdown)

Open the review with a GitHub alert callout matching the verdict:

- ✅ PASS  → `> [!TIP]`      (green)
- ⚠️ WARN  → `> [!WARNING]`  (amber)
- ❌ BLOCK → `> [!CAUTION]`  (red)
- N/A      → `> [!NOTE]`     (blue — docs-only PRs where no rules apply)

Do NOT include opening or closing `---` separators in the body.
Use the section headers below exactly as written.

ENDSTATIC

# Append the PR number (script-controlled integer — safe to expand here).
printf '## Spec-enforcer Review — PR #%s\n\n' "$PR_NUMBER" >> "$PROMPT_FILE"

cat >> "$PROMPT_FILE" <<'ENDTEMPLATE'
> [!TIP]
> **Verdict:** ✅ PASS — <one-sentence reason>

### Findings

For each finding:
- **[BLOCK | WARN | INFO]** `file:line` — description
  - **Rule:** skill filename or CLAUDE.md section
  - **Fix:** specific remediation

(If no findings: "No findings. All checked patterns pass.")

### Summary

2–3 sentence plain-English summary of what the PR does and whether it can merge.

ENDTEMPLATE

# Append untrusted content with printf '%s' (argument, not format string) to
# prevent shell metacharacter expansion from PR titles, bodies, or diff lines.
printf '## PR metadata\n' >> "$PROMPT_FILE"
printf '%s\n' "$PR_META" >> "$PROMPT_FILE"
printf '\n## Diff\n' >> "$PROMPT_FILE"
printf '%s\n' "$PR_DIFF" >> "$PROMPT_FILE"

# ── Run review ───────────────────────────────────────────────────────────────
echo "→ Running Spec-enforcer review via claude -p --model ${MODEL}..." >&2

START_TIME=$(date +%s)

# Pass prompt via stdin (< file) — avoids command-line length limits and
# prevents any $(...) or backtick expansion in untrusted diff content.
#
# --allowedTools restricts Claude to read-only file operations so that
# prompt-injected instructions in an attacker's PR diff cannot execute
# arbitrary shell commands. The reviewer only needs to read skills/ and
# docs/planning/ — no write, bash, or network access required.
REVIEW=$(claude -p --model "$MODEL" \
  --allowedTools "Read,Glob,Grep" \
  < "$PROMPT_FILE") || {
  echo "Error: claude -p exited with code $?" >&2
  echo "Tip: check \`claude auth status\` and confirm PATH includes homebrew bins." >&2
  exit 1
}

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

# ── Append footer ────────────────────────────────────────────────────────────
TZ_LOCAL="$(cfg_get '.project.timezone' 'America/Los_Angeles')"
TIMESTAMP_LOCAL=$(TZ="$TZ_LOCAL" date '+%Y-%m-%d %H:%M %Z' 2>/dev/null || date '+%Y-%m-%d %H:%M %Z')
TIMESTAMP_UTC=$(date -u '+%Y-%m-%d %H:%M UTC')

REVIEW="${REVIEW}

---
🤖 **Reviewer:** Spec-enforcer · **Project:** ${PROJECT_NAME} · **Model:** \`${MODEL}\` · **Elapsed:** ${ELAPSED}s · **Generated:** ${TIMESTAMP_LOCAL} · ${TIMESTAMP_UTC}"

# ── Output ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "$REVIEW"
echo "════════════════════════════════════════"

if [[ "$POST_TO_GH" == "true" ]]; then
  echo "" >&2
  echo "→ Posting review to PR #${PR_NUMBER}..." >&2
  gh pr review "$PR_NUMBER" --comment --body "$REVIEW"
  echo "→ Posted." >&2
fi
