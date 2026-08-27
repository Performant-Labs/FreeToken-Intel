# shell-aliases.sh — Project shell shortcuts for PR review
#
# Generated/installed by setup-pr-review.sh based on local_review.shell_alias
# in .agents/pr-review.yml.
#
# Source this from your shell profile (zsh only — colon function names like
# `pr:review` are legal in zsh but a parse error in bash):
#
#   echo 'source /path/to/project/.agents/scripts/shell-aliases.sh' >> ~/.zshrc
#
# Or source it per-session:
#   source .agents/scripts/shell-aliases.sh
#
# The PROJECT_DIR variable is auto-detected from the location of this script.
# Override it in your profile if needed:
#   export PROJECT_DIR="$HOME/alt/path/to/project"

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"

# @@ALIAS@@ — run the Spec-enforcer PR review via `claude -p`
# Usage:
#   @@ALIAS@@ <PR-number>              print review to stdout
#   @@ALIAS@@ <PR-number> --post       also post as a PR comment
#   @@ALIAS@@ <PR-number> --model X    override the default model
@@ALIAS@@() {
  if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "@@ALIAS@@: PROJECT_DIR not found: $PROJECT_DIR" >&2
    echo "         Set PROJECT_DIR in your shell profile if the repo moved." >&2
    return 1
  fi
  ( cd "$PROJECT_DIR" && ./.agents/scripts/pr-review.sh "$@" )
}
