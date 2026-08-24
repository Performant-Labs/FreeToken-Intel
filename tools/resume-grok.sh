#!/usr/bin/env bash
# Resume the FreeToken-Intel Grok session after a reboot (or any new terminal).
# Session is stored under ~/.grok/sessions/ — reboot does not wipe it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# This conversation (title: "FreeToken Intel B70 stubs and epic").
SESSION_ID="01a030db-8c69-79d3-90c8-28aba5e92cdb"

if command -v grok >/dev/null 2>&1; then
  exec grok --resume "$SESSION_ID"
fi

echo "grok is not on PATH. Install/open Grok, then from $ROOT run:" >&2
echo "  grok --resume $SESSION_ID" >&2
echo "or: grok --resume    # most recent session in this directory" >&2
exit 1
