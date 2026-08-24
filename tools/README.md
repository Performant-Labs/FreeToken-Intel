# tools

Agent prompts and helpers. These are not the runtime (`ft`) and not
build scripts (`scripts/`).

| File | How to use |
| --- | --- |
| [preflight-check.md](preflight-check.md) | Give this prompt to a coding agent in this repo. It runs a CPU/XPU pre-flight and reports PASS/FAIL/SKIP. It must not install or edit files unless you then ask it to fix failures. |
| [resume-grok.sh](resume-grok.sh) | After a reboot: `./tools/resume-grok.sh` from this repo (or `grok --resume`). Reopens this Grok conversation. |

Example:

```
Execute tools/preflight-check.md
```

Resume this chat (session `01a030db-8c69-79d3-90c8-28aba5e92cdb`):

```bash
cd ~/Projects/FreeToken-Intel
./tools/resume-grok.sh
# or: grok --resume
# or: grok --resume 01a030db-8c69-79d3-90c8-28aba5e92cdb
```
