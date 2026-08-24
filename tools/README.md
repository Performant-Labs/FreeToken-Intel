# tools

Agent prompts and helpers. These are not the runtime (`ft`) and not
build scripts (`scripts/`).

| File | How to use |
| --- | --- |
| [preflight-check.md](preflight-check.md) | Give this prompt to a coding agent in this repo. It runs a CPU/XPU pre-flight and reports PASS/FAIL/SKIP. It must not install or edit files unless you then ask it to fix failures. |

Example:

```
Execute tools/preflight-check.md
```
