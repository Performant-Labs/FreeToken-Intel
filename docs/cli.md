# CLI

`ft` is the FreeToken-Intel command (same name as upstream so docs and agent
wrappers stay familiar).

```
ft --version
ft --help
ft <command> --help
```

| Command | Role | Issue |
| --- | --- | --- |
| `serve` | OpenAI/Anthropic HTTP server | `server-openai` |
| `shell` | TUI client | `shell-daemon` |
| `ctl` | Inspect / control a running server | `shell-daemon` |
| `daemon` | Persistent supervisor | `shell-daemon` |
| `launch` | Wire coding agents to the server | `agent-launch` |
| `checkpoint` | HF → FTW | `ftw-checkpoint` |
| `bench bw` | CPU vs PCIe vs HBM profile | `moe-hybrid` |
| `device` | Print XPU / Xe2 detection | `device-layer` |

Listen port (planned, matching upstream): **1919**.
