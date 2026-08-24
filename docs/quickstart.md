# Quick start

The engine, HTTP server, and kernels are still stubs. The CLI surface is
wired so later issues can fill commands without renaming flags.

```bash
ft --help
ft --version
ft device          # XPU / Xe2 probe (works without a full engine)
```

Once `server-openai` and `models-qwen35` land:

```bash
ft serve --model Qwen/Qwen3.6-35B-A3B
ft shell --server http://127.0.0.1:1919
```

Default listen address matches upstream FreeToken: `127.0.0.1:1919`.

MoE backend selection (same flags as NVIDIA FreeToken):

```
ft serve --moe-backend {auto,fused,offload,cpu,hybrid}
```

Run `ft bench bw` after `moe-hybrid` is implemented so `auto` can pick
hybrid vs offload from a measured CPU vs PCIe vs HBM profile.
