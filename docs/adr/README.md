# Architecture Decision Records

ADRs capture a *choice* we made, the options we had, and why we picked one.
They stay in git so later work does not re-litigate a settled call.

The living description of the stack is [../stack.md](../stack.md). Use an
ADR when that stack (or the engine layout) *changes*, or when a choice is
narrower than the whole stack (for example “MXFP4 vs INT4 as the default
expert dtype”).

## Format

Numbered Markdown files:

```
NNNN-short-title.md
```

Copy [0000-template.md](0000-template.md). Status is one of:

| Status | Meaning |
| --- | --- |
| Proposed | Under discussion |
| Accepted | Current decision |
| Superseded | Replaced by a later ADR (link it) |
| Deprecated | No longer applies |

Keep the body short: context, options, decision, consequences. Link GitHub
issues. Do not duplicate the stack page.

## Index

| ADR | Title | Status |
| --- | --- | --- |
| [0001](0001-freetoken-on-intel-sycl-xpu.md) | Port FreeToken onto Intel SYCL / XPU for Arc Pro B70 | Accepted |
| [0002](0002-moe-expert-host-offload.md) | MoE experts live in host RAM, streamed into an XPU LRU slot pool | Accepted |
