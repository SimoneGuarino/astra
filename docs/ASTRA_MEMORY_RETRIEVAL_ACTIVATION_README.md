# AstraOS v0.5.2 — Memory Retrieval & Activation Context

This patch connects the Cognitive Memory Graph foundation to Astra's runtime in a governed, LLM-first way.

## What changed

- Adds `src-tauri/src/memory/retrieval/context_pack.rs`.
- Exposes `MemoryContextPacket` from `src-tauri/src/memory/retrieval/mod.rs`.
- Retrieves relevant memory nodes for each assistant request.
- Records bounded memory activation events for the UI brain/electricity visualization.
- Injects compact memory context into:
  - normal LLM chat context;
  - Work Session tool router prompt;
  - Work Session empty-content repair prompt.

## Important architecture notes

The memory graph does **not** execute actions and does **not** bypass governance.

Memory is passed to the model as background context only. Rust still validates:

- tool selection;
- permissions;
- policy;
- approvals;
- audit boundaries;
- Work Session execution targets.

This preserves the intended model:

```txt
LLM understands and proposes
Memory supplies experience/context
Rust validates, governs and executes
UI visualizes activation
```

## New UI event

The backend emits:

```txt
memory-activation
```

Payload includes:

- `requestId`
- `rootQuery`
- `activatedNodeIds`
- `activatedEdgeIds`
- `intensity`
- `metadata.ui_hint = electricity_reached_nodes`

The existing Memory Graph UI can consume this for the future live “electricity reaching nodes” effect.

## Validation

After importing:

```powershell
cd src-tauri
cargo check

cd ..
npm run build
```

Then ask Astra something related to a previous Work Session/recap. You should see new `activation_recorded` entries in:

```txt
.astra/memory/journal/memory_events.jsonl
```

and the memory router context should remain metadata-only / governed.
