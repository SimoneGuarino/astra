# AstraOS v0.5.32 — Memory Quality Dashboard & Stability Metrics

This patch adds a governed quality dashboard for the Cognitive Memory Graph.

## Goal

Before adding more autonomy, Astra must be able to measure whether the Brain RAG is healthy or noisy. This patch introduces a typed memory quality snapshot that evaluates:

- total nodes / edges / chunks / activations
- semantic memory coverage
- raw episode-only conversation memories
- pending reconsolidation candidates
- embedding index coverage
- governance state distribution
- recent activation activity
- practical warnings and recommendations

## Architecture

- SQLite remains the source of truth.
- Vector embeddings remain an advisory retrieval index.
- The dashboard is read-only and does not mutate memory.
- No tool execution, policy, approval, audit or governance checks are bypassed.
- The UI exposes the quality snapshot through a compact `Q` overlay in the Memory Graph toolbar.

## New Tauri command

```txt
get_memory_quality_dashboard
```

## Modified files

```txt
src-tauri/src/lib.rs
src-tauri/src/memory/types.rs
src-tauri/src/memory/commands/mod.rs
src-tauri/src/memory/store/sqlite_store.rs
src/types/memory.ts
src/hooks/useMemoryGraph.ts
src/features/memory/components/MemoryGraphPanel.tsx
src/App.css
```

## Validation

```powershell
cd src-tauri
cargo check

cd ..
npm run build
```

Open the Memory Graph and click `Q` to inspect the quality snapshot.
