# AstraOS v0.5.1 - Memory Graph Ingestion Foundation

This patch wires the existing Cognitive Memory Graph foundation into real Astra Work Session activity.

## Modified files

- `src-tauri/src/lib.rs`
- `src-tauri/src/memory/store/sqlite_store.rs`

## What changes

- Work Session chat memory snapshots are now ingested into the Memory Graph when Astra produces session answers/recaps/recall results.
- The ingestion creates typed nodes for:
  - `work_session`
  - `summary`
  - `tool_use`
  - `transcript_segment` evidence
- The ingestion creates typed relationships:
  - `summary derived_from work_session`
  - `summary used_tool tool_use`
  - `summary derived_from transcript_segment`
  - `transcript_segment part_of work_session`
- The ingestion emits a bounded activation event so the UI can later animate the involved nodes/edges.
- `MemoryGraphStore` now supports `find_node_by_source` and `create_node_once_by_source` to prevent obvious duplicate nodes for stable sources.

## Architecture notes

- No semantic hard-code routing was added.
- The LLM remains responsible for intent/tool choice.
- Rust only persists typed memory after a governed Work Session result exists.
- The Memory Graph remains separate from audit and LLM trace storage.
- Failed ingestion is intentionally non-fatal: Astra should never fail the user-facing response because memory persistence had a local issue.

## Validation

After extracting the patch:

```powershell
cd src-tauri
cargo check
cd ..
npm run build
```

Runtime validation:

1. Start Astra.
2. Ask for a Work Session recap.
3. Check `.astra/memory/graph/astra_memory.sqlite` and `.astra/memory/journal/memory_events.jsonl`.
4. Call/export the memory snapshot from the UI/API and verify new nodes/edges are present.
