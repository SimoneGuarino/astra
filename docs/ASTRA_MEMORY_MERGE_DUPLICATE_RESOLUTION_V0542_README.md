# AstraOS v0.5.42 — Memory Merge / Duplicate Resolution

This patch adds a governed, user-confirmed duplicate resolution workflow for the Cognitive Memory Graph.

## What changed

- Adds typed duplicate candidate DTOs to Rust and TypeScript.
- Adds `list_memory_duplicate_candidates` Tauri command.
- Adds `merge_memory_nodes` Tauri command.
- Adds soft-merge canonicalization in SQLite without hard delete.
- Adds an `MG` toolbar action in the Memory Graph.
- Adds a duplicate candidate overlay with score, reasons and merge action.

## Architecture notes

This is not semantic hard-coding. Candidate detection is structural and advisory, while the user confirms merge operations explicitly. The merge is soft:

- original nodes are retained as evidence;
- duplicate nodes can be marked `deprecated`;
- a `same_topic_as` edge with `semantic_relation: merged_into` links duplicate to canonical node;
- the target node keeps `merged_from_node_ids` metadata;
- retrieval normally excludes deprecated aliases unless explicitly included.

## Validation

Run:

```powershell
npm run build
cd src-tauri
cargo check
```

Smoke test:

1. Open Memory Graph.
2. Click `MG`.
3. Click `Find duplicates`.
4. Review a candidate.
5. Click `Merge`.
6. Refresh the graph and inspect both nodes.

## Safety invariants

- No hard delete.
- No automatic merge without user action.
- No bypass of governance, audit, policy, approval or tool validation.
- Memory remains advisory-only.
