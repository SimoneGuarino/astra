# AstraOS v0.5.8 — Memory Governance & User Control Panel

This patch introduces the first user-governed control layer for Astra's Cognitive Memory Graph.

## Added

- User-visible memory governance policy snapshot.
- Soft-state node governance updates:
  - `user_confirmed`
  - `unverified`
  - `contradicted`
  - `deprecated`
- Salience adjustment from the Memory Graph UI.
- Retrieval suppression for `deprecated` and `contradicted` memories unless explicitly requested.
- Governance journal events under `.astra/memory/journal/memory_events.jsonl`.
- Frontend controls in the Memory node inspector.

## Architectural guarantees

- No hard delete is introduced in this phase.
- Cognitive memory remains advisory context only.
- Memory cannot bypass Rust governance, policy, approvals, or tool validation.
- LLM-inferred memories are weighted lower than user-confirmed/system-verified memories.
- Deprecated/contradicted memories remain inspectable in the graph, but are excluded from normal retrieval.

## Validation

After importing:

```powershell
cd src-tauri
cargo check

cd ..
npm run build
```

Then open the Memory tab, select a node, and test:

- Confirm
- Unverify
- Contradict
- Deprecate
- Salience slider

Check `.astra/memory/journal/memory_events.jsonl` for `node_governance_updated`.
