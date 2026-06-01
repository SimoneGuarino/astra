# AstraOS v0.5.12 — Brain Graph Layout & RAG Context Hardening

## Scope

This patch fixes the Memory Graph UI behavior and strengthens the chat-to-memory retrieval path without changing Astra's Rust governance model.

## Changed files

- `src/features/memory/components/MemoryGraphPanel.tsx`
- `src/App.css`
- `src-tauri/src/lib.rs`
- `src-tauri/src/memory/retrieval/context_pack.rs`

## What changed

### Brain-first graph visualization

The graph view now behaves more like a cognitive vault / Obsidian-style graph:

- large primary graph canvas
- pan and zoom controls
- fit/reset controls
- drag-and-drop nodes
- linked nodes react to a dragged node through force attraction
- connected components are clustered in a brain-like layout instead of being pushed to the borders
- soft boundary force instead of hard clamping
- dark graph surface for better edge/node readability
- activated edges retain the electricity effect
- activated/selected nodes use a stable halo, not an ambiguous moving transparent dot

### Scalability and UI bounds

The UI now caps graph rendering at a bounded number of high-value nodes/edges:

- up to 420 graph nodes
- up to 900 graph edges

This is a visualization cap only. The Memory Graph database remains the source of truth.

### RAG brain usage hardening

The chat path now retrieves a larger Memory Context Packet:

- request context limit increased from 8 to 12 nodes
- memory preamble exposes up to 10 nodes to the assistant context
- retrieval context packet limit can now scale up to 18 hits
- graph activation expands up to 40 nodes
- the memory prompt explicitly tells the model not to claim “no memory” when relevant nodes are present

Memory still remains contextual and advisory. It does not bypass governed tools, policy, permissions, approvals, or audit.

## Validation

After import:

```powershell
cd src-tauri
cargo check

cd ..
npm run build
```

Runtime checks:

1. Open the Memory tab.
2. Drag a connected node and verify linked nodes react instead of staying pinned to the borders.
3. Use mouse wheel zoom and canvas drag/pan.
4. Search the memory for identity/preference facts.
5. Ask Astra a memory question such as `sai chi sono?` and verify that relevant memory nodes activate.

## Notes

This patch is UI/RAG-context hardening, not a new autonomous skill runtime. It preserves the LLM-first, Rust-governed design.
