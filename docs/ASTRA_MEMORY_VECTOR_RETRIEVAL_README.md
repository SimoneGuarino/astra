# AstraOS v0.5.6 — Memory Vector Retrieval Adapter

## Scope

This patch introduces the first governed vector retrieval layer for the AstraOS Cognitive Memory Graph.

It keeps the existing architecture intact:

- SQLite remains the local source of truth for memory nodes, edges, chunks, activations, verification state and graph relationships.
- Vector retrieval is an advisory retrieval index, not an authority layer.
- Rust keeps validation, persistence, governance and tool boundaries.
- The LLM receives memory as context only; it does not gain permission to execute actions from memory.

## What changed

### Rust

Added/extended:

- `src-tauri/src/memory/embeddings/provider.rs`
- `src-tauri/src/memory/embeddings/mod.rs`
- `src-tauri/src/memory/retrieval/vector.rs`
- `src-tauri/src/memory/retrieval/context_pack.rs`
- `src-tauri/src/memory/store/sqlite_store.rs`
- `src-tauri/src/memory/commands/mod.rs`
- `src-tauri/src/memory/types.rs`
- `src-tauri/src/lib.rs`

The new vector foundation includes:

- deterministic local embedding provider: `StableHashEmbeddingProvider`
- SQLite-backed embedding cache: `memory_embeddings`
- embedding rebuild command
- embedding status command
- hybrid query command
- context-packet retrieval upgraded from lexical+graph to lexical+vector+graph

### Frontend

Updated:

- `src/types/memory.ts`
- `src/hooks/useMemoryGraph.ts`
- `src/features/memory/components/MemoryGraphPanel.tsx`

The Memory Graph UI can now:

- show embedding status
- rebuild vector index
- use hybrid query for search
- display vector backend/chunk coverage

## New Tauri commands

- `get_memory_embedding_status`
- `rebuild_memory_embedding_index`
- `query_memory_graph_hybrid`

## Runtime behavior

After importing this patch, existing memory nodes remain valid. The vector index is built from `memory_chunks`.

Use the Memory Graph panel button:

```txt
Rebuild vectors
```

or call the command:

```ts
invoke("rebuild_memory_embedding_index", {
  request: { limit: 1000, force: false }
});
```

Then searches use hybrid ranking:

```txt
final score = lexical score + vector score + graph/salience/verification boosts
```

## Important architectural note

The included local embedding provider is deterministic and dependency-free. It is not a true semantic embedding model. It exists to stabilize the provider boundary and make the system fully local/testable.

The next step should replace or augment this provider with a real embedding backend through the same adapter boundary, for example:

- Ollama embeddings
- sentence-transformers Python worker
- sqlite-vec
- Qdrant local backend

without changing the Memory Graph source of truth.

## Validation

Run:

```powershell
cd src-tauri
cargo check

cd ..
npm run build
```

Then:

1. Open Astra.
2. Open the Memory Graph tab.
3. Click `Rebuild vectors`.
4. Search for a known topic already present in memory.
5. Confirm `.astra/memory/journal/memory_events.jsonl` contains `embedding_upserted` records.

## What this does not do

- It does not bypass Rust governance.
- It does not authorize tools from memory.
- It does not make vector hits trusted facts.
- It does not replace graph relations.
- It does not introduce an external VectorDB yet.
