# AstraOS v0.5.31 — Automatic Memory Embedding Lifecycle

## Scope

This patch stabilizes the Brain RAG lifecycle by removing the hard dependency on a manual "Rebuild vectors" action after Astra learns new memory.

## What changed

- Adds typed maintenance request/receipt structures for memory embeddings.
- Adds a governed `run_memory_embedding_maintenance` Tauri command.
- Adds bounded automatic indexing after conversation memory consolidation creates new nodes.
- Adds UI-triggered maintenance for pending vectors in the Memory Graph.
- Keeps the existing manual full rebuild action as a separate debug/admin action.
- Keeps SQLite Memory Graph as the source of truth; embeddings remain an index/cache.

## Runtime behavior

When conversation consolidation creates memory nodes, Astra now schedules a bounded embedding maintenance pass:

```text
conversation memory consolidation
  -> nodes/chunks created
  -> pending embeddings detected
  -> bounded batch indexing
  -> embedding_upserted / embedding_maintenance_ran journal events
```

The maintenance pass is intentionally bounded and non-authoritative. It does not execute tools, does not bypass governance, and does not modify memory semantics.

## Environment controls

```env
ASTRA_MEMORY_EMBEDDING_AUTO_INDEX=true
ASTRA_MEMORY_EMBEDDING_BATCH_SIZE=24
ASTRA_MEMORY_EMBEDDING_PROVIDER=ollama
ASTRA_MEMORY_EMBEDDING_MODEL=nomic-embed-text
ASTRA_MEMORY_EMBEDDING_OLLAMA_ENDPOINT=http://127.0.0.1:11434
```

To disable automatic indexing:

```env
ASTRA_MEMORY_EMBEDDING_AUTO_INDEX=false
```

## Validation

Run:

```powershell
cd src-tauri
cargo check

cd ..
npm run build
```

Then create a new memory by chatting with Astra and verify:

- `.astra/memory/journal/memory_events.jsonl` contains `embedding_maintenance_ran` and/or `embedding_upserted`.
- Memory Graph vector pending count decreases without manual rebuild.

## Architecture notes

This remains LLM-first and Rust-governed:

- LLM distills/uses memory.
- Rust owns persistence, validation, bounded indexing and diagnostics.
- Embeddings are retrieval infrastructure, not a source of truth.
