# AstraOS Cognitive Memory Graph - v0.5 foundation

This patch introduces the first governed foundation for AstraOS long-term cognitive memory.

## What is included

- Rust module `src-tauri/src/memory/` with typed memory nodes, edges, activations, SQLite-backed persistence, FTS5 lexical retrieval, JSONL journal and vector-provider abstraction.
- Tauri commands for graph status, node/edge creation, query, activation and snapshot export.
- React feature module `src/features/memory/` with a first Memory Graph panel and activation visualization.
- Python `python_services/memory/` worker structure for future embeddings, extraction and deep-research normalization.

## Architecture

SQLite is the source of truth for the memory graph. Vector retrieval is intentionally represented as a provider-agnostic adapter, not as the primary source of truth.

Rust remains responsible for validation, persistence, privacy, bounded activation and governance. Python workers are specialist providers only. The LLM can propose memory candidates, but Rust must validate before persistence.

## Runtime storage

Astra creates local memory files under:

`.astra/memory/graph/astra_memory.sqlite`
`.astra/memory/journal/memory_events.jsonl`

## Notes

`cargo check` could not be executed in the generation environment because `cargo` is not installed there. The patch is intentionally additive and keeps the current runtime governed model intact.
