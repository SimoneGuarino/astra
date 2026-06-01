# AstraOS v0.5.7 — Real Embedding Provider Integration

This patch introduces a real semantic embedding provider path for the Cognitive Memory Graph while preserving the existing Rust-governed architecture.

## What changed

- `StableHashEmbeddingProvider` remains available as deterministic local fallback.
- New `OllamaEmbeddingProvider` implements the same provider trait.
- `ASTRA_MEMORY_EMBEDDING_PROVIDER=ollama` enables Ollama embeddings.
- `ASTRA_MEMORY_EMBEDDING_MODEL` selects the embedding model, default `nomic-embed-text` for Ollama.
- Hybrid memory retrieval no longer fails hard if query embedding generation fails; it falls back to lexical/graph retrieval.
- Rebuild indexing can fallback to the deterministic local provider if Ollama is unavailable.
- UI displays the active adapter/fallback info after rebuild.
- Optional Python embedding worker helper added for future sentence-transformers/provider expansion. Rust remains the source of truth.

## Recommended env for semantic retrieval

```powershell
$env:ASTRA_MEMORY_EMBEDDING_PROVIDER="ollama"
$env:ASTRA_MEMORY_EMBEDDING_MODEL="nomic-embed-text"
$env:ASTRA_MEMORY_EMBEDDING_OLLAMA_ENDPOINT="http://127.0.0.1:11434"
```

Then make sure the model is available:

```powershell
ollama pull nomic-embed-text
```

Then launch Astra and click `Rebuild vectors` in the Memory tab.

## Safety / governance

- SQLite Memory Graph remains the source of truth.
- Vector embeddings are advisory retrieval indexes only.
- Python does not persist memory directly.
- Memory retrieval remains contextual; it never bypasses policy, approvals, or tool validation.

## Fallback behavior

By default, if Ollama embeddings fail during rebuild, Astra indexes the chunk with the deterministic local fallback so the Memory Graph remains usable.

Disable fallback only for strict testing:

```powershell
$env:ASTRA_MEMORY_EMBEDDING_DISABLE_FALLBACK="true"
```
