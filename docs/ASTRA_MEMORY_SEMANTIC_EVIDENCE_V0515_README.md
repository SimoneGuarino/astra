# AstraOS v0.5.15 — LLM Semantic Memory Distillation & Evidence Binding

## Goal

This patch fixes a critical gap in the Cognitive Memory Graph: short but durable user-declared facts such as “io sono Simone” must not remain only as raw episodic conversation strings. They must be distilled by the LLM into semantic memory atoms that the brain RAG can retrieve and pass back to the LLM as contextual evidence.

The implementation remains LLM-first and Rust-governed:

- the LLM proposes semantic memory atoms;
- Rust validates and persists typed Memory Graph nodes/edges;
- retrieval remains bounded and advisory;
- memory never bypasses policy, approval, audit, or governed tool validation.

## What changed

### Rust conversation consolidation

`ConversationMemoryBundle` now supports `semantic_atoms`.

Each semantic atom can include:

- `subject`
- `predicate`
- `object`
- `evidence`
- `kind`
- `confidence`
- `tags`

During consolidation, semantic atoms become typed graph nodes, usually `claim`, `entity`, `user_preference`, `procedure`, `decision`, or `concept`, linked back to the originating conversation turn through `derived_from`.

This allows the memory graph to store durable facts as contextualized knowledge instead of raw phrases.

### LLM extractor prompt

The conversation memory extractor now explicitly asks the model to distill user-declared facts into semantic atoms.

Example intent:

User says: `io sono Simone`

The LLM should produce a semantic atom similar to:

```json
{
  "subject": "user",
  "predicate": "preferred_name",
  "object": "Simone",
  "evidence": "The user said: io sono Simone",
  "kind": "profile_fact",
  "tags": ["user_profile", "identity", "name"]
}
```

### Memory context packet

Memory context nodes now include an optional `content_excerpt`, so the final LLM can see the actual evidence behind episodic or semantic nodes, not just a short title/summary.

### Cognitive working memory backfill

The LLM-integrated retrieval path now includes a bounded working-memory backfill from recent/salient graph nodes. This is not an intent-specific hard-coded path. It is a general brain-like recall mechanism that helps the model recover context even when lexical/vector retrieval misses a relevant memory.

Config:

```env
ASTRA_MEMORY_WORKING_BACKFILL=true
ASTRA_MEMORY_WORKING_BACKFILL_SCAN_LIMIT=60
ASTRA_MEMORY_WORKING_BACKFILL_LIMIT=8
```

## Validation scenario

1. Start Astra.
2. Say: `io sono Simone`.
3. Wait a few seconds for async memory consolidation.
4. Refresh Memory tab.
5. Rebuild vectors if needed.
6. Restart the app.
7. Ask: `sai chi sono?`.

Expected result:

Astra should answer using the Memory Graph evidence, for example:

> Dalla memoria locale risulta che ti sei presentato come Simone. Questa informazione è memoria LLM-inferred finché non la confermi esplicitamente.

## Governance

This patch does not authorize actions from memory. Memory remains contextual evidence only. All tool usage remains governed by Rust policy, permissions, approval checks, and audit.
