# AstraOS v0.5.5 — Conversation Memory Consolidation

This patch extends the Cognitive Memory Graph so Astra can learn from ordinary conversations, not only from Work Sessions, tools, or Deep Research.

## What changed

- Adds typed conversation-memory consolidation in Rust.
- Automatically runs a bounded LLM-first extractor after completed assistant responses.
- Persists conversation turns as `conversation_turn` nodes.
- Lets the LLM propose durable memory candidates:
  - important points
  - entities
  - user preferences
  - procedures
  - decisions
- Rust validates and persists only typed records.
- Adds `consolidate_conversation_memory_bundle` Tauri command.
- Adds frontend TypeScript types and hook support.
- Adds a Python normalization helper for future extraction pipelines.

## Governance

Conversation memory is advisory context only. It never authorizes actions and never bypasses permissions, policy, approval checks, or governed tool execution.

The model extracts candidate memories. Rust validates, caps, types, persists, and links them.

## Runtime behavior

After a normal or grounded assistant response:

```txt
conversation turn
  -> LLM conversation-memory extractor
  -> typed ConversationMemoryBundle
  -> Rust validation
  -> Memory Graph nodes/edges
  -> activation event
```

If the extractor fails, Astra falls back to an episode-only conversation node instead of losing the event.

## New trace stage

LLM extraction calls are traced under:

```txt
.astra/diagnostics/llm/conversation_memory_extractor/
```

## Env

Optional:

```env
ASTRA_CONVERSATION_MEMORY_TIMEOUT_MS=25000
ASTRA_LLM_TRACE_LEVEL=metadata
```

## Validation

Run:

```powershell
cd src-tauri
cargo check

cd ..
npm run build
```

Then have a normal conversation with Astra and inspect:

```txt
.astra/memory/journal/memory_events.jsonl
```

You should see `node_created`, `edge_created`, and `activation_recorded` entries with `conversation_memory_consolidation` metadata.
