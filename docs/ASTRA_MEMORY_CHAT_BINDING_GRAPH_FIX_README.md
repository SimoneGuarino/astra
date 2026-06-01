# AstraOS v0.5.11 — Memory Chat Binding & Interactive Graph Fix

## Scope

This patch fixes two issues observed after v0.5.10:

1. Cognitive Memory was being created, but short durable conversation facts such as `mi chiamo Simone` could be skipped by the conversation-memory consolidation pre-filter.
2. The automatic Memory Context Packet used the deterministic local hash embedding provider even when a real embedding provider was configured, so semantic recall could be weaker than the UI hybrid search.
3. The Memory Graph view was static and secondary, while the product direction requires an Obsidian-like interactive graph where nodes can be dragged and linked nodes react.
4. The animated translucent activation pulse could look like an unexpected moving bubble. This is replaced with a stable halo while keeping electric edge animation.

## Changed files

- `src-tauri/src/lib.rs`
- `src-tauri/src/memory/retrieval/context_pack.rs`
- `src/features/memory/components/MemoryGraphPanel.tsx`
- `src/App.css`

## Runtime behavior

### Conversation memory binding

The conversation consolidation threshold is now configurable and much less restrictive by default:

```env
ASTRA_CONVERSATION_MEMORY_MIN_USER_CHARS=4
ASTRA_CONVERSATION_MEMORY_MIN_ASSISTANT_CHARS=2
ASTRA_CONVERSATION_MEMORY_MIN_COMBINED_CHARS=16
```

This keeps the LLM-first extraction model: Rust does not classify `mi chiamo ...` through a hard-coded semantic rule. Rust only decides whether the exchange is large enough to be worth sending to the memory extractor. The model is instructed to extract explicit durable profile facts, preferred names, user preferences, project constraints and stable working context.

### Memory retrieval binding

Automatic memory injection now uses `build_embedding_provider()` instead of always using `StableHashEmbeddingProvider`. This means that if the real Ollama embedding provider is configured, normal chat memory context retrieval uses the same semantic provider as the Memory UI hybrid search.

### Memory preamble

The model receives a stronger but still governed memory preamble. It may use retrieved memory as durable context for questions about previous facts, preferences, user profile, or what Astra remembers, but memory remains advisory and never authorizes tool execution.

### Interactive Memory Graph

The graph now uses an internal bounded force layout:

- nodes are draggable;
- connected nodes are pulled by edges;
- repulsion reduces overlap;
- graph remains inside the canvas bounds;
- the canvas is larger and more primary;
- active nodes use a stable halo instead of the previous animated translucent pulse;
- active edges still show electric flow.

## Validation

After importing:

```powershell
cd src-tauri
cargo check

cd ..
npm run build
```

Recommended runtime test:

1. Start Astra.
2. Say: `mi chiamo Simone`.
3. Wait a few seconds for the async memory extractor.
4. Click `Rebuild vectors` if embeddings are pending.
5. Restart/reload the app.
6. Ask: `sai chi sono?`

Expected behavior: Astra should retrieve the profile/name memory when it exists and answer from memory, indicating uncertainty if the node is still LLM-inferred/unconfirmed.

## Governance

This patch does not bypass governance, approvals, policy validation, tool validation or audit. Memory remains context only.
