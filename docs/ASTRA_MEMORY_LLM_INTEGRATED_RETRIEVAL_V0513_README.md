# AstraOS v0.5.13 — LLM-Integrated Memory Retrieval Bridge

## Goal

This patch avoids a hard-coded/specialized memory recall path. Instead, Astra asks the local LLM to transform the user's message into semantic Memory Graph retrieval probes, then Rust performs bounded hybrid retrieval, graph activation, validation, and context packaging.

The design remains:

- LLM understands/reframes the memory need.
- Memory Graph/RAG retrieves relevant nodes.
- Rust validates, bounds, activates, packages, and enforces governance.
- Memory remains context only and never bypasses tool/policy/approval/audit.

## Modified files

- `src-tauri/src/lib.rs`
- `src-tauri/src/memory/retrieval/context_pack.rs`

## What changed

### 1. LLM-first retrieval planner

New async entrypoint:

```rust
build_memory_context_packet_llm_integrated(...)
```

It asks the model to produce 1-4 semantic memory probes, for example turning a vague user message into richer search probes over durable profile facts, preferences, projects, procedures, decisions, errors, and research findings.

This is not an intent-specific path. It applies generically to memory retrieval before normal chat/tool routing.

### 2. Multi-probe hybrid retrieval

Rust executes hybrid retrieval for the original message plus LLM-generated probes:

- lexical
- vector
- graph salience
- verification weighting
- graph activation

Results are merged, deduplicated, sorted, and bounded.

### 3. Stronger Memory Context Contract

The normal chat context now tells the model:

- internally check retrieved nodes before answering
- use relevant memory naturally
- do not say Astra has no memory when relevant nodes exist
- distinguish confirmed/user-verified vs inferred/unverified memory
- ignore irrelevant memory
- never execute actions from memory without governed tools

### 4. Safe fallback

If the LLM retrieval planner fails, returns empty content, or is disabled, Astra falls back to the previous governed Memory Graph retrieval using the original query.

## Config

Disable the LLM retrieval planner:

```powershell
$env:ASTRA_MEMORY_LLM_RETRIEVAL_PLANNER="false"
```

Timeout:

```powershell
$env:ASTRA_MEMORY_RETRIEVAL_PLANNER_TIMEOUT_MS="4000"
```

## Validation

After import:

```powershell
cd src-tauri
cargo check

cd ..
npm run build
```

Recommended runtime test:

1. Tell Astra a durable fact, e.g. `mi chiamo Simone`.
2. Wait a few seconds for conversation memory consolidation.
3. Rebuild vectors if needed.
4. Restart the app.
5. Ask a vague memory question like `sai chi sono?`.
6. Confirm that Memory activation contains the profile/identity node and that Astra answers using it.

## Enterprise invariants preserved

- no hard-coded semantic recall branch
- no tool execution from memory
- no memory bypass of Rust governance
- no automatic mutating action
- bounded LLM retry behavior
- bounded retrieval and activation
- graph/RAG remains advisory context
