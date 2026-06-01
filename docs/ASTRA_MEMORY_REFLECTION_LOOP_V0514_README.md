# AstraOS v0.5.14 — Cognitive Memory Reflection & Confidence Loop

This patch adds a governed, LLM-integrated reflection loop for Astra's Cognitive Memory Graph.

## Goal

After Astra answers, the runtime now performs a bounded async reflection step:

1. Retrieve relevant Memory Graph context using the existing LLM-integrated brain RAG retrieval.
2. Ask the local LLM to evaluate whether the answer used the available memory appropriately.
3. Persist only advisory reflection nodes/edges when there is a durable lesson, underused memory, correction signal, or quality issue.
4. Keep Rust as the validator/governance layer. The reflection never executes tools and never mutates existing memory governance directly.

## Files changed

```txt
src-tauri/src/lib.rs
src-tauri/src/memory/consolidation/mod.rs
src-tauri/src/memory/consolidation/reflection.rs
ASTRA_MEMORY_REFLECTION_LOOP_V0514_README.md
```

## Architecture

The flow is:

```txt
assistant answer
  -> conversation memory consolidation
  -> memory reflection retrieval
  -> LLM reflection verifier
  -> Rust validation
  -> reflection node / advisory lesson nodes
  -> edges to used / ignored / possibly contradicted memory nodes
  -> activation event
```

This is intentionally not a hard-coded memory QA path. The LLM evaluates memory usage generally for every eligible answer.

## Safety / governance

- Reflection is advisory-only.
- Existing memories are not automatically confirmed, deprecated, contradicted, or deleted by the LLM.
- Potential contradictions are stored as reflection edges requiring user/runtime governance.
- The Memory Graph remains consultive context and never bypasses policy, approvals, audit, or tool validation.

## Diagnostics

LLM reflection calls are traced under:

```txt
.astra/diagnostics/llm/memory_reflection_verifier/
```

Memory events are written to:

```txt
.astra/memory/journal/memory_events.jsonl
```

Relevant events include:

```txt
memory_reflection_consolidation
memory_reflection_skipped
node_created
edge_created
activation_recorded
```

## Configuration

```env
ASTRA_MEMORY_REFLECTION_ENABLED=true
ASTRA_MEMORY_REFLECTION_TIMEOUT_MS=30000
```

Set `ASTRA_MEMORY_REFLECTION_ENABLED=false` to disable the loop.

## Validation

After importing:

```powershell
cd src-tauri
cargo check

cd ..
npm run build
```

Then test:

1. Tell Astra a durable fact or preference.
2. Ask a follow-up that should use memory.
3. Check Memory tab activation timeline.
4. Check `.astra/diagnostics/llm/memory_reflection_verifier/`.
5. Check `.astra/memory/journal/memory_events.jsonl`.

## Next step

v0.5.15 should introduce Memory Evidence Binding UI: show which memory nodes influenced a specific answer, their verification status, and allow the user to confirm/correct them from the answer itself.
