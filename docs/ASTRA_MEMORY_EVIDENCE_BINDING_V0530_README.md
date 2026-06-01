# AstraOS v0.5.30 — Generic Memory Evidence Binding Verifier

This patch adds a generic, LLM-first Memory Evidence Binding loop for normal chat responses.

## Purpose

Astra already retrieves Memory Graph / Brain RAG context, but a local LLM can still ignore useful retrieved memory and answer as if no memory exists. This patch adds a verifier that checks the relationship between:

- user message
- retrieved Memory Context Packet
- draft assistant answer

If the verifier detects that memory was underused, contradicted, overclaimed, or used with the wrong certainty, Astra performs one bounded regeneration using the same memory evidence.

## Architectural constraints preserved

- No hard-coded path such as `if user asks who am I`.
- The LLM performs semantic verification.
- Rust validates, bounds, traces, emits diagnostics, and persists journal metadata.
- Memory remains advisory context.
- Memory never authorizes tools or actions.
- Policy, approval, audit, and governed execution remain unchanged.

## New files

```txt
src-tauri/src/memory/verification/mod.rs
```

## Modified files

```txt
src-tauri/src/lib.rs
src-tauri/src/memory/mod.rs
```

## Runtime flow

```txt
normal chat draft answer
  ↓
build LLM-integrated Memory Context Packet
  ↓
LLM evidence verifier
  ↓
answer_consistent → final answer unchanged
memory_underused / uncertainty_mismatch / contradiction → one bounded regeneration
  ↓
final answer emitted via existing assistant-request-finished replacement path
```

## Diagnostics

Trace files are written under:

```txt
.astra/diagnostics/llm/memory_evidence_binding_verifier/
.astra/diagnostics/llm/memory_evidence_binding_regenerator/
```

A memory journal note is also appended with event name:

```txt
memory_evidence_binding
```

The UI can listen for:

```txt
memory-evidence-binding
```

## Environment flags

```env
ASTRA_MEMORY_EVIDENCE_BINDING_ENABLED=true
ASTRA_MEMORY_EVIDENCE_BINDING_REGENERATE=true
ASTRA_MEMORY_EVIDENCE_BINDING_TIMEOUT_MS=8000
ASTRA_MEMORY_EVIDENCE_BINDING_MIN_REGEN_CONFIDENCE=0.55
```

## Validation

```powershell
cd src-tauri
cargo check

cd ..
npm run build
```

Suggested functional test:

1. Tell Astra a durable fact.
2. Confirm it appears in Memory Graph / Brain RAG.
3. Ask a related question.
4. Verify that if the draft answer ignores retrieved memory, the final response is corrected by evidence binding.
