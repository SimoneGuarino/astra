# AstraOS v0.5.4 — Deep Research Memory & Consolidation

This patch adds the first governed consolidation path for deep-research output into the AstraOS Cognitive Memory Graph.

## What changed

### Rust

- Added `src-tauri/src/memory/consolidation/research.rs`.
- Added typed research bundle models:
  - `ResearchMemoryBundle`
  - `ResearchSource`
  - `ResearchFinding`
  - `ResearchClaim`
  - `ResearchProcedure`
  - `ResearchRecommendation`
- Added a Rust-governed consolidation function:
  - `consolidate_research_bundle(...)`
- Added Tauri command:
  - `consolidate_research_memory_bundle`

The consolidation converts a research bundle into typed Memory Graph nodes and relations:

- `research_topic`
- `source_document`
- `research_finding`
- `claim`
- `procedure`
- `decision` / recommendation

and links them through existing relation types:

- `derived_from`
- `about`
- `supports`
- `learned_from`
- `related_to`

It also records a bounded activation event so the UI can visualize research-related memory activation.

## Architecture notes

The model may propose research findings, claims, procedures, and recommendations, but Rust remains responsible for:

- schema validation
- persistence
- relation creation
- bounded activation
- confidence/status handling
- governance boundary preservation

Research memory is advisory context only. It never authorizes autonomous action and never bypasses governed tools, permissions, approvals, or audit.

## Frontend

- Extended `src/types/memory.ts` with research bundle types.
- Extended `useMemoryGraph` with `consolidateResearchBundle(...)`.
- Updated Memory Graph labels for research nodes.

## Python

- Updated the research worker normalization schema to emit Rust-compatible research bundles.
- Python remains an extraction/normalization helper only; Rust is still the source of truth.

## Validation

After importing:

```powershell
cd src-tauri
cargo check

cd ..
npm run build
```

Then a deep-research flow can call `consolidate_research_memory_bundle` with a typed bundle and verify that new research nodes appear in the Memory Graph UI.
