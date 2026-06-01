# AstraOS v0.5.9 — Procedural Skills Candidate System

This patch introduces the first governed procedural-skill layer on top of the Cognitive Memory Graph.

## What this adds

- `memory_skill_candidates` SQLite table.
- Typed Rust skill candidate models.
- Extraction of skill candidates from existing `procedure`, `workflow`, and `code_pattern` memory nodes.
- User-visible candidate governance: approve, disable, deprecate.
- Skill candidates are advisory only and do not execute autonomous actions.
- UI panel inside the Memory Graph for extraction and candidate review.

## Safety model

A skill candidate does **not** grant execution rights.

Astra may use approved candidates as procedural context, but any real action still passes through:

- Rust tool validation
- permission checks
- policy checks
- approval gates
- audit logging
- bounded execution/recovery

## Commands added

- `extract_memory_skill_candidates`
- `list_memory_skill_candidates`
- `update_memory_skill_candidate`

## Validation

After importing the patch:

```powershell
cd src-tauri
cargo check

cd ..
npm run build
```

Then open Astra > Approval Center > Memory and click `Extract skills`.

You should see journal events in:

```txt
.astra/memory/journal/memory_events.jsonl
```

Expected events:

- `skill_candidate_created`
- `skill_candidate_governance_updated`
- `activation_recorded`

## Next step

The next architectural step is `v0.6.0 — Governed Skill Runtime Planner`.

That does not execute skills autonomously. It introduces a typed planning layer where Astra can propose using an approved skill, show the plan, bind it to governed tools, calculate risk, and ask for user approval before any mutating or high-risk action.
