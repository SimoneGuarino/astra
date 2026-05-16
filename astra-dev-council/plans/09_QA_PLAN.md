# 09 QA Plan

## Purpose

QA/regression checklist, validation matrix, pass/fail evidence.

## Current status

- Status: bootstrap-complete
- Owner: `09_qa_agent.md`
- Last updated by: QA Agent (2026-05-01)

## Scope

QA Agent responsibilities for Phase 0 (bootstrap):
- Define regression checklist and validation commands in `QA_REPORT.md`
- Establish baseline validation matrix for future source-code changes
- Document safety gates for frontend, Rust runtime, screen workflow, voice/audio, and AI orchestration
- Prepare QA gate status tracking
- Coordinate with Security Agent on high-risk change categories

## Inputs read

- [x] `astra-dev-council/TASK.md`
- [x] `astra-dev-council/ACTIVE_PLAN.md`
- [x] `astra-dev-council/AGENT_BOARD.md`
- [x] `astra-dev-council/FILE_LOCKS.md`
- [x] `astra-dev-council/DECISIONS.md`
- [x] `astra-dev-council/CHANGELOG_AGENTIC.md`
- [x] `astra-dev-council/SECURITY_REVIEW.md`
- [x] `astra-dev-council/QA_REPORT.md`

## Proposed work

**Phase 0 (Bootstrap) — Completed:**

1. Read all council governance files to understand current state
2. Updated `QA_REPORT.md` with comprehensive regression checklist covering:
   - Frontend validation (TypeScript, components, hooks)
   - Rust runtime validation (lib.rs, Tauri commands, desktop agent contracts)
   - Screen workflow validation (capture, vision, grounding, continuation)
   - Voice/audio validation (TTS, STT, VAD, session lifecycle)
   - AI orchestration validation (model routing, conversation routes, capability context)
3. Established QA gate status as `bootstrap-docs-only`
4. Documented baseline validation commands (`npm run build`, `cargo check`, `cargo test`)
5. Created this plan file with validation matrix and handoff notes

**Future phases (pending Architect/Security review):**

- Phase 1: Validate read-only UI panel for council status
- Phase 2: Validate controlled task runner with policy checks
- Phase 3: Validate patch proposal mode with approval gates

## Files or areas involved

| File/Area | Intended action | Lock required | Risk |
|---|---|---:|---|
| `QA_REPORT.md` | Define regression checklist | yes | low |
| `AGENT_BOARD.md` | Update QA agent status | no | low |
| `CHANGELOG_AGENTIC.md` | Record council doc changes | no | low |
| `FILE_LOCKS.md` | Update QA lock status | yes | low |

## Risks and constraints

- No application source code changes in Phase 0 (documentation only)
- Future phases require Architect plan approval and Security review before implementation
- Screen workflow and voice/audio modules are high-risk and require careful regression testing
- Rust/TypeScript contract mismatches in `desktopAgent.ts` could cause runtime failures

## Validation

**Phase 0 validation (completed):**
- All council files read and understood
- `QA_REPORT.md` contains comprehensive regression checklist
- Validation commands documented for future use
- No build/test commands required for documentation-only phase

**Future validation (when source changes begin):**
```bash
npm run build          # Frontend TypeScript compilation
cd src-tauri && cargo check  # Rust compilation check
cd src-tauri && cargo test   # Rust unit tests
npm run tauri dev      # Interactive runtime validation (optional)
```

## Handoffs

| To agent | Reason | Status |
|---|---|---|
| Architect Agent | QA ready to review validation commands for Phase 2 | pending |
| Security Agent | QA will verify audit events once task runner implemented | pending |
| Release Manager | QA will provide regression checklist before any release | pending |

## Final notes

QA Agent has completed Phase 0 bootstrap. All governance documentation is in place:
- Regression checklist defined in `QA_REPORT.md`
- Validation commands documented
- Safety gates aligned with `SECURITY_REVIEW.md`
- Ready to support future phases when Architect and Security approve

No blockers. Awaiting Phase 1 readiness signal from Architect Agent.
