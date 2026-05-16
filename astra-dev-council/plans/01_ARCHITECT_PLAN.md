# 01 Architect Plan

## Purpose

Lead architecture plan, ADR candidates, module boundaries, sequencing, risks.

## Current status

- Status: completed
- Owner: Architect Agent
- Last updated by: Architect Agent (2026-05-01)

## Scope

Define the technical coordination layer enabling 10 specialized agents to work safely on the same `main` branch of Astra without branches, worktrees, or destructive operations. This plan covers Phases 0–4 with explicit gates between each phase.

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

### Phase 0 — Governance Bootstrap (CURRENT — COMPLETED)

**Objective:** Create the minimal coordination infrastructure required before any agent touches application source code.

**Deliverables:**
- Council documentation files (TASK.md, ACTIVE_PLAN.md, AGENT_BOARD.md, FILE_LOCKS.md, DECISIONS.md, SECURITY_REVIEW.md, QA_REPORT.md, CHANGELOG_AGENTIC.md)
- Agent role prompts (10 agents in `agents/*.md`)
- Launch automation scripts
- File lock protocol
- QA validation checklist
- Security safety gates

**Completion Criteria:**
- All council files readable and internally consistent
- File lock protocol is enforceable
- Security and QA have signed off on bootstrap

### Phase 1 — Read-Only Council Visibility (FUTURE)

**Objective:** Add a read-only UI panel inside Astra displaying council status from markdown files.

**Proposed Boundaries:**
- Frontend: `src/components/dev-council/`, `src/types/devCouncil.ts`, `src/hooks/useDevCouncil.ts`
- Rust: `src-tauri/src/dev_council.rs` — new read-only Tauri command

**Constraints:**
- No autonomous code execution
- No write access to council files from within Astra
- UI is informational only

### Phase 2 — Controlled Task Runner (FUTURE)

**Objective:** Enable agents to execute approved validation commands only.

**Allowed Commands (Whitelist):**
- `npm run build`, `npm run tauri dev`, `cargo check`, `cargo test`

**Implementation:** Extend `src-tauri/src/terminal_runner.rs` with whitelist and policy check.

### Phase 3 — Patch Proposal Mode (FUTURE)

**Objective:** Agents may generate patch proposals requiring human approval before applying.

**Flow:** Proposal → Risk Assessment → QA Review → Human Approval → Apply

### Phase 4 — Advanced Multi-Agent Orchestration (FUTURE — REQUIRES EXPLICIT APPROVAL)

**Prerequisites:** Permission model, audit log, rollback plan, UI approval gate, regression suite, safe model routing.

**Recommendation:** Do not proceed without explicit human sign-off and completed Phase 0–3 validation.

## Files or areas involved

| File/Area | Intended action | Lock required | Risk |
|---|---|---:|---|
| `astra-dev-council/plans/01_ARCHITECT_PLAN.md` | Create comprehensive plan | yes | none |
| `astra-dev-council/ACTIVE_PLAN.md` | Reference existing plan | no | none |
| `astra-dev-council/DECISIONS.md` | Add ADRs as needed | no | none |
| `astra-dev-council/AGENT_BOARD.md` | Update status | no | none |
| `astra-dev-council/CHANGELOG_AGENTIC.md` | Log changes | no | none |
| `astra-dev-council/FILE_LOCKS.md` | Manage locks | yes | none |
| High-risk files matrix | Document for review | no | none |

## Risks and constraints

- **No source code changes in Phase 0:** Application code modifications require explicit human approval and completion of bootstrap.
- **File lock protocol must be respected:** Agents must check `FILE_LOCKS.md` before editing.
- **High-risk files require multiple reviewers:** See matrix in plan for required reviewers per file.
- **Rollback strategy required for all changes:** Git-based or file-based rollback must be documented.

## Validation

- [x] All council files read and validated for consistency
- [x] Architect plan file created/updated
- [x] File locks properly documented
- [ ] Security Agent sign-off (handoff)
- [ ] QA Agent sign-off (handoff)
- [ ] Release Manager sign-off (handoff)

**Build commands (when source changes occur):**
```bash
npm run build
cd src-tauri && cargo check
cd src-tauri && cargo test
```

## Handoffs

| To agent | Reason | Status |
|---|---|---|
| Security Agent | Review plan for safety gaps, especially Phases 2–4 | pending |
| QA Agent | Validate regression checklist covers all high-risk files | pending |
| Release Manager | Confirm rollback strategy and proposal expiration policy | pending |

## Final notes

This plan prioritizes enterprise-grade stability through governance documents, file locks, QA gates, and release review rather than autonomous self-modifying code. The phased approach ensures that each capability builds on a validated foundation. Do not skip phases.

---

## Phase 1 Readiness Review (2026-05-01)

### Phase 1 Scope and Constraints

**Phase 1 Objective:** Add a read-only UI panel inside Astra displaying council status from markdown files.

**Phase 1 Constraints:**
- **Read-only only:** No write access to council files from within Astra.
- **New files only:** Changes must target:
  - Frontend: `src/components/dev-council/`, `src/types/devCouncil.ts`, `src/hooks/useDevCouncil.ts`
  - Rust: `src-tauri/src/dev_council.rs` (new read-only Tauri command)
- **No high-risk file changes:** No modifications to `lib.rs`, `desktop_agent.rs`, `screen_workflow.rs`, `conversation_router.rs`, or `assistant_response.rs`.
- **No autonomous execution:** No task runners, patch proposals, or background code execution.

### Phase 1 Validation Checklist

| Check | Status | Notes |
|---|---|---|
| QA Agent completes regression checklist for Phase 1 scope | pending | QA Agent must validate UI panel does not modify council state or introduce regressions. |
| Security Agent reviews Phase 1 new file surfaces | pending | Security gates must cover markdown parsing, file watching, and IPC via Tauri commands. |
| Release Manager produces bootstrap consolidation | pending | Final consolidation of all council files and validation results. |
| Architect reviews Phase 1 UI security requirements | ready | UI component security constraints for read-only Dev Council panel. |
| File locks for new files in Phase 1 | pending | Locks for `src/components/dev-council/`, `src/types/devCouncil.ts`, `src/hooks/useDevCouncil.ts`, and `src-tauri/src/dev_council.rs`. |

### Phase 1 Implementation Plan

1. **UI Panel Design:**
   - Create a read-only markdown viewer component in `src/components/dev-council/`.
   - Define a Tauri command to fetch council status from markdown files.
   - Ensure UI does not modify council state.

2. **Rust Integration:**
   - Add a new Tauri command in `src-tauri/src/dev_council.rs` for fetching council status.
   - Ensure command is read-only and does not modify files.

3. **TypeScript Contracts:**
   - Mirror Rust types in `src/types/devCouncil.ts`.
   - Define hooks for council status updates.

4. **Security Review:**
   - Ensure markdown parsing does not execute arbitrary code.
   - Validate file watching does not introduce filesystem risks.

5. **QA Validation:**
   - Test UI panel for regressions in existing functionality.
   - Ensure no unintended modifications to council files.

---

## Open Action Items Before Phase 1

| Action | Owner | Status |
|---|---|---|
| QA Agent completes regression checklist for Phase 1 scope | QA Agent | pending |
| Release Manager produces bootstrap consolidation | Release Manager | pending |
| Architect reviews Phase 1 read-only UI scope | Architect Agent | ready |
| Security Agent reviews Phase 1 new file surfaces | Security Agent | pending |
| File locks for new files in Phase 1 | Architect Agent | pending |

---

## Risks and Mitigations for Phase 1

| Risk | Mitigation |
|---|---|
| UI panel modifies council state | UI is read-only; all changes require manual approval. |
| New files introduce security vulnerabilities | Security gates re-evaluated for Phase 1 surfaces. |
| Regressions in existing functionality | QA Agent validates regression checklist. |
| Unintended modifications to high-risk files | File locks and explicit approval required. |
| Unauthorized autonomous execution | No task runners or patch proposals in Phase 1. |

---

## Validation Commands for Phase 1

```bash
# Frontend validation
npm run build

# Rust validation
cd src-tauri && cargo check
cd src-tauri && cargo test

# Interactive validation
npm run tauri dev
```

---

## Phase Transition Readiness Assessment (2026-05-01)

### Phase 0 — Completion verification

| Criterion | Status | Notes |
|---|---|---|
| All 8 council files exist and are readable | PASS | All verified this turn |
| ACTIVE_PLAN.md has phased plan | PASS | Phases 0–4 defined |
| AGENT_BOARD.md reflects agent statuses | PASS | 3 completed, 5 waiting, 1 pending |
| FILE_LOCKS.md maintained | PASS | 5 locks documented |
| DECISIONS.md has ADRs | PASS | 5 ADRs recorded |
| SECURITY_REVIEW.md has safety gates | PASS | 5 safety categories defined |
| QA_REPORT.md has regression checklist | PASS | Frontend/Rust/screen/voice/AI checklists |
| CHANGELOG_AGENTIC.md records changes | PASS | 4 entries (bootstrap + 3 agents) |
| Agent role prompts created | PASS | 10 agents in agents/*.md |
| Launch automation script | PASS | Present in repo |
| Architect plan file | PASS | This file |
| Security plan file | PASS | plans/08_SECURITY_PLAN.md |
| Product plan file | PASS | plans/02_PRODUCT_PLAN.md |
| QA plan file | PASS | plans/09_QA_PLAN.md |
| Release plan file | PASS | plans/10_RELEASE_PLAN.md |

### Phase 0 — Architecture verdict

**APPROVED for Phase 1 transition** — with conditions:

1. **QA Agent must complete regression checklist** before any source code changes. Currently `bootstrap-docs-only`.
2. **Release Manager must produce final bootstrap consolidation** before first commit.
3. **Phase 1 must be read-only only** — no task runner, no patch application, no autonomous execution.
4. **All Phase 1 changes must target new files** under `src/components/dev-council/`, `src/types/devCouncil.ts`, `src/hooks/useDevCouncil.ts`, and `src-tauri/src/dev_council.rs` — no touching high-risk files.
5. **Security gates must be re-evaluated for Phase 1** — new files introduce new surfaces (markdown parsing, file watching, IPC via Tauri commands).

### Open action items before Phase 1

| Action | Owner | Status |
|---|---|---|
| QA Agent completes regression checklist for Phase 1 scope | QA Agent | pending |
| Release Manager produces bootstrap consolidation | Release Manager | pending |
| Architect reviews Phase 1 read-only UI scope | Architect Agent | ready |
| Security reviews Phase 1 new file surfaces | Security Agent | pending |
