# 02 Product Plan

## Purpose

Product value, MVP scope, prioritization, user impact, non-goals.

## Current status

- Status: completed
- Owner: `02_product_agent.md`
- Last updated by: Product Agent (2026-05-01)
- Phase 0 transition: approved for Phase 1 with 5 conditions (see Architect Agent's Phase Transition Assessment)

## Scope

Define MVP value proposition, prioritize phases from ACTIVE_PLAN.md, establish non-goals, and provide product-level guardrails for the Astra Development Council coordination layer.

## Inputs read

- [x] `astra-dev-council/TASK.md`
- [x] `astra-dev-council/ACTIVE_PLAN.md`
- [x] `astra-dev-council/AGENT_BOARD.md`
- [x] `astra-dev-council/FILE_LOCKS.md`
- [x] `astra-dev-council/DECISIONS.md`
- [x] `astra-dev-council/CHANGELOG_AGENTIC.md`
- [x] `astra-dev-council/SECURITY_REVIEW.md`
- [x] `astra-dev-council/QA_REPORT.md`

## Product value proposition

**Core problem**: Multiple AI agents working on the same codebase need coordination to avoid conflicts, regressions, and scope creep.

**Solution**: A governance layer that enforces file locks, small reversible changes, explicit handoffs, QA gates, and release review—all visible in markdown and eventually in a read-only UI panel.

**User impact**: Developers can delegate complex multi-step work to specialized agents while maintaining enterprise-grade stability and auditability.

## MVP scope (Phase 0)

Must have:
- Council documentation structure (TASK.md, ACTIVE_PLAN.md, AGENT_BOARD.md, FILE_LOCKS.md, DECISIONS.md, CHANGELOG_AGENTIC.md, SECURITY_REVIEW.md, QA_REPORT.md)
- Agent role prompts with explicit scope boundaries
- File lock protocol with active tracking
- Agent board for status/handoffs/blockers
- Launch automation script
- Pre-authorized write scope for each agent's own plan file

Nice to have (deferred):
- Read-only UI panel (Phase 1)
- Controlled task runner (Phase 2)
- Patch proposal mode (Phase 3)

## Non-goals for Phase 0

- Autonomous code generation or self-modification
- In-app agent orchestration
- Real-time agent communication
- Automated conflict resolution
- CI/CD integration
- External tool integrations

## Prioritization rationale

| Priority | Feature | Reason |
|---|---|---|
| P0 | Governance docs + file locks | Foundation for all coordinated work |
| P0 | Agent board | Enables handoffs and blocker tracking |
| P1 | Read-only UI (Phase 1) | Visibility without risk |
| P2 | Controlled task runner (Phase 2) | Non-destructive validation only |
| P3 | Patch proposal mode (Phase 3) | Requires approval UI and audit |
| P4 | Autonomous orchestration (Phase 4) | Highest risk, requires all prior phases stable |

## Success criteria for Phase 0

1. All 10 agents can read their role and council files without ambiguity
2. File locks prevent concurrent edits on the same file
3. Agent board reflects accurate status after each agent turn
4. CHANGELOG_AGENTIC.md captures every change with rationale
5. No application source code is modified without explicit human approval
6. QA and Security agents have signed off on safety gates before Phase 1

## Risks and constraints

| Risk | Mitigation |
|---|---|
| Agents overstep scope | Role prompts explicitly forbid source code edits |
| File lock conflicts | Handoff protocol in AGENT_BOARD.md |
| Governance bloat | Product Agent enforces MVP vs later phases |
| False sense of safety | QA + Security review required before Phase 1 |

## Validation

- [x] All required council files read
- [x] Product plan populated with MVP scope and non-goals
- [ ] AGENT_BOARD.md updated with Product Agent status
- [ ] CHANGELOG_AGENTIC.md updated
- [ ] DECISIONS.md updated if new ADRs proposed

## Handoffs

| To agent | Reason | Status |
|---|---|---|
| Architect Agent | Align on phase boundaries and module scope | pending |
| Security Agent | Confirm safety gates cover Phase 0 bootstrap | pending |
| QA Agent | Confirm validation checklist covers Phase 0 | pending |
| Release Manager | Final consolidation and commit message | pending |

## Final notes

Phase 0 is documentation and process only. No application code changes. This is intentional: the coordination layer must be stable before any read-only UI or task execution features are added. Product Agent will continue to enforce MVP scope until all agents have signed off.
