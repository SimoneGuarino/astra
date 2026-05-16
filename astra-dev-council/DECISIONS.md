# DECISIONS.md

## ADR-0001 — Same-branch multi-agent workflow

Decision: all agents may work on the same `main` branch, but coordination is mandatory through council files, file locks, small changes, QA gates, and release review.

Reason:

- The user prefers direct work on `main` and accepts rollback as recovery.
- Uncoordinated concurrent modifications are too risky for Astra because the current codebase has tightly coupled runtime, screen workflow, voice/session, and frontend contracts.

Consequences:

- Agents must not use branches/worktrees by default.
- File locks are mandatory.
- Commits must be small and reversible.
- Release Manager prepares rollback notes.

## ADR-0002 — No autonomous code execution in bootstrap phase

Decision: the initial council package only creates coordination, prompts, launch scripts and governance docs.

Reason:

- Astra already has desktop control, terminal, browser, screen and approval surfaces.
- A self-modifying agent system must not be added before permissioning, audit, QA and rollback are defined.

Consequences:

- First milestone is documentation and process.
- Future in-app Dev Council features should start read-only.

## ADR-0003 — Preserve existing Astra architecture

Decision: agents must extend current modules and contracts rather than replacing them.

Reason:

- v0.4.29 contains substantial Tauri/Rust orchestration, model routing, screen perception, workflow continuation, desktop actions, approvals, audit, voice lifecycle and frontend hooks.

Consequences:

- High-risk modules require review.
- Public contracts mirrored between Rust and TypeScript must be handled carefully.

## ADR-0004 — Phase 0 is documentation and process only

Decision: Phase 0 (governance bootstrap) must not modify application source code.

Reason:

- The coordination layer must be stable before any read-only UI or task execution features are added.
- Safety gates, file locks, and handoff protocols must be proven in documentation before being encoded in UI or runtime.

Consequences:

- Phase 0 deliverables are limited to council markdown files, agent prompts, and launch scripts.
- Architect, Security, and QA agents must sign off before Phase 1 (read-only UI) begins.
- Product Agent enforces MVP scope to prevent feature creep into Phase 0.

## ADR-0005 — Phased gates and high-risk file matrix

Decision: each phase requires explicit completion criteria and sign-off before the next phase begins. High-risk files require mandatory multi-agent review.

Reason:

- Astra's v0.4.29 architecture has tightly coupled runtime, screen workflow, voice/session, and frontend contracts.
- Unreviewed changes to critical modules (conversation_router, desktop_agent, screen_workflow, voice_session) could introduce regressions that are difficult to diagnose.
- Phased gates prevent cascading failures from premature feature activation.

Consequences:

- Phase transitions require Architect, Security, and QA sign-off.
- High-risk file matrix defines required reviewers per file (see `plans/01_ARCHITECT_PLAN.md`).
- Changes to Critical-risk files require Architect + Security + QA review before apply.
- Changes to High-risk files require Architect + domain specialist review.
- Release Manager verifies all sign-offs are present before commit.
