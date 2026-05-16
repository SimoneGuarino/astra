# Astra Development Council

This folder defines a coordinated 10-agent engineering workflow for Astra.

The goal is to let multiple AI engineering agents collaborate on the same `main` branch while preserving stability through file locks, shared planning, audit notes, QA gates, and release review.

## Operating model

Astra Development Council is not a free-for-all. It is a controlled software-engineering workflow:

1. **Architect** defines direction and module boundaries.
2. **Product** validates priority and user value.
3. **Security** defines safety constraints and gates.
4. **Implementation agents** work only inside their scopes.
5. **QA** verifies regressions, builds, contracts, and test coverage.
6. **Release Manager** consolidates the final state and prepares the commit/changelog.

All agents may operate on `main`, but they must coordinate through:

- `TASK.md`
- `ACTIVE_PLAN.md`
- `AGENT_BOARD.md`
- `FILE_LOCKS.md`
- `DECISIONS.md`
- `CHANGELOG_AGENTIC.md`
- `SECURITY_REVIEW.md`
- `QA_REPORT.md`

## Analysis summary from v0.4.29 zips

The analyzed Astra codebase already has significant architecture:

- Tauri v2/Rust orchestration in `src-tauri/src`.
- React/TypeScript UI in `src`.
- Desktop action requests, permissions, pending approvals, audit events and policy snapshots.
- Screen capture, structured vision, semantic frames, UI target grounding, accessibility enrichment, screen workflow execution and continuation handling.
- Voice session management, STT/TTS clients, VAD, metrics and audio file lifecycle.
- Model routing, assistant response rendering, conversation routing and capability context.

Therefore, every agent must extend the existing system instead of replacing it.

## Recommended launch order

1. Architect Agent
2. Product Agent
3. Security Agent
4. Relevant implementation agents
5. QA Agent
6. Release Manager Agent

Do not start all 10 agents writing source code at the same time. They may be running simultaneously, but only the right agents should hold locks and implement code for the current phase.
