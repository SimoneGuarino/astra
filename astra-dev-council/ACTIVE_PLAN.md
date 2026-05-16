# ACTIVE_PLAN.md

## Current plan: Astra Development Council coordination layer

### Phase 0 — Governance bootstrap

Create the shared coordination layer used by all agents.

Deliverables:

- Council docs.
- Agent role prompts.
- File lock protocol.
- Agent board.
- QA checklist.
- Security review.
- Launch automation script.

No application source code should be changed in this phase unless the human explicitly asks.

### Phase 1 — Read-only Dev Council visibility inside Astra

After Phase 0 is stable, add a read-only UI panel that displays council status from markdown/json files.

Suggested boundaries:

- Frontend: `src/components/dev-council/*`, `src/types/devCouncil.ts`, optional hook `src/hooks/useDevCouncil.ts`.
- Rust: optional read-only Tauri command for council status, implemented as a new module rather than modifying unrelated logic.

Do not allow autonomous code execution in this phase.

### Phase 2 — Controlled task runner

Add a controlled task-runner that can execute approved validation commands only.

Allowed examples:

- `npm run build`
- `cargo check`
- `cargo test`
- project-specific non-destructive validation commands

Constraints:

- Must go through policy and audit.
- Must be visible in the UI.
- Must not execute arbitrary shell commands without explicit approval.

### Phase 3 — Patch proposal mode

Agents may generate patch proposals, but human approval is required before applying them.

Design principle:

- Proposal first.
- Risk assessment second.
- QA third.
- Human approval last.

### Phase 4 — Advanced multi-agent orchestration

Only after prior phases are stable, consider deeper integration where Astra can orchestrate agent roles internally.

Required before Phase 4:

- Explicit permission model.
- Audit log.
- Rollback plan.
- UI approval gate.
- Regression suite.
- Safe model routing.

## Current architectural boundaries

### Rust/Tauri core

Key modules observed:

- `lib.rs`: application runtime and command registration surface.
- `desktop_agent.rs`: desktop action runtime, planning, approval, audit, screen workflow execution and dispatch.
- `desktop_agent_types.rs`: central shared Rust types for permissions, risk, policy, screen frames, goals, planner records, capability manifest, route diagnostics.
- `conversation_router.rs`: high-impact routing surface; changes here require QA and AI Orchestration review.
- `assistant_response.rs`: speech/display separation; changes here require Voice/AI Orchestration review.
- `screen_vision.rs`, `semantic_frame.rs`, `structured_vision.rs`, `screen_workflow.rs`, `ui_target_grounding.rs`, `workflow_continuation.rs`: high-risk perception/action pipeline; no hardcoded coordinates.
- `voice_session.rs`, `tts_client.rs`, `stt_client.rs`, `vad.rs`, `speech_events.rs`: voice lifecycle; avoid regressions in interruption and request handling.

### React/TypeScript UI

Key files observed:

- `App.tsx`, `App.css`.
- `DesktopAgentPanel.tsx`.
- `AssistantChat.tsx`, `AssistantInputBar.tsx`, `AssistantHeader.tsx`, `AstraOrb.tsx`.
- `useAssistantSession.ts`, `useAssistantEvents.ts`, `useDesktopAgent.ts`, `useVoiceSession.ts`, `useAssistantAudio.ts`.
- `src/types/desktopAgent.ts` mirrors important Rust contracts.

## Implementation policy

Any future source-code implementation must satisfy:

- Small changes.
- Explicit ownership.
- File locks.
- Documented contract updates.
- Build validation.
- QA and Security review.
- Commit message prepared by Release Manager.
