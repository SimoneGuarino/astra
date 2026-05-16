# 08 Security Plan

## Purpose

Security gates, policy risks, approvals, audit expectations.

## Current status

- **Status:** completed (Phase 0 bootstrap)
- **Owner:** `08_security_agent.md`
- **Last updated by:** Security Agent
- **Date:** 2026-05-01

## Scope

Security and Policy Agent responsibilities for Astra Development Council:

1. **Block unsafe or unstable changes** — Review filesystem, terminal, browser, and desktop-control changes
2. **Ensure confirmation gates** — Risky actions require explicit approval and audit logs remain intact
3. **Block dangerous patterns** — Hidden background execution, weakened risk classification, destructive commands, writing outside project root, hardcoded UI click coordinates
4. **Define security gates per phase** — From bootstrap (Phase 0) through advanced orchestration (Phase 4)

## Inputs read

- [x] `astra-dev-council/TASK.md`
- [x] `astra-dev-council/ACTIVE_PLAN.md`
- [x] `astra-dev-council/AGENT_BOARD.md`
- [x] `astra-dev-council/FILE_LOCKS.md`
- [x] `astra-dev-council/DECISIONS.md`
- [x] `astra-dev-council/CHANGELOG_AGENTIC.md`
- [x] `astra-dev-council/SECURITY_REVIEW.md`
- [x] `astra-dev-council/QA_REPORT.md`

## Security Gates by Phase

### Phase 0 — Governance Bootstrap (CURRENT)

**Status:** ✅ Complete

**Gates:**

| Gate | Status | Notes |
|------|--------|-------|
| Council docs created | ✅ | All 8 core files present |
| Agent role prompts defined | ✅ | agents/*.md |
| File lock protocol | ✅ | FILE_LOCKS.md defines lock/unlock/handoff |
| Launch automation | ✅ | scripts/ present |
| No application code changes | ✅ | Bootstrap is docs-only |

**Risk Level:** LOW (documentation only)

### Phase 1 — Read-only Dev Council UI

**Status:** Pending Security review

**Security requirements:**

1. Tauri commands must validate file paths (no `..` traversal)
2. Read-only commands restricted to `astra-dev-council/` directory
3. UI components must not inject innerHTML without sanitization
4. Return structured data, not raw file contents

### Phase 2 — Controlled Task Runner

**Status:** Pending (HIGH-RISK PHASE)

**Mandatory gates:**

- Whitelist of allowed commands (hardcoded, no string interpolation)
- No arbitrary shell execution (enum-dispatched commands)
- Output sanitization (no raw stdout to TTS/UI)
- Timeout enforcement (max 30s)
- Audit logging for every command execution

**Allowed commands (Phase 2 only):**
```
npm run build, npm run check, cargo check, cargo test, cargo clippy --no-deps
```

### Phase 3 — Patch Proposal Mode

**Status:** Pending (HIGHEST-RISK PHASE)

**Security requirements:**

1. Proposal isolation (temp files, never applied directly)
2. Diff review UI (human must see unified diff)
3. File lock check before proposing
4. Rollback notes auto-generated
5. No cross-module changes

### Phase 4 — Advanced Orchestration

**Status:** Not started

**Prerequisites:** Permission model, audit log persistence, rollback plan tested, UI approval gate, regression suite passing, safe model routing verified

## High-Risk Module Watchlist

| Module | Risk | Reason |
|--------|------|--------|
| `terminal_runner.rs` | CRITICAL | Arbitrary command execution surface |
| `filesystem_service.rs` | CRITICAL | Full project read/write access |
| `desktop_agent.rs` | HIGH | Desktop control, permissions, audit |
| `desktop_agent_types.rs` | HIGH | Policy/risk type definitions |
| `ui_control.rs` | HIGH | UI automation surface |
| `screen_workflow.rs` | HIGH | Screen capture and interaction |
| `workflow_continuation.rs` | HIGH | Trust boundaries for provider hints |
| `conversation_router.rs` | HIGH | Model routing, user-facing output |
| `assistant_response.rs` | MEDIUM | TTS output — internal JSON leakage risk |
| `voice_session.rs` | MEDIUM | Interruption/cancel — infinite loop risk |
| `src/types/desktopAgent.ts` | HIGH | Public contract — frontend/backend sync |
| `src/components/DesktopAgentPanel.tsx` | MEDIUM | Approval UI surface |

## Audit Log Expectations

Any risky action must emit an audit event with:

```rust
struct DesktopAuditEvent {
    timestamp: String,      // ISO 8601
    action_type: String,    // e.g., "PATCH_PROPOSAL", "TERMINAL_EXEC"
    risk_level: String,     // "low" | "medium" | "high" | "critical"
    target: String,         // File path or command
    actor: String,          // Agent role or "human"
    approval_id: Option<String>,
    outcome: String,        // "pending" | "approved" | "rejected" | "executed"
}
```

## Terminal Safety Rules

**Blocked commands (must fail with error):**
```
rm -rf /, rm -rf ~, del /F /Q *, rmdir /S /Q, format, fdisk, mkfs, dd if=/dev/zero, chmod -R 777, chown -R, curl | bash, wget | bash
```

**Restricted commands (require explicit approval):**
```
npm install/uninstall, cargo install, pip install, git reset --hard, git clean -fd
```

## Filesystem Safety Rules

1. No writes outside project root
2. No symlink following for writes
3. No hidden file creation (except council files)
4. No credential files (`.env`, `*.key`, `*.pem`, `credentials*`)

## Model/Orchestration Safety

| Risk | Mitigation |
|------|------------|
| Internal JSON spoken by TTS | Filter `assistant_response.rs` — block raw `serde_json::Value` output |
| Prompt injection via file content | Sanitize markdown before feeding to LLM |
| Token exhaustion | Set max_tokens on all requests |
| Context window overflow | Implement truncation strategy |
| Model routing bypass | All model calls go through `conversation_router.rs` |

## Files or areas involved

| File/Area | Intended action | Lock required | Risk |
|-----------|-----------------|---------------|------|
| `astra-dev-council/plans/08_SECURITY_PLAN.md` | Create security gates doc | Yes | LOW |
| `astra-dev-council/SECURITY_REVIEW.md` | Reference for phase gates | Yes | LOW |
| `astra-dev-council/AGENT_BOARD.md` | Status update | No | LOW |
| `astra-dev-council/CHANGELOG_AGENTIC.md` | Record bootstrap changes | No | LOW |
| `astra-dev-council/FILE_LOCKS.md` | Lock management | Yes | LOW |

## Risks and constraints

- **Current risk:** NONE (Phase 0 is documentation-only)
- **Constraint:** No application code changes until Phase 1+ with proper review
- **Constraint:** All agents must work on `main` branch only (ADR-0001)

## Validation

- [x] Council files created and consistent
- [x] Security gates defined per phase
- [x] High-risk modules identified
- [x] Audit log expectations documented
- [x] Terminal/filesystem safety rules defined

## Handoffs

| To agent | Reason | Status |
|----------|--------|--------|
| Architect Agent | Review Phase 1 UI security requirements | Ready — Phase 0 security gates complete, Phase 1 UI security review pending
| QA Agent | Sync validation checklist with security gates | Ready — Security gates defined, QA checklist needs alignment with high-risk modules
| Release Manager | Final bootstrap consolidation | Ready — All Phase 0 council files complete, awaiting final sign-off

## Current Status

- **Phase 0:** ✅ Complete — Security gates defined, no application code changes
- **Phase 1:** ✅ Ready for review — Security requirements for read-only UI defined
- **Phase 2:** ⚠️ Pending — High-risk task runner gates defined, awaiting QA validation
- **Phase 3:** ⚠️ Pending — Patch proposal mode security requirements defined
- **Phase 4:** ❌ Not started — Prerequisites not met

## Validation Status

- [x] All council files created and consistent
- [x] Security gates defined per phase
- [x] High-risk modules identified and documented
- [x] Audit log expectations documented
- [x] Terminal/filesystem safety rules defined
- [x] Model/orchestration safety mitigations implemented in documentation

## Risks and Constraints

- **Current risk:** NONE (Phase 0 is documentation-only)
- **Constraint:** No application code changes until Phase 1+ with proper review
- **Constraint:** All agents must work on `main` branch only (ADR-0001)
- **Constraint:** Phase 1 must be read-only only — no task runner, no patch application, no autonomous execution

## Next Steps

1. **Architect Agent:** Review Phase 1 UI security requirements and update `plans/01_ARCHITECT_PLAN.md` with UI security constraints
2. **QA Agent:** Align validation checklist with security gates, especially for high-risk modules
3. **Release Manager:** Finalize bootstrap consolidation and sign-off

## Final notes

**Phase 0 bootstrap is SECURE.** No application code has been modified. Council files establish clear role boundaries, file lock protocol, agent board for coordination, QA validation checklist, and changelog for audit trail.

**Next action:** Await Architect plan finalization, then review Phase 1 UI proposal for security gates.

**Security Agent Status:** ✅ GREEN — No risks identified in bootstrap phase.
