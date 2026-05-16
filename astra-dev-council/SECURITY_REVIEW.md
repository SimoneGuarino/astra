# SECURITY_REVIEW.md

## Security posture

Astra already includes desktop automation concepts such as permissions, risk levels, pending approvals, audit events, browser/desktop control, terminal execution, filesystem access, screen observation, UI control and workflow execution.

A multi-agent development system must not weaken those controls.

## Mandatory safety gates

### Git and repository safety

- No destructive git command without explicit human approval.
- No `git reset --hard`, `git clean`, force push, history rewrite or bulk deletion by default.
- Prefer `git diff`, `git status`, small commits and documented rollback notes.

### Filesystem safety

- No deletion of project files unless explicitly authorized.
- No editing outside the project root unless explicitly authorized.
- No secrets extraction or credential harvesting.
- No writing to global system locations.

### Terminal safety

- Validation commands are allowed when non-destructive.
- Package installation requires explicit approval.
- System configuration changes require explicit approval.
- Long-running processes must be documented.

### Application safety

- Do not bypass `DesktopActionRequest`, approval, policy or audit flows.
- Do not add hidden background execution.
- Do not weaken confirmation gates for high-risk actions.
- Do not hardcode UI click coordinates.
- Do not remove uncertainty handling from screen workflow.

### Model/orchestration safety

- Internal tool/planner JSON must not be spoken by TTS.
- Model routing changes must preserve user-facing vs internal-output separation.
- Agents must not leak system prompts or hidden chain-of-thought.
- Agents must document model assumptions and fallback behavior.

## High-risk change categories

Require Security + QA review:

- `terminal_runner.rs`
- `filesystem_service.rs`
- `desktop_agent.rs` dispatch/policy/pending approvals
- `ui_control.rs`
- `screen_workflow.rs`
- `workflow_continuation.rs`
- `conversation_router.rs`
- `assistant_response.rs`
- voice session interruption/cancel logic
- frontend approval UI

## Current security decision

During bootstrap, agents may create and edit council markdown files and launch scripts. Application code changes should wait until Architect, Security and QA initial reviews are completed.
