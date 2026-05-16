# QA_REPORT.md

## QA objective

Prevent regressions while multiple agents work on the same `main` branch.

## Baseline validation commands

Run from the project root as applicable:

```bash
npm run build
```

```bash
cd src-tauri
cargo check
cargo test
```

For interactive runtime validation:

```bash
npm run tauri dev
```

On Windows PowerShell equivalents are acceptable.

## Regression checklist

### Frontend

- [ ] App starts without TypeScript errors.
- [ ] `DesktopAgentPanel.tsx` still matches `src/types/desktopAgent.ts`.
- [ ] Assistant chat/input/header/orb remain functional.
- [ ] Hooks do not leak event listeners.
- [ ] Voice buttons and desktop-agent controls do not regress.
- [ ] Dev Council panel (if Phase 1) is read-only and does not modify council state.
- [ ] No hardcoded coordinates in any UI action.

### Rust runtime

- [ ] `lib.rs` still composes `AssistantRuntime` correctly.
- [ ] Tauri commands still compile.
- [ ] Desktop action request/response contracts are stable.
- [ ] Pending approvals still persist and resolve correctly.
- [ ] Audit events remain emitted for risky actions.
- [ ] No public Rust types changed without updating TypeScript mirror.

### Screen workflow

- [ ] Screen capture path remains valid.
- [ ] Structured vision output is not replaced with prose-only output.
- [ ] UI target grounding does not use hardcoded coordinates.
- [ ] Accessibility enrichment remains optional and non-blocking.
- [ ] Workflow continuation does not trust stale or wrong provider hints.
- [ ] No regression in semantic frame types (list/page-state extraction).

### Voice/audio

- [ ] TTS cancellation still works.
- [ ] Request replacement still cleans old audio files.
- [ ] STT/TTS clients remain reachable.
- [ ] VAD/follow-up flow does not trigger infinite loops.
- [ ] Internal JSON/tool output is not spoken via TTS.
- [ ] Voice session interruption/cancel semantics preserved.

### AI orchestration

- [ ] Model routing remains explicit.
- [ ] Conversation routes do not degrade into user-visible tool traces.
- [ ] Capability context remains accurate.
- [ ] Planner failures produce safe fallback messages.
- [ ] No unauthorized model routing changes (requires AI Orchestrator review).

### Security gates (aligned with SECURITY_REVIEW.md)

- [ ] No bypass of `DesktopActionRequest`, approval, policy, or audit flows.
- [ ] No hidden background execution added.
- [ ] No weakening of confirmation gates for high-risk actions.
- [ ] No secrets extraction or credential exposure in council files.
- [ ] Model/orchestration internal JSON not spoken by TTS.

### Cross-cutting

- [ ] Rust/TypeScript contract alignment verified for `desktopAgent.ts` mirror.
- [ ] No untracked `node_modules` or `target/` artifacts committed.
- [ ] File locks respected during all multi-agent changes.
- [ ] CHANGELOG_AGENTIC.md updated for each meaningful change.
- [ ] AGENT_BOARD.md reflects current status.

## QA gate status

Current status: `bootstrap-docs-complete`

**Phase 0 validation complete:**
- Regression checklist defined and aligned with `SECURITY_REVIEW.md` high-risk modules
- Validation commands documented for future use
- No application code changes required for documentation-only phase

**Ready for Phase 1:**
- QA Agent has completed Phase 0 governance tasks
- Regression checklist covers all high-risk files and security gates
- Validation commands ready for use when source changes begin

**Next steps:**
- Architect Agent to approve Phase 1 scope
- Security Agent to review Phase 1 security gates
- Release Manager to coordinate bootstrap consolidation
