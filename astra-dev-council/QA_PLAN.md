# QA_PLAN.md — QA Validation Plan for Call Meeting Intelligence

## Validation Strategy

Testing approach for Astra's **Call Meeting Intelligence Engine** — ensuring MVP functionality works correctly without breaking existing features.

## Testing Categories

### Functional Testing

| Test Case | Priority | Description | Expected Result |
|---|-|--|-|
| TC-001 | P0 | Capture audio during Teams/Meet/Discord call | Audio captured correctly |
| TC-002 | P0 | Start meeting session | Session starts, panel opens |
| TC-003 | P0 | Stop meeting session | All data saved to ~/.astra/meetings/ |
| TC-004 | P1 | Pause meeting session | Recording paused, resumed correctly |
| TC-005 | P0 | Live transcript display | Real-time text updates in panel |
| TC-006 | P0 | Live summarizer | Rolling summary updates every 30s |
| TC-007 | P1 | Action item extraction | Correct action items identified in summary |
| TC-008 | P1 | Decision log | Correct decisions recorded in summary |
| TC-009 | P2 | Follow-up draft | Draft email composed with meeting summary |
| TC-010 | P1 | Export meeting | JSON/Markdown/CSV files generated correctly |

### Integration Testing

| Test Case | Priority | Description | Expected Result |
|---|-|--|-|
| TC-011 | P1 | Meeting panel opens/closes | Panel opens/closes without errors |
| TC-012 | P1 | Meeting data saved correctly | All files saved to ~/.astra/meetings/ |
| TC-013 | P1 | Audio capture works on Windows/Mac/Linux | Works on all platforms |
| TC-014 | P2 | Meeting panel opens/closes | Panel opens/closes without errors |

### Security Testing

| Test Case | Priority | Description | Expected Result |
|---|-|--|-|
| TC-015 | P0 | Privacy consent prompt | Prompt appears before first recording |
| TC-016 | P0 | Consent flow | User can grant/deny consent (pause/resume) |
| TC-017 | P0 | Meeting data stays local | All meeting data stays local (~/.astra/meetings/) |
| TC-018 | P1 | No cloud upload | No outbound network calls from meeting engine |

### Regression Testing

| Test Case | Priority | Description | Expected Result |
|---|-||--|
| | P2 | | No regression in existing voice/stt/tts features |
| TC-019 | P1 | Existing voice features still work | Voice/stt/tts work correctly |
| TC-020 | P1 | Existing screen features still work | Screen/capture features work correctly |
| TC-021 | P1 | Existing approval features still work | Approvals work correctly |
| TC-022 | P1 | Existing tool features still work | Tool detection works correctly |

## Validation Checklist

### Build & Compilation

- [ ] `cargo build --release` compiles without errors
- [ ] `cargo test` all existing tests pass
- [ ] No warnings or errors in build output

### Functional Validation

- [ ] Call detection works (Teams, Meet, Discord)
- [ ] Audio capture works
- [ ] Meeting panel opens/closes correctly
- [ ] Transcript updates in real-time
- [ ] Summary updates correctly
- [ ] Action items extracted correctly
- [ ] Decisions logged correctly
- [ ] Stop meeting saves output correctly
- [ ] Export JSON/Markdown/CSV works
- [ ] Follow-up draft composed correctly

### Security Validation

- [ ] Privacy consent prompt appears before first call
- [ ] User can grant/deny consent correctly
- [ ] Pause/resume/recording works
- [ ] All meeting data stays local
- [ ] No unauthorized network calls
- [ ] Data retention policy works
- [ ] Audit logging works

### Platform Validation

- [ ] Audio capture works on Windows
- [ ] Audio capture works on Mac
- [ ] Audio capture works on Linux
- [ ] Meeting panel works on all platforms

## QA Sign-off

- [ ] Architect Agent
- [ ] QA Agent
- [ ] Security Agent
