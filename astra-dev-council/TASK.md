# TASK.md — Feature: Astra Meeting Intelligence Engine

## Executive Summary

Add a **Call Meeting Intelligence Engine** that captures Teams/Meet/Discord calls in real-time, transcribes live, extracts meetings, produces summary/action-items/decision-log/follow-up during and after the call, and saves structured notes — built as layered subsystems extending existing voice/screen/orchestration infrastructure.

## User Story

> "As a knowledge worker, I want Astra to automatically detect and join my video calls (Teams/Meet/Discord), transcribe them in real-time, and produce a structured summary with action items and decisions — all privately, without interrupting my workflow. After the call, I get follow-up emails drafted automatically."

## MVP Scope

### Must Have (MVP)
1. **Call detection** — detect active call (Teams, Meet, Discord) by process/window name
2. **System audio capture** — capture system audio (not mic) for transcription
3. **Live transcription** — real-time text display of what's being spoken
4. **Live summarizer** — rolling 30-second summary updates during call
5. **Action item extraction** — automatically identified action items with assignees + deadlines
6. **Decision log** — automatically extracted decisions made during the call
7. **Post-call summary** — structured JSON + Markdown saved to `~/.astra/meetings/`
8. **Meeting panel** — live dashboard in Astra showing transcript + summary + actions
9. **Privacy controls** — opt-in consent per call, local-only storage, no cloud upload

### Nice to Have (v2)
- Speaker diarization (voice ID per speaker)
- iCalendar export for follow-ups
- Multilingual summaries
- Zoom/Slack Huddles support
- "Pause recording" button during sensitive moments

### Non-Goals (Explicitly Out of Scope)

- Cloud-based processing (all local, no Whisper/ Whisper model download
- Modification of call audio itself (only copies system audio)
- Real-time translation during call
- Calendar integration (scheduling, reminders)

## Architecture Overview

### System-Level Design (Google Meet + Zoom Meeting Notes pattern)

```
┌──────────────────────────────────────────────────────────────────────┐
│                          MEETING PANEL                              │
│  (MeetingLivePanel.tsx)                                              │
│                                                                      │
│  ┌──────────┬──────────┬──────────┬──────────┐                      │
│  │ LIVE     │ SUMMARY  │ ACTIONS  │ NOTES    │                      │
│  │ TRANSCRIPT│          │          │          │                      │
│  └──────────┴──────────┴──────────┴──────────┘                      │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ CALL STATUS: ACTIVE | 23:45 elapsed | 4 participants          │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │     MEETING ENGINE            │
│                                                      │
│  ┌───────────────┴──────────────────────┐          │
│  │ Audio Capture Layer                  │          │
│  │ (PipeWire/CoreAudio/WASAPI)          │          │
│  └──────────────────────────────────────┘          │
│  ┌──────────────────────────────────────┐          │
│  │ Transcription Layer                  │          │
│  │ (Streaming STT, Local Whisper)       │          │
│  └──────────────────────────────────────┘          │
│  ┌──────────────────────────────────────┐          │
│  │ Intelligence Layer                   │          │
│  │ (summarizer, action items, decisions)│          │
│  └──────────────────────────────────────┘          │
│  ┌──────────────────────────────────────┐          │
│  │ Note Organizer Layer                 │          │
│  │ (.astra/meetings/ {session}.json)    │          │
│  └──────────────────────────────────────┘          │
└────────────────────────────────────────────────────────────

### Module Layout

```
src-tauri/src/meeting/
├─ mod.rs
├─ lib.rs (main modules)
├─ session_registry.rs   — meeting session lifecycle
├─ call_detector.rs      — detect active call platform
├─ audio_capture.rs      — system audio (PipeWire/CoreAudio/WASAPI)
├─ transcription_stream.rs  — streaming STT (local, Whisper/Wiper)
├─ speaker_diarization.rs  — voice ID per speaker
├─ live_summarizer.rs      — rolling 30-second summary
├─ action_item_tracker.rs  — extract/track action items
├─ decision_log.rs         — extract/track decisions
├─ follow_up_sender.rs     — compose follow-up drafts
├─ note_organizer.rs       — save structured notes to disk
└─ privacy_control.rs      — consent, retention, encryption

src/components/
├─ MeetingLivePanel.tsx       — live dashboard
├─ MeetingSummary.tsx         — post-call summary card
└─ MeetingNotesPanel.tsx      — meeting notes

src/hooks/
├─ useMeetingEngine.ts        — Rust Tauri invoke calls
├─ useLiveTranscript.ts       — real-time transcript streaming
└─ useMeetingSummary.ts       — summarizer logic
```

## Implementation Phases

### Phase A: Core Infrastructure (Rust Backend Agent)
- Create `meeting/` directory structure
- Implement `session_registry.rs` — meeting session lifecycle
- Implement `call_detector.rs` — detect active call platforms
- Implement `privacy_control.rs` — explicit consent flow

### Phase B: Audio Capture + Transcription (Rust Backend + Voice/Audio Agent)
- Implement `audio_capture.rs` — cross-platform system audio capture
- Implement `transcription_stream.rs` — streaming STT client
- Reuse existing SttClient infrastructure

### Phase C: Intelligence Processing (Rust Backend + AI Orchestration Agent)
- Implement `live_summarizer.rs` — rolling 30-second summary
- Implement `action_item_tracker.rs` — NLP action item extraction
- Implement `decision_log.rs` — decision extraction
- Implement `speaker_diarization.rs` — voice ID per speaker

### Phase D: Frontend Integration (Frontend UI Agent)
- Create `MeetingLivePanel.tsx` — live dashboard
- Create `useMeetingEngine.ts` — Rust invoke bindings
- Create `useLiveTranscript.ts` — real-time transcript management
- Create `useMeetingSummary.ts` — summarizer hooks
- Add `MeetingNotesPanel.tsx` — post-call summary view

### Phase E: Security Governance (Security Agent)
- Design privacy consent flow
- Write security review of meeting data flow
- Define data retention policy
- Create audit logging for meeting access

### Phase F: QA Validation (QA Agent)
- Define validation checklist
- Write test commands
- Verify no regression in existing features

### Phase G: Release (Release Manager Agent)
- Final commit message
- Version bump
- Changelog entry
- Rollback notes

## Success Criteria

- MVP feature works end-to-end (detect -> capture -> transcribe -> summarize -> export)
- No regression in existing voice, screen, or approval features
- Privacy: explicit consent per call, local-only storage, no cloud
- Panel UI: live transcript, summary, actions, decisions, notes — all in Astra UI
- Export: JSON structured, Markdown readable, CSV for action items
- Follow-up: automatic draft email with meeting summary sent

## Rollback Notes

- If meeting feature causes issues:
  1. Remove `meeting/` directory from src-tauri/
  2. Remove Meeting panels from src/components/
  3. Restore original App.tsx
  4. Revert Cargo.toml changes
  5. Revert lib.rs changes

## QA Validation Checklist

- [ ] `cargo build --release` — compiles without errors
- [ ] `cargo test` — all existing tests pass
- [ ] Active call detection works (Teams, Meet, Discord)
- [ ] Start meeting — audio capture starts
- [ ] Transcription transcribes during calls
- [ ] Meeting panel updates in real-time
- [ ] Summary updates correctly
- [ ] Action items extracted correctly
- [ ] Decisions logged correctly
- [ ] Stop meeting — saved files in ~astra/meetings/
- [ ] No regression in existing voice/screen/approval features
- [ ] Privacy consent prompt appears for first call
- [ ] All meeting data stays local (~.astra/meetings/)

## Dependencies

- **Cargo.toml additions**: audio_capturer, whisper-rs, sysinfo
- **Existing infrastructure reused**: SttClient, TTS, conversation_router
- **New infrastructure**: meeting/ module, Meeting live panels

## Handoffs

| From | To | Status |
|---
| Architect --> Rust Backend | Meeting modules + Tauri commands | READY |
| Architect --> Frontend UI | Meeting live panel + hooks | READY |
| Architect --> Voice/Audio | System audio + diarization | READY |
| Architect --> AI Orchestration | Live summarizer + action items | READY |
| Architect --> Security | Privacy control module | READY |
| Rust Backend --> Frontend UI | Tauri command contracts | PENDING |
| Meeting Engine --> QA | Validation checklist | PENDING |
## Final Notes

- The meeting engine should be a **Tauri feature flag** — disabled by default in release build
- All meeting data stored in `~/.astra/meetings/` (not in existing audio/stt/tts dirs)
- Respect existing permission model — meeting recording is "high_risk" requiring explicit approval
