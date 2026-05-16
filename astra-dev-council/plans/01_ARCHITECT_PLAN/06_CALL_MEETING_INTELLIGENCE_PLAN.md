# ADR-0006 — Call Meeting Intelligence Engine

## Summary

Add an enterprise-grade **Meeting Intelligence Engine** that captures Teams/Meet/Discord calls in real-time, transcribes live, produces summary/action-items/decision-log/follow-up during and after the call, and saves structured notes — built as layered subsystems extending existing voice/screen/orchestration infrastructure.

## Architecture Pattern — Google/Slack-style

Follow Google Meet + Slack Huddles + Notion Calendar pattern:
- **Capture Layer** — system audio capture, not mic-only (like Google Meet)
- **Transcription Layer** — streaming STT with speaker diarization (like Zoom)
- **Intelligence Layer** — live summarization, extraction, classification (like Notion AI)
- **Presentation Layer** — live dashboard + post-call workspace (like Google Meet Summary)

## Design Principles

1. **Privacy-first** — all audio, transcription, and summaries stored locally. No cloud upload.
2. **Layered extraction** — each layer (capture -> stt -> processing) is optional and toggleable
3. **Non-invasive** — does not modify call audio itself; only copies system audio
4. **Opt-in consent** — explicit toggle per call type to record
5. **Extensible plugins** — future provider-agnostic extraction (Zoom, Meet, Teams, Discord, WebRTC)
6. **Structured output** — all meeting data in JSON-first, markdown/CSV/ICalendar export

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   PRESENTER PANEL                    │
│  (MeetingLivePanel.tsx — live UI dashboard)           │
│  ┌──────────┬──────────┬──────────┬──────────┐      │
│  │ LIVE     │ SUMMARY  │ ACTION   │ DECISION  │      │
│  │ TRANSCRIPT│        │ ITEMS    │ LOG        │      │
│  └──────────┴──────────┴──────────┴──────────┘      │
│  ┌──────────┬──────────┬──────────┬──────────┐      │
│  │ FOLLOW-  │ NOTES    │ EXPORT   │ CALL DETECT│      │
│  │ UPS      │          │          │            │      │
│  └──────────┴──────────┴──────────┴──────────┘      │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                   AUDIO CAPTURE LAYER                │
│                                                      │
│  src-tauri/src/meeting/                               │
│  ├─ session_registry.rs  — track active call sessions│
│  ├─ audio_capture.rs     — system audio (PipeWire/   │
│  │                         CoreAudio/WASAPI)          │
│  ├─ call_detector.rs     — detect Meet/Teams/Discord │
│  │                         by process/window title     │
│  └─ audio_router.rs      — route capture to stt       │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                   TRANSCRIPTION LAYER                 │
│                                                      │
│  src-tauri/src/meeting/                               │
│  ├─ transcription_stream.rs  — streaming STT client  │
│  │                           (WIPER/Local Whisper)    │
│  ├─ speaker_diarization.rs   — voice ID per speaker  │
│  └─ transcription_buffer.rs  — deduplicate + segments│
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                   INTELLIGENCE LAYER                  │
│                                                      │
│  src-tauri/src/meeting/                               │
│  ├─ live_summarizer.rs     — rolling summary engine  │
│  ├─ action_item_tracker.rs — extract/track actions   │
│  ├─ decision_log.rs        — extract/track decisions │
│  ├─ follow_up_sender.rs    — send follow-up emails   │
│  │                         (IMAP via external script) │
│  └─ note_organizer.rs      — save structured notes   │
│                              ~/.astra/meetings/        │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                   DATA LAYER                          │
│                                                      │
│  ~/.astra/meetings/                                   │
│  ├─ {session_id}.json       — full meeting data      │
│  ├─ {session_id}_transcript.md — human-readable      │
│  ├─ {session_id}_summary.md  — executive summary     │
│  └─ {session_id}_actions/   — action items export    │
└─────────────────────────────────────────────────────┘
```

## File Locks — New files per agent

| Agent | New Files | Lock Required | Risk |
|---|---|---|---|
| Rust Backend | src-tauri/src/meeting/session_registry.rs | NEW | high |
| Rust Backend | src-tauri/src/meeting/audio_capture.rs | NEW | high |
| Rust Backend | src-tauri/src/meeting/call_detector.rs | NEW | medium |
| Rust Backend | src-tauri/src/meeting/transcription_stream.rs | NEW | high |
| Rust Backend | src-tauri/src/meeting/speaker_diarization.rs | NEW | medium |
| Rust Backend | src-tauri/src/meeting/live_summarizer.rs | NEW | high |
| Rust Backend | src-tauri/src/meeting/action_item_tracker.rs | NEW | low |
| Rust Backend | src-tauri/src/meeting/decision_log.rs | NEW | low |
| Rust Backend | src-tauri/src/lib.rs | MODIFIED | critical |
| Rust Backend | src-tauri/Cargo.toml | MODIFIED | critical |
| Rust Backend | src-tauri/build.rs | MODIFIED | medium |
| Frontend UI | src/components/MeetingLivePanel.tsx | NEW | high |
| Frontend UI | src/hooks/useMeetingEngine.ts | NEW | high |
| Frontend UI | src/hooks/useLiveTranscript.ts | NEW | medium |
| Frontend UI | src/hooks/useMeetingSummary.ts | NEW | medium |
| Frontend UI | src/hooks/useMeetingActions.ts | NEW | low |
| Frontend UI | src/types/meeting.ts | NEW | high |
| Frontend UI | App.tsx | MODIFIED | critical |
| Voice/Audio | src-tauri/src/meeting/audio_capture.rs | NEW | high |
| Voice/Audio | src-tauri/src/meeting/speaker_diarization.rs | NEW | medium |
| Security | src-tauri/src/meeting/privacy_control.rs | NEW | critical |
| Security | DECISIONS.md | MODIFIED | medium |
| AI Orchestration | src-tauri/src/meeting/llm_router.rs | NEW | high |
| AI Orchestration | src-tauri/src/meeting/live_summarizer.rs | NEW | high |

## Implementation Sequence

### Phase A: Core Infrastructure (Rust Backend)
1. `meeting/` directory structure with Cargo.toml features
2. `session_registry.rs` — meeting session lifecycle (start/stop/pause/resume)
3. `call_detector.rs` — detect active call platform by process/window name
4. `privacy_control.rs` — explicit consent flow, data retention, encryption

### Phase B: Audio Capture (Rust Backend + Voice/Audio)
5. `audio_capture.rs` — cross-platform system audio capture
6. `transcription_stream.rs` — streaming STT to model (local Whisper/Wiper)

### Phase C: Intelligence Processing (Rust Backend + AI Orchestration)
7. `live_summarizer.rs` — rolling 30-second summary updates
8. `action_item_tracker.rs` — extract action items (NLP)
9. `decision_log.rs` — extract and log decisions
10. `speaker_diarization.rs` — voice ID per speaker
11. `follow_up_sender.rs` — compose follow-up drafts
12. `note_organizer.rs` — save structured notes to disk

### Phase D: Frontend Panel (Frontend UI Agent)
13. `meeting.ts` — TypeScript types, hooks, event contracts
14. `MeetingLivePanel.tsx` — live dashboard with tabs
15. `useMeetingEngine.ts` — Rust invoke bindings
16. `useLiveTranscript.ts` — real-time transcript streaming
17. `useMeetingSummary.ts` — summary panel logic
18. Integration into App.tsx — meeting button in toolbar

## Risk Assessment

| Risk | Mitigation |
|---|---|
| System audio capture is platform-specific | Abstract behind 3 backends: PipeWire(Linux), CoreAudio(macOS), WASAPI(Windows) |
| STT accuracy in noisy environments | Configurable models; allow manual correction |
| Privacy/consent legal concerns | Opt-in only; explicit per-call; local-only storage; no cloud |
| Tightly coupled Rust lib.rs | Isolate meeting code in `meeting/` submodule; minimal changes to lib.rs |
| Audio processing overhead | Async processing; configurable interval; resource budget monitoring |

## API Contracts (Tauri commands to expose)

| Command | Return | Description |
|---|---|---|
| `meeting_detect_active_call` | `Option<CallInfo>` | Detect if user is in a call |
| `meeting_start` | `MeetingSession` | Start a new meeting session |
| `meeting_stop` | `()` | Stop and save meeting data |
| `meeting_pause` | `()` | Pause transcription (redaction mode) |
| `meeting_resume` | `()` | Resume transcription |
| `meeting_get_summary` | `MeetingSummary` | Get current rolling summary |
| `meeting_get_transcript` | `Vec<TranscriptEntry>` | Get full transcript transcript |
| `meeting_get_action_items` | `Vec<ActionItem>` | Get extracted action items |
| `meeting_get_decisions` | `Vec<DecisionLogEntry>` | Get decision log |
| `meeting_export_json` | `String` | Export entire meeting to JSON |
| `meeting_export_markdown` | `String` | Export transcript to markdown |
| `meeting_get_status` | `MeetingStatus` | Get current meeting state |

## Validation Steps

1. `cargo build --release` — compiles without errors
2. `cargo test` — all existing tests pass
3. Detect active call in Teams/Meet/Discord — verify detection
4. Start meeting — verify audio capture starts correctly
5. Transcribe live — verify real-time transcription appears in panel
6. Test summary generation — verify live summary updates
7. Test action item detection — verify extraction works
8. Stop meeting — verify saved output files
9. Verify no regression in existing voice/session/screen features
10. Test privacy controls — verify consent prompt appears

## Dependencies to add to Cargo.toml

```toml
[dependencies.meeting_engine]
# Optional audio capture backends
audio_capturer = { version = "0.4", features = ["pipewire", "wasapi", "coreaudio"], optional = true }
# Whisper for local STT (reuse existing pattern)
whisper-rs = { version = "0.12", optional = true }
# For process detection
sysinfo = "0.32"
# For structured notes
chrono = "0.4"
uuid = { version = "1", features = ["v4"] }
```

## Handoffs

| From | To | Status |
|---|---|---|
| Architect --> Rust Backend | All meeting modules to implement | READY |
| Architect --> Frontend UI | Meeting live panel + hooks | READY |
| Architect --> Voice/Audio | System audio capture + diarization | READY |
| Architect --> AI Orchestration | Live summarizer + action item NLP | READY |
| Architect --> Security | Privacy control module review | READY |
| Rust Backend --> Frontend UI | Tauri command contracts to implement hooks | PENDING |
| Meeting Engine --> QA | Validation checklist for meeting feature | PENDING |

## Final Notes

- The meeting engine should be a **Tauri feature flag** — disabled by default in release build
- All meeting data must be stored in `~/.astra/meetings/` (not in existing audio/stt/tts dirs)
- Respect existing permission model — meeting recording is "high_risk" requiring explicit approval
- The live panel uses the same design patterns as DesktopAgentPanel (cards, tabs, grid layout)
- Export options: JSON (structured), Markdown (readable), CSV (action items), iCalendar (follow-ups)
