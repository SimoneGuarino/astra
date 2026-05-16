# Rust Backend Implementation Plan

## Architecture Pattern
Follow Rust/async pattern in lib.rs. Keep meeting engine isolated in `meeting/` module directory. All new code must compile with Rust 1.75+.

## Module Structure

Create directory: `src-tauri/src/meeting/`
- mod.rs
- session_registry.rs
- call_detector.rs  
- audio_capture.rs
- transcription_stream.rs
- speaker_diarization.rs
- live_summarizer.rs
- action_item_tracker.rs
- decision_log.rs
- note_organizer.rs
- privacy_control.rs

## Implementation Sequence

### Phase A: Module skeleton (5 files)
- mod.rs, session_registry.rs, call_detector.rs
- audio_capture.rs, transcription_stream.rs

### Phase B: Session lifecycle (5 files)
- session_registry.rs, privacy_control.rs, note_organizer.rs
- transcription_stream.rs, speaker_diarization.rs

### Phase C: Intelligence (5 files)
- live_summarizer.rs, action_item_tracker.rs
- decision_log.rs, follow_up_sender.rs
- privacy_control.rs

### Phase D: Tauri commands (lib.rs integration)
- Add meeting module to lib.rs
- Implement all meeting Tauri commands
- Update Cargo.toml with new dependencies

## Data Contracts

### Types to define (in types.rs or inline)

```rust
// Meeting types
pub struct MeetingSession {
    pub session_id: String,
    pub platform: String,
    pub status: MeetingStatus,
    pub started_at: DateTime<Utc>,
    pub participants: Vec<ParticipantInfo>,
}

pub enum MeetingStatus {
    Idle,
    Capturing,
    Transcribing,
    Summarizing,
    Paused,
    Completed,
}

pub struct TranscriptEntry {
    pub timestamp: DateTime<Utc>,
    pub speaker: String,
    pub text: String,
}

pub struct SummaryEntry {
    pub timestamp: DateTime<Utc>,
    pub summary: String,
}

pub struct ActionItem {
    pub timestamp: DateTime<Utc>,
    pub description: String,
    pub assignee: Option<String>,
    pub deadline: Option<DateTime<Utc>>,
}

pub struct DecisionLogEntry {
    pub timestamp: DateTime<Utc>,
    pub decision: String,
    pub rationale: String,
}
```

## Tauri Commands to implement

| Command | Return | Description |
|---|-||
| `meeting_detect_active_call` | `Option<CallInfo>` | Detect active call platform |
| `meeting_start` | `MeetingSession` | Start new meeting session |
| `meeting_stop` | `Vec<MeetingFile>` | Stop and save meeting data |
| `meeting_pause` | `()` | Pause transcription (redact mode) |
| `meeting_resume` | `()` | Resume transcription |
| `meeting_get_summary` | `Vec<SummaryEntry>` | Rolling summary |
| `meeting_get_action_items` | `Vec<ActionItem>` | Extracted action items |
| `meeting_get_decisions` | `Vec<DecisionLogEntry>` | Decisions logged |
| `meeting_export_json` | `String` | Export entire meeting to JSON |
| `meeting_export_markdown` | `String` | Export transcript to markdown |
| `meeting_get_status` | `MeetingStatus` | Current meeting state |

## Cargo.toml changes

```toml
[dependencies]
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1", features = ["v4"] }
sysinfo = "0.32"
```

## Validation Checklist

- [ ] `cargo build --release` compiles
- [ ] `cargo test` passes
- [ ] All meeting Tauri commands work
- [ ] No regressions in existing features
- [ ] Meeting data stored in ~/.astra/meetings/
