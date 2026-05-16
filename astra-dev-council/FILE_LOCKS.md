# FILE_LOCKS.md — Active Locks for Call Meeting Intelligence Feature

## Lock Policy

Each agent MUST hold a file lock before writing to any file. Release the lock once written. Other agents must wait or skip that file.

## Active Locks

### Rust Backend Agent
| File | Agent | Timestamp |
|---|-|--|
| src-tauri/src/meeting/* (new) | Rust Backend Agent | 2026-05-01 |
| src-tauri/src/lib.rs | Rust Backend Agent | 2026-05-01 |
| src-tauri/Cargo.toml | Rust Backend Agent | 2026-05-01 |
| src-tauri/build.rs | Rust Backend Agent | 2026-05-01 |
| src-tauri/src/meeting/mod.rs | Rust | Rust Backend Agent |

### Frontend UI Agent
| File | Agent | Timestamp |
|---|---|---|
| src/components/MeetingLivePanel.tsx (new) | Frontend UI Agent | 2026-05-01 |
| src/hooks/useMeetingEngine.ts (new) | Frontend UI Agent | 2026-05-01 |
| src/hooks/useMeetingSummary.ts (new) | Frontend UI Agent | 2026-05-01 |
| src/services/meeting_service.ts (new) | Frontend UI Agent | 2026-05-01 |
| src/types/meeting_types.ts (new) | Frontend UI Agent | 2026-05-01 |
| src/types/meeting_types.ts (new) | Frontend UI Agent | 2026-05-01 |

### Voice/Audio Agent
| File | Agent | Timestamp |
|---|-|--|
| src-tauri/src/meeting/audio_capture.rs | Voice/Audio Agent | 2026-05-01 |
| src-tauri/src/meeting/transcription_stream.rs | Voice/Audio Agent | 2026-05-01 |
| src-tauri/src/meeting/speaker_diarization.rs | Voice/Audio Agent | 2026-05-01 |

### Security Agent
| File | Agent | Timestamp |
|---|---|-|
| src-tauri/src/meeting/privacy_control.rs | Security Agent | 2026-05-01 |
| DECISIONS.md | Security Agent | 2026-05-01 |

### AI Orchestration Agent
| File | Agent | Timestamp |
|---|---|-|
| src-tauri/src/meeting/llm_router.rs | AI Orchestration Agent | 2026-05-01 |
| src-tauri/src/meeting/live_summarizer.rs | AI Orchestration Agent | 2026
