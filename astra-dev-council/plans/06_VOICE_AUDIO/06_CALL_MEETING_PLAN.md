# Voice/Audio Agent Implementation Plan

## Architecture Pattern
Follow audio pipeline pattern in existing codebase. Reuse SttClient.rs, TtsClient.rs, vad.rs, voice_session.rs infrastructure. Audio capture extends existing voice session system to capture system audio in addition to mic.

## Module Structure

### src-tauri/src/meeting/
- `audio_capture.rs` — cross-platform system audio capture
- `transcription_stream.rs` — streaming STT client (extends SttClient)
- `speaker_diarization.rs` — voice ID per speaker
- `audio_router.rs` — route captured audio to meeting engine
- `meeting_audio_pipeline.rs` — orchestrate audio capture -> STT -> meeting engine

## Audio Pipeline Architecture

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Audio   │  →  │  Buffer  │  →  │  VAD     │  →  │  Speaker │
│ Capture  │     │ & Queue  │     │ (VAD)    │     │  Diar.   │
│          │     │          │     │          │     │          │
│ PipeWire │     │ Ringbuf  │     │ rms >    │     │  voice   │
│ CoreAudio│     │ FIFO     │     │ threshold│     │  feature │
│ WASAPI   │     │          │     │ → speech │     │  extract │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
                                                   │
                                                   │
                                              ┌──────────┐
                                              │ STT      │
                                              │ (Whisper │
                                              │  / Wiper)│
                                              └──────────┘
                                                   │
                                                   │
                                              ┌──────────┐
                                              │ Transcript│
                                              │ → Meeting │
                                              │ Engine   │
                                              └──────────┘
```

## Data Contracts

### Types for audio capture

```rust
pub struct AudioCaptureConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub buffer_size: usize,
    pub backend: AudioBackend,
}

pub enum AudioBackend {
    PipeWire,
    CoreAudio,
    WASAPI,
    Default,
}

pub struct AudioFrame {
    pub timestamp: DateTime<Utc>,
    pub samples: Vec<f32>,
    pub device_id: String,
}

pub struct SpeakerFrame {
    pub timestamp: DateTime<Utc>,
    pub voice_features: Vec<f32>,
    pub speaker_id: Option<String>,
    pub confidence: f32,
    pub audio_frame: AudioFrame,
}
```

## Implementation Sequence

### Phase 1: Audio Capture Backend (src-tauri/src/meeting/audio_capture.rs)
1. Cross-platform audio capture abstraction (3 backends)
2. Default to PulseAudio/PipeWire on Linux, CoreAudio on macOS, WASAPI on Windows
3. Capture system audio (not mic) — use loopback device

### Phase 2: VAD + Speaker Diarization (src-tauri/src/meeting/speaker_diarization.rs)
1. VAD (Voice Activity Detection) — reuse existing vad.rs
2. Voice feature extraction — MFCC-like features
3. Speaker clustering — group audio segments by speaker

### Phase 3: Audio Router (src-tauri/src/meeting/audio_router.rs)
1. Route captured audio to meeting engine
2. Buffer frames for meeting session lifecycle
3. Support for pause/resume (redaction mode)

### Phase 4: Meeting Audio Pipeline (src-tauri/src/meeting/meeting_audio_pipeline.rs)
1. Orchestrate audio capture -> VAD -> diarization -> STT
2. Manage session lifecycle (start/stop/pause)
3. Reuse existing SttClient for streaming STT

## Tauri Commands to expose

| Command | Return | Description |
|---|-||
| `meeting_audio_start` | `()` | Start system audio capture |
| `meeting_audio_stop` | `()` | Stop system audio capture |
| `meeting_audio_pause` | `()` | Pause recording (redact mode) |
| `meeting_audio_resume` | `()` | Resume recording |
| `meeting_audio_get_devices` | `Vec<AudioDevice>` | List available audio devices |
| `meeting_audio_get_sample_rate` | `u32` | Current sample rate |
| `meeting_audio_get_backend` | `String` | Current audio backend |

## Cargo.toml changes

```toml
[dependencies.meeting_audio]
audio_capturer = { version = "0.4", optional = true }
libvorbis = { version = "1.0", optional = true }
```

## Validation Checklist

- [ ] `cargo build --release` compiles
- [ ] `cargo test` passes
- [ ] Audio capture works on current platform
- [ ] VAD detects speech correctly
- [ ] Speaker diarization works for >1 speaker
- [ ] Audio routing to meeting engine works
- [ ] Pause/resume/redaction works
- [ ] No audio latency issues (>500ms)
- [ ] Resource usage <15% CPU during capture
