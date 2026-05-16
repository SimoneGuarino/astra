# Meeting Intelligence Engine

The Meeting Intelligence Engine is currently a governed foundation, not a live audio recorder.
It is exposed through `MeetingRuntime`, Tauri commands routed through governed direct actions,
and the internal desktop agent meeting debug panel.

## Supported now

- Scoped meeting consent read, grant, and revoke.
- Conservative call detection with explicit confidence states.
- Manual meeting sessions for safe metadata and manual transcript testing.
- Manual transcript, action item, and decision insertion during valid lifecycle states.
- File-based transcription of a validated local `.wav` file into the active meeting through the existing `SttClient::transcribe(Path)` boundary.
- Managed meeting segment copies under `.astra/meetings/{session_id}/segments/`, with cleanup success and warning metadata.
- Bounded managed `.wav` segment writing for future capture backends using generated segment names only.
- Internal captured-segment transcription through the existing file-based meeting STT bridge.
- Redacted audit for meeting file transcription.
- Session pause, resume, stop, and export persistence.
- Full meeting data clearing for runtime state and `.astra/meetings` files after typed confirmation.
- Granular meeting capability reporting through the existing tool registry and capability manifest.
- A persistent `CaptureController` boundary that owns the real capture handle lifecycle and reports health/readiness without starting fake capture.
- Windows WASAPI loopback capture into bounded managed WAV segments when run on Windows with an available default render endpoint.
- Redacted segment health metrics for written/transcribed/failed managed segments.
- A `MeetingSttAdapter` boundary that separately reports file transcription, live transcription, and chunk streaming capability states.

## Unsupported by design

- OS audio capture on non-Windows platforms, including CoreAudio and PipeWire.
- Live STT streaming or chunking from system audio.
- Speaker diarization from meeting audio.
- Live summarization from live transcript input.
- Follow-up email or message sending.

Unsupported features must return explicit errors or unavailable capability states. They must not
produce fake capture state, fake transcript text, or outbound side effects.

## Privacy and consent

Meeting start requires explicit scoped consent for the requested app or platform. File-based
meeting transcription also requires an active session, still-valid consent for that session
platform, a governed high-risk tool permission check, direct confirmation metadata, post-governance
path validation, managed copy into `.astra/meetings/{session_id}/segments/`, and audit redaction.
No recording, capture, or transcription path may start without consent. Consent operations are
governed and audited through the desktop agent runtime.

## Lifecycle

The session lifecycle is explicit:

```txt
Idle -> ConsentRequired -> Detecting -> Ready -> Capturing -> Transcribing -> Summarizing
     -> Paused(previous_state) -> Completed
     -> Failed(reason)
```

Manual sessions start in `Ready` with `capture_active = false`. Transcript insertion moves a
manual session to `Transcribing`. Paused and completed sessions reject live mutation.

## Future Capture Architecture

`MeetingRuntime` owns these meeting boundaries:

```txt
MeetingRuntime
  -> SessionRegistry
  -> PrivacyState
  -> NoteOrganizer
  -> CaptureController
  -> MeetingSttAdapter
```

`CaptureController` is the persistent owner for real/future audio capture. It models
`Idle`, `Unsupported`, `Starting`, `Capturing`, `Paused`, `Stopping`, and `Failed`, and exposes
`prepare`, `start_real_capture`, `pause_capture`, `resume_capture`, `stop_capture`,
`abort_capture`, and `health_snapshot`. CoreAudio and PipeWire return typed unsupported errors and
do not create active handles. Windows WASAPI loopback is implemented through a direct `windows`
crate dependency, initializes COM on the capture thread, and keeps the active capture handle owned
by `CaptureController`.

Real capture starts only after consent, governed command execution, `meeting.audio.capture`
permission preflight, and, when automatic segment transcription is enabled,
`meeting.transcription.segment` permission preflight.

The future live audio contract is metadata-first and bounded:

```txt
sample_rate
channels
sample format
monotonic timestamp
sequence number
duration
source backend
byte length
frame count
```

The default pipeline budget is bounded to 64 queued chunks, 1000 ms chunks, an 8 MiB memory
budget, a reject-newest overflow policy, two bounded retries, and 240 managed segments per
session. Raw audio is not written to audit logs.

Completed segment files are bounded before they enter STT:

```txt
generated UUID .wav.tmp
-> RIFF/WAVE PCM16 header and samples
-> byte and duration bound checks
-> atomic-ish rename to generated UUID .wav
-> managed path only under .astra/meetings/{session_id}/segments/
```

## File STT Bridge

Voice sessions already use `SttClient::transcribe(Path)` with the existing Python STT worker.
Meeting audio does not create a second STT worker. The current real meeting transcription path is
file-based only:

```txt
user-selected audio file
-> governed high-risk command confirmation
-> validate/canonicalize/size-check/extension-check
-> validate minimal WAV RIFF/WAVE header bytes
-> copy to managed meeting segment storage
-> SttClient::transcribe(Path)
-> TranscriptEntry
-> SessionRegistry::add_transcript()
```

Captured segments use the same transcription boundary after a backend has already produced a
managed segment:

```txt
managed captured WAV segment
-> consent re-check
-> managed path and WAV header validation
-> SttClient::transcribe(Path)
-> TranscriptEntry
-> SessionRegistry::add_transcript()
-> cleanup metadata
```

Windows capture uses this path:

```txt
default render endpoint
-> WASAPI loopback shared-mode capture
-> PCM16 conversion for float32 or PCM16 mix formats
-> bounded managed WAV segment
-> existing captured-segment transcription bridge
-> SttClient::transcribe(Path)
-> TranscriptEntry insertion
```

Only `.wav` is accepted until the supported worker/FFmpeg contract is made explicit. Transcript
confidence is stored as `0.0` because the existing STT boundary returns text only and no model
confidence.

Audit and privacy guarantees for `meeting.transcription.file`:

- No raw source audio path is written to audit.
- No managed audio path is written to audit.
- No source or managed audio filename is written to audit.
- No raw transcript text is written to audit.
- No meeting audio content is written to audit.
- No filesystem metadata read, canonicalization, existence check, WAV header read, managed copy, or STT call happens before the governed command starts.
- Pre-governance audit params are string-derived and redacted only, such as path presence, extension hint, speaker presence, cleanup preference, and redaction flags.
- Post-governance audit summaries may include safe metadata such as audio extension, file size, result kind, text length, transcript insertion status, cleanup requested/performed, cleanup-error presence, and redaction flags.

Audit and privacy guarantees for capture segments:

- No raw segment path is written to audit.
- No managed segment filename is written to audit.
- No raw transcript text is written to audit.
- No raw audio samples are written to audit.
- No device friendly name is written to audit.
- Segment health stores redacted counters and status strings only.

The meeting adapter reports the file bridge separately from unsupported live modes:

```txt
file_transcription: ready when the existing SttClient is attached
live_transcription: unavailable
chunk_streaming: unavailable
existing_stt_boundary: SttClient::transcribe(Path)
top_level_reason: none when file_transcription is ready
```

Cleanup after successful transcript insertion is reported as result metadata instead of turning a
successful transcript mutation into a hard command failure. If transcription fails after a managed
copy and cleanup was requested, cleanup failure is surfaced as sanitized warning metadata on the
runtime error and in redacted audit failure details. Raw file paths are not included in cleanup
warnings.

Future live integration must add a safe segment API or write governed bounded segments before
calling the existing client.

## Testing

Rust validation:

```bash
cd src-tauri
cargo test --test meeting_governance -- --nocapture
cargo test
```

Frontend validation:

```bash
npm run build
```

Manual UI validation is available in the Desktop Agent panel under the `meeting` tab.

## Future work

- Add CoreAudio/PipeWire `CaptureController` backends after Windows capture is validated.
- Add hardware/manual validation coverage for more Windows output device formats.
- Extend `SttClient` only if future governed live meeting segments need a richer file/metadata contract.
- Add diarization and summarization only after real transcript input exists.
- Implement follow-up as draft-first, policy-gated, approval-gated, and audited.
