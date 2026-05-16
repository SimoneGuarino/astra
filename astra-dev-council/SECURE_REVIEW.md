# SECURE_REVIEW.md — Security Review for Call Meeting Intelligence

## Feature Security Context

Add **Call Meeting Intelligence Engine** — captures Teams/Meet/Discord calls, transcribes live, produces summary/action-items/decision-log/follow-up, saves structured notes.

## Threat Model

### Attack Vectors

1. **Unauthorized recording** — meeting starts without consent -> Mitigation: opt-in consent per call
2. **Data leak** — meeting data stored insecurely -> Mitigation: local-only, encrypted at rest
3. **Privacy violation** — captures audio from other applications -> Mitigation: system audio capture only for the meeting call window
4. **Model prompt injection** — meeting transcript used as LLM context -> Mitigation: context isolation, input sanitization
5. **Follow-up email interception** — follow-up drafts could leak sensitive info -> Mitigation: review before send, no auto-send MVP

### Data Flow Analysis

### Data Classification

| Data Type | Sensitivity | Storage Location | Retention | Encryption |
|---|---|-|---|---|
| Raw audio | HIGH | ~/.astra/meetings/ (temporary) | 24h auto-delete | Yes |
| Transcript | HIGH | ~/.astra/meetings/ | 30d configurable | Optional |
| Summary | MEDIUM | ~/.astra/meetings/ | 90d | No |
| Action items | LOW | ~/.astra/meetings/ | Indefinite | No |
| Decisions | MEDIUM | ~/.astra/meetings/ | 90d | No |
| Speaker VOS | MEDIUM | ~/.astra/meetings/ | 30d | Optional |

## Security Requirements

### Phase 1 (MVP) — Security Gates

- [ ] **Explicit consent required** — meeting cannot start without explicit consent
- [ ] **Local-only storage** — meeting data never leaves the device
- [ ] **Privacy controls** — user can pause recording at any time
- [ ] **Data retention policy** — raw audio 24h, transcript 30d, summarize/export 90d
- [ ] **Audit logging** — all meeting data access logged
- [ ] **No cloud upload** — all processing local, no Whisper/ Whisper download
- [ ] **Explicit consent required** — meeting cannot start without explicit consent

### Phase 2 (v2) — Enhanced Security

- [ ] End-to-end encryption for meeting data
- [ ] Automatic redaction of sensitive patterns (credit card, SSN, etc.)
- [ ] Meeting data export with password protection
- [ ] Biometric unlock for meeting panel access
- [ ] Meeting data cleanup on device shutdown

## Privacy Controls

### Consent Flow

1. First meeting detection -> consent prompt
2. User grants permission to record
3. Meeting starts
4. User can pause/resume recording
5. User can stop recording
6. Meeting data is processed

### UI Privacy Features

- Privacy indicator when recording
- Visual indicator when meeting data is being captured
- Clear consent prompt with explanation of data usage
- "Pause recording" available during sensitive moments
- "Never record from this app" option per app

### Data Lifecycle

1. **Captured** -> raw audio (24h)
2. **Processed** -> transcript (30d)
3. **Summarized** -> summary (90d)
4. **Exported** -> JSON/Markdown (indefinite)
5. **Deleted** -> all data deleted after retention

## Implementation Checklist

- [ ] Meeting data stored in ~/.astra/meetings/ (not audio/stt/tts)
- [ ] Meeting recording is opt-in only
- [ ] Consent prompt appears on first meeting detection
- [ ] Privacy controls available in meeting panel
- [ ] Data retention policy enforced
- [ ] Audit logging for meeting data access
- [ ] No cloud upload of meeting data
- [ ] Meeting export is optional and user-initiated

## Security Review Sign-off

- [ ] Architect Agent
- [ ] Security Agent
- [ ] QA Agent
