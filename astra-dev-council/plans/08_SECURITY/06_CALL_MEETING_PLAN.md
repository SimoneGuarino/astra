# Security Agent Implementation Plan

## Architecture Pattern
Extend existing Tauri permissions/security model. Meeting recording is "high_risk" requiring explicit privacy approval. Privacy controls follow existing permission flow (pending_approvals_store, action_policy).

## Module Structure

### src-tauri/src/meeting/
- `privacy_control.rs` — consent flow, data retention, encryption
- `audit_log.rs` — meeting access audit logging (reuse existing pattern)
- `data_retention.rs` — automatic data cleanup after retention
- `consent_flow.rs` — explicit consent prompt and state management

## Privacy & Security Design

### Consent Flow
1. **First meeting detection** → explicit consent prompt
2. **User grants** → meeting starts recording
3. **User denies** → meeting panel opens but no audio capture
4. **User can revoke** → at any time via meeting panel

### Data Classification
| Data | Sensitivity | Retention | Encryption |
|---|-||
| Raw audio | HIGH | 24h auto-delete | AES-256 |
| Transcript | HIGH | 30d (configurable) | Optional |
| Summary | MEDIUM | 90d | No |
| Action items | LOW | Indefinite | No |
| Decisions | MEDIUM | 90d | No |
| Speaker profiles | MEDIUM | 30d | Optional |

### Privacy Controls

#### Per-call Controls
- ✅ Opt-in consent before recording
- ✅ Pause/resume during call
- ✅ Delete recording at any time

#### Per-app Controls
- ✅ "Never record from this app" option
- ✅ Remember consent per app

#### Global Controls
- ✅ Master toggle to disable meeting features
- ✅ Data retention settings
- ✅ Audit log of all meeting data access

## Implementation Sequence

### Phase 1: Privacy Control (src-tauri/src/meeting/privacy_control.rs)
1. Consent state management (granted/denied/revoked)
2. Per-app consent storage (JSON config)
3. Global master toggle
4. Data retention enforcement

### Phase 2: Audit Logging (src-tauri/src/meeting/audit_log.rs)
1. Meeting audit event types (access, modification, export, deletion)
2. Audit log storage (JSON file in ~/.astra/meetings/audit/)
3. Audit log query/API

### Phase 3: Data Retention (src-tauri/src/meeting/data_retention.rs)
1. Timer for raw audio deletion (24h)
2. Timer for transcript deletion (30d)
3. Notification before deletion
4. Manual override (force delete)

### Phase 4: Consent Flow (src-tauri/src/meeting/consent_flow.rs)
1. Consent prompt UI (via Tauri dialog)
2. Consent state persistence
3. Consent revocation flow

## Tauri Commands to expose

| Command | Return | Description |
|---|-||
| `meeting_consent_get` | `ConsentState` | Current consent state |
| `meeting_consent_grant` | `()` | Grant consent for specific app |
| `meeting_consent_revoke` | `()` | Revoke consent |
| `meeting_consent_global_toggle` | `()` | Master toggle for feature |
| `meeting_data_get_retention` | `DataRetention` | Current retention settings |
| `meeting_data_set_retention` | `()` | Update retention settings |
| `meeting_audit_get_recent` | `Vec<AuditEvent>` | Recent audit events |
| `meeting_data_delete` | `()` | Delete all meeting data |

## Security Review Checklist

- [ ] All meeting data stored locally (~/.astra/meetings/)
- [ ] No cloud upload of any meeting data
- [ ] Explicit consent required before recording
- [ ] User can pause/resume/delete recording at any time
- [ ] Data retention policy enforced (24h/30d/90d)
- [ ] Audit log of all meeting data access
- [ ] Per-app consent management works
- [ ] Global master toggle works
- [ ] No unauthorized network calls from meeting engine
- [ ] Raw audio encrypted at rest (AES-256)
- [ ] Speaker profiles encrypted (optional)
- [ ] Audit logging comprehensive

## Implementation Checklist

- [ ] `privacy_control.rs` implements correctly
- [ ] `audit_log.rs` implements correctly
- [ ] `data_retention.rs` implements correctly
- [ ] `consent_flow.rs` implements correctly
- [ ] All Tauri commands work correctly
- [ ] No regressions in existing permission model
- [ ] Security review passed by architect
