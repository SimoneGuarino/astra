//! Session registry - meeting session lifecycle management.

use super::types::*;
use super::{
    action_item_tracker::ActionItemTracker,
    decision_log::DecisionLog,
    intelligence_engine::{MeetingIntelligenceEngine, MeetingIntelligenceInput},
    live_summarizer::LiveSummarizer,
};
use chrono::Utc;
use serde_json::json;
use std::path::{Path, PathBuf};
use uuid::Uuid;

fn storage_error(operation: &str, error: std::io::Error) -> MeetingRuntimeError {
    MeetingRuntimeError::StorageError {
        message: format!("{operation} failed: {}", error.kind()),
    }
}

fn save_session_metadata(
    storage_dir: &Path,
    session: &MeetingSession,
) -> Result<(), MeetingRuntimeError> {
    std::fs::create_dir_all(storage_dir.join(session.session_id.clone()))
        .map_err(|error| storage_error("create meeting session directory", error))?;
    let metadata_path = storage_dir.join(format!("{}.json.meta", session.session_id));
    let data = serde_json::to_vec_pretty(&json!({
        "session_id": &session.session_id,
        "platform": &session.platform,
        "started_at": session.started_at.to_rfc3339(),
        "status": format!("{:?}", session.status),
        "participants": session.participants,
        "config": &session.config,
        "session_mode": session.session_mode,
        "capture_active": session.capture_active,
        "capture_backend_status": session.capture_backend_status,
    }))
    .map_err(|error| MeetingRuntimeError::SerializationError {
        message: format!("serialize meeting session metadata failed: {error}"),
    })?;
    std::fs::write(metadata_path, data)
        .map_err(|error| storage_error("write meeting session metadata", error))
}

fn save_partial_state(
    storage_dir: &Path,
    state: &MeetingSessionState,
) -> Result<(), MeetingRuntimeError> {
    std::fs::create_dir_all(storage_dir)
        .map_err(|error| storage_error("create meeting state directory", error))?;
    let state_path = storage_dir.join(format!("{}.partial", state.session.session_id));
    let data = serde_json::to_vec_pretty(&json!({
        "session_id": &state.session.session_id,
        "status": format!("{:?}", state.status),
        "paused_from": state.paused_from.as_ref().map(|status| format!("{status:?}")),
        "last_updated_at": state.last_updated_at.to_rfc3339(),
        "transcript_count": state.transcript.len(),
        "summary_count": state.summary.len(),
        "action_items_count": state.action_items.len(),
        "decisions_count": state.decisions.len(),
        "intelligence_present": state.intelligence.is_some(),
    }))
    .map_err(|error| MeetingRuntimeError::SerializationError {
        message: format!("serialize meeting partial state failed: {error}"),
    })?;
    std::fs::write(state_path, data)
        .map_err(|error| storage_error("write meeting partial state", error))
}

pub struct SessionRegistry {
    pub storage_dir: PathBuf,
    pub active_session: Option<MeetingSession>,
    pub active_state: MeetingSessionState,
    pub last_completed_state: Option<MeetingSessionState>,
    pub paused_from: Option<MeetingStatus>,
}

impl SessionRegistry {
    pub fn new(project_root: PathBuf) -> Self {
        let storage_dir = project_root.join(".astra/meetings");
        let _ = std::fs::create_dir_all(&storage_dir);
        Self {
            storage_dir: storage_dir.clone(),
            active_session: None,
            active_state: idle_state(),
            last_completed_state: None,
            paused_from: None,
        }
    }

    pub fn start(
        &mut self,
        platform: String,
        config: MeetingConfig,
        initial_status: MeetingStatus,
        capture_active: bool,
        capture_backend_status: Option<String>,
    ) -> Result<MeetingSession, MeetingRuntimeError> {
        if let Some(session) = self.active_session.as_ref() {
            return Err(MeetingRuntimeError::ActiveSessionExists {
                session_id: session.session_id.clone(),
            });
        }

        let session_id = Uuid::new_v4().to_string();
        let started_at = Utc::now();
        let session = MeetingSession {
            session_id,
            platform,
            status: initial_status.clone(),
            started_at,
            participants: Vec::new(),
            session_mode: config.session_mode,
            capture_active,
            capture_backend_status,
            config,
        };

        let active_state = MeetingSessionState {
            session: session.clone(),
            transcript: Vec::new(),
            summary: Vec::new(),
            action_items: Vec::new(),
            decisions: Vec::new(),
            notes: Vec::new(),
            intelligence: None,
            speakers: default_session_speakers(),
            speaker_rename_count: 0,
            status: initial_status,
            paused_from: None,
            diagnostics: initial_session_diagnostics(&session),
            started_at,
            last_updated_at: started_at,
        };

        save_session_metadata(&self.storage_dir, &session)?;
        self.active_session = Some(session.clone());
        self.active_state = active_state;
        self.paused_from = None;

        Ok(session)
    }

    pub fn get_active_session(&self) -> Option<&MeetingSession> {
        self.active_session.as_ref()
    }

    pub fn get_active_state(&self) -> &MeetingSessionState {
        &self.active_state
    }

    pub fn get_last_completed_state(&self) -> Option<&MeetingSessionState> {
        self.last_completed_state.as_ref()
    }

    pub fn update_capture_status(
        &mut self,
        capture_active: bool,
        capture_backend_status: Option<String>,
    ) -> Result<(), MeetingRuntimeError> {
        self.ensure_active()?;
        if let Some(ref mut session) = self.active_session {
            session.capture_active = capture_active;
            session.capture_backend_status = capture_backend_status;
            self.active_state.session = session.clone();
            save_session_metadata(&self.storage_dir, session)?;
            save_partial_state(&self.storage_dir, &self.active_state)?;
        }
        Ok(())
    }

    pub fn has_runtime_state(&self) -> bool {
        self.active_session.is_some()
            || !matches!(self.active_state.status, MeetingStatus::Idle)
            || self.last_completed_state.is_some()
    }

    pub fn pause(&mut self) -> Result<(), MeetingRuntimeError> {
        self.ensure_active()?;
        if matches!(self.active_state.status, MeetingStatus::Paused) {
            return Err(MeetingRuntimeError::SessionPaused {
                previous_status: self.paused_from.clone(),
            });
        }
        if matches!(
            self.active_state.status,
            MeetingStatus::Completed | MeetingStatus::Stopped
        ) {
            return Err(MeetingRuntimeError::SessionCompleted);
        }

        let previous_status = self.active_state.status.clone();
        self.transition_to(MeetingStatus::Paused)?;
        self.paused_from = Some(previous_status.clone());
        self.active_state.paused_from = Some(previous_status);

        if let Some(ref mut session) = self.active_session {
            session.status = MeetingStatus::Paused;
        }

        save_partial_state(&self.storage_dir, &self.active_state)?;
        Ok(())
    }

    pub fn resume(&mut self) -> Result<(), MeetingRuntimeError> {
        self.ensure_active()?;
        if !matches!(self.active_state.status, MeetingStatus::Paused) {
            return Err(MeetingRuntimeError::InvalidLifecycleTransition {
                from: self.active_state.status.clone(),
                to: MeetingStatus::Ready,
            });
        }

        let resumed_status = self.paused_from.take().unwrap_or(MeetingStatus::Ready);
        self.active_state.status = resumed_status.clone();
        self.active_state.paused_from = None;
        if let Some(ref mut session) = self.active_session {
            session.status = resumed_status;
        }

        self.active_state.last_updated_at = Utc::now();
        save_partial_state(&self.storage_dir, &self.active_state)?;
        Ok(())
    }

    pub fn stop(&mut self) -> Result<ExportedMeeting, MeetingRuntimeError> {
        let mut session = self
            .active_session
            .take()
            .ok_or(MeetingRuntimeError::NoActiveSession)?;

        session.status = MeetingStatus::Stopped;
        session.capture_active = false;
        let session_id = session.session_id.clone();
        let platform = session.platform.clone();
        let started_at = session.started_at;
        let participants = session.participants.clone();

        self.active_state.status = MeetingStatus::Stopped;
        self.active_state.session = session.clone();
        self.active_state.paused_from = None;
        self.paused_from = None;
        let ended_at = Utc::now();

        let completed_state = self.active_state.clone();
        save_partial_state(&self.storage_dir, &completed_state)?;
        self.last_completed_state = Some(completed_state.clone());
        self.active_state = idle_state();

        Ok(ExportedMeeting {
            session_id,
            platform,
            started_at,
            ended_at,
            participants,
            transcript: completed_state.transcript.clone(),
            summary: completed_state.summary.clone(),
            action_items: completed_state.action_items.clone(),
            decisions: completed_state.decisions.clone(),
            notes: completed_state.notes.clone(),
            intelligence: completed_state.intelligence.clone(),
            metadata: json!({
                "capture_backend": completed_state.session.config.capture_backend,
                "transcription_model": completed_state.session.config.transcription_model,
                "diarization_enabled": completed_state.session.config.diarization_enabled,
                "session_mode": completed_state.session.session_mode,
                "capture_active": false,
                "capture_backend_status": completed_state.session.capture_backend_status,
                "speakers": completed_state.speakers,
                "speaker_rename_count": completed_state.speaker_rename_count,
            }),
        })
    }

    pub fn add_transcript(
        &mut self,
        mut entry: TranscriptEntry,
    ) -> Result<(), MeetingRuntimeError> {
        self.ensure_can_accept_transcript()?;
        if matches!(self.active_state.status, MeetingStatus::Ready) {
            self.transition_to(MeetingStatus::Transcribing)?;
        }
        if let Some(session) = self.active_session.as_ref() {
            entry.session_id = session.session_id.clone();
        }
        if entry.segment_id.trim().is_empty() {
            entry.segment_id = new_meeting_artifact_id();
        }
        if entry.created_at.timestamp_millis() == 0 {
            entry.created_at = entry.timestamp;
        }
        apply_source_default_speaker(&mut self.active_state.speakers, &mut entry);
        self.active_state.intelligence = None;
        self.active_state.transcript.push(entry.clone());
        self.active_state.transcript.sort_by(transcript_order);
        self.derive_artifacts_from_transcript(&entry);
        self.active_state.last_updated_at = Utc::now();
        save_partial_state(&self.storage_dir, &self.active_state)?;
        Ok(())
    }

    pub fn add_diagnostic(
        &mut self,
        code: impl Into<String>,
        severity: MeetingDiagnosticSeverity,
        message: impl Into<String>,
    ) -> Result<(), MeetingRuntimeError> {
        self.ensure_active()?;
        self.active_state.diagnostics.push(MeetingDiagnostic {
            code: code.into(),
            severity,
            message: message.into(),
            created_at: Utc::now(),
        });
        self.active_state.last_updated_at = Utc::now();
        save_partial_state(&self.storage_dir, &self.active_state)?;
        Ok(())
    }

    pub fn add_action_item(&mut self, item: ActionItem) -> Result<(), MeetingRuntimeError> {
        self.ensure_can_mutate_notes()?;
        self.active_state.action_items.push(item);
        self.active_state.last_updated_at = Utc::now();
        save_partial_state(&self.storage_dir, &self.active_state)?;
        Ok(())
    }

    pub fn add_decision(&mut self, entry: DecisionLogEntry) -> Result<(), MeetingRuntimeError> {
        self.ensure_can_mutate_notes()?;
        self.active_state.decisions.push(entry);
        self.active_state.last_updated_at = Utc::now();
        save_partial_state(&self.storage_dir, &self.active_state)?;
        Ok(())
    }

    pub fn add_summary(&mut self, entry: SummaryEntry) -> Result<(), MeetingRuntimeError> {
        self.ensure_can_mutate_notes()?;
        self.active_state.summary.push(entry);
        self.active_state.last_updated_at = Utc::now();
        save_partial_state(&self.storage_dir, &self.active_state)?;
        Ok(())
    }

    pub fn add_note(&mut self, note: NoteEntry) -> Result<(), MeetingRuntimeError> {
        self.ensure_can_mutate_notes()?;
        self.active_state.notes.push(note);
        self.active_state.last_updated_at = Utc::now();
        save_partial_state(&self.storage_dir, &self.active_state)?;
        Ok(())
    }

    pub fn get_transcript(&self) -> &[TranscriptEntry] {
        &self.active_state.transcript
    }

    pub fn get_summary(&self) -> &[SummaryEntry] {
        &self.active_state.summary
    }

    pub fn get_action_items(&self) -> &[ActionItem] {
        &self.active_state.action_items
    }

    pub fn get_decisions(&self) -> &[DecisionLogEntry] {
        &self.active_state.decisions
    }

    pub fn get_notes(&self) -> &[NoteEntry] {
        &self.active_state.notes
    }

    pub fn get_intelligence(&self) -> Option<&MeetingIntelligenceResult> {
        if self.active_session.is_some() {
            self.active_state.intelligence.as_ref()
        } else {
            self.last_completed_state
                .as_ref()
                .and_then(|state| state.intelligence.as_ref())
        }
    }

    pub fn generate_intelligence(
        &mut self,
        options: MeetingIntelligenceGenerationOptions,
    ) -> Result<MeetingIntelligenceResult, MeetingRuntimeError> {
        let storage_dir = self.storage_dir.clone();
        let target = self.current_or_completed_state_mut()?;
        if target.transcript.is_empty() {
            return Err(MeetingRuntimeError::InvalidConfig {
                message: "meeting intelligence requires transcript evidence".to_string(),
            });
        }

        target.diagnostics.push(MeetingDiagnostic {
            code: "meeting_intelligence_generation_started".to_string(),
            severity: MeetingDiagnosticSeverity::Info,
            message: "Meeting intelligence generation started; transcript text is not written to audit logs".to_string(),
            created_at: Utc::now(),
        });

        let input = MeetingIntelligenceInput {
            session_id: target.session.session_id.clone(),
            transcript_entries: target.transcript.clone(),
            speakers: target.speakers.clone(),
            generation_options: options.clone(),
        };
        let result = MeetingIntelligenceEngine::generate_with_llm_json_or_rule_based(
            input,
            None,
            Some(&target.session.config.transcription_model),
        )?;

        target.intelligence = Some(result.clone());
        target.last_updated_at = Utc::now();
        target.diagnostics.push(MeetingDiagnostic {
            code: match result.status {
                MeetingIntelligenceStatus::Generated => "meeting_intelligence_generated",
                MeetingIntelligenceStatus::Degraded => "meeting_intelligence_degraded",
                MeetingIntelligenceStatus::Failed => "meeting_intelligence_failed",
                MeetingIntelligenceStatus::Idle | MeetingIntelligenceStatus::Generating => {
                    "meeting_intelligence_status"
                }
            }
            .to_string(),
            severity: if result.status == MeetingIntelligenceStatus::Generated {
                MeetingDiagnosticSeverity::Info
            } else {
                MeetingDiagnosticSeverity::Warning
            },
            message: format!(
                "Meeting intelligence generated from {} transcript segment(s); fallback_used={}; audit_redacted=true; transcript_text_logged=false",
                result.source_transcript_segment_count,
                result.diagnostics.fallback_used
            ),
            created_at: Utc::now(),
        });
        save_partial_state(&storage_dir, target)?;
        Ok(result)
    }

    pub fn clear_intelligence(&mut self) -> Result<(), MeetingRuntimeError> {
        let storage_dir = self.storage_dir.clone();
        let target = self.current_or_completed_state_mut()?;
        target.intelligence = None;
        target.last_updated_at = Utc::now();
        target.diagnostics.push(MeetingDiagnostic {
            code: "meeting_intelligence_cleared".to_string(),
            severity: MeetingDiagnosticSeverity::Info,
            message: "Generated meeting intelligence artifacts were cleared; transcript text was not changed".to_string(),
            created_at: Utc::now(),
        });
        save_partial_state(&storage_dir, target)
    }

    pub fn rename_speaker(
        &mut self,
        speaker_id: &str,
        display_name: &str,
    ) -> Result<RenameSpeakerResult, MeetingRuntimeError> {
        let speaker_id = speaker_id.trim();
        let display_name = normalize_speaker_display_name(display_name)?;
        if speaker_id.is_empty() {
            return Err(MeetingRuntimeError::InvalidConfig {
                message: "speaker_id is required".to_string(),
            });
        }

        let result = if self.active_session.is_some() {
            let result =
                rename_speaker_in_state(&mut self.active_state, speaker_id, &display_name)?;
            save_partial_state(&self.storage_dir, &self.active_state)?;
            result
        } else if let Some(state) = self.last_completed_state.as_mut() {
            let result = rename_speaker_in_state(state, speaker_id, &display_name)?;
            save_partial_state(&self.storage_dir, state)?;
            result
        } else {
            return Err(MeetingRuntimeError::NoActiveSession);
        };

        Ok(result)
    }

    fn current_or_completed_state_mut(
        &mut self,
    ) -> Result<&mut MeetingSessionState, MeetingRuntimeError> {
        if self.active_session.is_some() {
            return Ok(&mut self.active_state);
        }
        self.last_completed_state
            .as_mut()
            .ok_or(MeetingRuntimeError::NoActiveSession)
    }

    pub fn clear(&mut self) {
        self.active_session = None;
        self.active_state = idle_state();
        self.last_completed_state = None;
        self.paused_from = None;
    }

    pub fn transition_to(&mut self, next: MeetingStatus) -> Result<(), MeetingRuntimeError> {
        self.ensure_active()?;
        let current = self.active_state.status.clone();
        if !Self::transition_allowed(&current, &next) {
            return Err(MeetingRuntimeError::InvalidLifecycleTransition {
                from: current,
                to: next,
            });
        }
        self.active_state.status = next.clone();
        self.active_state.last_updated_at = Utc::now();
        if let Some(ref mut session) = self.active_session {
            session.status = next;
        }
        save_partial_state(&self.storage_dir, &self.active_state)?;
        Ok(())
    }

    pub fn ensure_active(&self) -> Result<(), MeetingRuntimeError> {
        if self.active_session.is_none() {
            return Err(MeetingRuntimeError::NoActiveSession);
        }
        Ok(())
    }

    pub fn ensure_can_accept_transcript(&self) -> Result<(), MeetingRuntimeError> {
        self.ensure_active()?;
        match &self.active_state.status {
            MeetingStatus::Ready | MeetingStatus::Capturing | MeetingStatus::Transcribing => Ok(()),
            MeetingStatus::Paused => Err(MeetingRuntimeError::SessionPaused {
                previous_status: self.paused_from.clone(),
            }),
            MeetingStatus::Completed | MeetingStatus::Stopped => {
                Err(MeetingRuntimeError::SessionCompleted)
            }
            status => Err(MeetingRuntimeError::InvalidLifecycleTransition {
                from: status.clone(),
                to: MeetingStatus::Transcribing,
            }),
        }
    }

    pub fn ensure_can_mutate_notes(&self) -> Result<(), MeetingRuntimeError> {
        self.ensure_active()?;
        match &self.active_state.status {
            MeetingStatus::Ready
            | MeetingStatus::Capturing
            | MeetingStatus::Transcribing
            | MeetingStatus::Summarizing => Ok(()),
            MeetingStatus::Paused => Err(MeetingRuntimeError::SessionPaused {
                previous_status: self.paused_from.clone(),
            }),
            MeetingStatus::Completed | MeetingStatus::Stopped => {
                Err(MeetingRuntimeError::SessionCompleted)
            }
            status => Err(MeetingRuntimeError::InvalidLifecycleTransition {
                from: status.clone(),
                to: MeetingStatus::Ready,
            }),
        }
    }

    fn transition_allowed(current: &MeetingStatus, next: &MeetingStatus) -> bool {
        match (current, next) {
            (MeetingStatus::Idle, MeetingStatus::ConsentRequired)
            | (MeetingStatus::Idle, MeetingStatus::Detecting)
            | (MeetingStatus::Idle, MeetingStatus::Ready)
            | (MeetingStatus::Detecting, MeetingStatus::Ready)
            | (MeetingStatus::Ready, MeetingStatus::Starting)
            | (MeetingStatus::Starting, MeetingStatus::Capturing)
            | (MeetingStatus::Starting, MeetingStatus::Failed(_))
            | (MeetingStatus::Detecting, MeetingStatus::Capturing)
            | (MeetingStatus::Ready, MeetingStatus::Capturing)
            | (MeetingStatus::Ready, MeetingStatus::Transcribing)
            | (MeetingStatus::Ready, MeetingStatus::Summarizing)
            | (MeetingStatus::Capturing, MeetingStatus::Transcribing)
            | (MeetingStatus::Transcribing, MeetingStatus::Summarizing)
            | (MeetingStatus::Summarizing, MeetingStatus::Transcribing)
            | (MeetingStatus::Ready, MeetingStatus::Stopping)
            | (MeetingStatus::Capturing, MeetingStatus::Stopping)
            | (MeetingStatus::Transcribing, MeetingStatus::Stopping)
            | (MeetingStatus::Paused, MeetingStatus::Stopping)
            | (MeetingStatus::Stopping, MeetingStatus::Stopped)
            | (MeetingStatus::Paused, MeetingStatus::Completed)
            | (MeetingStatus::Paused, MeetingStatus::Stopped) => true,
            (_, MeetingStatus::Paused) => !matches!(
                current,
                MeetingStatus::Idle
                    | MeetingStatus::Paused
                    | MeetingStatus::Stopped
                    | MeetingStatus::Completed
                    | MeetingStatus::Failed(_)
                    | MeetingStatus::Error(_)
            ),
            (_, MeetingStatus::Stopped) => !matches!(
                current,
                MeetingStatus::Idle
                    | MeetingStatus::Stopped
                    | MeetingStatus::Completed
                    | MeetingStatus::Failed(_)
            ),
            (_, MeetingStatus::Completed) => !matches!(
                current,
                MeetingStatus::Idle
                    | MeetingStatus::Stopped
                    | MeetingStatus::Completed
                    | MeetingStatus::Failed(_)
            ),
            (_, MeetingStatus::Failed(_)) => !matches!(current, MeetingStatus::Completed),
            (from, to) => from == to,
        }
    }

    fn derive_artifacts_from_transcript(&mut self, entry: &TranscriptEntry) {
        if entry.text.trim().is_empty() {
            return;
        }

        let recent = std::slice::from_ref(entry);
        let mut action_tracker = ActionItemTracker::new();
        for mut item in action_tracker.track(recent) {
            item.session_id = entry.session_id.clone();
            if item.title.trim().is_empty() {
                item.title = item.description.chars().take(80).collect();
            }
            if item.evidence_segment_ids.is_empty() {
                item.evidence_segment_ids.push(entry.segment_id.clone());
            }
            if !self
                .active_state
                .action_items
                .iter()
                .any(|existing| existing.description == item.description)
            {
                self.active_state.action_items.push(item);
            }
        }

        let mut decision_log = DecisionLog::new();
        for mut decision in decision_log.track(recent) {
            decision.session_id = entry.session_id.clone();
            if decision.evidence_segment_ids.is_empty() {
                decision.evidence_segment_ids.push(entry.segment_id.clone());
            }
            if !self
                .active_state
                .decisions
                .iter()
                .any(|existing| existing.decision == decision.decision)
            {
                self.active_state.decisions.push(decision);
            }
        }

        if !self
            .active_state
            .notes
            .iter()
            .any(|note| note.evidence_segment_ids.contains(&entry.segment_id))
        {
            let now = Utc::now();
            self.active_state.notes.push(NoteEntry {
                id: new_meeting_artifact_id(),
                session_id: entry.session_id.clone(),
                timestamp: now,
                created_at: now,
                content: format!(
                    "[{}] {}",
                    entry.speaker_display_name(),
                    entry.text.chars().take(220).collect::<String>()
                ),
                evidence_segment_ids: vec![entry.segment_id.clone()],
            });
        }

        let transcript_len = self.active_state.transcript.len();
        if transcript_len == 1 || transcript_len.is_multiple_of(3) {
            let mut summarizer = LiveSummarizer::new(0, 3);
            let summary = summarizer.generate(&self.active_state.transcript);
            if !summary.evidence_segment_ids.is_empty() {
                self.active_state.summary.push(summary);
            }
        }
    }
}

fn idle_state() -> MeetingSessionState {
    let now = Utc::now();
    MeetingSessionState {
        session: MeetingSession {
            session_id: String::new(),
            platform: String::new(),
            status: MeetingStatus::Idle,
            started_at: now,
            participants: Vec::new(),
            config: MeetingConfig {
                platform: String::new(),
                capture_backend: CaptureBackend::Default,
                transcription_model: "local".to_string(),
                sample_rate: 44100,
                diarization_enabled: false,
                privacy_mode: "default".to_string(),
                session_mode: MeetingSessionMode::Manual,
                live_transcription_enabled: false,
                capture_options: MeetingCaptureOptions::manual(),
            },
            session_mode: MeetingSessionMode::Manual,
            capture_active: false,
            capture_backend_status: None,
        },
        transcript: Vec::new(),
        summary: Vec::new(),
        action_items: Vec::new(),
        decisions: Vec::new(),
        notes: Vec::new(),
        intelligence: None,
        speakers: default_session_speakers(),
        speaker_rename_count: 0,
        status: MeetingStatus::Idle,
        paused_from: None,
        diagnostics: Vec::new(),
        started_at: now,
        last_updated_at: now,
    }
}

fn transcript_order(left: &TranscriptEntry, right: &TranscriptEntry) -> std::cmp::Ordering {
    (
        left.start_ms.unwrap_or(u64::MAX),
        transcript_source_rank(left.source),
        left.created_at,
        left.segment_id.as_str(),
    )
        .cmp(&(
            right.start_ms.unwrap_or(u64::MAX),
            transcript_source_rank(right.source),
            right.created_at,
            right.segment_id.as_str(),
        ))
}

fn transcript_source_rank(source: TranscriptSource) -> u8 {
    match source {
        TranscriptSource::SystemAudio => 0,
        TranscriptSource::Microphone => 1,
        TranscriptSource::ImportedFile => 2,
        TranscriptSource::Manual => 3,
        TranscriptSource::Unknown => 4,
    }
}

fn default_session_speakers() -> Vec<SpeakerLabel> {
    [
        TranscriptSource::Microphone,
        TranscriptSource::SystemAudio,
        TranscriptSource::Manual,
        TranscriptSource::ImportedFile,
        TranscriptSource::Unknown,
    ]
    .into_iter()
    .map(SpeakerLabel::source_default)
    .collect()
}

fn normalize_speaker_display_name(display_name: &str) -> Result<String, MeetingRuntimeError> {
    let normalized = display_name
        .trim()
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if normalized.is_empty() {
        return Err(MeetingRuntimeError::InvalidConfig {
            message: "speaker display_name is required".to_string(),
        });
    }
    if normalized.chars().count() > 80 {
        return Err(MeetingRuntimeError::InvalidConfig {
            message: "speaker display_name must be 80 characters or fewer".to_string(),
        });
    }
    Ok(normalized)
}

fn speaker_label_for_entry(
    speakers: &mut Vec<SpeakerLabel>,
    entry: &TranscriptEntry,
) -> SpeakerLabel {
    let requested_name = entry.speaker.trim();
    let requested_is_explicit = !requested_name.is_empty()
        && !matches!(
            requested_name.to_ascii_lowercase().as_str(),
            "unknown" | "stt" | "manual" | "imported"
        );

    let mut default_label = SpeakerLabel::source_default(entry.source);
    if matches!(
        entry.source,
        TranscriptSource::Manual | TranscriptSource::ImportedFile
    ) && requested_is_explicit
    {
        default_label.speaker_id = stable_user_assigned_speaker_id(entry.source, requested_name);
        default_label.display_name = requested_name.to_string();
        default_label.confidence = 1.0;
        default_label.attribution_method = SpeakerAttributionMethod::UserAssigned;
    }

    if let Some(existing) = speakers
        .iter()
        .find(|speaker| speaker.speaker_id == default_label.speaker_id)
        .cloned()
    {
        return existing;
    }

    speakers.push(default_label.clone());
    default_label
}

fn apply_source_default_speaker(speakers: &mut Vec<SpeakerLabel>, entry: &mut TranscriptEntry) {
    let speaker = speaker_label_for_entry(speakers, entry);
    entry.speaker_id = Some(speaker.speaker_id.clone());
    entry.speaker = speaker.display_name.clone();
    entry.speaker_label = Some(speaker.display_name.clone());
    entry.speaker_confidence = Some(speaker.confidence);
    entry.speaker_attribution_method = speaker.attribution_method;
}

fn stable_user_assigned_speaker_id(source: TranscriptSource, display_name: &str) -> String {
    let source_prefix = match source {
        TranscriptSource::Manual => "manual",
        TranscriptSource::ImportedFile => "imported",
        TranscriptSource::Microphone => "microphone",
        TranscriptSource::SystemAudio => "system",
        TranscriptSource::Unknown => "unknown",
    };
    let normalized = display_name
        .to_ascii_lowercase()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '_'
            }
        })
        .collect::<String>()
        .split('_')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("_");
    format!(
        "{source_prefix}_{}",
        normalized.chars().take(48).collect::<String>()
    )
}

fn rename_speaker_in_state(
    state: &mut MeetingSessionState,
    speaker_id: &str,
    display_name: &str,
) -> Result<RenameSpeakerResult, MeetingRuntimeError> {
    let Some(speaker_index) = state
        .speakers
        .iter()
        .position(|speaker| speaker.speaker_id == speaker_id)
    else {
        return Err(MeetingRuntimeError::InvalidConfig {
            message: format!("unknown speaker_id: {speaker_id}"),
        });
    };

    state.speakers[speaker_index].display_name = display_name.to_string();
    state.speakers[speaker_index].confidence = 1.0;
    state.speakers[speaker_index].attribution_method = SpeakerAttributionMethod::UserAssigned;
    let speaker = state.speakers[speaker_index].clone();

    let mut renamed_entries = 0;
    for entry in &mut state.transcript {
        if entry.speaker_id.as_deref() == Some(speaker_id) {
            entry.speaker = speaker.display_name.clone();
            entry.speaker_label = Some(speaker.display_name.clone());
            entry.speaker_confidence = Some(speaker.confidence);
            entry.speaker_attribution_method = SpeakerAttributionMethod::UserAssigned;
            renamed_entries += 1;
        }
    }

    for item in &mut state.action_items {
        if let Some(assignee) = item.assignee.as_mut() {
            if assignee.speaker_id.as_deref() == Some(speaker_id) {
                assignee.name = speaker.display_name.clone();
            }
        }
    }

    for decision in &mut state.decisions {
        if let Some(made_by) = decision.made_by.as_mut() {
            if made_by.speaker_id.as_deref() == Some(speaker_id) {
                made_by.name = speaker.display_name.clone();
            }
        }
    }

    if let Some(intelligence) = state.intelligence.as_mut() {
        rename_speaker_in_intelligence(intelligence, speaker_id, &speaker.display_name);
    }

    for note in &mut state.notes {
        if note.evidence_segment_ids.len() == 1 {
            if let Some(source_entry) = state
                .transcript
                .iter()
                .find(|entry| entry.segment_id == note.evidence_segment_ids[0])
            {
                note.content = format!(
                    "[{}] {}",
                    source_entry.speaker_display_name(),
                    source_entry.text.chars().take(220).collect::<String>()
                );
            }
        }
    }

    state.speaker_rename_count = state.speaker_rename_count.saturating_add(1);
    state.last_updated_at = Utc::now();
    state.diagnostics.push(MeetingDiagnostic {
        code: "speaker_label_renamed".to_string(),
        severity: MeetingDiagnosticSeverity::Info,
        message: format!(
            "Speaker label metadata was renamed for speaker_id {}; transcript text was not changed; display_name_length={}",
            speaker_id,
            display_name.chars().count()
        ),
        created_at: Utc::now(),
    });

    Ok(RenameSpeakerResult {
        speaker,
        renamed_entries,
    })
}

fn rename_speaker_in_intelligence(
    intelligence: &mut MeetingIntelligenceResult,
    speaker_id: &str,
    display_name: &str,
) {
    for decision in &mut intelligence.decisions {
        if decision.made_by_speaker_id.as_deref() == Some(speaker_id) {
            decision.made_by_display_name = Some(display_name.to_string());
        }
    }
    for item in &mut intelligence.action_items {
        if item.assignee_speaker_id.as_deref() == Some(speaker_id) {
            item.assignee_display_name = Some(display_name.to_string());
        }
    }
    for question in &mut intelligence.open_questions {
        if question.asked_by_speaker_id.as_deref() == Some(speaker_id) {
            question.asked_by_display_name = Some(display_name.to_string());
        }
    }
    for item in &mut intelligence.timeline {
        if item.speaker_id.as_deref() == Some(speaker_id) {
            item.speaker_display_name = Some(display_name.to_string());
        }
    }
    intelligence.diagnostics.warnings.push(format!(
        "Speaker display labels were refreshed after metadata rename for speaker_id {speaker_id}; evidence IDs were unchanged"
    ));
}

fn initial_session_diagnostics(session: &MeetingSession) -> Vec<MeetingDiagnostic> {
    let mut diagnostics = Vec::new();
    diagnostics.push(MeetingDiagnostic {
        code: "ui_live_updates_event_subscription_supported".to_string(),
        severity: MeetingDiagnosticSeverity::Info,
        message: "Meeting UI subscribes to meeting update events and uses bounded polling while the panel is open".to_string(),
        created_at: Utc::now(),
    });
    diagnostics.push(MeetingDiagnostic {
        code: "transcript_ordering_chronological_storage_newest_first_ui".to_string(),
        severity: MeetingDiagnosticSeverity::Info,
        message:
            "Meeting transcript is stored chronologically and presented newest-first in the UI"
                .to_string(),
        created_at: Utc::now(),
    });
    diagnostics.push(MeetingDiagnostic {
        code: "speaker_attribution_source_default".to_string(),
        severity: MeetingDiagnosticSeverity::Info,
        message: "Speaker attribution uses source defaults: microphone is You, system audio is Speaker 1; real diarization is not claimed".to_string(),
        created_at: Utc::now(),
    });
    diagnostics.push(MeetingDiagnostic {
        code: "diarization_unsupported".to_string(),
        severity: MeetingDiagnosticSeverity::Info,
        message: "True speaker diarization is unsupported in this milestone; user speaker renames are metadata only".to_string(),
        created_at: Utc::now(),
    });
    if session.session_mode == MeetingSessionMode::Manual {
        diagnostics.push(MeetingDiagnostic {
            code: "manual_fallback_active".to_string(),
            severity: MeetingDiagnosticSeverity::Info,
            message: "Manual session is active; no audio capture or STT pipeline was started"
                .to_string(),
            created_at: Utc::now(),
        });
    }
    if session.config.session_mode == MeetingSessionMode::RealCapture
        && session.config.capture_backend == CaptureBackend::Wasapi
    {
        if session.config.capture_options.system_audio {
            diagnostics.push(MeetingDiagnostic {
                code: "system_audio_capture_requested".to_string(),
                severity: MeetingDiagnosticSeverity::Info,
                message:
                    "System audio capture uses Windows WASAPI render loopback managed segments"
                        .to_string(),
                created_at: Utc::now(),
            });
        }
        if session.config.capture_options.microphone {
            diagnostics.push(MeetingDiagnostic {
                code: "microphone_capture_requested".to_string(),
                severity: MeetingDiagnosticSeverity::Info,
                message: "Microphone capture uses the Windows default WASAPI capture endpoint"
                    .to_string(),
                created_at: Utc::now(),
            });
        }
        diagnostics.push(MeetingDiagnostic {
            code: "streaming_stt_unsupported".to_string(),
            severity: MeetingDiagnosticSeverity::Info,
            message: "Streaming STT is unsupported; enabled transcription uses completed managed WAV segment files".to_string(),
            created_at: Utc::now(),
        });
    }
    diagnostics
}
