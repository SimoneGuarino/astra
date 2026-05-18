//! Local meeting session memory archive and lexical search.
//!
//! This module deliberately uses bounded local files rather than a database or
//! vector store. Transcript text remains in archived session documents only;
//! list/search audit payloads are supplied by the Tauri boundary as metadata.

use super::{note_organizer::NoteOrganizer, types::*};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};
use uuid::Uuid;

const ARCHIVE_SCHEMA_VERSION: u32 = 2;
const INDEX_SCHEMA_VERSION: u32 = 2;
const MAX_LIST_LIMIT: usize = 50;
const MAX_SEARCH_LIMIT: usize = 50;
const MAX_QUERY_CHARS: usize = 120;
const MAX_SNIPPET_CHARS: usize = 180;
const MAX_PREVIEW_CHARS: usize = 220;

#[derive(Debug, Clone)]
pub struct SessionMemoryStore {
    storage_base: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct MeetingSessionMemoryIndex {
    schema_version: u32,
    updated_at: Option<chrono::DateTime<Utc>>,
    sessions: Vec<MeetingSessionListItem>,
}

impl SessionMemoryStore {
    pub fn new(storage_base: PathBuf) -> Self {
        Self { storage_base }
    }

    pub fn archive_completed_session(
        &self,
        state: &MeetingSessionState,
        exported: &ExportedMeeting,
        capture_health: &CaptureHealth,
        system_capture_health: &CaptureHealth,
        microphone_capture_health: &CaptureHealth,
    ) -> Result<MeetingSessionListItem, MeetingRuntimeError> {
        let now = Utc::now();
        let document = MeetingSessionArchiveDocument {
            schema_version: ARCHIVE_SCHEMA_VERSION,
            session_id: exported.session_id.clone(),
            archived_at: now,
            updated_at: now,
            state: state.clone(),
            exported: exported.clone(),
            screen_contexts: state.screen_contexts.clone(),
            capture_health: capture_health.clone(),
            system_capture_health: system_capture_health.clone(),
            microphone_capture_health: microphone_capture_health.clone(),
        };
        let session_dir = self.session_dir(&document.session_id)?;
        fs::create_dir_all(&session_dir)
            .map_err(|error| storage_error("create meeting session archive directory", error))?;

        atomic_write_json(&session_dir.join("session.json"), &document)?;
        atomic_write_json(
            &session_dir.join("transcript.json"),
            &document.state.transcript,
        )?;
        atomic_write_json(
            &session_dir.join("screen_context.json"),
            &document.screen_contexts,
        )?;
        if let Some(intelligence) = &document.state.intelligence {
            atomic_write_json(&session_dir.join("intelligence.json"), intelligence)?;
        }
        let markdown = NoteOrganizer::new(self.storage_base.clone())
            .to_markdown(exported)
            .map_err(|message| MeetingRuntimeError::StorageError {
                message: format!("render meeting archive markdown failed: {message}"),
            })?;
        atomic_write_string(&session_dir.join("export.md"), &markdown)?;

        let item = list_item_from_document(&document);
        self.upsert_index_item(item.clone())?;
        Ok(item)
    }

    pub fn list_sessions(
        &self,
        request: MeetingSessionListRequest,
    ) -> Result<MeetingSessionListResponse, MeetingRuntimeError> {
        let mut diagnostics = Vec::new();
        let mut index = self.read_index_or_rebuild(&mut diagnostics)?;
        index.sessions.sort_by(|left, right| {
            right
                .ended_at
                .cmp(&left.ended_at)
                .then_with(|| right.started_at.cmp(&left.started_at))
        });

        let query = normalize_query(request.query.as_deref().unwrap_or_default());
        let mut sessions = index
            .sessions
            .into_iter()
            .filter(|item| {
                request
                    .date_from
                    .is_none_or(|date_from| item.started_at >= date_from)
                    && request
                        .date_to
                        .is_none_or(|date_to| item.started_at <= date_to)
                    && request
                        .has_intelligence
                        .is_none_or(|expected| item.intelligence_present == expected)
                    && (query.is_empty() || list_item_matches_query(item, &query))
            })
            .collect::<Vec<_>>();

        let limit = bounded_limit(request.limit, MAX_LIST_LIMIT);
        let offset = request
            .cursor
            .as_deref()
            .and_then(|cursor| cursor.parse::<usize>().ok())
            .unwrap_or_default();
        let next_cursor = if offset.saturating_add(limit) < sessions.len() {
            Some(offset.saturating_add(limit).to_string())
        } else {
            None
        };
        sessions = sessions.into_iter().skip(offset).take(limit).collect();

        Ok(MeetingSessionListResponse {
            sessions,
            next_cursor,
            diagnostics,
        })
    }

    pub fn read_session(
        &self,
        request: MeetingSessionReadRequest,
    ) -> Result<MeetingSessionReadResponse, MeetingRuntimeError> {
        let mut archive = self.read_archive(&request.session_id)?;
        if !request.include_transcript {
            archive.state.transcript.clear();
            archive.exported.transcript.clear();
        }
        if !request.include_intelligence {
            archive.state.intelligence = None;
            archive.exported.intelligence = None;
        }
        if !request.include_diagnostics {
            archive.state.diagnostics.clear();
        }
        Ok(MeetingSessionReadResponse {
            archive,
            diagnostics: Vec::new(),
        })
    }

    pub fn search_sessions(
        &self,
        request: MeetingSessionSearchRequest,
    ) -> Result<MeetingSessionSearchResponse, MeetingRuntimeError> {
        let normalized_query = normalize_query(&request.query);
        if normalized_query.is_empty() {
            return Ok(MeetingSessionSearchResponse {
                results: Vec::new(),
                searched_session_count: 0,
                matched_session_count: 0,
                truncated: false,
                corrupt_archive_count: 0,
                diagnostics: vec![diagnostic(
                    "search_query_empty",
                    MeetingDiagnosticSeverity::Warning,
                    "Session memory search query was empty after normalization",
                )],
            });
        }

        let limit = bounded_limit(request.limit, MAX_SEARCH_LIMIT);
        let mut diagnostics = Vec::new();
        let index = self.read_index_or_rebuild(&mut diagnostics)?;
        let mut searched_session_count = 0usize;
        let mut corrupt_archive_count = 0usize;
        let mut matched_sessions = HashSet::new();
        let mut results = Vec::new();

        for item in index.sessions {
            searched_session_count = searched_session_count.saturating_add(1);
            let archive = match self.read_archive(&item.session_id) {
                Ok(archive) => archive,
                Err(_) => {
                    corrupt_archive_count = corrupt_archive_count.saturating_add(1);
                    diagnostics.push(diagnostic(
                        "archive_corrupt",
                        MeetingDiagnosticSeverity::Warning,
                        format!(
                            "Archived meeting session {} could not be read during search",
                            item.session_id
                        ),
                    ));
                    continue;
                }
            };
            let session_results = search_archive(&archive, &normalized_query, limit);
            if !session_results.is_empty() {
                matched_sessions.insert(item.session_id);
            }
            results.extend(session_results);
            if results.len() >= limit {
                break;
            }
        }

        results.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let truncated = results.len() > limit;
        results.truncate(limit);

        Ok(MeetingSessionSearchResponse {
            results,
            searched_session_count,
            matched_session_count: matched_sessions.len(),
            truncated,
            corrupt_archive_count,
            diagnostics,
        })
    }

    pub fn export_session(
        &self,
        request: MeetingSessionExportRequest,
    ) -> Result<MeetingSessionExportResponse, MeetingRuntimeError> {
        let archive = self.read_archive(&request.session_id)?;
        let diagnostics = export_diagnostics(&archive);
        let (filename, content) = match request.format {
            MeetingSessionExportFormat::Json => {
                let content = serde_json::to_string_pretty(&archive).map_err(|error| {
                    MeetingRuntimeError::SerializationError {
                        message: format!("serialize archived meeting export failed: {error}"),
                    }
                })?;
                (format!("{}_session.json", archive.session_id), content)
            }
            MeetingSessionExportFormat::Markdown => {
                let content = NoteOrganizer::new(self.storage_base.clone())
                    .to_markdown(&archive.exported)
                    .map_err(|message| MeetingRuntimeError::StorageError {
                        message: format!("render archived meeting markdown failed: {message}"),
                    })?;
                (format!("{}_session.md", archive.session_id), content)
            }
        };
        Ok(MeetingSessionExportResponse {
            session_id: archive.session_id,
            format: request.format,
            content_length: content.len(),
            filename,
            content,
            diagnostics,
        })
    }

    pub fn rebuild_index(&self) -> Result<MeetingSessionListResponse, MeetingRuntimeError> {
        let mut diagnostics = Vec::new();
        let index = self.rebuild_index_internal(&mut diagnostics)?;
        Ok(MeetingSessionListResponse {
            sessions: index.sessions,
            next_cursor: None,
            diagnostics,
        })
    }

    fn read_archive(
        &self,
        session_id: &str,
    ) -> Result<MeetingSessionArchiveDocument, MeetingRuntimeError> {
        let session_dir = self.session_dir(session_id)?;
        let path = session_dir.join("session.json");
        let bytes = fs::read(&path)
            .map_err(|error| storage_error("read meeting session archive", error))?;
        serde_json::from_slice(&bytes).map_err(|error| MeetingRuntimeError::SerializationError {
            message: format!("parse meeting session archive failed: {error}"),
        })
    }

    fn upsert_index_item(&self, item: MeetingSessionListItem) -> Result<(), MeetingRuntimeError> {
        let mut diagnostics = Vec::new();
        let mut index = self.read_index_or_rebuild(&mut diagnostics)?;
        index
            .sessions
            .retain(|existing| existing.session_id != item.session_id);
        index.sessions.push(item);
        index.sessions.sort_by(|left, right| {
            right
                .ended_at
                .cmp(&left.ended_at)
                .then_with(|| right.started_at.cmp(&left.started_at))
        });
        index.schema_version = INDEX_SCHEMA_VERSION;
        index.updated_at = Some(Utc::now());
        self.write_index(&index)
    }

    fn read_index_or_rebuild(
        &self,
        diagnostics: &mut Vec<MeetingDiagnostic>,
    ) -> Result<MeetingSessionMemoryIndex, MeetingRuntimeError> {
        let path = self.index_path();
        if !path.exists() {
            diagnostics.push(diagnostic(
                "index_rebuilt",
                MeetingDiagnosticSeverity::Info,
                "Meeting session memory index was missing and has been rebuilt",
            ));
            return self.rebuild_index_internal(diagnostics);
        }
        match fs::read(&path)
            .map_err(|error| storage_error("read meeting session memory index", error))
            .and_then(|bytes| {
                serde_json::from_slice::<MeetingSessionMemoryIndex>(&bytes).map_err(|error| {
                    MeetingRuntimeError::SerializationError {
                        message: format!("parse meeting session memory index failed: {error}"),
                    }
                })
            }) {
            Ok(index) if index.schema_version == INDEX_SCHEMA_VERSION => Ok(index),
            Ok(_) | Err(_) => {
                diagnostics.push(diagnostic(
                    "index_rebuilt",
                    MeetingDiagnosticSeverity::Warning,
                    "Meeting session memory index was corrupt or from an older schema and has been rebuilt",
                ));
                self.rebuild_index_internal(diagnostics)
            }
        }
    }

    fn rebuild_index_internal(
        &self,
        diagnostics: &mut Vec<MeetingDiagnostic>,
    ) -> Result<MeetingSessionMemoryIndex, MeetingRuntimeError> {
        let mut sessions = Vec::new();
        let dir = sessions_dir(&self.storage_base);
        fs::create_dir_all(&dir)
            .map_err(|error| storage_error("create meeting sessions archive directory", error))?;
        for entry in fs::read_dir(&dir)
            .map_err(|error| storage_error("scan meeting session archive directory", error))?
        {
            let Ok(entry) = entry else {
                continue;
            };
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let Some(session_id) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            match self.read_archive(session_id) {
                Ok(archive) => sessions.push(list_item_from_document(&archive)),
                Err(_) => diagnostics.push(diagnostic(
                    "archive_corrupt",
                    MeetingDiagnosticSeverity::Warning,
                    format!("Archived meeting session {session_id} could not be indexed"),
                )),
            }
        }
        sessions.sort_by(|left, right| {
            right
                .ended_at
                .cmp(&left.ended_at)
                .then_with(|| right.started_at.cmp(&left.started_at))
        });
        let index = MeetingSessionMemoryIndex {
            schema_version: INDEX_SCHEMA_VERSION,
            updated_at: Some(Utc::now()),
            sessions,
        };
        self.write_index(&index)?;
        Ok(index)
    }

    fn write_index(&self, index: &MeetingSessionMemoryIndex) -> Result<(), MeetingRuntimeError> {
        atomic_write_json(&self.index_path(), index)
    }

    fn index_path(&self) -> PathBuf {
        self.storage_base.join("index.json")
    }

    fn session_dir(&self, session_id: &str) -> Result<PathBuf, MeetingRuntimeError> {
        let session_id = validate_session_id(session_id)?;
        Ok(sessions_dir(&self.storage_base).join(session_id))
    }
}

fn sessions_dir(storage_base: &Path) -> PathBuf {
    storage_base.join("sessions")
}

fn list_item_from_document(document: &MeetingSessionArchiveDocument) -> MeetingSessionListItem {
    let intelligence = document.state.intelligence.as_ref();
    let summary_preview = summary_preview(&document.state);
    let title = session_title(&document.state, &summary_preview);
    let metadata = &document.exported.metadata;
    let drain_status = metadata
        .get("segment_stt_diagnostics")
        .and_then(|value| value.get("drain_status"))
        .and_then(|value| value.as_str())
        .unwrap_or_else(|| {
            document
                .capture_health
                .metrics
                .segment_transcription_drain_status
                .as_deref()
                .unwrap_or("unknown")
        })
        .to_string();
    let fallback_report = derive_meeting_stt_completeness(
        &document.system_capture_health,
        &document.microphone_capture_health,
    );
    let stt_completeness_status = metadata
        .get("stt_completeness")
        .and_then(|value| value.get("overall"))
        .and_then(|value| value.as_str())
        .map(ToString::to_string)
        .unwrap_or_else(|| fallback_report.overall.as_str().to_string());
    let stt_completeness_detail = metadata
        .get("stt_completeness")
        .map(stt_completeness_detail)
        .unwrap_or_else(|| stt_completeness_detail_from_report(&fallback_report));

    MeetingSessionListItem {
        session_id: document.session_id.clone(),
        title,
        platform: document.state.session.platform.clone(),
        session_mode: document.state.session.session_mode,
        started_at: document.exported.started_at,
        ended_at: document.exported.ended_at,
        duration_ms: document
            .exported
            .ended_at
            .signed_duration_since(document.exported.started_at)
            .num_milliseconds()
            .max(0) as u64,
        transcript_count: document.state.transcript.len(),
        intelligence_present: intelligence.is_some(),
        summary_preview,
        action_item_count: document.state.action_items.len()
            + intelligence
                .map(|value| value.action_items.len())
                .unwrap_or_default(),
        decision_count: document.state.decisions.len()
            + intelligence
                .map(|value| value.decisions.len())
                .unwrap_or_default(),
        open_question_count: intelligence
            .map(|value| value.open_questions.len())
            .unwrap_or_default(),
        risk_count: intelligence
            .map(|value| value.risks.len())
            .unwrap_or_default(),
        technical_recap_present: intelligence
            .and_then(|value| value.technical_recap.as_ref())
            .is_some_and(|recap| {
                !recap.bullets.is_empty()
                    || !recap.mentioned_files.is_empty()
                    || !recap.mentioned_commands.is_empty()
                    || !recap.mentioned_errors.is_empty()
            }),
        screen_context_count: document
            .screen_contexts
            .len()
            .max(document.state.screen_contexts.len()),
        speakers_preview: document
            .state
            .speakers
            .iter()
            .map(|speaker| speaker.display_name.clone())
            .filter(|name| !name.trim().is_empty())
            .take(6)
            .collect(),
        capture_sources: capture_sources(&document.state.transcript),
        stt_completeness_status,
        stt_completeness_detail,
        drain_status,
        last_updated_at: document.state.last_updated_at,
    }
}

fn search_archive(
    archive: &MeetingSessionArchiveDocument,
    query: &str,
    limit: usize,
) -> Vec<MeetingSessionSearchResult> {
    let mut results = Vec::new();
    let item = list_item_from_document(archive);

    for entry in &archive.state.transcript {
        maybe_push_result(
            &mut results,
            query,
            SearchCandidate {
                session_id: &archive.session_id,
                session_title: &item.title,
                matched_kind: "transcript",
                title: entry.speaker_display_name().to_string(),
                text: &entry.text,
                evidence_segment_ids: vec![entry.segment_id.clone()],
                speaker_display_name: Some(entry.speaker_display_name().to_string()),
                timestamp_ms: entry.start_ms,
                screen_context_id: None,
            },
        );
        if results.len() >= limit {
            return results;
        }
    }

    for summary in &archive.state.summary {
        maybe_push_result(
            &mut results,
            query,
            SearchCandidate {
                session_id: &archive.session_id,
                session_title: &item.title,
                matched_kind: "summary",
                title: "Summary".to_string(),
                text: &summary.summary,
                evidence_segment_ids: summary.evidence_segment_ids.clone(),
                speaker_display_name: None,
                timestamp_ms: None,
                screen_context_id: None,
            },
        );
    }
    for decision in &archive.state.decisions {
        let text = format!("{} {}", decision.decision, decision.rationale);
        maybe_push_result(
            &mut results,
            query,
            SearchCandidate {
                session_id: &archive.session_id,
                session_title: &item.title,
                matched_kind: "decision",
                title: "Decision".to_string(),
                text: &text,
                evidence_segment_ids: decision.evidence_segment_ids.clone(),
                speaker_display_name: decision.made_by.as_ref().map(|value| value.name.clone()),
                timestamp_ms: None,
                screen_context_id: None,
            },
        );
    }
    for action in &archive.state.action_items {
        let text = format!("{} {}", action.title, action.description);
        maybe_push_result(
            &mut results,
            query,
            SearchCandidate {
                session_id: &archive.session_id,
                session_title: &item.title,
                matched_kind: "action_item",
                title: "Action item".to_string(),
                text: &text,
                evidence_segment_ids: action.evidence_segment_ids.clone(),
                speaker_display_name: action.assignee.as_ref().map(|value| value.name.clone()),
                timestamp_ms: None,
                screen_context_id: None,
            },
        );
    }

    for context in screen_contexts_for_archive(archive) {
        maybe_push_result(
            &mut results,
            query,
            SearchCandidate {
                session_id: &archive.session_id,
                session_title: &item.title,
                matched_kind: "screen_context",
                title: "Screen context".to_string(),
                text: &context.summary,
                evidence_segment_ids: context.linked_transcript_segment_ids.clone(),
                speaker_display_name: None,
                timestamp_ms: Some(context.captured_at.timestamp_millis().max(0) as u64),
                screen_context_id: Some(context.context_id.clone()),
            },
        );
    }

    if let Some(intelligence) = &archive.state.intelligence {
        search_intelligence(&mut results, &item, intelligence, query, limit);
    }

    results.truncate(limit);
    results
}

fn screen_contexts_for_archive(
    archive: &MeetingSessionArchiveDocument,
) -> Vec<&MeetingScreenContext> {
    if !archive.screen_contexts.is_empty() {
        archive.screen_contexts.iter().collect()
    } else {
        archive.state.screen_contexts.iter().collect()
    }
}

fn search_intelligence(
    results: &mut Vec<MeetingSessionSearchResult>,
    item: &MeetingSessionListItem,
    intelligence: &MeetingIntelligenceResult,
    query: &str,
    limit: usize,
) {
    if let Some(summary) = &intelligence.summary {
        maybe_push_result(
            results,
            query,
            SearchCandidate {
                session_id: &intelligence.session_id,
                session_title: &item.title,
                matched_kind: "intelligence_summary",
                title: "Intelligence summary".to_string(),
                text: &summary.text,
                evidence_segment_ids: summary.evidence_segment_ids.clone(),
                speaker_display_name: None,
                timestamp_ms: None,
                screen_context_id: None,
            },
        );
    }
    for decision in &intelligence.decisions {
        let text = format!(
            "{} {}",
            decision.decision,
            decision.rationale.clone().unwrap_or_default()
        );
        maybe_push_result(
            results,
            query,
            SearchCandidate {
                session_id: &intelligence.session_id,
                session_title: &item.title,
                matched_kind: "intelligence_decision",
                title: "Intelligence decision".to_string(),
                text: &text,
                evidence_segment_ids: decision.evidence_segment_ids.clone(),
                speaker_display_name: decision.made_by_display_name.clone(),
                timestamp_ms: None,
                screen_context_id: None,
            },
        );
    }
    for action in &intelligence.action_items {
        maybe_push_result(
            results,
            query,
            SearchCandidate {
                session_id: &intelligence.session_id,
                session_title: &item.title,
                matched_kind: "intelligence_action_item",
                title: "Intelligence action item".to_string(),
                text: &action.task,
                evidence_segment_ids: action.evidence_segment_ids.clone(),
                speaker_display_name: action.assignee_display_name.clone(),
                timestamp_ms: None,
                screen_context_id: None,
            },
        );
    }
    for question in &intelligence.open_questions {
        maybe_push_result(
            results,
            query,
            SearchCandidate {
                session_id: &intelligence.session_id,
                session_title: &item.title,
                matched_kind: "open_question",
                title: "Open question".to_string(),
                text: &question.question,
                evidence_segment_ids: question.evidence_segment_ids.clone(),
                speaker_display_name: question.asked_by_display_name.clone(),
                timestamp_ms: None,
                screen_context_id: None,
            },
        );
    }
    for risk in &intelligence.risks {
        maybe_push_result(
            results,
            query,
            SearchCandidate {
                session_id: &intelligence.session_id,
                session_title: &item.title,
                matched_kind: "risk",
                title: "Risk / blocker".to_string(),
                text: &risk.risk,
                evidence_segment_ids: risk.evidence_segment_ids.clone(),
                speaker_display_name: None,
                timestamp_ms: None,
                screen_context_id: None,
            },
        );
    }
    if let Some(recap) = &intelligence.technical_recap {
        let text = [
            recap.bullets.join(" "),
            recap.mentioned_files.join(" "),
            recap.mentioned_commands.join(" "),
            recap.mentioned_errors.join(" "),
        ]
        .join(" ");
        maybe_push_result(
            results,
            query,
            SearchCandidate {
                session_id: &intelligence.session_id,
                session_title: &item.title,
                matched_kind: "technical_recap",
                title: "Technical recap".to_string(),
                text: &text,
                evidence_segment_ids: recap.evidence_segment_ids.clone(),
                speaker_display_name: None,
                timestamp_ms: None,
                screen_context_id: None,
            },
        );
    }
    if let Some(draft) = &intelligence.follow_up_draft {
        let text = format!("{} {}", draft.subject, draft.body);
        maybe_push_result(
            results,
            query,
            SearchCandidate {
                session_id: &intelligence.session_id,
                session_title: &item.title,
                matched_kind: "follow_up_draft",
                title: "Follow-up draft".to_string(),
                text: &text,
                evidence_segment_ids: draft.evidence_segment_ids.clone(),
                speaker_display_name: None,
                timestamp_ms: None,
                screen_context_id: None,
            },
        );
    }
    for timeline in &intelligence.timeline {
        let text = format!("{} {}", timeline.title, timeline.detail);
        maybe_push_result(
            results,
            query,
            SearchCandidate {
                session_id: &intelligence.session_id,
                session_title: &item.title,
                matched_kind: "timeline",
                title: "Timeline".to_string(),
                text: &text,
                evidence_segment_ids: timeline.evidence_segment_ids.clone(),
                speaker_display_name: timeline.speaker_display_name.clone(),
                timestamp_ms: timeline.timestamp_ms,
                screen_context_id: None,
            },
        );
    }
    results.truncate(limit);
}

struct SearchCandidate<'a> {
    session_id: &'a str,
    session_title: &'a str,
    matched_kind: &'a str,
    title: String,
    text: &'a str,
    evidence_segment_ids: Vec<String>,
    speaker_display_name: Option<String>,
    timestamp_ms: Option<u64>,
    screen_context_id: Option<String>,
}

fn maybe_push_result(
    results: &mut Vec<MeetingSessionSearchResult>,
    query: &str,
    candidate: SearchCandidate<'_>,
) {
    let score = lexical_score(candidate.text, query);
    if score <= 0.0 {
        return;
    }
    results.push(MeetingSessionSearchResult {
        session_id: candidate.session_id.to_string(),
        session_title: candidate.session_title.to_string(),
        matched_kind: candidate.matched_kind.to_string(),
        title: candidate.title,
        snippet: snippet(candidate.text, query),
        score,
        evidence_segment_ids: candidate.evidence_segment_ids,
        speaker_display_name: candidate.speaker_display_name,
        timestamp_ms: candidate.timestamp_ms,
        screen_context_id: candidate.screen_context_id,
    });
}

fn lexical_score(text: &str, query: &str) -> f32 {
    let text = text.to_lowercase();
    let tokens = query.split_whitespace().collect::<Vec<_>>();
    if tokens.is_empty() {
        return 0.0;
    }
    let matches = tokens.iter().filter(|token| text.contains(**token)).count() as f32;
    if matches == 0.0 {
        0.0
    } else {
        matches / tokens.len() as f32 + if text.contains(query) { 1.0 } else { 0.0 }
    }
}

fn snippet(text: &str, query: &str) -> String {
    let compact = compact_text(text);
    let lower = compact.to_lowercase();
    let byte_index = query
        .split_whitespace()
        .find_map(|token| lower.find(token))
        .unwrap_or_default();
    let match_char_index = lower[..byte_index].chars().count();
    let start = match_char_index.saturating_sub(50);
    let chars = compact.chars().collect::<Vec<_>>();
    let end = chars.len().min(start.saturating_add(MAX_SNIPPET_CHARS));
    chars[start..end]
        .iter()
        .collect::<String>()
        .trim()
        .to_string()
}

fn normalize_query(query: &str) -> String {
    compact_text(query)
        .chars()
        .take(MAX_QUERY_CHARS)
        .collect::<String>()
        .to_lowercase()
}

fn list_item_matches_query(item: &MeetingSessionListItem, query: &str) -> bool {
    let haystack = [
        item.title.as_str(),
        item.platform.as_str(),
        item.summary_preview.as_str(),
        &item.speakers_preview.join(" "),
        &item.capture_sources.join(" "),
        item.stt_completeness_status.as_str(),
    ]
    .join(" ")
    .to_lowercase();
    query
        .split_whitespace()
        .all(|token| haystack.contains(token))
}

fn session_title(state: &MeetingSessionState, summary_preview: &str) -> String {
    if !summary_preview.is_empty() {
        return bounded_text(first_sentence(summary_preview), 90);
    }
    format!(
        "Work Session {}",
        state.session.started_at.format("%Y-%m-%d %H:%M")
    )
}

fn summary_preview(state: &MeetingSessionState) -> String {
    if let Some(summary) = state
        .intelligence
        .as_ref()
        .and_then(|intelligence| intelligence.summary.as_ref())
    {
        return bounded_text(&summary.text, MAX_PREVIEW_CHARS);
    }
    if let Some(summary) = state.summary.first() {
        return bounded_text(&summary.summary, MAX_PREVIEW_CHARS);
    }
    bounded_text(
        &state
            .transcript
            .iter()
            .take(2)
            .map(|entry| entry.text.as_str())
            .collect::<Vec<_>>()
            .join(" "),
        MAX_PREVIEW_CHARS,
    )
}

fn first_sentence(text: &str) -> &str {
    text.split(['.', '!', '?']).next().unwrap_or(text).trim()
}

fn capture_sources(transcript: &[TranscriptEntry]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut sources = Vec::new();
    for source in transcript.iter().map(|entry| entry.source.as_str()) {
        if seen.insert(source) {
            sources.push(source.to_string());
        }
    }
    sources
}

fn export_diagnostics(archive: &MeetingSessionArchiveDocument) -> Vec<MeetingDiagnostic> {
    let mut diagnostics = Vec::new();
    let overall = archive
        .exported
        .metadata
        .get("stt_completeness")
        .and_then(|value| value.get("overall"))
        .and_then(|value| value.as_str());
    if overall
        .map(|status| status.starts_with("incomplete_"))
        .unwrap_or_else(|| {
            archive
                .exported
                .metadata
                .get("meeting_segment_transcription_incomplete")
                .and_then(|value| value.as_bool())
                .unwrap_or(false)
        })
    {
        diagnostics.push(diagnostic(
            "export_incomplete_transcription",
            MeetingDiagnosticSeverity::Warning,
            "Exported archived session has captured audio that was not fully transcribed",
        ));
    }
    if matches!(overall, Some("incomplete_drain_timeout"))
        || archive.capture_health.metrics.drain_timeout
    {
        diagnostics.push(diagnostic(
            "stt_drain_incomplete",
            MeetingDiagnosticSeverity::Warning,
            "Segment STT drain timed out for this archived session",
        ));
    }
    diagnostics
}

fn stt_completeness_detail(value: &serde_json::Value) -> String {
    let system = value.get("system_audio");
    let microphone = value.get("microphone");
    let overall = value
        .get("overall")
        .and_then(|status| status.as_str())
        .unwrap_or("unknown");
    format!(
        "{}; system {}; microphone {}",
        overall,
        source_stt_detail(system),
        source_stt_detail(microphone)
    )
}

fn stt_completeness_detail_from_report(report: &MeetingSttCompletenessReport) -> String {
    format!(
        "{}; system {}; microphone {}",
        report.overall.as_str(),
        source_stt_detail_from_source(&report.system_audio),
        source_stt_detail_from_source(&report.microphone)
    )
}

fn source_stt_detail(value: Option<&serde_json::Value>) -> String {
    let Some(value) = value else {
        return "unknown".to_string();
    };
    format!(
        "{} ({}/{} transcribed, {} queued, {} in-flight, {} failed, {} timeout)",
        value
            .get("status")
            .and_then(|status| status.as_str())
            .unwrap_or("unknown"),
        value
            .get("segments_transcribed")
            .and_then(|count| count.as_u64())
            .unwrap_or_default(),
        value
            .get("segments_written")
            .and_then(|count| count.as_u64())
            .unwrap_or_default(),
        value
            .get("current_queue_depth")
            .and_then(|count| count.as_u64())
            .unwrap_or_default(),
        value
            .get("segments_in_flight")
            .and_then(|count| count.as_u64())
            .unwrap_or_default(),
        value
            .get("segments_failed")
            .and_then(|count| count.as_u64())
            .unwrap_or_default(),
        value
            .get("timeouts")
            .and_then(|count| count.as_u64())
            .unwrap_or_default()
    )
}

fn source_stt_detail_from_source(source: &MeetingSttCompletenessSource) -> String {
    format!(
        "{} ({}/{} transcribed, {} queued, {} in-flight, {} failed, {} timeout)",
        source.status.as_str(),
        source.segments_transcribed,
        source.segments_written,
        source.current_queue_depth,
        source.segments_in_flight,
        source.segments_failed,
        source.timeouts
    )
}

fn bounded_limit(limit: usize, max: usize) -> usize {
    if limit == 0 {
        20.min(max)
    } else {
        limit.min(max)
    }
}

fn bounded_text(text: &str, max_chars: usize) -> String {
    let compact = compact_text(text);
    if compact.chars().count() <= max_chars {
        return compact;
    }
    let mut output = compact.chars().take(max_chars).collect::<String>();
    output.push_str("...");
    output
}

fn compact_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn validate_session_id(session_id: &str) -> Result<String, MeetingRuntimeError> {
    let trimmed = session_id.trim();
    if trimmed.is_empty()
        || trimmed.len() > 128
        || !trimmed
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        return Err(MeetingRuntimeError::InvalidConfig {
            message: "invalid archived meeting session id".to_string(),
        });
    }
    Ok(trimmed.to_string())
}

fn atomic_write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), MeetingRuntimeError> {
    let data = serde_json::to_vec_pretty(value).map_err(|error| {
        MeetingRuntimeError::SerializationError {
            message: format!("serialize meeting session memory document failed: {error}"),
        }
    })?;
    atomic_write_bytes(path, &data)
}

fn atomic_write_string(path: &Path, value: &str) -> Result<(), MeetingRuntimeError> {
    atomic_write_bytes(path, value.as_bytes())
}

fn atomic_write_bytes(path: &Path, bytes: &[u8]) -> Result<(), MeetingRuntimeError> {
    let parent = path
        .parent()
        .ok_or_else(|| MeetingRuntimeError::StorageError {
            message: format!("meeting memory path has no parent: {}", path.display()),
        })?;
    fs::create_dir_all(parent)
        .map_err(|error| storage_error("create meeting session memory directory", error))?;
    let tmp_path = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("meeting-memory"),
        Uuid::new_v4()
    ));
    fs::write(&tmp_path, bytes)
        .map_err(|error| storage_error("write temporary meeting session memory file", error))?;
    fs::rename(&tmp_path, path)
        .map_err(|error| storage_error("commit meeting session memory file", error))
}

fn diagnostic(
    code: impl Into<String>,
    severity: MeetingDiagnosticSeverity,
    message: impl Into<String>,
) -> MeetingDiagnostic {
    MeetingDiagnostic {
        code: code.into(),
        severity,
        message: message.into(),
        created_at: Utc::now(),
    }
}

fn storage_error(operation: &str, error: std::io::Error) -> MeetingRuntimeError {
    MeetingRuntimeError::StorageError {
        message: format!("{operation} failed: {}", error.kind()),
    }
}
