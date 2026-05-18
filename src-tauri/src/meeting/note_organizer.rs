//! Meeting notes organizer — save structured meeting data to disk
//!
//! Handles export of meeting sessions to JSON, Markdown, and CSV formats.
//! Saves files in ~/.astra/meetings/{session_id}/ for organized storage.

use super::types::*;
use chrono::Utc;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// The organizer module handles saving meeting data to disk.
#[derive(Debug, Clone)]
pub struct NoteOrganizer {
    pub storage_base: PathBuf,
}

impl NoteOrganizer {
    pub fn new(storage_base: PathBuf) -> Self {
        let _ = std::fs::create_dir_all(&storage_base);
        NoteOrganizer { storage_base }
    }

    /// Save all meeting data for a session to the storage directory.
    pub fn save_meeting_data(
        &self,
        exported: &ExportedMeeting,
    ) -> Result<Vec<MeetingFile>, String> {
        let session_dir = self.storage_base.join(&exported.session_id);
        fs::create_dir_all(&session_dir).map_err(|e| e.to_string())?;

        let mut files = Vec::new();

        // 1. Save JSON export (structured)
        let json_content = match serde_json::to_string_pretty(exported) {
            Ok(content) => content,
            Err(e) => return Err(format!("Failed to serialize JSON: {}", e)),
        };
        let json_file = session_dir.join(format!("{}.json", exported.session_id.clone()));
        let mut f1 = fs::File::create(&json_file).map_err(|e| e.to_string())?;
        f1.write_all(json_content.as_bytes())
            .map_err(|e| e.to_string())?;
        f1.flush().map_err(|e| e.to_string())?;
        files.push(MeetingFile {
            path: json_file.to_string_lossy().to_string(),
            filename: file_name_string(&json_file)?,
            size: json_file.metadata().map_err(|e| e.to_string())?.len(),
            checksum: Self::sha256_hex(&json_content),
        });

        // 2. Save Markdown export (readable)
        let markdown = self.to_markdown(exported).map_err(|e| e.to_string())?;
        let md_file = session_dir.join(format!("{}_transcript.md", exported.session_id));
        let mut f2 = fs::File::create(&md_file).map_err(|e| e.to_string())?;
        f2.write_all(markdown.as_bytes())
            .map_err(|e| e.to_string())?;
        f2.flush().map_err(|e| e.to_string())?;
        files.push(MeetingFile {
            path: md_file.to_string_lossy().to_string(),
            filename: file_name_string(&md_file)?,
            size: md_file.metadata().map_err(|e| e.to_string())?.len(),
            checksum: Self::sha256_hex(&markdown),
        });

        // 3. Save CSV for action items (spreadsheet import)
        let csv = self.to_csv(exported).map_err(|e| e.to_string())?;
        let csv_file = session_dir.join("action_items.csv");
        let mut f3 = fs::File::create(&csv_file).map_err(|e| e.to_string())?;
        f3.write_all(csv.as_bytes()).map_err(|e| e.to_string())?;
        f3.flush().map_err(|e| e.to_string())?;
        files.push(MeetingFile {
            path: csv_file.to_string_lossy().to_string(),
            filename: file_name_string(&csv_file)?,
            size: csv_file.metadata().map_err(|e| e.to_string())?.len(),
            checksum: Self::sha256_hex(&csv),
        });

        // 4. Save action items only
        let action_csv = self
            .action_items_to_csv(exported)
            .map_err(|e| e.to_string())?;
        let action_file = session_dir.join("actions.csv");
        let mut f4 = fs::File::create(&action_file).map_err(|e| e.to_string())?;
        f4.write_all(action_csv.as_bytes())
            .map_err(|e| e.to_string())?;
        f4.flush().map_err(|e| e.to_string())?;
        files.push(MeetingFile {
            path: action_file.to_string_lossy().to_string(),
            filename: file_name_string(&action_file)?,
            size: action_file.metadata().map_err(|e| e.to_string())?.len(),
            checksum: Self::sha256_hex(&action_csv),
        });

        Ok(files)
    }

    /// Convert exported meeting to Markdown text
    pub fn to_markdown(&self, exported: &ExportedMeeting) -> Result<String, String> {
        let mut md = String::new();

        md.push_str(&format!(
            "# Meeting Session Recap\n\nSession: `{}`\n\n",
            exported.session_id
        ));
        md.push_str("## Metadata\n");
        md.push_str(&format!("- Platform: {}\n", exported.platform));
        md.push_str(&format!(
            "- Started: {}\n",
            exported.started_at.format("%Y-%m-%dT%H:%M:%SZ")
        ));
        md.push_str(&format!(
            "- Ended: {}\n",
            exported.ended_at.format("%Y-%m-%dT%H:%M:%SZ")
        ));
        if let Some(session_mode) = exported.metadata.get("session_mode") {
            md.push_str(&format!("- Session mode: {}\n", session_mode));
        }
        if let Some(completeness) = exported.metadata.get("stt_completeness") {
            md.push_str(&format!(
                "- STT completeness: {}\n",
                stt_status_markdown_label(
                    completeness
                        .get("overall")
                        .and_then(|value| value.as_str())
                        .unwrap_or("unknown")
                )
            ));
            md.push_str(&format!(
                "- System audio STT: {}\n",
                stt_source_markdown_label(completeness.get("system_audio"))
            ));
            md.push_str(&format!(
                "- Microphone STT: {}\n",
                stt_source_markdown_label(completeness.get("microphone"))
            ));
        } else if let Some(status) = exported
            .metadata
            .get("meeting_segment_transcription_incomplete")
            .and_then(|value| value.as_bool())
        {
            md.push_str(&format!(
                "- STT completeness: {}\n",
                if status { "incomplete" } else { "complete" }
            ));
        }
        if let Some(diagnostics) = exported.metadata.get("segment_stt_diagnostics") {
            md.push_str(&format!("- Segment STT diagnostics: `{}`\n", diagnostics));
        }
        md.push('\n');

        if exported
            .metadata
            .get("meeting_segment_transcription_incomplete")
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
        {
            md.push_str(
                "> Warning: STT drain timed out or captured segments were not fully transcribed before export. The saved transcript may be incomplete.\n\n",
            );
        }

        if !exported.participants.is_empty() {
            md.push_str("## Participants\n");
            for p in &exported.participants {
                md.push_str(&format!("- {}\n", p.name));
            }
            md.push('\n');
        }

        md.push_str("## Summary\n");
        if let Some(intelligence) = &exported.intelligence {
            if let Some(summary) = &intelligence.summary {
                md.push_str(&summary.text);
                md.push_str("\n\n");
                for bullet in &summary.bullets {
                    md.push_str(&format!("- {}\n", bullet));
                }
                if !summary.evidence_segment_ids.is_empty() {
                    md.push_str(&format!(
                        "\nEvidence: {}\n",
                        summary.evidence_segment_ids.join(", ")
                    ));
                }
            }
        }
        if exported
            .intelligence
            .as_ref()
            .and_then(|value| value.summary.as_ref())
            .is_none()
        {
            for entry in &exported.summary {
                md.push_str(&format!(
                    "- [{}] {}\n",
                    entry.timestamp.format("%H:%M:%S"),
                    entry.summary
                ));
            }
            if exported.summary.is_empty() {
                md.push_str("_No summary available._\n");
            }
        }

        md.push_str("\n## Decisions\n");
        let mut decision_no = 0;
        if let Some(intelligence) = &exported.intelligence {
            for decision in &intelligence.decisions {
                decision_no += 1;
                md.push_str(&format!("{}. {}\n", decision_no, decision.decision));
                if let Some(rationale) = &decision.rationale {
                    md.push_str(&format!("   - Rationale: {}\n", rationale));
                }
                if let Some(speaker) = &decision.made_by_display_name {
                    md.push_str(&format!("   - By: {}\n", speaker));
                }
                if !decision.evidence_segment_ids.is_empty() {
                    md.push_str(&format!(
                        "   - Evidence: {}\n",
                        decision.evidence_segment_ids.join(", ")
                    ));
                }
            }
        }
        for decision in &exported.decisions {
            decision_no += 1;
            md.push_str(&format!("{}. {}\n", decision_no, decision.decision));
            if !decision.rationale.is_empty() {
                md.push_str(&format!("   - Rationale: {}\n", decision.rationale));
            }
            if !decision.evidence_segment_ids.is_empty() {
                md.push_str(&format!(
                    "   - Evidence: {}\n",
                    decision.evidence_segment_ids.join(", ")
                ));
            }
        }
        if decision_no == 0 {
            md.push_str("_No evidence-backed decisions recorded._\n");
        }

        md.push_str("\n## Action Items\n");
        let mut item_no = 0;
        if let Some(intelligence) = &exported.intelligence {
            for item in &intelligence.action_items {
                item_no += 1;
                md.push_str(&format!("{}. {}\n", item_no, item.task));
                md.push_str(&format!(
                    "   - Assignee: {}\n",
                    item.assignee_display_name
                        .as_deref()
                        .unwrap_or("not detected")
                ));
                if let Some(due) = &item.due_date {
                    md.push_str(&format!("   - Due: {}\n", due));
                }
                md.push_str(&format!("   - Status: {}\n", item.status));
                if !item.evidence_segment_ids.is_empty() {
                    md.push_str(&format!(
                        "   - Evidence: {}\n",
                        item.evidence_segment_ids.join(", ")
                    ));
                }
            }
        }
        for item in &exported.action_items {
            item_no += 1;
            let assignee = item
                .assignee
                .as_ref()
                .map(|p| p.name.clone())
                .unwrap_or("tbd".to_string());
            let deadline = item.deadline.map(|d| d.format("%Y-%m-%d").to_string());
            md.push_str(&format!(
                "{}. {}\n   - Assignee: {}{} {}{}\n",
                item_no,
                item.description,
                assignee,
                deadline
                    .map(|d| format!(" (deadline: {})", d))
                    .unwrap_or_default(),
                if item.status == ActionItemStatus::Closed {
                    " (closed)"
                } else {
                    ""
                },
                if item.evidence_segment_ids.is_empty() {
                    String::new()
                } else {
                    format!("\n   - Evidence: {}", item.evidence_segment_ids.join(", "))
                },
            ));
        }
        if item_no == 0 {
            md.push_str("_No evidence-backed action items recorded._\n");
        }

        if let Some(intelligence) = &exported.intelligence {
            md.push_str("\n## Open Questions\n");
            if !intelligence.open_questions.is_empty() {
                for question in &intelligence.open_questions {
                    md.push_str(&format!("- {}\n", question.question));
                    if !question.evidence_segment_ids.is_empty() {
                        md.push_str(&format!(
                            "  - Evidence: {}\n",
                            question.evidence_segment_ids.join(", ")
                        ));
                    }
                }
            } else {
                md.push_str("_No open questions detected._\n");
            }

            md.push_str("\n## Risks / Blockers\n");
            if !intelligence.risks.is_empty() {
                for risk in &intelligence.risks {
                    md.push_str(&format!("- {:?}: {}\n", risk.severity, risk.risk));
                    if !risk.evidence_segment_ids.is_empty() {
                        md.push_str(&format!(
                            "  - Evidence: {}\n",
                            risk.evidence_segment_ids.join(", ")
                        ));
                    }
                }
            } else {
                md.push_str("_No grounded risks detected._\n");
            }

            md.push_str("\n## Technical Recap\n");
            if let Some(recap) = &intelligence.technical_recap {
                for bullet in &recap.bullets {
                    md.push_str(&format!("- {}\n", bullet));
                }
                if !recap.mentioned_files.is_empty() {
                    md.push_str(&format!(
                        "- Files/modules: {}\n",
                        recap.mentioned_files.join(", ")
                    ));
                }
                if !recap.mentioned_commands.is_empty() {
                    md.push_str(&format!(
                        "- Commands: {}\n",
                        recap.mentioned_commands.join(", ")
                    ));
                }
                if !recap.mentioned_errors.is_empty() {
                    md.push_str(&format!(
                        "- Errors: {}\n",
                        recap.mentioned_errors.join(", ")
                    ));
                }
                if !recap.evidence_segment_ids.is_empty() {
                    md.push_str(&format!(
                        "- Evidence: {}\n",
                        recap.evidence_segment_ids.join(", ")
                    ));
                }
            } else {
                md.push_str("_No grounded technical details detected._\n");
            }

            md.push_str("\n## Timeline\n");
            if intelligence.timeline.is_empty() {
                md.push_str("_No timeline generated._\n");
            }
            for item in &intelligence.timeline {
                md.push_str(&format!(
                    "- {} {}: {}\n",
                    item.timestamp_ms
                        .map(|value| format!("{}ms", value))
                        .unwrap_or_else(|| "time_unknown".to_string()),
                    item.speaker_display_name.as_deref().unwrap_or("Unknown"),
                    if item.detail.is_empty() {
                        &item.title
                    } else {
                        &item.detail
                    }
                ));
                if !item.evidence_segment_ids.is_empty() {
                    md.push_str(&format!(
                        "  - Evidence: {}\n",
                        item.evidence_segment_ids.join(", ")
                    ));
                }
            }

            if let Some(draft) = &intelligence.follow_up_draft {
                md.push_str("\n## Follow-up Draft\n");
                md.push_str(&format!("Subject: {}\n\n{}\n", draft.subject, draft.body));
                if !draft.evidence_segment_ids.is_empty() {
                    md.push_str(&format!(
                        "\nEvidence: {}\n",
                        draft.evidence_segment_ids.join(", ")
                    ));
                }
            }
        }

        md.push_str("\n## Screen Context\n");
        if exported.screen_contexts.is_empty() {
            md.push_str("_No screen context attachments were saved for this session._\n");
        }
        for context in &exported.screen_contexts {
            md.push_str(&format!(
                "- `{}` at {}: {}\n",
                context.context_id,
                context.captured_at.format("%Y-%m-%dT%H:%M:%SZ"),
                context.summary
            ));
            md.push_str(&format!(
                "  - Linked transcript segments: {}\n",
                if context.linked_transcript_segment_ids.is_empty() {
                    "none".to_string()
                } else {
                    context.linked_transcript_segment_ids.join(", ")
                }
            ));
            md.push_str(&format!(
                "  - Screenshot: {}\n",
                if context.screenshot_ref.is_some() {
                    "stored"
                } else {
                    "not stored"
                }
            ));
            if !context.diagnostics.is_empty() {
                md.push_str(&format!(
                    "  - Diagnostics: {}\n",
                    context
                        .diagnostics
                        .iter()
                        .map(|diagnostic| diagnostic.code.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }

        md.push_str("\n## Transcript\n");
        if exported.transcript.is_empty() {
            md.push_str("_No transcript entries were available at export time._\n");
        }
        for entry in &exported.transcript {
            md.push_str(&format!(
                "- `{}` [{}] {}: {}\n",
                entry.segment_id,
                entry.source.as_str(),
                entry.speaker_display_name(),
                entry.text
            ));
        }

        md.push_str("\n## Diagnostics\n");
        if let Some(speakers) = exported.metadata.get("speakers") {
            md.push_str(&format!("- Speakers: `{}`\n", speakers));
        }
        if let Some(intelligence) = &exported.intelligence {
            md.push_str(&format!(
                "- Intelligence: {:?}; generator={:?}; fallback_used={}; audit_redacted={}\n",
                intelligence.status,
                intelligence.diagnostics.generator,
                intelligence.diagnostics.fallback_used,
                intelligence.diagnostics.audit_redacted
            ));
        }

        Ok(md)
    }

    /// Convert all participants to CSV
    pub fn to_csv(&self, exported: &ExportedMeeting) -> Result<String, String> {
        let mut csv = "speaker,source,segment_id,timestamp,text\n".to_string();
        for entry in &exported.transcript {
            let text = entry.text.replace('"', "").replace('\n', " ");
            // Escape CSV fields
            csv.push_str(&format!(
                "\"{}\",{},\"{}\",{},\"{}\"\n",
                entry.speaker_display_name().replace('"', "\"\""),
                entry.source.as_str(),
                entry.segment_id.replace('"', "\"\""),
                entry.timestamp.format("%Y-%m-%dT%H:%M:%S"),
                text.replace('"', "\"\"")
            ));
        }
        csv.push('\n');

        csv.push_str("timestamp,description,assignee,status\n");
        for item in &exported.action_items {
            let assignee = item
                .assignee
                .as_ref()
                .map(|p| p.name.clone())
                .unwrap_or("tbd".to_string());
            csv.push_str(&format!(
                "{},{},{},{}\n",
                item.timestamp.format("%Y-%m-%dT%H:%M:%S"),
                item.description.replace('"', ""),
                assignee.replace('"', ""),
                item.status
            ));
        }
        csv.push('\n');

        Ok(csv)
    }

    /// Convert action items only to CSV
    pub fn action_items_to_csv(&self, exported: &ExportedMeeting) -> Result<String, String> {
        let mut csv = "timestamp,description,assignee,deadline,status\n".to_string();
        for item in &exported.action_items {
            let assignee = item
                .assignee
                .as_ref()
                .map(|p| p.name.clone())
                .unwrap_or("tbd".to_string());
            _ = item
                .deadline
                .map(|d| d.format("%Y-%m-%d").to_string())
                .unwrap_or("tbd".to_string());
            csv.push_str(&format!(
                "{},{},{},,{}\n",
                item.timestamp.format("%Y-%m-%dT%H:%M:%S"),
                item.description.replace('"', ""),
                assignee.replace('"', ""),
                item.status
            ));
        }
        Ok(csv)
    }

    /// Export meeting data as a tar.gz archive to a temp path
    pub fn export_as_archive(&self, exported: &ExportedMeeting) -> Result<String, String> {
        // MVP: just save all files to path and return path
        let files = self
            .save_meeting_data(exported)
            .map_err(|e| e.to_string())?;
        let path = files
            .first()
            .map(|f| f.path.clone())
            .unwrap_or(String::new());
        Ok(path)
    }

    /// Clean up meeting files older than retention policy
    pub fn apply_retention_policy(&self, policy: &DataRetentionPolicy) -> Result<(), String> {
        let now = Utc::now();
        // In a real implementation, this would:
        // 1. List directories in .astra/meetings/
        // 2. Check modified date
        // 3. Delete raw audio after 24h
        // 4. Delete transcript after 30d
        // 5. Keep summary + action items for 90d
        // 6. Keep action items indefinitely

        let raw_audio_days = policy.raw_audio_days as i64;
        let transcript_days = policy.transcript_days as i64;

        if raw_audio_days > 0 {
            let _cutoff = now - chrono::Duration::days(raw_audio_days);
            // TODO: Iterate .astra/meetings/ and delete raw audio older than cutoff
        }

        if transcript_days > 0 {
            let _cutoff = now - chrono::Duration::days(transcript_days);
            // TODO: Iterate .astra/meetings/ and delete transcript entries older than cutoff
        }

        Ok(())
    }

    pub fn preview_clear_all_meeting_data(&self) -> Result<MeetingDataClearPreview, String> {
        Ok(MeetingDataClearPreview {
            scope: MeetingClearScope::All,
            runtime_state_present: false,
            persisted_entries: self.storage_entry_count()?,
            storage_path: self.storage_base.to_string_lossy().to_string(),
        })
    }

    pub fn clear_all_meeting_data(&self) -> Result<MeetingDataClearResult, String> {
        fs::create_dir_all(&self.storage_base).map_err(|e| e.to_string())?;
        let mut removed = 0usize;
        for entry in fs::read_dir(&self.storage_base).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.is_dir() {
                fs::remove_dir_all(&path).map_err(|e| e.to_string())?;
            } else {
                fs::remove_file(&path).map_err(|e| e.to_string())?;
            }
            removed += 1;
        }
        Ok(MeetingDataClearResult {
            runtime_state_cleared: true,
            persisted_entries_removed: removed,
            storage_path: self.storage_base.to_string_lossy().to_string(),
            capture_stop_attempted: false,
            capture_stop_succeeded: false,
            capture_stop_error_kind: None,
            clear_aborted: false,
        })
    }

    pub fn sha256_hex(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn storage_entry_count(&self) -> Result<usize, String> {
        if !self.storage_base.exists() {
            return Ok(0);
        }
        fs::read_dir(&self.storage_base)
            .map_err(|e| e.to_string())?
            .try_fold(0usize, |count, entry| {
                entry.map_err(|e| e.to_string()).map(|_| count + 1)
            })
    }
}

fn file_name_string(path: &Path) -> Result<String, String> {
    path.file_name()
        .map(|name| name.to_string_lossy().to_string())
        .ok_or_else(|| format!("Path has no file name: {}", path.display()))
}

fn stt_status_markdown_label(status: &str) -> String {
    match status {
        "complete" => "complete".to_string(),
        "complete_no_speech" => "complete/no speech".to_string(),
        "incomplete_drain_timeout" => "incomplete (drain timed out)".to_string(),
        "incomplete_pending_queue" => "incomplete (pending queue)".to_string(),
        "incomplete_in_flight" => "incomplete (segment still in flight)".to_string(),
        "incomplete_failed_segments" => "incomplete (failed segments)".to_string(),
        "incomplete_timeouts" => "incomplete (STT timeouts)".to_string(),
        "unavailable" => "unavailable".to_string(),
        "unknown" => "unknown".to_string(),
        other => other.replace('_', " "),
    }
}

fn stt_source_markdown_label(source: Option<&serde_json::Value>) -> String {
    let Some(source) = source else {
        return "unknown".to_string();
    };
    let status = source
        .get("status")
        .and_then(|value| value.as_str())
        .unwrap_or("unknown");
    let written = source
        .get("segments_written")
        .and_then(|value| value.as_u64())
        .unwrap_or_default();
    let transcribed = source
        .get("segments_transcribed")
        .and_then(|value| value.as_u64())
        .unwrap_or_default();
    let queued = source
        .get("current_queue_depth")
        .and_then(|value| value.as_u64())
        .unwrap_or_default();
    let in_flight = source
        .get("segments_in_flight")
        .and_then(|value| value.as_u64())
        .unwrap_or_default();
    let failed = source
        .get("segments_failed")
        .and_then(|value| value.as_u64())
        .unwrap_or_default();
    let timeouts = source
        .get("timeouts")
        .and_then(|value| value.as_u64())
        .unwrap_or_default();
    let silence = source
        .get("dropped_silence_segments")
        .and_then(|value| value.as_u64())
        .unwrap_or_default();
    format!(
        "{} ({}/{} transcribed, {} queued, {} in-flight, {} failed, {} timeout, {} silence dropped)",
        stt_status_markdown_label(status),
        transcribed,
        written,
        queued,
        in_flight,
        failed,
        timeouts,
        silence
    )
}

#[cfg(test)]
mod tests {
    use super::NoteOrganizer;

    #[test]
    fn exported_meeting_uses_real_sha256() {
        assert_eq!(
            NoteOrganizer::sha256_hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
