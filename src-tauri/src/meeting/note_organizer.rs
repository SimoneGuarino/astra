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
            "# Meeting: {} ({})\n\n",
            exported.platform, exported.session_id
        ));
        md.push_str(&format!(
            "Started: {}\n",
            exported.started_at.format("%Y-%m-%dT%H:%M:%SZ")
        ));
        md.push_str(&format!(
            "Ended: {}\n\n",
            exported.ended_at.format("%Y-%m-%dT%H:%M:%SZ")
        ));

        if !exported.participants.is_empty() {
            md.push_str("## Participants\n");
            for p in &exported.participants {
                md.push_str(&format!("- {}\n", p.name));
            }
            md.push('\n');
        }

        md.push_str("## Transcript\n");
        for entry in &exported.transcript {
            md.push_str(&format!(
                "{} [{}] ({}): {}\n",
                entry.speaker,
                entry.source.as_str(),
                entry.timestamp.format("%H:%M:%S"),
                entry.text
            ));
        }

        md.push_str("\n## Summary\n");
        for entry in &exported.summary {
            md.push_str(&format!(
                "[{}] {}\n",
                entry.timestamp.format("%H:%M:%S"),
                entry.summary
            ));
        }

        md.push_str("\n## Decisions\n");
        let mut decision_no = 0;
        for decision in &exported.decisions {
            decision_no += 1;
            md.push_str(&format!("\nD{}: {}\n", decision_no, decision.decision));
            if !decision.rationale.is_empty() {
                md.push_str(&format!("Rationale: {}\n", decision.rationale));
            }
            if !decision.evidence_segment_ids.is_empty() {
                md.push_str(&format!(
                    "Evidence: {}\n",
                    decision.evidence_segment_ids.join(", ")
                ));
            }
        }

        md.push_str("\n## Action items\n");
        let mut item_no = 0;
        for item in &exported.action_items {
            item_no += 1;
            let assignee = item
                .assignee
                .as_ref()
                .map(|p| p.name.clone())
                .unwrap_or("tbd".to_string());
            let deadline = item.deadline.map(|d| d.format("%Y-%m-%d").to_string());
            md.push_str(&format!(
                "\n{}- {}\n  Assignee: {}{} {}{}\n",
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
                    format!("\n  Evidence: {}", item.evidence_segment_ids.join(", "))
                },
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
                entry.speaker.replace('"', "\"\""),
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
