//! Action item tracker — extract and track action items from meeting transcript
//!
//! Scans transcript entries for action-oriented language and extracts structured
//! action items with assignees and inferred deadlines.

use super::types::*;
use chrono::Utc;

/// Tracks action items extracted from the meeting transcript.
pub struct ActionItemTracker {
    /// All extracted action items (accumulated during the meeting)
    items: Vec<ActionItem>,
    /// Keywords that indicate an action item
    action_keywords: Vec<String>,
    // Patterns for extracting assignees (names that look like names or @mentions)
}

impl ActionItemTracker {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            action_keywords: vec![
                "action item".to_string(),
                "will do".to_string(),
                "i'll".to_string(),
                "let me".to_string(),
                "please check".to_string(),
                "follow up".to_string(),
                "assign to".to_string(),
                "deadline".to_string(),
                "by this week".to_string(),
                "next sprint".to_string(),
                "need to".to_string(),
                "should".to_string(),
            ],
        }
    }

    /// Scan the transcript for action items and extract them.
    pub fn track(&mut self, recent_entries: &[TranscriptEntry]) -> Vec<ActionItem> {
        let mut new_items = Vec::new();

        for entry in recent_entries {
            let text_lower = entry.text.to_lowercase();
            let is_action = self
                .action_keywords
                .iter()
                .any(|kw| text_lower.contains(kw));

            // Also check for imperative verb patterns: "X please do Y"
            let has_verb = entry.text.contains("please")
                || entry.text.contains("need")
                || entry.text.contains("should")
                || entry.text.contains("must")
                || entry.text.contains("let's");

            if is_action || has_verb {
                let assignee =
                    ActionItemTracker::extract_assignee(&entry.text, entry.speaker.clone());
                let deadline = ActionItemTracker::extract_deadline(&entry.text);
                let status = ActionItemStatus::Open;
                let now = Utc::now();

                let item = ActionItem {
                    id: new_meeting_artifact_id(),
                    session_id: entry.session_id.clone(),
                    timestamp: now,
                    created_at: now,
                    title: entry.text.chars().take(80).collect(),
                    description: entry.text.clone(),
                    assignee,
                    deadline,
                    status,
                    evidence_segment_ids: vec![entry.segment_id.clone()],
                };

                // Avoid duplicate descriptions (if not already tracked)
                if !self
                    .items
                    .iter()
                    .any(|ex| ex.description == item.description)
                {
                    self.items.push(item.clone());
                    new_items.push(item);
                }
            }
        }

        new_items
    }

    /// Get all tracked action items
    pub fn get_items(&self) -> &[ActionItem] {
        &self.items
    }

    /// Clear all action items
    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Extract assignee from text and current speaker as fallback
    fn extract_assignee(text: &str, default_speaker: String) -> Option<ParticipantInfo> {
        // Try to find @mention or name pattern
        if text.contains('@') {
            let parts: Vec<&str> = text.split('@').collect();
            for part in &parts[1..] {
                let name: String = part
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '.')
                    .collect();
                if !name.is_empty() {
                    return Some(ParticipantInfo {
                        name,
                        speaker_id: None,
                    });
                }
            }
        }
        Some(ParticipantInfo {
            name: default_speaker,
            speaker_id: None,
        })
    }

    /// Extract deadline from text if present
    fn extract_deadline(text: &str) -> Option<chrono::DateTime<Utc>> {
        let lower = text.to_lowercase();
        if lower.contains("next sprint") || lower.contains("next week") {
            let now = Utc::now();
            Some(now + chrono::Duration::days(7))
        } else if lower.contains("by tomorrow")
            || lower.contains("today")
            || lower.contains("tonight")
        {
            Some(Utc::now() + chrono::Duration::days(1))
        } else if lower.contains("by this week") {
            let now = Utc::now();
            Some(now + chrono::Duration::days(7))
        } else {
            None
        }
    }
}

impl Default for ActionItemTracker {
    fn default() -> Self {
        Self::new()
    }
}
