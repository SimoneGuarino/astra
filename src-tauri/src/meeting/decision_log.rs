//! Decision log — extract and track decisions made during the meeting
//!
//! Scans transcript entries for decision-indicative language and logs structured
//! decisions with context, rationale, and decision maker.

use super::types::*;
use chrono::Utc;

/// Manages the decision log during a meeting.
pub struct DecisionLog {
    /// All decisions extracted from the meeting
    entries: Vec<DecisionLogEntry>,
    /// Keywords/patterns that indicate a decision was made
    decision_patterns: Vec<&'static str>,
}

impl DecisionLog {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            decision_patterns: vec![
                "decide",
                "decision",
                "agreed",
                "let's go with",
                "we'll",
                "we need to",
                "should we do",
                "we choose",
                "the plan is",
                "the decision is",
                "finalized",
                "approved",
                "resolved",
                "confirmed",
                "settled on",
                "consensus",
                "moving forward with",
                "go ahead with",
                "pick",
            ],
        }
    }

    /// Scan transcript entries and extract decisions.
    pub fn track(&mut self, recent_entries: &[TranscriptEntry]) -> Vec<DecisionLogEntry> {
        let mut new_entries = Vec::new();

        for entry in recent_entries {
            let text_lower = entry.text.to_lowercase();
            let is_decision = self
                .decision_patterns
                .iter()
                .any(|pattern| text_lower.contains(pattern));

            if is_decision {
                let now = Utc::now();
                let entry = DecisionLogEntry {
                    id: new_meeting_artifact_id(),
                    session_id: entry.session_id.clone(),
                    timestamp: now,
                    created_at: now,
                    decision: entry.text.clone(),
                    rationale: Self::extract_rationale(&entry.text),
                    made_by: Some(ParticipantInfo {
                        name: entry.speaker.clone(),
                        speaker_id: entry.speaker_id.clone(),
                    }),
                    evidence_segment_ids: vec![entry.segment_id.clone()],
                };

                // Avoid duplicate decisions (exact match on text)
                let already_exists = self.entries.iter().any(|ex| ex.decision == entry.decision);

                if !already_exists {
                    self.entries.push(entry.clone());
                    new_entries.push(entry);
                }
            }
        }

        new_entries
    }

    /// Get all decisions from the meeting
    pub fn get_entries(&self) -> &[DecisionLogEntry] {
        &self.entries
    }

    /// Get decisions filtered by status/context
    pub fn get_by_status(&self, _status: DecisionStatus) -> Vec<&DecisionLogEntry> {
        self.entries.iter().filter(|_| true).collect()
    }

    /// Clear all decisions
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Extract rationale from the decision text — this is a simplification
    /// In production, this would use a local LLM or rule-based extraction.
    fn extract_rationale(text: &str) -> String {
        // Simple rationale extraction:
        // Look for "because", "since", "so that", "to", "for" patterns
        let lower = text.to_lowercase();

        if let Some(rationale) = lower.find("because").map(|i| text[i + 9..].trim()) {
            rationale.to_string()
        } else if let Some(rationale) = lower.find("since").map(|i| text[i + 5..].trim()) {
            rationale.to_string()
        } else if let Some(rationale) = lower.find("for").map(|i| text[i + 3..].trim()) {
            rationale.to_string()
        } else {
            "No explicit rationale found. See transcript for context.".to_string()
        }
    }
}

impl Default for DecisionLog {
    fn default() -> Self {
        Self::new()
    }
}

/// Optional: status for decisions
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionStatus {
    Open,
    InProgress,
    Closed,
    Overruled,
}
