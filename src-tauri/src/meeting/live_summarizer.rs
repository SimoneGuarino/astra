//! Live summarizer — rolling 30-second summary updates during calls
//!
//! Generates rolling summaries of recent transcript segments to provide
//! real-time context during meetings.

use super::types::*;
use chrono::Utc;

pub struct LiveSummarizer {
    /// How often to generate a new summary (in seconds between updates)
    summary_interval: u64,
    /// Time of last summary
    last_summary_time: Option<chrono::DateTime<Utc>>,
    /// How many recent transcript entries to consider
    window_size: usize,
}

impl LiveSummarizer {
    pub fn new(summary_interval_secs: u64, window_size: usize) -> Self {
        Self {
            summary_interval: summary_interval_secs,
            last_summary_time: None,
            window_size,
        }
    }

    /// Check if it's time to generate a new summary
    pub fn should_summarize(&self) -> bool {
        match self.last_summary_time {
            Some(t) => {
                let elapsed = (Utc::now() - t).num_seconds();
                elapsed >= self.summary_interval as i64
            }
            None => true, // First summary should always happen
        }
    }

    /// Generate a summarized update from the recent transcript
    pub fn generate(&mut self, transcript: &[TranscriptEntry]) -> SummaryEntry {
        // Get recent transcript entries within window
        let recent = if self.window_size == 0 {
            transcript
        } else {
            let start = if transcript.len() > self.window_size {
                transcript.len() - self.window_size
            } else {
                0
            };
            &transcript[start..]
        };

        let summary_text = if recent.is_empty() {
            "No transcript data yet.".to_string()
        } else {
            // MVP: aggregate speaker names + simple topic extraction
            let speakers: Vec<&str> = recent
                .iter()
                .map(TranscriptEntry::speaker_display_name)
                .collect::<std::collections::HashSet<_>>()
                .iter()
                .copied()
                .collect();

            let mut summary_parts = vec![format!("Speaking: {}", speakers.join(", "))];

            // Extract recurring keywords
            let topics = Self::extract_keywords(recent);
            if !topics.is_empty() {
                summary_parts.push(format!("Topics: {}", topics.join(", ")));
            }

            summary_parts.join(". ")
        };

        self.last_summary_time = Some(Utc::now());

        let now = Utc::now();
        SummaryEntry {
            id: new_meeting_artifact_id(),
            session_id: recent
                .last()
                .map(|entry| entry.session_id.clone())
                .unwrap_or_default(),
            timestamp: now,
            created_at: now,
            summary: summary_text,
            evidence_segment_ids: recent
                .iter()
                .map(|entry| entry.segment_id.clone())
                .collect(),
        }
    }

    /// Get the last generated summary (if any)
    pub fn get_last_summary(&self) -> Option<&SummaryEntry> {
        // This would store the last summary in self — for MVP returning None
        None
    }

    /// Extract recurring keywords from transcript entries
    fn extract_keywords(entries: &[TranscriptEntry]) -> Vec<String> {
        let common_words = [
            "todo",
            "next",
            "action",
            "check",
            "follow up",
            "review",
            "design",
            "build",
            "ship",
            "release",
            "deploy",
            "fix",
            "improve",
            "optimize",
            "test",
            "document",
            "plan",
            "meeting",
            "sprint",
            "deadline",
            "milestone",
            "goal",
        ];

        let mut keyword_counts = std::collections::HashMap::<String, usize>::new();

        for entry in entries {
            let text = entry.text.to_lowercase();
            for keyword in common_words {
                if text.contains(keyword) {
                    *keyword_counts.entry(keyword.to_string()).or_insert(0) += 1;
                }
            }
        }

        // Return sorted by frequency
        let mut topics: Vec<(String, usize)> = keyword_counts.into_iter().collect();
        topics.sort_by(|(_, a), (_, b)| b.cmp(a));

        topics.into_iter().take(5).map(|(topic, _)| topic).collect()
    }
}
