//! Session accounting for bounded autonomous deep-search runs.

use super::types::{DeepSearchPassSummary, DeepSearchRunSummary, DeepSearchStopReason};
use crate::memory::types::now_ms;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DeepSearchSessionState {
    pub started_at: i64,
    pub completed_at: Option<i64>,
    pub passes: Vec<DeepSearchPassSummary>,
    pub stop_reason: Option<DeepSearchStopReason>,
}

impl DeepSearchSessionState {
    pub fn new() -> Self {
        Self { started_at: now_ms(), completed_at: None, passes: Vec::new(), stop_reason: None }
    }

    pub fn complete(&mut self, stop_reason: DeepSearchStopReason) -> i64 {
        let completed_at = now_ms();
        self.completed_at = Some(completed_at);
        self.stop_reason = Some(stop_reason);
        completed_at
    }

    pub fn run_summary(&self, topic: &str, objective: Option<String>, seen: usize, accepted: usize, rejected: usize) -> DeepSearchRunSummary {
        let completed_at = self.completed_at.unwrap_or_else(now_ms);
        DeepSearchRunSummary {
            id: format!("deep_search_{}", self.started_at),
            topic: topic.to_string(),
            objective,
            started_at: self.started_at,
            completed_at,
            duration_ms: completed_at.saturating_sub(self.started_at),
            sources_seen: seen,
            sources_accepted: accepted,
            sources_rejected: rejected,
            status: if accepted > 0 { "completed".into() } else { "no_sources_accepted".into() },
            passes_executed: self.passes.len(),
            stop_reason: self.stop_reason.clone(),
        }
    }
}
