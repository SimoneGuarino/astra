use crate::memory::deep_search::{DeepSearchReceipt, DeepSearchRequest};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchKnowledgeAutopilotRequest {
    /// Master enable flag. Kept explicit so UI/schedulers can persist an off state.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// If true, build the learning agenda but do not run deep-search.
    #[serde(default)]
    pub dry_run: bool,
    /// Maximum agenda items returned to the caller.
    #[serde(default = "default_max_topics")]
    pub max_topics: usize,
    /// Maximum deep-search runs executed in this cycle.
    #[serde(default = "default_max_runs")]
    pub max_runs: usize,
    /// Maximum sources per topic deep-search run.
    #[serde(default = "default_max_sources_per_topic")]
    pub max_sources_per_topic: usize,
    /// Minimum topic priority before a mined topic can be scheduled.
    #[serde(default = "default_min_topic_priority")]
    pub min_topic_priority: f32,
    /// Include low-confidence or unverified claim/finding nodes as refresh targets.
    #[serde(default = "default_true")]
    pub include_low_confidence_claims: bool,
    /// Include recent user preference / workflow / task memories as personalization topics.
    #[serde(default)]
    pub include_user_context_topics: bool,
    /// Include broad memory topics inferred from tags/titles.
    #[serde(default = "default_true")]
    pub include_topic_mining: bool,
    /// Optional user/topic seed list. These are merged with mined agenda items.
    #[serde(default)]
    pub seed_topics: Vec<String>,
    /// Optional blocked topic substrings, evaluated case-insensitively before scheduling.
    #[serde(default)]
    pub blocked_topics: Vec<String>,
    /// Optional provider allow-list forwarded to deep-search.
    #[serde(default)]
    pub search_providers: Vec<String>,
    /// Reason written to the memory journal.
    #[serde(default)]
    pub reason: Option<String>,
    /// Optional deep-search defaults merged into every generated request.
    #[serde(default)]
    pub deep_search_defaults: Option<DeepSearchRequest>,
    #[serde(default)]
    pub metadata: Value,
}

fn default_true() -> bool { true }
fn default_max_topics() -> usize { 8 }
fn default_max_runs() -> usize { 3 }
fn default_max_sources_per_topic() -> usize { 10 }
fn default_min_topic_priority() -> f32 { 0.38 }

impl Default for DeepSearchKnowledgeAutopilotRequest {
    fn default() -> Self {
        Self {
            enabled: true,
            dry_run: false,
            max_topics: default_max_topics(),
            max_runs: default_max_runs(),
            max_sources_per_topic: default_max_sources_per_topic(),
            min_topic_priority: default_min_topic_priority(),
            include_low_confidence_claims: true,
            include_user_context_topics: false,
            include_topic_mining: true,
            seed_topics: Vec::new(),
            blocked_topics: Vec::new(),
            search_providers: Vec::new(),
            reason: Some("deep_search_knowledge_autopilot".into()),
            deep_search_defaults: None,
            metadata: Value::Null,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchLearningAgendaItem {
    pub topic: String,
    pub objective: String,
    pub reason: String,
    pub priority: f32,
    #[serde(default)]
    pub source_node_ids: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub signals: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchLearningRunReceipt {
    pub agenda_item: DeepSearchLearningAgendaItem,
    pub accepted: bool,
    pub reason: String,
    pub accepted_sources: usize,
    pub extracted_claims: usize,
    pub extracted_findings: usize,
    pub promoted_claims: usize,
    pub candidate_claims: usize,
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub receipt: Option<DeepSearchReceipt>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchKnowledgeAutopilotReceipt {
    pub accepted: bool,
    pub reason: String,
    pub started_at: i64,
    pub completed_at: i64,
    pub dry_run: bool,
    pub agenda_items: usize,
    pub runs_executed: usize,
    pub sources_accepted: usize,
    pub claims_extracted: usize,
    pub findings_extracted: usize,
    pub claims_promoted: usize,
    pub candidate_claims: usize,
    #[serde(default)]
    pub agenda: Vec<DeepSearchLearningAgendaItem>,
    #[serde(default)]
    pub runs: Vec<DeepSearchLearningRunReceipt>,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub recommendations: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchKnowledgeRefreshRequest {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub dry_run: bool,
    #[serde(default = "default_refresh_snapshot_limit")]
    pub snapshot_limit: usize,
    #[serde(default = "default_refresh_max_candidates")]
    pub max_candidates: usize,
    #[serde(default = "default_refresh_stale_after_days")]
    pub stale_after_days: u64,
    #[serde(default = "default_refresh_temporal_stale_after_days")]
    pub temporal_stale_after_days: u64,
    #[serde(default = "default_refresh_low_confidence_threshold")]
    pub low_confidence_threshold: f32,
    #[serde(default = "default_true")]
    pub include_low_confidence_candidates: bool,
    #[serde(default = "default_true")]
    pub tag_candidates_for_refresh: bool,
    #[serde(default = "default_refresh_max_tags")]
    pub max_tags: usize,
    #[serde(default = "default_true")]
    pub run_refresh_research: bool,
    #[serde(default = "default_refresh_max_topics")]
    pub max_refresh_topics: usize,
    #[serde(default = "default_refresh_max_runs")]
    pub max_refresh_runs: usize,
    #[serde(default = "default_refresh_sources_per_topic")]
    pub max_sources_per_topic: usize,
    #[serde(default)]
    pub blocked_topics: Vec<String>,
    #[serde(default)]
    pub search_providers: Vec<String>,
    #[serde(default)]
    pub deep_search_defaults: Option<DeepSearchRequest>,
    #[serde(default)]
    pub metadata: Value,
}

fn default_refresh_snapshot_limit() -> usize { 320 }
fn default_refresh_max_candidates() -> usize { 24 }
fn default_refresh_stale_after_days() -> u64 { 45 }
fn default_refresh_temporal_stale_after_days() -> u64 { 7 }
fn default_refresh_low_confidence_threshold() -> f32 { 0.58 }
fn default_refresh_max_tags() -> usize { 24 }
fn default_refresh_max_topics() -> usize { 8 }
fn default_refresh_max_runs() -> usize { 3 }
fn default_refresh_sources_per_topic() -> usize { 8 }

impl Default for DeepSearchKnowledgeRefreshRequest {
    fn default() -> Self {
        Self {
            enabled: true,
            dry_run: false,
            snapshot_limit: default_refresh_snapshot_limit(),
            max_candidates: default_refresh_max_candidates(),
            stale_after_days: default_refresh_stale_after_days(),
            temporal_stale_after_days: default_refresh_temporal_stale_after_days(),
            low_confidence_threshold: default_refresh_low_confidence_threshold(),
            include_low_confidence_candidates: true,
            tag_candidates_for_refresh: true,
            max_tags: default_refresh_max_tags(),
            run_refresh_research: true,
            max_refresh_topics: default_refresh_max_topics(),
            max_refresh_runs: default_refresh_max_runs(),
            max_sources_per_topic: default_refresh_sources_per_topic(),
            blocked_topics: Vec::new(),
            search_providers: Vec::new(),
            deep_search_defaults: None,
            metadata: Value::Null,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchKnowledgeRefreshCandidate {
    pub node: crate::memory::types::MemoryNode,
    pub topic: String,
    pub reason: String,
    pub priority: f32,
    pub age_days: f32,
    pub temporal: bool,
    pub low_confidence: bool,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchKnowledgeRefreshReceipt {
    pub accepted: bool,
    pub reason: String,
    pub started_at: i64,
    pub completed_at: i64,
    pub dry_run: bool,
    pub candidates_scanned: usize,
    pub stale_candidates: usize,
    pub tagged_for_refresh: usize,
    pub refresh_runs: usize,
    pub sources_accepted: usize,
    pub claims_promoted: usize,
    pub candidate_claims: usize,
    #[serde(default)]
    pub candidates: Vec<DeepSearchKnowledgeRefreshCandidate>,
    #[serde(default)]
    pub autopilot: Option<DeepSearchKnowledgeAutopilotReceipt>,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub recommendations: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
}
