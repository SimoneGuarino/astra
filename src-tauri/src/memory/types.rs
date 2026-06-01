use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryNodeKind {
    ConversationTurn,
    WorkSession,
    TranscriptSegment,
    Summary,
    Concept,
    Entity,
    Task,
    ToolUse,
    Error,
    Fix,
    ResearchTopic,
    ResearchFinding,
    SourceDocument,
    CodePattern,
    UserPreference,
    Workflow,
    Claim,
    Decision,
    Procedure,
    Unknown,
}

impl Default for MemoryNodeKind {
    fn default() -> Self {
        Self::Unknown
    }
}

impl MemoryNodeKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ConversationTurn => "conversation_turn",
            Self::WorkSession => "work_session",
            Self::TranscriptSegment => "transcript_segment",
            Self::Summary => "summary",
            Self::Concept => "concept",
            Self::Entity => "entity",
            Self::Task => "task",
            Self::ToolUse => "tool_use",
            Self::Error => "error",
            Self::Fix => "fix",
            Self::ResearchTopic => "research_topic",
            Self::ResearchFinding => "research_finding",
            Self::SourceDocument => "source_document",
            Self::CodePattern => "code_pattern",
            Self::UserPreference => "user_preference",
            Self::Workflow => "workflow",
            Self::Claim => "claim",
            Self::Decision => "decision",
            Self::Procedure => "procedure",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryRelationKind {
    Mentions,
    About,
    DerivedFrom,
    Supports,
    Contradicts,
    Caused,
    ResolvedBy,
    Follows,
    PartOf,
    SameTopicAs,
    PreferredByUser,
    UsedTool,
    VerifiedBy,
    LearnedFrom,
    Triggered,
    ImplementedIn,
    RelatedToCodebase,
    DependsOn,
    RelatedTo,
}

impl Default for MemoryRelationKind {
    fn default() -> Self {
        Self::RelatedTo
    }
}

impl MemoryRelationKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Mentions => "mentions",
            Self::About => "about",
            Self::DerivedFrom => "derived_from",
            Self::Supports => "supports",
            Self::Contradicts => "contradicts",
            Self::Caused => "caused",
            Self::ResolvedBy => "resolved_by",
            Self::Follows => "follows",
            Self::PartOf => "part_of",
            Self::SameTopicAs => "same_topic_as",
            Self::PreferredByUser => "preferred_by_user",
            Self::UsedTool => "used_tool",
            Self::VerifiedBy => "verified_by",
            Self::LearnedFrom => "learned_from",
            Self::Triggered => "triggered",
            Self::ImplementedIn => "implemented_in",
            Self::RelatedToCodebase => "related_to_codebase",
            Self::DependsOn => "depends_on",
            Self::RelatedTo => "related_to",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryVerificationStatus {
    Unverified,
    LlmInferred,
    UserConfirmed,
    SystemVerified,
    Contradicted,
    Deprecated,
}

impl Default for MemoryVerificationStatus {
    fn default() -> Self {
        Self::Unverified
    }
}

impl MemoryVerificationStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Unverified => "unverified",
            Self::LlmInferred => "llm_inferred",
            Self::UserConfirmed => "user_confirmed",
            Self::SystemVerified => "system_verified",
            Self::Contradicted => "contradicted",
            Self::Deprecated => "deprecated",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode {
    pub id: String,
    pub kind: MemoryNodeKind,
    pub title: String,
    pub summary: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub source: Option<String>,
    pub confidence: f32,
    pub verification_status: MemoryVerificationStatus,
    pub salience: f32,
    pub created_at: i64,
    pub updated_at: i64,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEdge {
    pub id: String,
    pub from_node_id: String,
    pub to_node_id: String,
    pub relation: MemoryRelationKind,
    pub weight: f32,
    pub confidence: f32,
    pub created_at: i64,
    #[serde(default)]
    pub last_activated_at: Option<i64>,
    pub activation_count: u64,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryChunk {
    pub id: String,
    pub node_id: String,
    pub text: String,
    pub ordinal: u32,
    pub created_at: i64,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryActivation {
    pub id: String,
    pub request_id: Option<String>,
    pub root_query: String,
    pub activated_node_ids: Vec<String>,
    pub activated_edge_ids: Vec<String>,
    pub intensity: Value,
    pub created_at: i64,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryGraphSnapshot {
    pub nodes: Vec<MemoryNode>,
    pub edges: Vec<MemoryEdge>,
    pub activations: Vec<MemoryActivation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateMemoryNodeRequest {
    pub kind: MemoryNodeKind,
    pub title: String,
    pub summary: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    #[serde(default)]
    pub verification_status: MemoryVerificationStatus,
    #[serde(default = "default_salience")]
    pub salience: f32,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateMemoryEdgeRequest {
    pub from_node_id: String,
    pub to_node_id: String,
    pub relation: MemoryRelationKind,
    #[serde(default = "default_weight")]
    pub weight: f32,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryQueryRequest {
    pub query: String,
    #[serde(default)]
    pub kinds: Vec<MemoryNodeKind>,
    #[serde(default = "default_query_limit")]
    pub limit: usize,
    #[serde(default)]
    pub include_edges: bool,
    #[serde(default)]
    pub include_deprecated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryQueryHit {
    pub node: MemoryNode,
    pub score: f32,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryQueryResponse {
    pub hits: Vec<MemoryQueryHit>,
    pub related_edges: Vec<MemoryEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryActivationRequest {
    #[serde(default)]
    pub request_id: Option<String>,
    pub root_query: String,
    pub seed_node_ids: Vec<String>,
    #[serde(default = "default_activation_depth")]
    pub max_depth: usize,
    #[serde(default = "default_activation_limit")]
    pub max_nodes: usize,
    #[serde(default)]
    pub metadata: Value,
}

pub fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

pub fn default_metadata() -> Value {
    json!({})
}

fn default_confidence() -> f32 { 0.7 }
fn default_salience() -> f32 { 0.5 }
fn default_weight() -> f32 { 0.5 }
fn default_query_limit() -> usize { 20 }
fn default_activation_depth() -> usize { 2 }
fn default_activation_limit() -> usize { 30 }


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNodeGovernanceUpdateRequest {
    pub node_id: String,
    #[serde(default)]
    pub verification_status: Option<MemoryVerificationStatus>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub salience: Option<f32>,
    #[serde(default)]
    pub add_tags: Vec<String>,
    #[serde(default)]
    pub remove_tags: Vec<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNodeGovernanceUpdateReceipt {
    pub accepted: bool,
    pub reason: String,
    pub node: MemoryNode,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryGovernancePolicySnapshot {
    pub version: String,
    pub user_control_enabled: bool,
    pub inferred_memory_default_weight: f32,
    pub user_confirmed_weight: f32,
    pub deprecated_memory_retrieval_enabled: bool,
    pub hard_delete_enabled: bool,
    #[serde(default)]
    pub allowed_statuses: Vec<MemoryVerificationStatus>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEmbeddingRecord {
    pub chunk_id: String,
    pub node_id: String,
    pub model: String,
    pub dimensions: usize,
    pub vector: Vec<f32>,
    pub created_at: i64,
    pub updated_at: i64,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEmbeddingIndexStatus {
    pub backend: String,
    pub provider: String,
    pub dimensions: usize,
    pub embedded_chunks: usize,
    pub total_chunks: usize,
    pub pending_chunks: usize,
    pub last_indexed_at: Option<i64>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEmbeddingRebuildRequest {
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub force: bool,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEmbeddingRebuildReceipt {
    pub accepted: bool,
    pub reason: String,
    pub indexed_chunks: usize,
    pub skipped_chunks: usize,
    pub failed_chunks: usize,
    pub model: String,
    pub dimensions: usize,
    #[serde(default)]
    pub sample_node_ids: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEmbeddingMaintenanceRequest {
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub force: bool,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEmbeddingMaintenanceReceipt {
    pub accepted: bool,
    pub reason: String,
    pub ran: bool,
    pub indexed_chunks: usize,
    pub skipped_chunks: usize,
    pub failed_chunks: usize,
    pub pending_before: usize,
    pub pending_after: usize,
    pub model: String,
    pub dimensions: usize,
    #[serde(default)]
    pub sample_node_ids: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryHybridQueryRequest {
    pub query: String,
    #[serde(default)]
    pub kinds: Vec<MemoryNodeKind>,
    #[serde(default = "default_query_limit")]
    pub limit: usize,
    #[serde(default)]
    pub include_edges: bool,
    #[serde(default)]
    pub include_deprecated: bool,
    #[serde(default = "default_vector_weight")]
    pub vector_weight: f32,
    #[serde(default = "default_lexical_weight")]
    pub lexical_weight: f32,
    #[serde(default = "default_graph_weight")]
    pub graph_weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryHybridQueryResponse {
    pub hits: Vec<MemoryQueryHit>,
    pub related_edges: Vec<MemoryEdge>,
    pub embedding_status: MemoryEmbeddingIndexStatus,
    #[serde(default)]
    pub metadata: Value,
}

fn default_vector_weight() -> f32 { 0.42 }
fn default_lexical_weight() -> f32 { 0.42 }
fn default_graph_weight() -> f32 { 0.16 }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemorySkillCandidateStatus {
    Candidate,
    Approved,
    Disabled,
    Deprecated,
}

impl Default for MemorySkillCandidateStatus {
    fn default() -> Self {
        Self::Candidate
    }
}

impl MemorySkillCandidateStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Candidate => "candidate",
            Self::Approved => "approved",
            Self::Disabled => "disabled",
            Self::Deprecated => "deprecated",
        }
    }

    pub fn from_str(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "approved" => Self::Approved,
            "disabled" => Self::Disabled,
            "deprecated" => Self::Deprecated,
            _ => Self::Candidate,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySkillCandidate {
    pub id: String,
    pub title: String,
    pub summary: String,
    #[serde(default)]
    pub source_node_id: Option<String>,
    pub status: MemorySkillCandidateStatus,
    pub confidence: f32,
    pub salience: f32,
    #[serde(default)]
    pub trigger_hints: Vec<String>,
    #[serde(default)]
    pub required_tools: Vec<String>,
    pub risk_level: String,
    pub created_at: i64,
    pub updated_at: i64,
    #[serde(default)]
    pub approved_by: Option<String>,
    #[serde(default)]
    pub approved_at: Option<i64>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySkillCandidateExtractionReceipt {
    pub accepted: bool,
    pub reason: String,
    #[serde(default)]
    pub candidates: Vec<MemorySkillCandidate>,
    #[serde(default)]
    pub activation: Option<MemoryActivation>,
    #[serde(default)]
    pub metadata: Value,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryQualityDashboard {
    pub schema_version: u32,
    pub generated_at: i64,
    pub status: String,
    pub score: f32,
    pub summary: String,
    pub totals: MemoryQualityTotals,
    pub semantic: MemoryQualitySemanticStats,
    pub governance: MemoryQualityGovernanceStats,
    pub embeddings: MemoryEmbeddingIndexStatus,
    pub reconsolidation: MemoryQualityReconsolidationStats,
    pub retrieval: MemoryQualityRetrievalStats,
    #[serde(default)]
    pub repair_plan: Option<MemoryHealthRepairPlan>,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub recommendations: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryHealthRepairPlan {
    pub schema_version: u32,
    pub generated_at: i64,
    pub status: String,
    pub summary: String,
    #[serde(default)]
    pub actions: Vec<MemoryHealthRepairAction>,
    pub automatic_action_count: usize,
    pub review_action_count: usize,
    pub blocked_action_count: usize,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryHealthRepairAction {
    pub id: String,
    pub kind: String,
    pub title: String,
    pub description: String,
    pub priority: String,
    pub risk_level: String,
    pub requires_user_review: bool,
    pub can_run_automatically: bool,
    pub status: String,
    pub affected_count: usize,
    pub confidence: f32,
    pub rationale: String,
    #[serde(default)]
    pub command_hint: Option<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryQualityTotals {
    pub nodes: usize,
    pub edges: usize,
    pub chunks: usize,
    pub activations: usize,
    pub skill_candidates: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryQualitySemanticStats {
    pub semantic_nodes: usize,
    pub episode_only_nodes: usize,
    pub conversation_turn_nodes: usize,
    pub semantic_ratio: f32,
    pub average_confidence: f32,
    pub average_salience: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryQualityGovernanceStats {
    pub unverified: usize,
    pub llm_inferred: usize,
    pub user_confirmed: usize,
    pub system_verified: usize,
    pub contradicted: usize,
    pub deprecated: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryQualityReconsolidationStats {
    pub pending_candidates: usize,
    pub reconsolidated_nodes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryQualityRetrievalStats {
    pub recent_activations: usize,
    pub average_activation_nodes: f32,
    pub last_activation_at: Option<i64>,
}



#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAutopilotRequest {
    #[serde(default = "default_memory_autopilot_reconsolidation_limit")]
    pub reconsolidation_limit: usize,
    #[serde(default = "default_memory_autopilot_embedding_limit")]
    pub embedding_limit: usize,
    #[serde(default)]
    pub run_skill_extraction: bool,
    #[serde(default)]
    pub run_candidate_discovery: bool,
    #[serde(default)]
    pub force_embeddings: bool,
    #[serde(default)]
    pub reason: Option<String>,
}

fn default_memory_autopilot_reconsolidation_limit() -> usize { 12 }
fn default_memory_autopilot_embedding_limit() -> usize { 48 }

impl Default for MemoryAutopilotRequest {
    fn default() -> Self {
        Self {
            reconsolidation_limit: default_memory_autopilot_reconsolidation_limit(),
            embedding_limit: default_memory_autopilot_embedding_limit(),
            run_skill_extraction: true,
            run_candidate_discovery: true,
            force_embeddings: false,
            reason: Some("memory_autopilot".into()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAutopilotReceipt {
    pub accepted: bool,
    pub reason: String,
    pub started_at: i64,
    pub completed_at: i64,
    pub reconsolidated_candidates: usize,
    pub semantic_nodes_created: usize,
    pub embeddings_indexed: usize,
    pub embeddings_failed: usize,
    pub pending_embeddings_after: usize,
    pub skill_candidates: usize,
    pub duplicate_candidates: usize,
    pub canonical_review_candidates: usize,
    pub quality_score: f32,
    pub quality_status: String,
    #[serde(default)]
    pub repair_plan: Option<MemoryHealthRepairPlan>,
    #[serde(default)]
    pub recommendations: Vec<String>,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySkillCandidateUpdateRequest {
    pub candidate_id: String,
    #[serde(default)]
    pub status: Option<MemorySkillCandidateStatus>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub salience: Option<f32>,
    #[serde(default)]
    pub add_trigger_hints: Vec<String>,
    #[serde(default)]
    pub remove_trigger_hints: Vec<String>,
    #[serde(default)]
    pub required_tools: Option<Vec<String>>,
    #[serde(default)]
    pub risk_level: Option<String>,
    #[serde(default)]
    pub approved_by: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySkillCandidateUpdateReceipt {
    pub accepted: bool,
    pub reason: String,
    pub candidate: MemorySkillCandidate,
    #[serde(default)]
    pub activation: Option<MemoryActivation>,
    #[serde(default)]
    pub metadata: Value,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDuplicateCandidateRequest {
    #[serde(default = "default_duplicate_candidate_limit")]
    pub limit: usize,
    #[serde(default = "default_duplicate_candidate_min_score")]
    pub min_score: f32,
    #[serde(default)]
    pub include_deprecated: bool,
    #[serde(default)]
    pub kinds: Vec<MemoryNodeKind>,
}

fn default_duplicate_candidate_limit() -> usize { 80 }
fn default_duplicate_candidate_min_score() -> f32 { 0.72 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDuplicateCandidate {
    pub canonical_node: MemoryNode,
    pub duplicate_node: MemoryNode,
    pub score: f32,
    pub reasons: Vec<String>,
    #[serde(default)]
    pub shared_tags: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
}



#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCanonicalReviewRequest {
    #[serde(default = "default_canonical_review_limit")]
    pub limit: usize,
    #[serde(default = "default_canonical_review_min_score")]
    pub min_score: f32,
    #[serde(default)]
    pub include_deprecated: bool,
    #[serde(default)]
    pub kinds: Vec<MemoryNodeKind>,
    #[serde(default)]
    pub llm_assist: bool,
}

fn default_canonical_review_limit() -> usize { 40 }
fn default_canonical_review_min_score() -> f32 { 0.62 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCanonicalReviewCandidate {
    pub id: String,
    pub target_node: MemoryNode,
    pub candidate_nodes: Vec<MemoryNode>,
    pub confidence: f32,
    pub rationale: String,
    pub proposed_title: String,
    pub proposed_summary: String,
    pub reasons: Vec<String>,
    #[serde(default)]
    pub shared_tags: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCanonicalReviewApplyRequest {
    pub candidate: MemoryCanonicalReviewCandidate,
    #[serde(default)]
    pub mark_sources_deprecated: bool,
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMergeNodesRequest {
    pub target_node_id: String,
    pub source_node_ids: Vec<String>,
    #[serde(default)]
    pub mark_sources_deprecated: bool,
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMergeNodesReceipt {
    pub accepted: bool,
    pub reason: String,
    pub target_node: MemoryNode,
    pub merged_node_ids: Vec<String>,
    pub created_edge_ids: Vec<String>,
    #[serde(default)]
    pub activation: Option<MemoryActivation>,
    #[serde(default)]
    pub metadata: Value,
}
