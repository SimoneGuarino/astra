use crate::memory::{consolidation::ResearchMemoryConsolidationReceipt, types::{MemoryNode, MemoryVerificationStatus}};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchRequest {
    pub topic: String,
    #[serde(default)]
    pub objective: Option<String>,
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub seed_urls: Vec<String>,

    /// Enables native multi-provider discovery when no explicit seed URLs are provided.
    /// Defaults to true. Rust still enforces bounded source limits and policy checks.
    #[serde(default)]
    pub enable_web_discovery: Option<bool>,
    /// Optional allow-list for discovery providers. Empty means the default provider set.
    /// Supported: duckduckgo_html, duckduckgo_lite, bing, wikipedia, arxiv, crossref,
    /// pubmed, semantic_scholar, europe_pmc.
    #[serde(default)]
    pub search_providers: Vec<String>,
    /// Enables general web discovery providers. Defaults to true.
    #[serde(default)]
    pub include_general_web: Option<bool>,
    /// Enables academic/scientific discovery providers. Defaults to true.
    #[serde(default)]
    pub include_academic_sources: Option<bool>,
    /// Enables document/academic landing-page ingestion. Defaults to true.
    #[serde(default)]
    pub document_ingestion: Option<bool>,
    /// Allows known PDF URLs to be normalized to abstract/landing pages. Defaults to true.
    #[serde(default)]
    pub prefer_academic_landing_pages: Option<bool>,
    /// Enables bounded full-text extraction for unencrypted PDF sources. Raw PDF bytes are never persisted.
    #[serde(default)]
    pub enable_pdf_text_extraction: Option<bool>,
    /// Bounded limit per discovery provider before policy filtering.
    #[serde(default)]
    pub max_discovery_results_per_provider: Option<usize>,
    /// Bounded total candidate URLs retained after discovery dedupe.
    #[serde(default)]
    pub max_discovered_sources: Option<usize>,
    /// Number of deterministic initial research queries derived from the user request.
    /// This prevents long natural-language prompts from becoming a single poor search query.
    #[serde(default)]
    pub initial_query_count: Option<usize>,
    #[serde(default)]
    pub allowed_domains: Vec<String>,
    #[serde(default)]
    pub blocked_domains: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub max_sources: Option<usize>,
    /// Enables the v0.6.3 autonomous multi-pass research loop. Defaults to true.
    #[serde(default)]
    pub autonomous_loop: Option<bool>,
    /// Maximum bounded research passes. This is never infinite; Rust owns the stop conditions.
    #[serde(default)]
    pub max_research_passes: Option<usize>,
    /// Minimum passes before saturation may stop the loop.
    #[serde(default)]
    pub min_research_passes: Option<usize>,
    /// Maximum accepted sources per pass.
    #[serde(default)]
    pub max_sources_per_pass: Option<usize>,
    /// Saturation threshold for new information gain. Lower means Astra keeps searching longer.
    #[serde(default)]
    pub min_new_information_gain: Option<f32>,
    /// Required coverage score before the loop may stop as saturated.
    #[serde(default)]
    pub min_coverage_score: Option<f32>,
    /// Required cross-source support ratio before the loop may stop as saturated.
    #[serde(default)]
    pub min_supported_claim_ratio: Option<f32>,
    /// Enables claim clustering and cross-source verification metadata. Defaults to true.
    #[serde(default)]
    pub enable_claim_graph: Option<bool>,
    /// Minimum independent domains/sources needed before a claim is treated as cross-source supported.
    #[serde(default)]
    pub min_independent_sources_for_claim: Option<usize>,
    /// Enables conservative contradiction-risk detection inside the claim graph. Defaults to true.
    #[serde(default)]
    pub enable_contradiction_detection: Option<bool>,
    /// Enables governed memory promotion decisions for external research claims. Defaults to true.
    #[serde(default)]
    pub enable_memory_promotion_policy: Option<bool>,
    /// Allows cross-source supported claims to become llm_inferred memory automatically.
    /// This never permits external content to become system_verified.
    #[serde(default)]
    pub auto_promote_supported_claims: Option<bool>,
    /// Keeps system_verified promotion behind a separate governed/user-confirmed path. Defaults to true.
    #[serde(default)]
    pub require_user_confirmation_for_system_verified: Option<bool>,
    /// Minimum claim confidence required before promotion to llm_inferred memory.
    #[serde(default)]
    pub min_promotion_confidence: Option<f32>,
    /// Minimum independent domains/sources required before promotion to llm_inferred memory.
    #[serde(default)]
    pub min_promotion_independent_sources: Option<usize>,
    /// Enables deterministic source reliability scoring. Defaults to true.
    #[serde(default)]
    pub enable_source_reliability_scoring: Option<bool>,
    /// Minimum average source reliability needed before automatic claim promotion may happen.
    #[serde(default)]
    pub min_reliable_source_score_for_promotion: Option<f32>,
    #[serde(default)]
    pub min_sources_for_learning: Option<usize>,
    #[serde(default)]
    pub max_bytes_per_source: Option<usize>,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    #[serde(default)]
    pub require_cross_source_verification: bool,
    #[serde(default)]
    pub allow_http_localhost: bool,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchPolicy {
    pub allowed_domains: Vec<String>,
    pub blocked_domains: Vec<String>,
    pub require_cross_source_verification: bool,
    pub allow_http_localhost: bool,
    pub max_sources: usize,
    pub max_bytes_per_source: usize,
}

impl DeepSearchPolicy {
    pub fn from_request(request: &DeepSearchRequest) -> Self {
        Self {
            allowed_domains: request.allowed_domains.clone(),
            blocked_domains: request.blocked_domains.clone(),
            require_cross_source_verification: request.require_cross_source_verification,
            allow_http_localhost: request.allow_http_localhost,
            max_sources: request.max_sources.unwrap_or(12).clamp(1, 48),
            max_bytes_per_source: request.max_bytes_per_source.unwrap_or(512_000).clamp(16_000, 2_000_000),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchBudget {
    pub max_passes: usize,
    pub min_passes: usize,
    pub max_total_sources: usize,
    pub max_sources_per_pass: usize,
    pub max_total_candidates: usize,
    pub max_runtime_ms: u64,
    pub min_new_information_gain: f32,
    pub min_coverage_score: f32,
    pub min_supported_claim_ratio: f32,
}

impl DeepSearchBudget {
    pub fn from_request(request: &DeepSearchRequest, policy: &DeepSearchPolicy) -> Self {
        let max_total_sources = policy.max_sources.clamp(1, 48);
        let max_passes = request.max_research_passes.unwrap_or(5).clamp(1, 8);
        let min_passes = request.min_research_passes.unwrap_or(2).clamp(1, max_passes);
        let max_sources_per_pass = request
            .max_sources_per_pass
            .unwrap_or_else(|| ((max_total_sources as f32 / max_passes as f32).ceil() as usize).max(2))
            .clamp(1, max_total_sources);
        Self {
            max_passes,
            min_passes,
            max_total_sources,
            max_sources_per_pass,
            max_total_candidates: request.max_discovered_sources.unwrap_or(192).clamp(8, 256),
            max_runtime_ms: request.timeout_ms.unwrap_or(180_000).clamp(10_000, 300_000),
            min_new_information_gain: request.min_new_information_gain.unwrap_or(0.12).clamp(0.02, 0.75),
            min_coverage_score: request.min_coverage_score.unwrap_or(0.58).clamp(0.10, 0.95),
            min_supported_claim_ratio: request.min_supported_claim_ratio.unwrap_or(0.50).clamp(0.10, 0.95),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchPassSummary {
    pub pass_index: usize,
    pub query: String,
    pub candidates_seen: usize,
    pub accepted_sources: usize,
    pub rejected_sources: usize,
    pub new_information_gain: f32,
    pub coverage_score: f32,
    pub saturation_score: f32,
    #[serde(default)]
    pub generated_follow_up_queries: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchCoverageReport {
    pub overall_score: f32,
    pub domain_diversity_score: f32,
    pub academic_coverage_score: f32,
    pub authoritative_source_score: f32,
    pub source_reliability_score: f32,
    pub topic_token_coverage: f32,
    pub query_diversity_score: f32,
    pub unique_domains: usize,
    pub academic_sources: usize,
    pub authoritative_sources: usize,
    #[serde(default)]
    pub high_reliability_sources: usize,
    #[serde(default)]
    pub low_reliability_sources: usize,
    #[serde(default)]
    pub source_reliability: DeepSearchSourceReliabilitySummary,
    #[serde(default)]
    pub missing_subtopics: Vec<String>,
}

impl Default for DeepSearchCoverageReport {
    fn default() -> Self {
        Self {
            overall_score: 0.0,
            domain_diversity_score: 0.0,
            academic_coverage_score: 0.0,
            authoritative_source_score: 0.0,
            source_reliability_score: 0.0,
            topic_token_coverage: 0.0,
            query_diversity_score: 0.0,
            unique_domains: 0,
            academic_sources: 0,
            authoritative_sources: 0,
            high_reliability_sources: 0,
            low_reliability_sources: 0,
            source_reliability: DeepSearchSourceReliabilitySummary::default(),
            missing_subtopics: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchSaturationReport {
    pub is_saturated: bool,
    pub score: f32,
    pub new_information_gain: f32,
    pub supported_claim_ratio: f32,
    pub duplicate_ratio: f32,
    #[serde(default)]
    pub missing_subtopics: Vec<String>,
    #[serde(default)]
    pub stop_reason: Option<DeepSearchStopReason>,
}

impl Default for DeepSearchSaturationReport {
    fn default() -> Self {
        Self {
            is_saturated: false,
            score: 0.0,
            new_information_gain: 1.0,
            supported_claim_ratio: 0.0,
            duplicate_ratio: 0.0,
            missing_subtopics: Vec::new(),
            stop_reason: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeepSearchStopReason {
    Saturated,
    BudgetExhausted,
    MaxPassesReached,
    PolicyBlocked,
    TooManyProviderFailures,
    NoFollowUpQueries,
    NoSourcesAccepted,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchClaimEvidenceRef {
    pub source_ref: String,
    pub url: String,
    pub title: String,
    #[serde(default)]
    pub document_kind: Option<String>,
    #[serde(default)]
    pub doi: Option<String>,
    #[serde(default)]
    pub academic_id: Option<String>,
    pub evidence: String,
    #[serde(default)]
    pub section_title: Option<String>,
    #[serde(default)]
    pub section_ordinal: Option<usize>,
    pub content_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchClaimCluster {
    pub id: String,
    pub normalized_claim: String,
    pub representative_claim: String,
    pub support_count: usize,
    #[serde(default)]
    pub source_refs: Vec<String>,
    pub independent_domain_count: usize,
    pub confidence: f32,
    pub verification_status: MemoryVerificationStatus,
    pub contradiction_risk: f32,
    #[serde(default)]
    pub evidence_refs: Vec<DeepSearchClaimEvidenceRef>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchClaimGraphReport {
    #[serde(default)]
    pub clusters: Vec<DeepSearchClaimCluster>,
    pub supported_claims: usize,
    pub contradicted_claims: usize,
    pub unverified_claims: usize,
    pub independent_source_ratio: f32,
    pub cross_source_verified_ratio: f32,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
}

impl Default for DeepSearchClaimGraphReport {
    fn default() -> Self {
        Self {
            clusters: Vec::new(),
            supported_claims: 0,
            contradicted_claims: 0,
            unverified_claims: 0,
            independent_source_ratio: 0.0,
            cross_source_verified_ratio: 0.0,
            warnings: Vec::new(),
            metadata: Value::Null,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeepSearchMemoryPromotionStage {
    EphemeralResearch,
    CandidateMemory,
    LlmInferredMemory,
    ReviewRequired,
    BlockedContradicted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchMemoryPromotionDecision {
    pub claim_cluster_id: String,
    pub stage: DeepSearchMemoryPromotionStage,
    pub verification_status: MemoryVerificationStatus,
    pub confidence: f32,
    pub salience: f32,
    pub reason: String,
    #[serde(default)]
    pub source_refs: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchPromotionReport {
    pub enabled: bool,
    pub promoted_claims: usize,
    pub candidate_claims: usize,
    pub review_required_claims: usize,
    pub blocked_claims: usize,
    #[serde(default)]
    pub decisions: Vec<DeepSearchMemoryPromotionDecision>,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
}

impl Default for DeepSearchPromotionReport {
    fn default() -> Self {
        Self {
            enabled: true,
            promoted_claims: 0,
            candidate_claims: 0,
            review_required_claims: 0,
            blocked_claims: 0,
            decisions: Vec::new(),
            warnings: Vec::new(),
            metadata: Value::Null,
        }
    }
}


#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum DeepSearchSourceReliabilityTier {
    GovernmentInstitutional,
    AcademicPeerReviewedLike,
    AcademicPreprintOrIndex,
    OfficialDocumentation,
    EncyclopedicReference,
    GeneralWebModerate,
    CommunityDiscussion,
    UnknownUnranked,
    LowQuality,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchSourceReliability {
    pub score: f32,
    pub tier: DeepSearchSourceReliabilityTier,
    pub reason: String,
    #[serde(default)]
    pub signals: Vec<String>,
    #[serde(default)]
    pub penalties: Vec<String>,
}

impl Default for DeepSearchSourceReliability {
    fn default() -> Self {
        Self {
            score: 0.45,
            tier: DeepSearchSourceReliabilityTier::UnknownUnranked,
            reason: "source reliability not scored yet".into(),
            signals: Vec::new(),
            penalties: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchSourceReliabilitySummary {
    pub average_score: f32,
    pub min_score: f32,
    pub max_score: f32,
    pub high_reliability_sources: usize,
    pub low_reliability_sources: usize,
    #[serde(default)]
    pub tier_counts: std::collections::BTreeMap<String, usize>,
}

impl Default for DeepSearchSourceReliabilitySummary {
    fn default() -> Self {
        Self {
            average_score: 0.0,
            min_score: 0.0,
            max_score: 0.0,
            high_reliability_sources: 0,
            low_reliability_sources: 0,
            tier_counts: std::collections::BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchRunSummary {
    pub id: String,
    pub topic: String,
    pub objective: Option<String>,
    pub started_at: i64,
    pub completed_at: i64,
    pub duration_ms: i64,
    pub sources_seen: usize,
    pub sources_accepted: usize,
    pub sources_rejected: usize,
    pub status: String,
    #[serde(default)]
    pub passes_executed: usize,
    #[serde(default)]
    pub stop_reason: Option<DeepSearchStopReason>,
}

impl DeepSearchRunSummary {
    pub fn new(request: &DeepSearchRequest, started_at: i64, completed_at: i64, seen: usize, accepted: usize, rejected: usize) -> Self {
        Self {
            id: format!("deep_search_{}", started_at),
            topic: request.topic.clone(),
            objective: request.objective.clone(),
            started_at,
            completed_at,
            duration_ms: completed_at.saturating_sub(started_at),
            sources_seen: seen,
            sources_accepted: accepted,
            sources_rejected: rejected,
            status: if accepted > 0 { "completed".into() } else { "no_sources_accepted".into() },
            passes_executed: 1,
            stop_reason: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchAcceptedSource {
    pub url: String,
    pub title: String,
    pub content_hash: String,
    pub fetched_at: i64,
    pub content_type: Option<String>,
    pub discovered_by: Option<String>,
    pub source_type: Option<String>,
    pub discovery_rank: Option<usize>,
    #[serde(default)]
    pub document_kind: Option<String>,
    #[serde(default)]
    pub doi: Option<String>,
    #[serde(default)]
    pub academic_id: Option<String>,
    #[serde(default)]
    pub published_at: Option<String>,
    #[serde(default)]
    pub abstract_present: bool,
    #[serde(default)]
    pub section_count: usize,
    #[serde(default)]
    pub section_titles: Vec<String>,
    #[serde(default)]
    pub pdf_extracted: bool,
    #[serde(default)]
    pub extraction_method: Option<String>,
    #[serde(default)]
    pub reliability_score: f32,
    #[serde(default)]
    pub reliability_tier: Option<DeepSearchSourceReliabilityTier>,
    #[serde(default)]
    pub reliability_reason: Option<String>,
    #[serde(default)]
    pub reliability_signals: Vec<String>,
    #[serde(default)]
    pub reliability_penalties: Vec<String>,
}

impl DeepSearchAcceptedSource {
    pub(crate) fn from_document(document: &super::DeepSearchDocument) -> Self {
        Self {
            url: document.url.clone(),
            title: document.title.clone(),
            content_hash: document.content_hash.clone(),
            fetched_at: document.fetched_at,
            content_type: document.content_type.clone(),
            discovered_by: document.discovered_by.clone(),
            source_type: document.source_type.clone(),
            discovery_rank: document.discovery_rank,
            document_kind: document.document_kind.clone(),
            doi: document.doi.clone(),
            academic_id: document.academic_id.clone(),
            published_at: document.published_at.clone(),
            abstract_present: document.abstract_present,
            section_count: document.section_count,
            section_titles: document.sections.iter().take(16).map(|section| section.title.clone()).collect(),
            pdf_extracted: document.pdf_extracted,
            extraction_method: document.extraction_method.clone(),
            reliability_score: document.reliability.score,
            reliability_tier: Some(document.reliability.tier.clone()),
            reliability_reason: Some(document.reliability.reason.clone()),
            reliability_signals: document.reliability.signals.clone(),
            reliability_penalties: document.reliability.penalties.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchRejectedSource {
    pub url: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchConsolidationSummary {
    pub accepted: bool,
    pub reason: String,
    pub topic_node: MemoryNode,
    pub created_node_ids: Vec<String>,
    pub created_edge_ids: Vec<String>,
    #[serde(default)]
    pub summary: Value,
}

impl From<ResearchMemoryConsolidationReceipt> for DeepSearchConsolidationSummary {
    fn from(receipt: ResearchMemoryConsolidationReceipt) -> Self {
        Self {
            accepted: receipt.accepted,
            reason: receipt.reason,
            topic_node: receipt.topic_node,
            created_node_ids: receipt.created_node_ids,
            created_edge_ids: receipt.created_edge_ids,
            summary: receipt.summary,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchReceipt {
    pub accepted: bool,
    pub reason: String,
    pub run: DeepSearchRunSummary,
    #[serde(default)]
    pub consolidated: Option<DeepSearchConsolidationSummary>,
    #[serde(default)]
    pub accepted_sources: Vec<DeepSearchAcceptedSource>,
    #[serde(default)]
    pub rejected_sources: Vec<DeepSearchRejectedSource>,
    pub extracted_claims: usize,
    pub extracted_findings: usize,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub passes: Vec<DeepSearchPassSummary>,
    #[serde(default)]
    pub coverage: DeepSearchCoverageReport,
    #[serde(default)]
    pub saturation: DeepSearchSaturationReport,
    #[serde(default)]
    pub claim_graph: Option<DeepSearchClaimGraphReport>,
    #[serde(default)]
    pub promotion: Option<DeepSearchPromotionReport>,
    #[serde(default)]
    pub metadata: Value,
}
