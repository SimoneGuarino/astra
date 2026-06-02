//! Governed memory promotion policy for Deep Search.
//!
//! Cross-source support is useful evidence, but it is not the same thing as
//! truth. This module decides which external research claims may be promoted
//! from ephemeral research context into candidate / inferred long-term memory.
//! It intentionally never upgrades untrusted web content to `SystemVerified`.

use super::types::{
    DeepSearchClaimGraphReport, DeepSearchCoverageReport, DeepSearchMemoryPromotionDecision,
    DeepSearchMemoryPromotionStage, DeepSearchPromotionReport, DeepSearchRequest,
    DeepSearchSaturationReport,
};
use crate::memory::types::MemoryVerificationStatus;
use serde_json::json;

pub(crate) fn apply_memory_promotion_policy(
    request: &DeepSearchRequest,
    claim_graph: &DeepSearchClaimGraphReport,
    coverage: &DeepSearchCoverageReport,
    saturation: &DeepSearchSaturationReport,
) -> DeepSearchPromotionReport {
    let enabled = request.enable_memory_promotion_policy.unwrap_or(true);
    let auto_promote_supported = request.auto_promote_supported_claims.unwrap_or(true);
    let require_user_confirmation_for_system_verified = request
        .require_user_confirmation_for_system_verified
        .unwrap_or(true);
    let min_confidence = request.min_promotion_confidence.unwrap_or(0.62).clamp(0.10, 0.95);
    let min_independent_sources = request
        .min_promotion_independent_sources
        .or(request.min_independent_sources_for_claim)
        .unwrap_or(2)
        .clamp(1, 4);
    let min_coverage = request.min_coverage_score.unwrap_or(0.58).clamp(0.10, 0.95);
    let reliability_enabled = request.enable_source_reliability_scoring.unwrap_or(true);
    let min_reliable_source_score = request
        .min_reliable_source_score_for_promotion
        .unwrap_or(0.50)
        .clamp(0.10, 0.95);

    let mut decisions = Vec::new();
    let mut warnings = Vec::new();

    if !enabled {
        warnings.push("memory promotion policy is disabled for this deep-search run; claims remain ephemeral research context".into());
    }
    if require_user_confirmation_for_system_verified {
        warnings.push("external deep-search content cannot become system_verified without a separate governed confirmation path".into());
    }
    if claim_graph.contradicted_claims > 0 {
        warnings.push(format!(
            "{} contradiction-risk claim cluster(s) were blocked from promotion",
            claim_graph.contradicted_claims
        ));
    }
    if reliability_enabled && coverage.source_reliability_score < min_reliable_source_score {
        warnings.push(format!(
            "source reliability average {:.2} is below automatic-promotion threshold {:.2}; claims may remain candidates",
            coverage.source_reliability_score, min_reliable_source_score
        ));
    }

    for cluster in &claim_graph.clusters {
        let decision = if !enabled {
            DeepSearchMemoryPromotionDecision {
                claim_cluster_id: cluster.id.clone(),
                stage: DeepSearchMemoryPromotionStage::EphemeralResearch,
                verification_status: MemoryVerificationStatus::Unverified,
                confidence: cluster.confidence.min(0.50),
                salience: 0.42,
                reason: "promotion policy disabled; retain as ephemeral research evidence only".into(),
                source_refs: cluster.source_refs.clone(),
                metadata: json!({
                    "rule": "promotion_disabled",
                    "external_content_untrusted": true,
                }),
            }
        } else if cluster.verification_status == MemoryVerificationStatus::Contradicted
            || cluster.contradiction_risk >= 0.70
        {
            DeepSearchMemoryPromotionDecision {
                claim_cluster_id: cluster.id.clone(),
                stage: DeepSearchMemoryPromotionStage::BlockedContradicted,
                verification_status: MemoryVerificationStatus::Contradicted,
                confidence: cluster.confidence.min(0.40),
                salience: 0.38,
                reason: "claim has contradiction risk and must not be promoted as usable long-term knowledge".into(),
                source_refs: cluster.source_refs.clone(),
                metadata: json!({
                    "rule": "blocked_contradiction_risk",
                    "contradiction_risk": cluster.contradiction_risk,
                    "external_content_untrusted": true,
                }),
            }
        } else if cluster.support_count >= 2
            && cluster.independent_domain_count >= min_independent_sources
            && cluster.confidence >= min_confidence
            && coverage.overall_score >= min_coverage
            && (!reliability_enabled || coverage.source_reliability_score >= min_reliable_source_score)
        {
            let stage = if auto_promote_supported {
                DeepSearchMemoryPromotionStage::LlmInferredMemory
            } else {
                DeepSearchMemoryPromotionStage::ReviewRequired
            };
            let status = if auto_promote_supported {
                MemoryVerificationStatus::LlmInferred
            } else {
                MemoryVerificationStatus::Unverified
            };
            DeepSearchMemoryPromotionDecision {
                claim_cluster_id: cluster.id.clone(),
                stage,
                verification_status: status,
                confidence: cluster.confidence.clamp(min_confidence, 0.86),
                salience: 0.72,
                reason: format!(
                    "claim met bounded promotion gates: support_count={}, independent_domains={}, confidence={:.2}, coverage={:.2}",
                    cluster.support_count, cluster.independent_domain_count, cluster.confidence, coverage.overall_score
                ),
                source_refs: cluster.source_refs.clone(),
                metadata: json!({
                    "rule": "cross_source_supported_to_llm_inferred",
                    "auto_promote_supported_claims": auto_promote_supported,
                    "never_system_verified_from_external_web": true,
                    "min_confidence": min_confidence,
                    "min_independent_sources": min_independent_sources,
                    "coverage_score": coverage.overall_score,
                    "source_reliability_score": coverage.source_reliability_score,
                    "min_reliable_source_score": min_reliable_source_score,
                    "saturation_score": saturation.score,
                    "external_content_untrusted": true,
                }),
            }
        } else if !cluster.evidence_refs.is_empty() {
            DeepSearchMemoryPromotionDecision {
                claim_cluster_id: cluster.id.clone(),
                stage: DeepSearchMemoryPromotionStage::CandidateMemory,
                verification_status: MemoryVerificationStatus::Unverified,
                confidence: cluster.confidence.min(0.60),
                salience: 0.58,
                reason: "claim has evidence but did not satisfy cross-source / confidence / coverage promotion gates".into(),
                source_refs: cluster.source_refs.clone(),
                metadata: json!({
                    "rule": "evidence_linked_candidate",
                    "support_count": cluster.support_count,
                    "independent_domains": cluster.independent_domain_count,
                    "confidence": cluster.confidence,
                    "coverage_score": coverage.overall_score,
                    "source_reliability_score": coverage.source_reliability_score,
                    "min_reliable_source_score": min_reliable_source_score,
                    "external_content_untrusted": true,
                }),
            }
        } else {
            DeepSearchMemoryPromotionDecision {
                claim_cluster_id: cluster.id.clone(),
                stage: DeepSearchMemoryPromotionStage::ReviewRequired,
                verification_status: MemoryVerificationStatus::Unverified,
                confidence: 0.30,
                salience: 0.30,
                reason: "claim lacks evidence references and requires review before memory use".into(),
                source_refs: cluster.source_refs.clone(),
                metadata: json!({
                    "rule": "missing_evidence_review_required",
                    "external_content_untrusted": true,
                }),
            }
        };
        decisions.push(decision);
    }

    let promoted_claims = decisions
        .iter()
        .filter(|decision| decision.stage == DeepSearchMemoryPromotionStage::LlmInferredMemory)
        .count();
    let candidate_claims = decisions
        .iter()
        .filter(|decision| decision.stage == DeepSearchMemoryPromotionStage::CandidateMemory)
        .count();
    let review_required_claims = decisions
        .iter()
        .filter(|decision| decision.stage == DeepSearchMemoryPromotionStage::ReviewRequired)
        .count();
    let blocked_claims = decisions
        .iter()
        .filter(|decision| decision.stage == DeepSearchMemoryPromotionStage::BlockedContradicted)
        .count();

    DeepSearchPromotionReport {
        enabled,
        promoted_claims,
        candidate_claims,
        review_required_claims,
        blocked_claims,
        decisions,
        warnings,
        metadata: json!({
            "schema_version": 1,
            "pipeline": "astra_deep_search_memory_promotion_policy_v0_6_7",
            "auto_promote_supported_claims": auto_promote_supported,
            "require_user_confirmation_for_system_verified": require_user_confirmation_for_system_verified,
            "min_promotion_confidence": min_confidence,
            "min_promotion_independent_sources": min_independent_sources,
            "min_coverage_score": min_coverage,
            "coverage_score": coverage.overall_score,
            "source_reliability_score": coverage.source_reliability_score,
            "min_reliable_source_score": min_reliable_source_score,
            "reliability_scoring_enabled": reliability_enabled,
            "saturation_score": saturation.score,
            "supported_claim_ratio": saturation.supported_claim_ratio,
            "external_content_untrusted": true,
            "system_verified_blocked_for_external_content": true,
        }),
    }
}
