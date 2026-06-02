use crate::{
    cognitive_thinking::{DeepSearchReason, ThinkingPlan, ThinkingRoute, ThinkingUncertaintyLevel},
    memory::consolidation::{
        ConversationDecision, ConversationImportantPoint, ConversationMemoryBundle,
        ConversationSemanticAtom,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const MAX_MESSAGE_CHARS: usize = 1_200;
const MAX_ANSWER_CHARS: usize = 2_400;
const MAX_TITLE_CHARS: usize = 160;
const MAX_SUMMARY_CHARS: usize = 900;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingMemoryFeedbackReceipt {
    pub accepted: bool,
    pub reason: String,
    pub request_id: Option<String>,
    pub learning_score: f32,
    pub durable_candidate_count: usize,
    pub review_required: bool,
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
}

pub fn build_thinking_memory_feedback_bundle(
    request_id: Option<String>,
    source: String,
    user_message: String,
    assistant_answer: String,
    plan: ThinkingPlan,
    min_score: f32,
) -> Result<(ConversationMemoryBundle, ThinkingMemoryFeedbackReceipt), ThinkingMemoryFeedbackReceipt> {
    let learning_score = score_learning_value(&plan, &user_message, &assistant_answer);
    let tags = feedback_tags(&plan);
    if learning_score < min_score {
        return Err(ThinkingMemoryFeedbackReceipt {
            accepted: false,
            reason: "thinking_feedback_below_learning_threshold".into(),
            request_id,
            learning_score,
            durable_candidate_count: 0,
            review_required: false,
            tags,
            metadata: json!({
                "min_score": min_score,
                "thinking_route": route_label(&plan.route),
                "planner_source": plan.planner_source.clone(),
                "metadata_only": true,
            }),
        });
    }

    let review_required = true;
    let mut semantic_atoms = Vec::new();
    let mut important_points = Vec::new();
    let mut decisions = Vec::new();

    semantic_atoms.push(ConversationSemanticAtom {
        title: Some(cap_text(
            format!("Thinking route for request: {}", route_label(&plan.route)),
            MAX_TITLE_CHARS,
        )),
        summary: Some(cap_text(
            format!(
                "Astra selected route '{}' with {:.0}% confidence. Intent: {}",
                route_label(&plan.route),
                plan.confidence.clamp(0.0, 1.0) * 100.0,
                plan.intent_summary
            ),
            MAX_SUMMARY_CHARS,
        )),
        subject: Some("astra_thinking_runtime".into()),
        predicate: Some("selected_route".into()),
        object: Some(route_label(&plan.route).into()),
        evidence: Some(cap_text(&user_message, MAX_MESSAGE_CHARS)),
        kind: Some("claim".into()),
        confidence: Some((plan.confidence * 0.82).clamp(0.35, 0.86)),
        tags: normalize_feedback_tags(vec![
            "thinking_feedback".into(),
            "requires_review".into(),
            "metacognitive_memory".into(),
            "not_canonical".into(),
        ], &tags),
        metadata: json!({
            "schema_version": 1,
            "source": "thinking_memory_feedback_loop",
            "thinking_route": route_label(&plan.route),
            "planner_source": plan.planner_source.clone(),
            "review_required": review_required,
            "auto_promote": false,
            "raw_chain_of_thought_included": false,
            "metadata_only": true,
        }),
    });

    if plan.deep_search.is_needed() {
        important_points.push(ConversationImportantPoint {
            title: "Deep Search decision candidate".into(),
            summary: cap_text(
                format!(
                    "Thinking decided that Deep Search was needed because '{}'. Query hint: {}",
                    deep_search_reason_label(&plan.deep_search.reason),
                    plan.deep_search.query_hint.as_deref().unwrap_or("none")
                ),
                MAX_SUMMARY_CHARS,
            ),
            kind: Some("research_decision".into()),
            confidence: Some((plan.confidence * 0.8).clamp(0.4, 0.84)),
            tags: normalize_feedback_tags(vec![
                "thinking_feedback".into(),
                "deep_search_decision".into(),
                "requires_review".into(),
            ], &tags),
            metadata: json!({
                "deep_search_reason": deep_search_reason_label(&plan.deep_search.reason),
                "query_hint": plan.deep_search.query_hint.clone(),
                "auto_promote": false,
                "metadata_only": true,
            }),
        });
    }

    if plan.memory_assessment.relevant || plan.memory_assessment.coverage > 0.0 {
        important_points.push(ConversationImportantPoint {
            title: "Memory coverage assessment candidate".into(),
            summary: cap_text(
                format!(
                    "Memory coverage was estimated at {:.0}% across {} node(s). Missing information: {}",
                    plan.memory_assessment.coverage.clamp(0.0, 1.0) * 100.0,
                    plan.memory_assessment.node_count,
                    if plan.memory_assessment.missing_information.is_empty() {
                        "none".into()
                    } else {
                        plan.memory_assessment.missing_information.join("; ")
                    }
                ),
                MAX_SUMMARY_CHARS,
            ),
            kind: Some("memory_assessment".into()),
            confidence: Some(0.7),
            tags: normalize_feedback_tags(vec![
                "thinking_feedback".into(),
                "memory_coverage".into(),
                "requires_review".into(),
            ], &tags),
            metadata: json!({
                "memory_nodes": plan.memory_assessment.node_count,
                "memory_coverage": plan.memory_assessment.coverage,
                "auto_promote": false,
                "metadata_only": true,
            }),
        });
    }

    if plan.evidence_assessment.requires_current_information
        || plan.evidence_assessment.requires_external_sources
        || matches!(&plan.uncertainty.level, ThinkingUncertaintyLevel::Medium | ThinkingUncertaintyLevel::High)
    {
        decisions.push(ConversationDecision {
            title: "Evidence/uncertainty learning candidate".into(),
            summary: cap_text(
                format!(
                    "Evidence summary: {}. Uncertainty: {}. Reasons: {}",
                    plan.evidence_assessment.evidence_summary,
                    uncertainty_label(&plan.uncertainty.level),
                    if plan.uncertainty.reasons.is_empty() {
                        "none".into()
                    } else {
                        plan.uncertainty.reasons.join("; ")
                    }
                ),
                MAX_SUMMARY_CHARS,
            ),
            confidence: Some((plan.confidence * 0.76).clamp(0.4, 0.82)),
            metadata: json!({
                "source": "thinking_memory_feedback_loop",
                "review_required": true,
                "auto_promote": false,
                "uncertainty_level": uncertainty_label(&plan.uncertainty.level),
                "metadata_only": true,
            }),
        });
    }

    let durable_candidate_count = semantic_atoms.len() + important_points.len() + decisions.len();
    if durable_candidate_count == 0 {
        return Err(ThinkingMemoryFeedbackReceipt {
            accepted: false,
            reason: "thinking_feedback_produced_no_durable_candidates".into(),
            request_id,
            learning_score,
            durable_candidate_count: 0,
            review_required,
            tags,
            metadata: json!({"metadata_only": true}),
        });
    }

    let topic = cap_text(
        format!("Thinking feedback: {}", plan.intent_summary),
        MAX_TITLE_CHARS,
    );
    let bundle = ConversationMemoryBundle {
        request_id: request_id.clone(),
        source: Some(source),
        user_message: cap_text(user_message, MAX_MESSAGE_CHARS),
        assistant_answer: cap_text(assistant_answer, MAX_ANSWER_CHARS),
        topic: Some(topic),
        summary: Some(cap_text(
            format!(
                "Governed Thinking feedback loop generated {} review-gated learning candidate(s) for route '{}'.",
                durable_candidate_count,
                route_label(&plan.route)
            ),
            MAX_SUMMARY_CHARS,
        )),
        importance: Some(learning_score.clamp(0.0, 1.0)),
        confidence: Some((plan.confidence * 0.78).clamp(0.35, 0.84)),
        tags: normalize_feedback_tags(vec![
            "thinking_feedback".into(),
            "metacognitive_memory".into(),
            "requires_review".into(),
            "not_canonical".into(),
            "no_raw_chain_of_thought".into(),
        ], &tags),
        semantic_atoms,
        important_points,
        entities: Vec::new(),
        preferences: Vec::new(),
        procedures: Vec::new(),
        decisions,
        metadata: json!({
            "schema_version": 1,
            "source": "thinking_memory_feedback_loop",
            "thinking_plan_request_id": plan.request_id.clone(),
            "thinking_route": route_label(&plan.route),
            "planner_source": plan.planner_source.clone(),
            "thinking_confidence": plan.confidence,
            "duration_ms": plan.duration_ms,
            "learning_score": learning_score,
            "review_required": review_required,
            "auto_promote": false,
            "canonical_memory_candidate": false,
            "raw_chain_of_thought_included": false,
            "governance_note": "Thinking feedback stores only metadata-safe review candidates; canonical promotion remains governed by Brain Review/autopilot policy.",
            "metadata_only": true,
        }),
    };

    let receipt = ThinkingMemoryFeedbackReceipt {
        accepted: true,
        reason: "thinking_feedback_bundle_ready_for_review_gated_consolidation".into(),
        request_id,
        learning_score,
        durable_candidate_count,
        review_required,
        tags,
        metadata: json!({
            "thinking_route": route_label(&plan.route),
            "planner_source": plan.planner_source.clone(),
            "auto_promote": false,
            "metadata_only": true,
        }),
    };
    Ok((bundle, receipt))
}

fn score_learning_value(plan: &ThinkingPlan, user_message: &str, assistant_answer: &str) -> f32 {
    let mut score = 0.22f32;
    if user_message.chars().count() >= 40 {
        score += 0.08;
    }
    if assistant_answer.chars().count() >= 220 {
        score += 0.08;
    }
    match &plan.route {
        ThinkingRoute::DeepSearchRequired => score += 0.24,
        ThinkingRoute::ToolArbitrationRequired => score += 0.18,
        ThinkingRoute::MemoryGroundedAnswer => score += 0.16,
        ThinkingRoute::ClarifyRequired | ThinkingRoute::Refuse => score += 0.12,
        ThinkingRoute::DirectAnswer => score += 0.04,
    }
    if plan.deep_search.is_needed() {
        score += 0.14;
    }
    if plan.memory_assessment.relevant || plan.memory_assessment.coverage >= 0.35 {
        score += 0.10;
    }
    if plan.evidence_assessment.requires_external_sources || plan.evidence_assessment.requires_current_information {
        score += 0.10;
    }
    if !plan.warnings.is_empty() {
        score += 0.04;
    }
    if matches!(&plan.uncertainty.level, ThinkingUncertaintyLevel::Medium | ThinkingUncertaintyLevel::High) {
        score += 0.08;
    }
    (score * plan.confidence.clamp(0.35, 1.0)).clamp(0.0, 1.0)
}

fn feedback_tags(plan: &ThinkingPlan) -> Vec<String> {
    let mut tags = vec![
        "thinking_feedback".into(),
        "review_gated".into(),
        route_label(&plan.route).into(),
    ];
    if plan.deep_search.is_needed() {
        tags.push("deep_search_related".into());
    }
    if plan.memory_assessment.relevant {
        tags.push("memory_related".into());
    }
    if plan.evidence_assessment.requires_external_sources {
        tags.push("external_evidence_related".into());
    }
    tags
}

fn normalize_feedback_tags(mut base: Vec<String>, extra: &[String]) -> Vec<String> {
    for tag in extra {
        if !base.iter().any(|existing| existing.eq_ignore_ascii_case(tag)) {
            base.push(cap_text(tag, 48));
        }
    }
    base.sort();
    base.dedup();
    base.truncate(24);
    base
}

fn route_label(route: &ThinkingRoute) -> &'static str {
    match route {
        ThinkingRoute::DirectAnswer => "direct_answer",
        ThinkingRoute::MemoryGroundedAnswer => "memory_grounded_answer",
        ThinkingRoute::ToolArbitrationRequired => "tool_arbitration_required",
        ThinkingRoute::DeepSearchRequired => "deep_search_required",
        ThinkingRoute::ClarifyRequired => "clarify_required",
        ThinkingRoute::Refuse => "refuse",
    }
}

fn deep_search_reason_label(reason: &DeepSearchReason) -> &'static str {
    match reason {
        DeepSearchReason::NotNeeded => "not_needed",
        DeepSearchReason::UnknownTopic => "unknown_topic",
        DeepSearchReason::CurrentInformation => "current_information",
        DeepSearchReason::LowMemoryCoverage => "low_memory_coverage",
        DeepSearchReason::HighStakes => "high_stakes",
        DeepSearchReason::BlockedByPolicy => "blocked_by_policy",
    }
}

fn uncertainty_label(level: &ThinkingUncertaintyLevel) -> &'static str {
    match level {
        ThinkingUncertaintyLevel::Low => "low",
        ThinkingUncertaintyLevel::Medium => "medium",
        ThinkingUncertaintyLevel::High => "high",
    }
}

fn cap_text(value: impl AsRef<str>, max_chars: usize) -> String {
    let trimmed = value.as_ref().trim().replace('\0', " ");
    if trimmed.chars().count() <= max_chars {
        return trimmed;
    }
    let mut capped = trimmed.chars().take(max_chars.saturating_sub(1)).collect::<String>();
    capped.push('…');
    capped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_thinking::{
        DeepSearchDecision, EvidenceAssessment, MemoryAssessment, ThinkingUncertainty,
        ToolDecision,
    };

    fn plan(route: ThinkingRoute) -> ThinkingPlan {
        ThinkingPlan {
            request_id: "req_test_thinking".into(),
            intent_summary: "test intent".into(),
            self_questions: Vec::new(),
            memory_assessment: MemoryAssessment {
                relevant: true,
                coverage: 0.62,
                node_count: 4,
                missing_information: Vec::new(),
            },
            evidence_assessment: EvidenceAssessment {
                has_local_evidence: true,
                has_current_session_evidence: false,
                requires_current_information: matches!(route, ThinkingRoute::DeepSearchRequired),
                requires_external_sources: matches!(route, ThinkingRoute::DeepSearchRequired),
                evidence_summary: "test evidence".into(),
            },
            route,
            deep_search: DeepSearchDecision {
                needed: true,
                reason: DeepSearchReason::CurrentInformation,
                query_hint: Some("test query".into()),
            },
            tool_decision: ToolDecision {
                tool_required: false,
                reason: None,
                candidate_tool: None,
            },
            uncertainty: ThinkingUncertainty {
                level: ThinkingUncertaintyLevel::Medium,
                reasons: vec!["needs current evidence".into()],
            },
            user_visible_trace: Vec::new(),
            confidence: 0.82,
            planner_source: "test".into(),
            duration_ms: 12,
            warnings: Vec::new(),
        }
    }

    #[test]
    fn feedback_bundle_is_review_gated_and_no_raw_cot() {
        let (bundle, receipt) = build_thinking_memory_feedback_bundle(
            Some("req_test_thinking".into()),
            "normal_chat".into(),
            "Fammi una ricerca aggiornata su una tecnologia recente e spiegami cosa fare.".into(),
            "Risposta lunga con sintesi, evidenze e conclusioni operative governate.".repeat(8),
            plan(ThinkingRoute::DeepSearchRequired),
            0.25,
        )
        .expect("feedback should be accepted above threshold");

        assert!(receipt.accepted);
        assert!(receipt.review_required);
        assert!(receipt.durable_candidate_count >= 2);
        assert!(bundle.tags.iter().any(|tag| tag == "requires_review"));
        assert_eq!(
            bundle
                .metadata
                .get("raw_chain_of_thought_included")
                .and_then(serde_json::Value::as_bool),
            Some(false)
        );
        assert_eq!(
            bundle
                .metadata
                .get("auto_promote")
                .and_then(serde_json::Value::as_bool),
            Some(false)
        );
    }

    #[test]
    fn feedback_respects_min_score_gate() {
        let skipped = build_thinking_memory_feedback_bundle(
            Some("req_test_skip".into()),
            "normal_chat".into(),
            "ciao".into(),
            "ok".into(),
            plan(ThinkingRoute::DirectAnswer),
            0.95,
        )
        .expect_err("high threshold should reject weak feedback");

        assert!(!skipped.accepted);
        assert_eq!(skipped.reason, "thinking_feedback_below_learning_threshold");
        assert_eq!(skipped.durable_candidate_count, 0);
    }
}
