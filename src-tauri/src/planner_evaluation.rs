use crate::{
    desktop_agent_types::{
        PlannerContractDecision, PlannerContractInput, PlannerContractSource,
        PlannerDecisionStatus, PlannerRejectionReason, PlannerScrollIntent,
        PlannerVisibilityAssessment,
    },
    semantic_frame::{deterministic_planner_contract_decision, validate_model_planner_decision},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct PlannerEvaluationCase {
    pub name: String,
    pub input: PlannerContractInput,
    #[serde(default)]
    pub model_decision: Option<PlannerContractDecision>,
    pub expected: PlannerExpectedOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct PlannerExpectedOutcome {
    #[serde(default)]
    pub source: Option<PlannerContractSource>,
    #[serde(default)]
    pub decision_status: Option<PlannerDecisionStatus>,
    #[serde(default)]
    pub rejection_code: Option<PlannerRejectionReason>,
    #[serde(default)]
    pub visibility_assessment: Option<PlannerVisibilityAssessment>,
    #[serde(default)]
    pub scroll_intent: Option<PlannerScrollIntent>,
    #[serde(default)]
    pub fallback_used: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct PlannerEvaluationResult {
    pub name: String,
    pub passed: bool,
    pub failures: Vec<String>,
    pub actual_decision: PlannerContractDecision,
}

#[allow(dead_code)]
pub fn evaluate_planner_case(case: PlannerEvaluationCase) -> PlannerEvaluationResult {
    let deterministic = deterministic_planner_contract_decision(&case.input);
    let actual_decision = case
        .model_decision
        .clone()
        .map(|model| validate_model_planner_decision(&case.input, model, deterministic.clone()))
        .unwrap_or(deterministic);
    let mut failures = Vec::new();

    if let Some(expected) = case.expected.source {
        if actual_decision.source != expected {
            failures.push(format!(
                "source mismatch: expected {:?}, got {:?}",
                expected, actual_decision.source
            ));
        }
    }
    if let Some(expected) = case.expected.decision_status {
        if actual_decision.decision_status != expected {
            failures.push(format!(
                "decision_status mismatch: expected {:?}, got {:?}",
                expected, actual_decision.decision_status
            ));
        }
    }
    if let Some(expected) = case.expected.rejection_code {
        if actual_decision.rejection_code != Some(expected.clone()) {
            failures.push(format!(
                "rejection_code mismatch: expected {:?}, got {:?}",
                expected, actual_decision.rejection_code
            ));
        }
    }
    if let Some(expected) = case.expected.visibility_assessment {
        if actual_decision.visibility_assessment != expected {
            failures.push(format!(
                "visibility_assessment mismatch: expected {:?}, got {:?}",
                expected, actual_decision.visibility_assessment
            ));
        }
    }
    if let Some(expected) = case.expected.scroll_intent {
        if actual_decision.scroll_intent != expected {
            failures.push(format!(
                "scroll_intent mismatch: expected {:?}, got {:?}",
                expected, actual_decision.scroll_intent
            ));
        }
    }
    if let Some(expected) = case.expected.fallback_used {
        if actual_decision.fallback_used != expected {
            failures.push(format!(
                "fallback_used mismatch: expected {}, got {}",
                expected, actual_decision.fallback_used
            ));
        }
    }

    PlannerEvaluationResult {
        name: case.name,
        passed: failures.is_empty(),
        failures,
        actual_decision,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        desktop_agent_types::{
            FocusedPerceptionRequest, GoalConstraints, GoalSpec, GoalType, PerceptionRequestMode,
            PerceptionRoutingDecision, PlannerContractDecision, PlannerContractSource, PlannerStep,
            PlannerStepKind, VisibleActionabilityDiagnostic, VisibleRefinementStrategy,
            VisibleResultKind,
        },
        semantic_frame::{
            goal_for_open_media_result, planner_contract_input,
            planner_contract_input_with_perception, semantic_frame_from_vision_value,
        },
        ui_target_grounding::{
            TargetGroundingSource, TargetRegion, UITargetCandidate, UITargetRole,
        },
    };
    use serde_json::{json, Value};

    fn youtube_first_video_goal() -> GoalSpec {
        goal_for_open_media_result(
            "aprimi il primo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        )
    }

    fn youtube_search_input(frame_json: Value, goal: GoalSpec) -> PlannerContractInput {
        let frame =
            semantic_frame_from_vision_value(&frame_json, 1_000, None, None, Vec::new()).unwrap();
        planner_contract_input(&goal, &frame, &[], &[], 3, 0)
    }

    fn model_result_click(item_id: &str, confidence: f32) -> PlannerContractDecision {
        PlannerContractDecision {
            source: PlannerContractSource::ModelAssisted,
            proposed_step: PlannerStep {
                step_id: "model_step".into(),
                kind: PlannerStepKind::ClickResultRegion,
                confidence,
                rationale: "model selected visible result".into(),
                target_item_id: Some(item_id.into()),
                target_entity_id: None,
                click_region_key: Some("title".into()),
                executable_candidate: None,
                expected_state: Some("media_watch_page_visible".into()),
            },
            strategy_rationale: "model_first_video".into(),
            focused_perception_needed: false,
            replan_needed: false,
            expected_verification_target: Some("media_watch_page_visible".into()),
            planner_confidence: confidence,
            accepted: false,
            fallback_used: false,
            rejection_reason: None,
            decision_status: PlannerDecisionStatus::Accepted,
            rejection_code: None,
            visibility_assessment: PlannerVisibilityAssessment::VisibleGrounded,
            scroll_intent: PlannerScrollIntent::NotNeeded,
            visible_actionability: VisibleActionabilityDiagnostic::default(),
            target_confidence: None,
            normalized: false,
            downgraded: false,
        }
    }

    fn model_likely_offscreen_replan(confidence: f32) -> PlannerContractDecision {
        PlannerContractDecision {
            source: PlannerContractSource::ModelAssisted,
            proposed_step: PlannerStep {
                step_id: "model_replan".into(),
                kind: PlannerStepKind::ReplanAfterPerception,
                confidence,
                rationale: "target is probably below the fold".into(),
                target_item_id: None,
                target_entity_id: None,
                click_region_key: None,
                executable_candidate: None,
                expected_state: Some("more_perception_or_scroll_required".into()),
            },
            strategy_rationale: "model_scroll".into(),
            focused_perception_needed: true,
            replan_needed: true,
            expected_verification_target: Some("more_perception_or_scroll_required".into()),
            planner_confidence: confidence,
            accepted: false,
            fallback_used: false,
            rejection_reason: Some("target likely offscreen".into()),
            decision_status: PlannerDecisionStatus::Accepted,
            rejection_code: Some(PlannerRejectionReason::ScrollRequiredButUnsupported),
            visibility_assessment: PlannerVisibilityAssessment::LikelyOffscreen,
            scroll_intent: PlannerScrollIntent::RequiredButUnsupported,
            visible_actionability: VisibleActionabilityDiagnostic::default(),
            target_confidence: None,
            normalized: false,
            downgraded: false,
        }
    }

    fn run_case(case: PlannerEvaluationCase) -> PlannerEvaluationResult {
        let result = evaluate_planner_case(case);
        assert!(result.passed, "{:?}", result.failures);
        result
    }

    #[test]
    fn evaluation_accepts_visible_youtube_first_video_model_decision() {
        let input = youtube_search_input(
            json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "search_results", "result_list_visible": true, "confidence": 0.94},
                "visible_result_items": [{
                    "item_id": "video",
                    "kind": "video",
                    "rank_within_kind": 1,
                    "click_regions": {"title": {"x": 450, "y": 300, "width": 400, "height": 50, "coordinate_space": "screen"}},
                    "confidence": 0.95
                }]
            }),
            youtube_first_video_goal(),
        );

        let result = run_case(PlannerEvaluationCase {
            name: "youtube_first_video_model_accept".into(),
            input,
            model_decision: Some(model_result_click("video", 0.92)),
            expected: PlannerExpectedOutcome {
                source: Some(PlannerContractSource::ModelAssisted),
                decision_status: Some(PlannerDecisionStatus::Normalized),
                rejection_code: None,
                visibility_assessment: Some(PlannerVisibilityAssessment::VisibleGrounded),
                scroll_intent: Some(PlannerScrollIntent::NotNeeded),
                fallback_used: Some(false),
            },
        });

        assert!(result
            .actual_decision
            .proposed_step
            .executable_candidate
            .is_some());
    }

    #[test]
    fn evaluation_accepts_visible_target_with_missing_item_confidence_when_region_is_strong() {
        let input = youtube_search_input(
            json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "search_results", "result_list_visible": true, "confidence": 0.94},
                "visible_result_items": [{
                    "item_id": "video",
                    "kind": "video",
                    "title": "Shiva - Intro",
                    "rank_within_kind": 1,
                    "click_regions": {"title": {
                        "region": {"x": 450, "y": 300, "width": 400, "height": 50, "coordinate_space": "screen"},
                        "confidence": 0.93
                    }}
                }]
            }),
            youtube_first_video_goal(),
        );

        let result = run_case(PlannerEvaluationCase {
            name: "missing_item_confidence_derived_from_region".into(),
            input,
            model_decision: Some(model_result_click("video", 0.91)),
            expected: PlannerExpectedOutcome {
                source: Some(PlannerContractSource::ModelAssisted),
                decision_status: Some(PlannerDecisionStatus::Normalized),
                rejection_code: None,
                visibility_assessment: Some(PlannerVisibilityAssessment::VisibleGrounded),
                scroll_intent: Some(PlannerScrollIntent::NotNeeded),
                fallback_used: Some(false),
            },
        });

        let confidence = result
            .actual_decision
            .target_confidence
            .as_ref()
            .expect("target confidence");
        assert_eq!(confidence.raw_item_confidence, None);
        assert!(confidence.confidence_was_derived);
        assert!(confidence.derived_confidence >= confidence.required_threshold);
    }

    #[test]
    fn evaluation_rejects_fabricated_target_and_falls_back() {
        let input = youtube_search_input(
            json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "search_results", "confidence": 0.94},
                "visible_result_items": [{
                    "item_id": "video",
                    "kind": "video",
                    "rank_within_kind": 1,
                    "click_regions": {"title": {"x": 450, "y": 300, "width": 400, "height": 50, "coordinate_space": "screen"}},
                    "confidence": 0.95
                }]
            }),
            youtube_first_video_goal(),
        );

        run_case(PlannerEvaluationCase {
            name: "fabricated_target_fallback".into(),
            input,
            model_decision: Some(model_result_click("not_in_frame", 0.92)),
            expected: PlannerExpectedOutcome {
                source: Some(PlannerContractSource::ModelAssistedFallback),
                decision_status: Some(PlannerDecisionStatus::FallbackUsed),
                rejection_code: Some(PlannerRejectionReason::FabricatedTarget),
                visibility_assessment: Some(PlannerVisibilityAssessment::VisibleGrounded),
                scroll_intent: Some(PlannerScrollIntent::NotNeeded),
                fallback_used: Some(true),
            },
        });
    }

    #[test]
    fn evaluation_preserves_provider_backend_separation() {
        let input = youtube_search_input(
            json!({
                "page_evidence": {
                    "content_provider_hint": "youtube",
                    "page_kind_hint": "search_results",
                    "capture_backend": "powershell_gdi",
                    "confidence": 0.94
                },
                "visible_result_items": [{
                    "item_id": "video",
                    "kind": "video",
                    "rank_within_kind": 1,
                    "click_regions": {"title": {"x": 450, "y": 300, "width": 400, "height": 50, "coordinate_space": "screen"}},
                    "confidence": 0.95
                }]
            }),
            youtube_first_video_goal(),
        );

        let result = run_case(PlannerEvaluationCase {
            name: "provider_backend_separation".into(),
            input,
            model_decision: Some(model_result_click("video", 0.91)),
            expected: PlannerExpectedOutcome {
                source: Some(PlannerContractSource::ModelAssisted),
                decision_status: Some(PlannerDecisionStatus::Normalized),
                rejection_code: None,
                visibility_assessment: Some(PlannerVisibilityAssessment::VisibleGrounded),
                scroll_intent: Some(PlannerScrollIntent::NotNeeded),
                fallback_used: Some(false),
            },
        });

        assert_ne!(
            result
                .actual_decision
                .proposed_step
                .executable_candidate
                .as_ref()
                .and_then(|candidate| candidate.content_provider_hint.as_deref()),
            Some("powershell_gdi")
        );
    }

    #[test]
    fn evaluation_marks_correct_page_missing_video_as_scroll_required() {
        let input = youtube_search_input(
            json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "search_results", "result_list_visible": true, "confidence": 0.94},
                "visible_result_items": [{
                    "item_id": "mix",
                    "kind": "mix",
                    "rank_overall": 1,
                    "rank_within_kind": 1,
                    "click_regions": {"thumbnail": {"x": 100, "y": 100, "width": 300, "height": 180, "coordinate_space": "screen"}},
                    "confidence": 0.95
                }]
            }),
            youtube_first_video_goal(),
        );

        run_case(PlannerEvaluationCase {
            name: "scroll_required_missing_video".into(),
            input,
            model_decision: None,
            expected: PlannerExpectedOutcome {
                source: Some(PlannerContractSource::RustDeterministic),
                decision_status: Some(PlannerDecisionStatus::Accepted),
                rejection_code: Some(PlannerRejectionReason::ScrollRequiredButUnsupported),
                visibility_assessment: Some(PlannerVisibilityAssessment::LikelyOffscreen),
                scroll_intent: Some(PlannerScrollIntent::RequiredButUnsupported),
                fallback_used: Some(false),
            },
        });
    }

    #[test]
    fn evaluation_preserves_focused_perception_for_visible_ambiguous_video() {
        let input = youtube_search_input(
            json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "search_results", "result_list_visible": true, "confidence": 0.94},
                "visible_result_items": [{
                    "item_id": "video",
                    "kind": "video",
                    "rank_within_kind": 1,
                    "attributes": {"region": {"x": 100, "y": 200, "width": 500, "height": 180, "coordinate_space": "screen"}},
                    "confidence": 0.95
                }]
            }),
            youtube_first_video_goal(),
        );

        run_case(PlannerEvaluationCase {
            name: "focused_perception_visible_ambiguous_video".into(),
            input,
            model_decision: None,
            expected: PlannerExpectedOutcome {
                source: Some(PlannerContractSource::RustDeterministic),
                decision_status: Some(PlannerDecisionStatus::Accepted),
                rejection_code: Some(PlannerRejectionReason::FocusedPerceptionRequested),
                visibility_assessment: Some(
                    PlannerVisibilityAssessment::VisibleTargetNeedsClickRegion,
                ),
                scroll_intent: Some(PlannerScrollIntent::NotNeeded),
                fallback_used: Some(false),
            },
        });
    }

    #[test]
    fn evaluation_marks_visible_under_grounded_results_as_refinement_before_scroll() {
        let input = youtube_search_input(
            json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "search_results", "result_list_visible": true, "confidence": 0.94},
                "visible_result_items": [{
                    "item_id": "candidate_1",
                    "kind": "unknown",
                    "attributes": {"region": {"x": 100, "y": 180, "width": 520, "height": 180, "coordinate_space": "screen"}},
                    "confidence": 0.9
                }],
                "visible_entities": [{
                    "entity_id": "title_1",
                    "kind": "title_link",
                    "name": "Shiva - Intro",
                    "region": {"x": 420, "y": 200, "width": 320, "height": 42, "coordinate_space": "screen"},
                    "confidence": 0.84
                }]
            }),
            youtube_first_video_goal(),
        );

        run_case(PlannerEvaluationCase {
            name: "visible_under_grounded_refinement_before_scroll".into(),
            input,
            model_decision: None,
            expected: PlannerExpectedOutcome {
                source: Some(PlannerContractSource::RustDeterministic),
                decision_status: Some(PlannerDecisionStatus::Accepted),
                rejection_code: Some(PlannerRejectionReason::VisiblePageRefinementRequested),
                visibility_assessment: Some(PlannerVisibilityAssessment::VisibleUnderGrounded),
                scroll_intent: Some(PlannerScrollIntent::NotNeeded),
                fallback_used: Some(false),
            },
        });
    }

    #[test]
    fn evaluation_downgrades_model_offscreen_when_visible_refinement_is_still_eligible() {
        let input = youtube_search_input(
            json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "search_results", "result_list_visible": true, "confidence": 0.94},
                "visible_result_items": [{
                    "item_id": "candidate_1",
                    "kind": "unknown",
                    "attributes": {"region": {"x": 100, "y": 180, "width": 520, "height": 180, "coordinate_space": "screen"}},
                    "confidence": 0.9
                }],
                "visible_entities": [{
                    "entity_id": "title_1",
                    "kind": "title_link",
                    "name": "Shiva - Intro",
                    "region": {"x": 420, "y": 200, "width": 320, "height": 42, "coordinate_space": "screen"},
                    "confidence": 0.84
                }]
            }),
            youtube_first_video_goal(),
        );

        run_case(PlannerEvaluationCase {
            name: "model_offscreen_downgraded_to_visible_refinement".into(),
            input,
            model_decision: Some(model_likely_offscreen_replan(0.88)),
            expected: PlannerExpectedOutcome {
                source: Some(PlannerContractSource::ModelAssistedFallback),
                decision_status: Some(PlannerDecisionStatus::FallbackUsed),
                rejection_code: Some(PlannerRejectionReason::VisiblePageRefinementRequested),
                visibility_assessment: Some(PlannerVisibilityAssessment::VisibleUnderGrounded),
                scroll_intent: Some(PlannerScrollIntent::NotNeeded),
                fallback_used: Some(true),
            },
        });
    }

    #[test]
    fn evaluation_covers_channel_goal_video_fallback() {
        let goal = GoalSpec {
            goal_id: "channel_goal".into(),
            goal_type: GoalType::OpenChannel,
            constraints: GoalConstraints {
                provider: Some("youtube".into()),
                result_kind: None,
                rank_within_kind: None,
                rank_overall: None,
                entity_name: Some("Shiva".into()),
                attributes: Value::Null,
            },
            success_condition: "channel_page_visible".into(),
            utterance: "vai sul canale di Shiva".into(),
            confidence: 0.86,
        };
        let input = youtube_search_input(
            json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "search_results", "confidence": 0.92},
                "visible_result_items": [{
                    "item_id": "video",
                    "kind": "video",
                    "title": "Shiva - Bacio di Giuda",
                    "channel_name": "SHIVA",
                    "rank_within_kind": 1,
                    "click_regions": {"title": {"x": 450, "y": 300, "width": 400, "height": 50, "coordinate_space": "screen"}},
                    "confidence": 0.94
                }]
            }),
            goal,
        );

        let result = run_case(PlannerEvaluationCase {
            name: "channel_goal_video_fallback".into(),
            input,
            model_decision: None,
            expected: PlannerExpectedOutcome {
                source: Some(PlannerContractSource::RustDeterministic),
                decision_status: Some(PlannerDecisionStatus::Accepted),
                rejection_code: None,
                visibility_assessment: Some(PlannerVisibilityAssessment::VisibleGrounded),
                scroll_intent: Some(PlannerScrollIntent::NotNeeded),
                fallback_used: Some(false),
            },
        });

        assert_eq!(
            result
                .actual_decision
                .proposed_step
                .expected_state
                .as_deref(),
            Some("media_watch_page_visible")
        );
    }

    #[test]
    fn evaluation_uses_legacy_candidate_fallback_after_regionless_refinement() {
        let goal = youtube_first_video_goal();
        let legacy_candidate = UITargetCandidate {
            candidate_id: "legacy_video".into(),
            role: UITargetRole::RankedResult,
            region: Some(TargetRegion {
                x: 430.0,
                y: 242.0,
                width: 300.0,
                height: 40.0,
                coordinate_space: "screen".into(),
            }),
            center_x: None,
            center_y: None,
            app_hint: None,
            browser_app_hint: None,
            provider_hint: Some("youtube".into()),
            content_provider_hint: Some("youtube".into()),
            page_kind_hint: Some("search_results".into()),
            capture_backend: None,
            observation_source: Some("structured_vision".into()),
            result_kind: Some("video".into()),
            confidence: 0.93,
            source: TargetGroundingSource::ScreenAnalysis,
            label: Some("Shiva - Intro".into()),
            rank: Some(1),
            observed_at_ms: Some(1_500),
            reuse_eligible: true,
            supports_focus: false,
            supports_click: true,
            rationale: "validated legacy target candidate".into(),
        };
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "search_results", "result_list_visible": true, "confidence": 0.94},
                "visible_result_items": [{
                    "item_id": "video",
                    "kind": "video",
                    "rank_within_kind": 1,
                    "title": "Shiva - Intro",
                    "confidence": 0.95
                }]
            }),
            1_000,
            None,
            None,
            vec![legacy_candidate],
        )
        .expect("frame");
        let refinement_request = FocusedPerceptionRequest {
            request_id: "refine_1".into(),
            iteration: 0,
            reason: "regionless visible target".into(),
            mode: PerceptionRequestMode::VisiblePageRefinement,
            routing_decision: PerceptionRoutingDecision::RegionlessTargetVisible,
            refinement_strategy: VisibleRefinementStrategy::VisibleCluster,
            target_item_id: Some("video".into()),
            target_entity_id: None,
            region: None,
            target_region_anchor_present: false,
        };
        let input = planner_contract_input_with_perception(
            &goal,
            &frame,
            &[],
            &[],
            &[refinement_request],
            3,
            0,
            1,
        );

        let result = run_case(PlannerEvaluationCase {
            name: "legacy_fallback_after_regionless_refinement".into(),
            input,
            model_decision: None,
            expected: PlannerExpectedOutcome {
                source: Some(PlannerContractSource::RustDeterministic),
                decision_status: Some(PlannerDecisionStatus::Accepted),
                rejection_code: None,
                visibility_assessment: Some(PlannerVisibilityAssessment::VisibleGrounded),
                scroll_intent: Some(PlannerScrollIntent::NotNeeded),
                fallback_used: Some(false),
            },
        });

        assert_eq!(
            result
                .actual_decision
                .proposed_step
                .executable_candidate
                .as_ref()
                .and_then(|candidate| candidate.observation_source.as_deref()),
            Some("planner_legacy_candidate_fallback")
        );
    }
}
