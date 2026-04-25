use crate::{
    desktop_agent_types::{
        ActionableControl, BrowserHandoffStatus, BrowserHandoffVerificationDecision,
        BrowserHandoffVerificationDiagnostic, BrowserPageSemanticKind, BrowserRecoveryStatus,
        BrowserVisualHandoffResult, ClickRegion, ConfidenceSignalState, ExecutableFallbackSource,
        ExecutableTargetConfidenceDiagnostic, FocusedPerceptionFailureReason,
        FocusedPerceptionRequest, FrameUncertainty, GoalConstraints, GoalLoopRun, GoalLoopStatus,
        GoalSpec, GoalType, GoalVerificationRecord, GoalVerificationStatus, InteractionSurfaceKind,
        OffscreenInferenceStage, PageEvidenceSource, PageSemanticEvidence, PageState,
        PerceptionRequestMode, PerceptionRoutingDecision, PlannerContractDecision,
        PlannerContractInput, PlannerContractSource, PlannerDecisionDiagnostic,
        PlannerDecisionStatus, PlannerRejectionReason, PlannerScrollIntent, PlannerStep,
        PlannerStepExecutionRecord, PlannerStepExecutionStatus, PlannerStepKind,
        PlannerVisibilityAssessment, PrimaryList, PrimaryListItem, SemanticScreenFrame,
        SurfaceOwnershipDiagnostic, SurfaceOwnershipStatus, VerifiedInteractionSurface,
        VisibleActionabilityDiagnostic, VisibleActionabilityStatus, VisibleEntity,
        VisibleEntityKind, VisibleGroundingGap, VisibleRefinementStrategy, VisibleResultItem,
        VisibleResultKind,
    },
    ui_target_grounding::{
        ground_targets_for_request, normalize_browser_app_hint, normalize_content_provider_hint,
        select_target_candidate, TargetAction, TargetGroundingRequest, TargetGroundingSource,
        TargetRegion, TargetSelectionPolicy, UITargetCandidate, UITargetRole,
    },
};
use serde_json::Value;
use std::{collections::HashMap, future::Future, pin::Pin};
use uuid::Uuid;

const MIN_PLANNER_CLICK_CONFIDENCE: f32 = 0.86;
const MAX_BROWSER_RECOVERY_ATTEMPTS: usize = 1;
const MAX_IDENTICAL_CLICK_ATTEMPTS: usize = 2;

pub fn parse_semantic_screen_frame(
    content: &str,
    captured_at: u64,
    image_path: Option<String>,
    fallback_page_evidence: Option<PageSemanticEvidence>,
    legacy_candidates: Vec<UITargetCandidate>,
) -> Result<SemanticScreenFrame, String> {
    let json_text = extract_json_object(content)
        .ok_or_else(|| "semantic frame response did not contain a JSON object".to_string())?;
    let parsed: Value = serde_json::from_str(json_text)
        .map_err(|error| format!("semantic frame JSON parse failed: {error}"))?;
    semantic_frame_from_vision_value(
        &parsed,
        captured_at,
        image_path,
        fallback_page_evidence,
        legacy_candidates,
    )
    .ok_or_else(|| "semantic frame response did not contain useful structured perception".into())
}

pub fn semantic_frame_from_vision_value(
    parsed: &Value,
    captured_at: u64,
    image_path: Option<String>,
    fallback_page_evidence: Option<PageSemanticEvidence>,
    legacy_candidates: Vec<UITargetCandidate>,
) -> Option<SemanticScreenFrame> {
    let frame_value = parsed
        .get("semantic_frame")
        .or_else(|| parsed.get("screen_frame"))
        .or_else(|| parsed.get("frame"))
        .unwrap_or(parsed);

    let page_evidence = page_evidence_from_value(
        frame_value
            .get("page_evidence")
            .or_else(|| parsed.get("page_evidence")),
    )
    .or(fallback_page_evidence)
    .unwrap_or_else(|| PageSemanticEvidence {
        browser_app_hint: None,
        content_provider_hint: None,
        page_kind_hint: None,
        query_hint: None,
        result_list_visible: None,
        raw_confidence: None,
        confidence: 0.0,
        evidence_sources: vec![PageEvidenceSource::Heuristic],
        capture_backend: None,
        observation_source: Some("semantic_frame_fallback".into()),
        uncertainty: vec!["missing_page_evidence".into()],
    });

    let scene_summary = frame_value
        .get("scene_summary")
        .or_else(|| frame_value.get("summary"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| summary_from_evidence(&page_evidence, &legacy_candidates));

    let visible_entities = parse_visible_entities(frame_value.get("visible_entities"));
    let mut visible_result_items =
        parse_visible_result_items(frame_value.get("visible_result_items"));
    if visible_result_items.is_empty() {
        visible_result_items = result_items_from_candidates(&legacy_candidates);
    }
    let (primary_list, primary_list_uncertainty) =
        parse_primary_list(frame_value.get("primary_list"));
    let page_state = parse_page_state(frame_value.get("page_state"));
    let actionable_controls = parse_actionable_controls(frame_value.get("actionable_controls"));
    let mut uncertainty = parse_uncertainty(frame_value.get("uncertainty"));
    uncertainty.extend(primary_list_uncertainty);

    let has_structured_content = !visible_entities.is_empty()
        || !visible_result_items.is_empty()
        || primary_list
            .as_ref()
            .is_some_and(|list| !list.items.is_empty())
        || page_state.is_some()
        || !actionable_controls.is_empty()
        || !legacy_candidates.is_empty()
        || page_evidence.confidence > 0.0;

    has_structured_content.then(|| SemanticScreenFrame {
        frame_id: frame_value
            .get("frame_id")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| Uuid::new_v4().to_string()),
        captured_at,
        image_path,
        page_evidence,
        scene_summary,
        visible_entities,
        visible_result_items,
        primary_list,
        page_state,
        actionable_controls,
        legacy_target_candidates: legacy_candidates,
        uncertainty,
    })
}

fn extract_json_object(content: &str) -> Option<&str> {
    let trimmed = content.trim();
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        return Some(trimmed);
    }
    if let Some(start) = trimmed.find("```json") {
        let after_fence = &trimmed[start + "```json".len()..];
        if let Some(end) = after_fence.find("```") {
            let body = after_fence[..end].trim();
            if body.starts_with('{') && body.ends_with('}') {
                return Some(body);
            }
        }
    }
    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    (end > start).then_some(trimmed[start..=end].trim())
}

pub fn semantic_frame_from_candidates(
    captured_at: u64,
    image_path: Option<String>,
    page_evidence: Option<PageSemanticEvidence>,
    legacy_candidates: Vec<UITargetCandidate>,
) -> SemanticScreenFrame {
    let page_evidence = page_evidence.unwrap_or_else(|| PageSemanticEvidence {
        browser_app_hint: None,
        content_provider_hint: None,
        page_kind_hint: None,
        query_hint: None,
        result_list_visible: None,
        raw_confidence: None,
        confidence: 0.0,
        evidence_sources: vec![PageEvidenceSource::TargetCandidate],
        capture_backend: None,
        observation_source: Some("candidate_bridge".into()),
        uncertainty: Vec::new(),
    });
    let visible_result_items = result_items_from_candidates(&legacy_candidates);
    SemanticScreenFrame {
        frame_id: Uuid::new_v4().to_string(),
        captured_at,
        image_path,
        scene_summary: summary_from_evidence(&page_evidence, &legacy_candidates),
        page_evidence,
        visible_entities: Vec::new(),
        visible_result_items,
        primary_list: None,
        page_state: None,
        actionable_controls: Vec::new(),
        legacy_target_candidates: legacy_candidates,
        uncertainty: Vec::new(),
    }
}

pub fn plan_next_step(goal: &GoalSpec, frame: &SemanticScreenFrame) -> PlannerStep {
    if let Some(status) = verify_goal_against_frame(goal, frame) {
        return PlannerStep {
            step_id: Uuid::new_v4().to_string(),
            kind: PlannerStepKind::VerifyGoal,
            confidence: 0.92,
            rationale: format!("goal already appears satisfied: {status}"),
            target_item_id: None,
            target_entity_id: None,
            click_region_key: None,
            executable_candidate: None,
            expected_state: Some(status),
        };
    }

    match goal.goal_type {
        GoalType::OpenListItem | GoalType::OpenMediaResult => plan_open_list_item(goal, frame),
        GoalType::OpenChannel => plan_open_channel(goal, frame),
        GoalType::InspectScreen => PlannerStep {
            step_id: Uuid::new_v4().to_string(),
            kind: PlannerStepKind::NoOp,
            confidence: frame.page_evidence.confidence,
            rationale: "screen inspection goal only requires perception".into(),
            target_item_id: None,
            target_entity_id: None,
            click_region_key: None,
            executable_candidate: None,
            expected_state: Some("semantic_frame_available".into()),
        },
        _ => PlannerStep {
            step_id: Uuid::new_v4().to_string(),
            kind: PlannerStepKind::Refuse,
            confidence: 0.0,
            rationale: "planner does not support this goal type yet".into(),
            target_item_id: None,
            target_entity_id: None,
            click_region_key: None,
            executable_candidate: None,
            expected_state: None,
        },
    }
}

pub fn run_goal_loop_once(goal: GoalSpec, frame: SemanticScreenFrame) -> GoalLoopRun {
    let step = plan_next_step(&goal, &frame);
    let status = match step.kind {
        PlannerStepKind::VerifyGoal | PlannerStepKind::NoOp => GoalLoopStatus::GoalAchieved,
        PlannerStepKind::ClickResultRegion | PlannerStepKind::ClickEntityRegion => {
            GoalLoopStatus::NeedsExecution
        }
        PlannerStepKind::ReplanAfterPerception => GoalLoopStatus::NeedsPerception,
        PlannerStepKind::RequestClarification => GoalLoopStatus::NeedsClarification,
        PlannerStepKind::Refuse => GoalLoopStatus::Refused,
    };
    GoalLoopRun {
        run_id: Uuid::new_v4().to_string(),
        goal,
        status,
        iteration_count: 1,
        retry_budget: 1,
        retries_used: 0,
        current_strategy: Some("single_pass_planner".into()),
        fallback_strategy_state: None,
        frames: vec![frame],
        planner_steps: vec![step],
        planner_diagnostics: Vec::new(),
        executed_steps: Vec::new(),
        verification_history: Vec::new(),
        focused_perception_requests: Vec::new(),
        browser_handoff_history: Vec::new(),
        browser_handoff: None,
        verified_surface: None,
        surface_diagnostics: Vec::new(),
        focused_perception_used: false,
        visible_refinement_used: false,
        stale_capture_reuse_prevented: false,
        browser_recovery_used: false,
        browser_recovery_status: BrowserRecoveryStatus::NotNeeded,
        post_action_progress_observed: false,
        surface_ownership_lost: false,
        focused_perception_failure_reason: None,
        repeated_click_protection_triggered: false,
        selected_target_candidate: None,
        verifier_status: None,
        failure_reason: None,
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GoalLoopRuntimeConfig {
    pub max_iterations: usize,
    pub retry_budget: usize,
    pub max_focused_perception_passes: usize,
    pub max_visible_refinement_passes: usize,
    pub min_execution_confidence: f32,
}

impl Default for GoalLoopRuntimeConfig {
    fn default() -> Self {
        Self {
            max_iterations: 6,
            retry_budget: 3,
            max_focused_perception_passes: 2,
            max_visible_refinement_passes: 1,
            min_execution_confidence: MIN_PLANNER_CLICK_CONFIDENCE,
        }
    }
}

#[allow(dead_code)]
pub type GoalLoopDriverFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[allow(dead_code)]
pub trait GoalLoopDriver {
    fn prepare_visual_handoff<'a>(
        &'a mut self,
        _goal: &'a GoalSpec,
        _iteration: usize,
    ) -> GoalLoopDriverFuture<'a, Result<Option<BrowserVisualHandoffResult>, String>> {
        Box::pin(async { Ok(None) })
    }

    fn perceive<'a>(
        &'a mut self,
        goal: &'a GoalSpec,
        iteration: usize,
        fresh_capture_required: bool,
    ) -> GoalLoopDriverFuture<'a, Result<SemanticScreenFrame, String>>;
    fn execute_planner_step<'a>(
        &'a mut self,
        step: &'a PlannerStep,
    ) -> GoalLoopDriverFuture<'a, PlannerStepExecutionRecord>;
    fn focused_perception<'a>(
        &'a mut self,
        request: &'a FocusedPerceptionRequest,
    ) -> GoalLoopDriverFuture<'a, Result<Option<SemanticScreenFrame>, String>>;
    fn plan<'a>(
        &'a mut self,
        _input: &'a PlannerContractInput,
    ) -> GoalLoopDriverFuture<'a, Result<Option<PlannerContractDecision>, String>> {
        Box::pin(async { Ok(None) })
    }

    fn recover_browser_surface<'a>(
        &'a mut self,
        _goal: &'a GoalSpec,
        _iteration: usize,
        _reason: &'a str,
    ) -> GoalLoopDriverFuture<'a, Result<Option<BrowserVisualHandoffResult>, String>> {
        Box::pin(async { Ok(None) })
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GoalLoopRuntime {
    config: GoalLoopRuntimeConfig,
}

#[derive(Debug, Clone)]
struct PendingPostClickVerification {
    step_index: usize,
    target_signature: Option<String>,
    geometry_signature: Option<String>,
    pre_click_frame_signature: String,
    pre_click_list_surface_visible: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PostClickVerificationOutcome {
    goal_achieved: bool,
    progress_observed: bool,
    frame_unchanged: bool,
    browser_surface_suspect: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenListItemVerificationState {
    Achieved(&'static str),
    Ambiguous(&'static str),
    NotAchieved,
}

#[allow(dead_code)]
impl GoalLoopRuntime {
    pub fn new(config: GoalLoopRuntimeConfig) -> Self {
        Self { config }
    }

    pub async fn run_goal_loop_until_complete<D: GoalLoopDriver>(
        &self,
        goal: GoalSpec,
        driver: &mut D,
    ) -> GoalLoopRun {
        let run_id = Uuid::new_v4().to_string();
        let mut run = GoalLoopRun {
            run_id,
            goal,
            status: GoalLoopStatus::Running,
            iteration_count: 0,
            retry_budget: self.config.retry_budget,
            retries_used: 0,
            current_strategy: None,
            fallback_strategy_state: None,
            frames: Vec::new(),
            planner_steps: Vec::new(),
            planner_diagnostics: Vec::new(),
            executed_steps: Vec::new(),
            verification_history: Vec::new(),
            focused_perception_requests: Vec::new(),
            browser_handoff_history: Vec::new(),
            browser_handoff: None,
            verified_surface: None,
            surface_diagnostics: Vec::new(),
            focused_perception_used: false,
            visible_refinement_used: false,
            stale_capture_reuse_prevented: false,
            browser_recovery_used: false,
            browser_recovery_status: BrowserRecoveryStatus::NotNeeded,
            post_action_progress_observed: false,
            surface_ownership_lost: false,
            focused_perception_failure_reason: None,
            repeated_click_protection_triggered: false,
            selected_target_candidate: None,
            verifier_status: None,
            failure_reason: None,
        };

        let mut focused_passes = 0usize;
        let mut should_perceive = true;
        let mut visual_handoff_checked = false;
        let mut force_fresh_capture = false;
        let mut browser_recovery_attempts = 0usize;
        let mut pending_post_click: Option<PendingPostClickVerification> = None;
        let mut repeated_click_attempts: HashMap<String, usize> = HashMap::new();
        let mut observed_geometry_frames: HashMap<String, String> = HashMap::new();
        let mut no_progress_geometry_failures: HashMap<String, usize> = HashMap::new();

        for iteration in 0..self.config.max_iterations {
            run.iteration_count = iteration + 1;

            if should_perceive || run.frames.is_empty() {
                let mut handoff_supplied_frame = false;
                if !visual_handoff_checked {
                    visual_handoff_checked = true;
                    match driver.prepare_visual_handoff(&run.goal, iteration).await {
                        Ok(Some(outcome)) => {
                            let record = outcome.record;
                            let verified = record.status == BrowserHandoffStatus::VisuallyVerified;
                            run.browser_handoff = Some(record.clone());
                            run.browser_handoff_history.push(record.clone());
                            if !verified {
                                run.status = GoalLoopStatus::BrowserHandoffFailed;
                                run.failure_reason =
                                    Some(record.reason.clone().unwrap_or_else(|| {
                                        "browser visual handoff did not produce a verified page frame"
                                            .into()
                                    }));
                                run.verifier_status = Some(format!("{:?}", record.status));
                                return run;
                            }
                            if let Some(frame) = outcome.verified_frame {
                                let surface =
                                    verified_browser_surface_from_frame(&run.goal, &frame);
                                run.verified_surface = Some(surface.clone());
                                run.surface_diagnostics.push(surface_verified_diagnostic(
                                    iteration,
                                    &surface,
                                    SurfaceOwnershipStatus::Verified,
                                    "browser visual handoff established browser surface ownership",
                                ));
                                run.frames.push(frame);
                                handoff_supplied_frame = true;
                            }
                        }
                        Ok(None) => {}
                        Err(error) => {
                            run.status = GoalLoopStatus::BrowserHandoffFailed;
                            run.failure_reason =
                                Some(format!("browser visual handoff failed: {error}"));
                            run.verifier_status = Some("browser_handoff_failed".into());
                            return run;
                        }
                    }
                }

                if !handoff_supplied_frame {
                    match driver
                        .perceive(&run.goal, iteration, force_fresh_capture)
                        .await
                    {
                        Ok(frame) => {
                            if force_fresh_capture {
                                run.stale_capture_reuse_prevented = true;
                                if let Some(pending) = pending_post_click.as_ref() {
                                    if let Some(execution) =
                                        run.executed_steps.get_mut(pending.step_index)
                                    {
                                        execution.fresh_capture_used = true;
                                    }
                                }
                            }
                            force_fresh_capture = false;
                            run.frames.push(frame);
                        }
                        Err(error) => {
                            run.status = GoalLoopStatus::NeedsPerception;
                            run.failure_reason =
                                Some(format!("semantic perception failed: {error}"));
                            run.verifier_status = Some("perception_failed".into());
                            return run;
                        }
                    }
                }
            }

            if let Some(pending) = pending_post_click.take() {
                let current_frame = run
                    .frames
                    .last()
                    .expect("frame after post-click perception");
                let mut surface_lost_after_click = false;
                if let Some(surface) = run.verified_surface.clone() {
                    let diagnostic = surface_continuity_diagnostic(
                        &run.goal,
                        &surface,
                        current_frame,
                        iteration,
                    );
                    surface_lost_after_click = diagnostic.status == SurfaceOwnershipStatus::Lost
                        && verify_goal_against_frame(&run.goal, current_frame).is_none();
                    if surface_lost_after_click {
                        run.surface_ownership_lost = true;
                        run.focused_perception_failure_reason =
                            Some(FocusedPerceptionFailureReason::SurfaceOwnershipLost);
                    } else if diagnostic.status == SurfaceOwnershipStatus::Verified {
                        run.verified_surface = Some(verified_browser_surface_from_frame(
                            &run.goal,
                            current_frame,
                        ));
                    }
                    run.surface_diagnostics.push(diagnostic);
                }
                let post_click = evaluate_post_click_frame(&run.goal, &pending, current_frame);
                if post_click.progress_observed {
                    run.post_action_progress_observed = true;
                }
                if let Some(signature) = pending.geometry_signature.clone() {
                    if post_click.progress_observed {
                        no_progress_geometry_failures.remove(&signature);
                    } else {
                        *no_progress_geometry_failures.entry(signature).or_insert(0) += 1;
                    }
                }
                if let Some(signature) = pending.target_signature.clone() {
                    if post_click.progress_observed {
                        repeated_click_attempts.remove(&signature);
                    } else {
                        let attempts = repeated_click_attempts
                            .entry(signature.clone())
                            .or_insert(0);
                        *attempts += 1;
                        if *attempts >= MAX_IDENTICAL_CLICK_ATTEMPTS {
                            run.status = GoalLoopStatus::VerificationFailed;
                            run.repeated_click_protection_triggered = true;
                            run.failure_reason = Some(format!(
                                "repeated governed clicks on target `{signature}` did not produce observable page progress"
                            ));
                            run.verifier_status =
                                Some("repeated_click_protection_triggered".into());
                            return run;
                        }
                    }
                }

                if !post_click.goal_achieved
                    && browser_recovery_attempts < MAX_BROWSER_RECOVERY_ATTEMPTS
                    && (surface_lost_after_click
                        || post_click.browser_surface_suspect
                        || post_click.frame_unchanged)
                {
                    let reason = if surface_lost_after_click {
                        "surface_ownership_lost_after_click"
                    } else if post_click.browser_surface_suspect {
                        "browser_surface_lost_after_click"
                    } else {
                        "post_click_page_unchanged"
                    };
                    run.browser_recovery_status = BrowserRecoveryStatus::Attempted;
                    match driver
                        .recover_browser_surface(&run.goal, iteration, reason)
                        .await
                    {
                        Ok(Some(outcome)) => {
                            browser_recovery_attempts += 1;
                            run.browser_recovery_used = true;
                            run.browser_recovery_status = BrowserRecoveryStatus::Reacquired;
                            run.browser_handoff = Some(outcome.record.clone());
                            run.browser_handoff_history.push(outcome.record.clone());
                            if let Some(frame) = outcome.verified_frame {
                                if outcome.record.status == BrowserHandoffStatus::VisuallyVerified {
                                    let surface =
                                        verified_browser_surface_from_frame(&run.goal, &frame);
                                    run.verified_surface = Some(surface.clone());
                                    run.surface_diagnostics.push(surface_verified_diagnostic(
                                        iteration,
                                        &surface,
                                        SurfaceOwnershipStatus::Reacquired,
                                        "bounded browser recovery reacquired browser surface ownership",
                                    ));
                                    run.surface_ownership_lost = false;
                                    run.focused_perception_failure_reason = None;
                                }
                                run.frames.push(frame);
                            }
                        }
                        Ok(None) => {
                            browser_recovery_attempts += 1;
                            run.browser_recovery_status = BrowserRecoveryStatus::Failed;
                        }
                        Err(error) => {
                            browser_recovery_attempts += 1;
                            run.browser_recovery_status = BrowserRecoveryStatus::Failed;
                            run.verifier_status = Some(format!("browser_recovery_failed:{error}"));
                        }
                    }
                }
                if surface_lost_after_click && run.surface_ownership_lost {
                    run.status = GoalLoopStatus::VerificationFailed;
                    run.failure_reason = Some(
                        "browser interaction surface ownership was lost after the governed click and bounded recovery did not reacquire it"
                            .into(),
                    );
                    run.verifier_status = Some("surface_ownership_lost".into());
                    return run;
                }
            }

            let frame = run.frames.last().expect("frame after perception");
            let verification = verify_goal_state(&run.goal, frame, iteration);
            run.verifier_status = Some(format!("{:?}", verification.status));
            run.verification_history.push(verification.clone());
            if verification.status == GoalVerificationStatus::GoalAchieved {
                run.status = GoalLoopStatus::GoalAchieved;
                return run;
            }

            let planner_input = planner_contract_input_with_perception(
                &run.goal,
                frame,
                &run.executed_steps,
                &run.verification_history,
                &run.focused_perception_requests,
                self.config.retry_budget,
                run.retries_used,
                self.config.max_visible_refinement_passes,
            );
            let deterministic = deterministic_planner_contract_decision(&planner_input);
            let decision = if deterministic_visible_ordinal_fast_path(
                &planner_input,
                &deterministic.proposed_step,
            ) {
                PlannerContractDecision {
                    strategy_rationale: format!(
                        "deterministic_visible_ordinal_fast_path: {}",
                        deterministic.strategy_rationale
                    ),
                    ..deterministic
                }
            } else {
                match driver.plan(&planner_input).await {
                    Ok(Some(model_decision)) => validate_model_planner_decision(
                        &planner_input,
                        model_decision,
                        deterministic,
                    ),
                    Ok(None) => deterministic,
                    Err(error) => planner_fallback_decision(
                        deterministic,
                        format!("model planner unavailable: {error}"),
                    ),
                }
            };
            let step = decision.proposed_step.clone();
            run.planner_diagnostics.push(PlannerDecisionDiagnostic {
                iteration,
                source: decision.source.clone(),
                accepted: decision.accepted,
                fallback_used: decision.fallback_used,
                planner_confidence: decision.planner_confidence,
                strategy: Some(decision.strategy_rationale.clone()),
                reason: decision.rejection_reason.clone(),
                decision_status: decision.decision_status.clone(),
                rejection_code: decision.rejection_code.clone(),
                visibility_assessment: decision.visibility_assessment.clone(),
                scroll_intent: decision.scroll_intent.clone(),
                visible_actionability: decision.visible_actionability.clone(),
                target_confidence: decision.target_confidence.clone(),
                normalized: decision.normalized,
                downgraded: decision.downgraded,
                focused_perception_needed: decision.focused_perception_needed,
                replan_needed: decision.replan_needed,
            });
            run.current_strategy = Some(strategy_for_step(&run.goal, &step));
            run.fallback_strategy_state = fallback_state_for_step(&step);
            run.selected_target_candidate = step.executable_candidate.clone();
            run.planner_steps.push(step.clone());

            if matches!(
                decision.scroll_intent,
                PlannerScrollIntent::RequiredButUnsupported
                    | PlannerScrollIntent::FuturePrimitiveRequired
            ) {
                run.status = GoalLoopStatus::ScrollRequiredButUnsupported;
                run.failure_reason = Some(
                    "current page matches the goal context, but the requested target is likely off-screen and ScrollViewport is not safely implemented yet".into(),
                );
                run.verifier_status = Some("scroll_required_but_unsupported".into());
                return run;
            }

            match step.kind {
                PlannerStepKind::ClickResultRegion | PlannerStepKind::ClickEntityRegion => {
                    let current_frame_signature = frame_progress_signature(frame);
                    if let Some(geometry_signature) = step_geometry_signature(&step) {
                        if no_progress_geometry_failures
                            .get(&geometry_signature)
                            .copied()
                            .unwrap_or(0)
                            > 0
                            && browser_recovery_attempts >= MAX_BROWSER_RECOVERY_ATTEMPTS
                        {
                            run.status = GoalLoopStatus::VerificationFailed;
                            run.failure_reason = Some(
                                "fresh recapture and bounded browser recovery did not produce progress; refusing to re-execute the same governed geometry"
                                    .into(),
                            );
                            run.verifier_status =
                                Some("suspicious_geometry_rejected_after_no_progress".into());
                            return run;
                        }
                        if let Some(previous_frame_signature) =
                            observed_geometry_frames.get(&geometry_signature)
                        {
                            if previous_frame_signature != &current_frame_signature {
                                run.status = GoalLoopStatus::Refused;
                                run.failure_reason = Some(
                                    "the same executable geometry was proposed on distinct fresh frames; rejecting it as suspicious copied or ungrounded vision geometry"
                                        .into(),
                                );
                                run.verifier_status = Some("suspicious_geometry_rejected".into());
                                return run;
                            }
                        }
                    }
                    if !step_is_executable(&step, self.config.min_execution_confidence) {
                        run.status = GoalLoopStatus::Refused;
                        run.failure_reason = Some(format!(
                            "planner step is not executable or below confidence threshold: {:.2}",
                            step.confidence
                        ));
                        return run;
                    }
                    let mut execution = driver.execute_planner_step(&step).await;
                    execution.target_signature = execution
                        .target_signature
                        .clone()
                        .or_else(|| step_target_signature(&step));
                    let executed = execution.status == PlannerStepExecutionStatus::Executed;
                    if executed {
                        execution.fresh_capture_required = true;
                    }
                    run.executed_steps.push(execution);
                    let mut recovery_supplied_frame = false;
                    if !executed {
                        if let Some(reason) = browser_recovery_reason_for_execution(
                            run.executed_steps.last().expect("execution record"),
                        ) {
                            if browser_recovery_attempts < MAX_BROWSER_RECOVERY_ATTEMPTS {
                                run.browser_recovery_status = BrowserRecoveryStatus::Attempted;
                                match driver
                                    .recover_browser_surface(&run.goal, iteration, &reason)
                                    .await
                                {
                                    Ok(Some(outcome)) => {
                                        browser_recovery_attempts += 1;
                                        run.browser_recovery_used = true;
                                        run.browser_recovery_status =
                                            BrowserRecoveryStatus::Reacquired;
                                        run.browser_handoff = Some(outcome.record.clone());
                                        run.browser_handoff_history.push(outcome.record.clone());
                                        if let Some(frame) = outcome.verified_frame {
                                            if outcome.record.status
                                                == BrowserHandoffStatus::VisuallyVerified
                                            {
                                                let surface = verified_browser_surface_from_frame(
                                                    &run.goal, &frame,
                                                );
                                                run.verified_surface = Some(surface.clone());
                                                run.surface_diagnostics.push(
                                                    surface_verified_diagnostic(
                                                        iteration,
                                                        &surface,
                                                        SurfaceOwnershipStatus::Reacquired,
                                                        "bounded browser recovery reacquired browser surface after execution failure",
                                                    ),
                                                );
                                                run.surface_ownership_lost = false;
                                                run.focused_perception_failure_reason = None;
                                            }
                                            run.frames.push(frame);
                                            recovery_supplied_frame = true;
                                        }
                                    }
                                    Ok(None) => {
                                        browser_recovery_attempts += 1;
                                        run.browser_recovery_status = BrowserRecoveryStatus::Failed;
                                    }
                                    Err(error) => {
                                        browser_recovery_attempts += 1;
                                        run.browser_recovery_status = BrowserRecoveryStatus::Failed;
                                        run.verifier_status = Some(format!(
                                            "browser_recovery_failed_after_geometry:{error}"
                                        ));
                                    }
                                }
                            }
                        }
                        run.retries_used += 1;
                        if run.retries_used >= run.retry_budget {
                            run.status = GoalLoopStatus::BudgetExhausted;
                            run.failure_reason = Some(
                                "execution retry budget exhausted before goal completion".into(),
                            );
                            return run;
                        }
                    } else {
                        let step_index = run.executed_steps.len() - 1;
                        if let Some(geometry_signature) = step_geometry_signature(&step) {
                            observed_geometry_frames
                                .insert(geometry_signature, current_frame_signature.clone());
                        }
                        pending_post_click = Some(PendingPostClickVerification {
                            step_index,
                            target_signature: run.executed_steps[step_index]
                                .target_signature
                                .clone(),
                            geometry_signature: step_geometry_signature(&step),
                            pre_click_frame_signature: current_frame_signature,
                            pre_click_list_surface_visible: structural_list_surface_visible(frame),
                        });
                        force_fresh_capture = true;
                    }
                    should_perceive = executed || !recovery_supplied_frame;
                }
                PlannerStepKind::ReplanAfterPerception => {
                    let Some(request) = focused_perception_request_for_step(
                        &run.goal,
                        frame,
                        &step,
                        &decision.visible_actionability,
                        iteration,
                    ) else {
                        if decision.visible_actionability.target_visible_evidence
                            && !decision.visible_actionability.refinement_eligible
                            && target_region_anchor_for_step(frame, &step).is_none()
                        {
                            run.status = GoalLoopStatus::BudgetExhausted;
                            run.failure_reason = Some(
                                "visible target remains non-executable after bounded regionless refinement; no trusted region anchor or safe fallback is available"
                                    .into(),
                            );
                            run.verifier_status =
                                Some("regionless_refinement_budget_exhausted".into());
                        } else {
                            run.status = GoalLoopStatus::NeedsPerception;
                            run.failure_reason = Some(
                                "planner requested more perception but no safe focus or refinement route is available"
                                    .into(),
                            );
                        }
                        return run;
                    };
                    let browser_owned_workflow = run.verified_surface.is_some()
                        || run.browser_handoff.as_ref().is_some_and(|record| {
                            record.status == BrowserHandoffStatus::VisuallyVerified
                        });
                    let request = match bind_focused_perception_request_to_surface(
                        request,
                        run.verified_surface.as_ref(),
                        browser_owned_workflow,
                        iteration,
                    ) {
                        Ok(request) => request,
                        Err(diagnostic) => {
                            run.focused_perception_failure_reason =
                                diagnostic.failure_reason.clone();
                            run.surface_diagnostics.push(diagnostic.clone());
                            run.status = GoalLoopStatus::Refused;
                            run.failure_reason = diagnostic.reason.clone().or_else(|| {
                                Some(
                                    "focused perception was refused outside the verified browser surface"
                                        .into(),
                                )
                            });
                            run.verifier_status =
                                Some("focused_perception_surface_bound_refused".into());
                            return run;
                        }
                    };
                    if focused_passes >= self.config.max_focused_perception_passes {
                        run.status = GoalLoopStatus::BudgetExhausted;
                        run.failure_reason =
                            Some("focused perception retry budget exhausted".into());
                        return run;
                    }
                    focused_passes += 1;
                    run.focused_perception_used = true;
                    if request.mode == PerceptionRequestMode::VisiblePageRefinement {
                        run.visible_refinement_used = true;
                    }
                    run.current_strategy = Some(strategy_for_perception_request(&request));
                    run.fallback_strategy_state = fallback_state_for_perception_request(&request);
                    run.focused_perception_requests.push(request.clone());
                    match driver.focused_perception(&request).await {
                        Ok(Some(frame)) => {
                            if let Some(surface) = run.verified_surface.clone() {
                                let diagnostic = surface_continuity_diagnostic(
                                    &run.goal, &surface, &frame, iteration,
                                );
                                if diagnostic.status == SurfaceOwnershipStatus::Lost {
                                    run.surface_ownership_lost = true;
                                    run.focused_perception_failure_reason =
                                        Some(FocusedPerceptionFailureReason::SurfaceOwnershipLost);
                                    run.surface_diagnostics.push(diagnostic);
                                    run.status = GoalLoopStatus::VerificationFailed;
                                    run.failure_reason = Some(
                                        "focused perception returned a frame outside the verified browser surface"
                                            .into(),
                                    );
                                    run.verifier_status =
                                        Some("focused_perception_surface_ownership_lost".into());
                                    return run;
                                }
                                run.verified_surface =
                                    Some(verified_browser_surface_from_frame(&run.goal, &frame));
                                run.surface_diagnostics.push(diagnostic);
                            }
                            run.frames.push(frame);
                            should_perceive = false;
                        }
                        Ok(None) => {
                            run.focused_perception_failure_reason =
                                Some(FocusedPerceptionFailureReason::StructuredPerceptionEmpty);
                            run.status = GoalLoopStatus::NeedsPerception;
                            run.failure_reason =
                                Some("focused perception returned no semantic frame".into());
                            return run;
                        }
                        Err(error) => {
                            run.focused_perception_failure_reason =
                                Some(FocusedPerceptionFailureReason::StructuredPerceptionEmpty);
                            run.status = GoalLoopStatus::NeedsPerception;
                            run.failure_reason =
                                Some(format!("focused perception failed: {error}"));
                            return run;
                        }
                    }
                }
                PlannerStepKind::VerifyGoal | PlannerStepKind::NoOp => {
                    run.status = GoalLoopStatus::GoalAchieved;
                    return run;
                }
                PlannerStepKind::RequestClarification => {
                    run.status = GoalLoopStatus::NeedsClarification;
                    run.failure_reason = Some(step.rationale);
                    return run;
                }
                PlannerStepKind::Refuse => {
                    run.status = GoalLoopStatus::Refused;
                    run.failure_reason = Some(step.rationale);
                    return run;
                }
            }
        }

        run.status = GoalLoopStatus::BudgetExhausted;
        run.failure_reason = Some("goal loop iteration budget exhausted".into());
        run
    }
}

pub fn verify_goal_against_frame(goal: &GoalSpec, frame: &SemanticScreenFrame) -> Option<String> {
    match goal.goal_type {
        GoalType::OpenListItem | GoalType::OpenMediaResult => {
            match structural_open_list_item_state(goal, frame) {
                OpenListItemVerificationState::Achieved(reason) => Some(reason.into()),
                OpenListItemVerificationState::Ambiguous(_)
                | OpenListItemVerificationState::NotAchieved => None,
            }
        }
        GoalType::OpenChannel => is_channel_page(frame).then_some("channel_page_visible".into()),
        _ => None,
    }
}

#[allow(dead_code)]
pub fn verify_goal_state(
    goal: &GoalSpec,
    frame: &SemanticScreenFrame,
    iteration: usize,
) -> GoalVerificationRecord {
    if let Some(reason) = verify_goal_against_frame(goal, frame) {
        return GoalVerificationRecord {
            iteration,
            status: GoalVerificationStatus::GoalAchieved,
            confidence: frame.page_evidence.confidence.max(0.86),
            reason,
            frame_id: Some(frame.frame_id.clone()),
        };
    }

    let status = match goal.goal_type {
        GoalType::OpenListItem | GoalType::OpenMediaResult
            if matches!(
                structural_open_list_item_state(goal, frame),
                OpenListItemVerificationState::Ambiguous(_)
            ) =>
        {
            GoalVerificationStatus::Ambiguous
        }
        GoalType::OpenChannel if is_watch_page(frame) => GoalVerificationStatus::ReplanRequired,
        GoalType::OpenChannel if is_search_results_page(frame) => {
            GoalVerificationStatus::GoalNotAchieved
        }
        _ if !frame.uncertainty.is_empty() => GoalVerificationStatus::Ambiguous,
        _ => GoalVerificationStatus::GoalNotAchieved,
    };

    GoalVerificationRecord {
        iteration,
        status,
        confidence: frame.page_evidence.confidence,
        reason: format!(
            "goal not satisfied on page kind {:?}",
            frame.page_evidence.page_kind_hint
        ),
        frame_id: Some(frame.frame_id.clone()),
    }
}

#[allow(dead_code)]
pub fn verify_browser_handoff_page(
    goal: &GoalSpec,
    frame: &SemanticScreenFrame,
) -> Result<BrowserHandoffVerificationDiagnostic, BrowserHandoffVerificationDiagnostic> {
    let provider_matches = goal
        .constraints
        .provider
        .as_deref()
        .map_or(true, |expected| {
            provider_context_compatible(
                Some(expected),
                frame.page_evidence.content_provider_hint.as_deref(),
            )
        });
    let mut diagnostic = browser_handoff_verification_diagnostic(goal, frame, provider_matches);
    if !provider_matches {
        diagnostic.reason = Some(format!(
            "visible page provider {:?} does not match expected provider {:?}",
            frame.page_evidence.content_provider_hint, goal.constraints.provider
        ));
        return Err(diagnostic);
    }

    if verify_goal_against_frame(goal, frame).is_some() {
        diagnostic.accepted = true;
        diagnostic.decision = BrowserHandoffVerificationDecision::GoalSatisfied;
        diagnostic.reason =
            Some("goal already appears satisfied on the current visible browser page".into());
        return Ok(diagnostic);
    }

    match goal.goal_type {
        GoalType::OpenListItem | GoalType::OpenMediaResult => {
            verify_list_handoff_context(diagnostic)
        }
        GoalType::OpenChannel => {
            if diagnostic.normalized_page_kind == BrowserPageSemanticKind::WatchPage {
                diagnostic.accepted = true;
                diagnostic.decision = BrowserHandoffVerificationDecision::NormalizedPageKind;
                diagnostic.reason = Some(format!(
                    "page kind {:?} normalized to watch_page and is valid for channel continuation",
                    diagnostic.raw_page_kind_hint
                ));
                Ok(diagnostic)
            } else {
                let mut diagnostic = verify_list_handoff_context(diagnostic)?;
                diagnostic.reason = Some(format!(
                    "page kind {:?} resolved as a verified result-list context for channel continuation",
                    diagnostic.raw_page_kind_hint
                ));
                Ok(diagnostic)
            }
        }
        GoalType::InspectScreen => {
            diagnostic.accepted = true;
            diagnostic.decision = BrowserHandoffVerificationDecision::NotRequired;
            diagnostic.reason =
                Some("inspect_screen goals do not require browser page-kind verification".into());
            Ok(diagnostic)
        }
        _ => {
            diagnostic.reason = Some(
                "browser handoff page verification is not supported for this goal type".into(),
            );
            Err(diagnostic)
        }
    }
}

fn browser_handoff_verification_diagnostic(
    goal: &GoalSpec,
    frame: &SemanticScreenFrame,
    provider_matches: bool,
) -> BrowserHandoffVerificationDiagnostic {
    let summary = summarize_visible_signals(frame);
    let raw_page_kind_hint = frame.page_evidence.page_kind_hint.clone();
    let normalized_page_kind = browser_page_semantic_kind(raw_page_kind_hint.as_deref());
    let generic_provider_page_kind_hint = raw_page_kind_hint.as_deref().is_some_and(|kind| {
        frame
            .page_evidence
            .content_provider_hint
            .as_deref()
            .is_some_and(|provider| labels_match(kind, provider))
    });
    let goal_expects_results_context = matches!(
        goal.goal_type,
        GoalType::OpenListItem | GoalType::OpenMediaResult | GoalType::OpenChannel
    );
    let query_hint_present = frame
        .page_evidence
        .query_hint
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty());
    let result_list_visible = frame.page_evidence.result_list_visible == Some(true);
    let primary_list_item_count = frame
        .primary_list
        .as_ref()
        .map(|list| list.items.len())
        .unwrap_or(0);
    let structural_list_surface = structural_list_surface_visible(frame);
    let structured_result_signal = primary_list_item_count > 0
        || summary.result_item_count > 0
        || summary.legacy_candidate_signal_count > 0;
    let supporting_signal_count = usize::from(result_list_visible)
        + usize::from(structural_list_surface)
        + usize::from(primary_list_item_count > 0)
        + usize::from(structured_result_signal)
        + usize::from(summary.entity_signal_count > 0)
        + usize::from(summary.scene_summary_result_hint)
        + usize::from(query_hint_present);
    let mut supporting_evidence = Vec::new();
    if provider_matches {
        supporting_evidence.push("provider_matches".into());
    }
    if goal_expects_results_context {
        supporting_evidence.push("goal_expects_results_context".into());
    }
    if generic_provider_page_kind_hint {
        supporting_evidence.push("generic_provider_page_kind_hint".into());
    }
    if result_list_visible {
        supporting_evidence.push("result_list_visible".into());
    }
    if structural_list_surface {
        supporting_evidence.push("structural_list_surface_visible".into());
    }
    if primary_list_item_count > 0 {
        supporting_evidence.push(format!("primary_list_items={primary_list_item_count}"));
    }
    if !frame.visible_result_items.is_empty() {
        supporting_evidence.push(format!(
            "visible_result_items={}",
            frame.visible_result_items.len()
        ));
    }
    if summary.entity_signal_count > 0 {
        supporting_evidence.push(format!(
            "visible_entity_signals={}",
            summary.entity_signal_count
        ));
    }
    if summary.legacy_candidate_signal_count > 0 {
        supporting_evidence.push(format!(
            "legacy_candidate_signals={}",
            summary.legacy_candidate_signal_count
        ));
    }
    if summary.scene_summary_result_hint {
        supporting_evidence.push("scene_summary_result_hint".into());
    }
    if query_hint_present {
        supporting_evidence.push("query_hint_present".into());
    }

    BrowserHandoffVerificationDiagnostic {
        raw_page_kind_hint,
        normalized_page_kind,
        decision: BrowserHandoffVerificationDecision::Rejected,
        accepted: false,
        provider_matches,
        goal_expects_results_context,
        generic_provider_page_kind_hint,
        query_hint_present,
        result_list_visible,
        visible_result_item_count: frame.visible_result_items.len(),
        primary_list_item_count,
        structural_list_surface_visible: structural_list_surface,
        page_state_kind: frame.page_state.as_ref().map(|state| state.kind.clone()),
        page_state_dominant_content: frame
            .page_state
            .as_ref()
            .map(|state| state.dominant_content.clone()),
        visible_entity_signal_count: summary.entity_signal_count,
        legacy_candidate_signal_count: summary.legacy_candidate_signal_count,
        scene_summary_result_hint: summary.scene_summary_result_hint,
        supporting_signal_count,
        supporting_evidence,
        reason: None,
    }
}

fn verify_list_handoff_context(
    mut diagnostic: BrowserHandoffVerificationDiagnostic,
) -> Result<BrowserHandoffVerificationDiagnostic, BrowserHandoffVerificationDiagnostic> {
    if diagnostic.normalized_page_kind == BrowserPageSemanticKind::SearchResults {
        diagnostic.accepted = true;
        diagnostic.decision = BrowserHandoffVerificationDecision::NormalizedPageKind;
        diagnostic.reason = Some(format!(
            "page kind {:?} normalized to search_results",
            diagnostic.raw_page_kind_hint
        ));
        return Ok(diagnostic);
    }

    let has_structural_result_signal = diagnostic.primary_list_item_count > 0
        || diagnostic.visible_result_item_count > 0
        || diagnostic.visible_entity_signal_count > 0
        || diagnostic.legacy_candidate_signal_count > 0;
    let has_scene_or_query_signal = diagnostic.scene_summary_result_hint
        || diagnostic.query_hint_present
        || diagnostic.structural_list_surface_visible;
    let has_supporting_result_context = has_structural_result_signal
        || diagnostic.scene_summary_result_hint
        || diagnostic.structural_list_surface_visible;
    if diagnostic.structural_list_surface_visible
        && diagnostic.primary_list_item_count > 0
        && diagnostic.supporting_signal_count >= 2
    {
        diagnostic.accepted = true;
        diagnostic.decision = BrowserHandoffVerificationDecision::SupportingEvidence;
        diagnostic.reason = Some(format!(
            "visible page was accepted as a provider-agnostic list context using primary_list/page_state evidence; raw page kind {:?}",
            diagnostic.raw_page_kind_hint
        ));
        return Ok(diagnostic);
    }

    let generic_provider_results_context = diagnostic.generic_provider_page_kind_hint
        && diagnostic.goal_expects_results_context
        && diagnostic.provider_matches
        && has_structural_result_signal
        && has_scene_or_query_signal
        && diagnostic.supporting_signal_count >= 3;

    if generic_provider_results_context {
        diagnostic.accepted = true;
        diagnostic.decision = BrowserHandoffVerificationDecision::SupportingEvidence;
        diagnostic.reason = Some(format!(
            "generic provider page kind {:?} was accepted as a result-list context using goal-aware supporting evidence",
            diagnostic.raw_page_kind_hint
        ));
        return Ok(diagnostic);
    }

    if diagnostic.result_list_visible
        && diagnostic.supporting_signal_count >= 2
        && has_supporting_result_context
    {
        diagnostic.accepted = true;
        diagnostic.decision = BrowserHandoffVerificationDecision::SupportingEvidence;
        diagnostic.reason = Some(format!(
            "page kind {:?} was accepted as a result-list context using supporting semantic evidence",
            diagnostic.raw_page_kind_hint
        ));
        return Ok(diagnostic);
    }

    diagnostic.reason = Some(if diagnostic.generic_provider_page_kind_hint {
        format!(
            "generic provider page kind {:?} did not normalize to search_results and the supporting evidence was not strong enough to infer a reusable result-list context",
            diagnostic.raw_page_kind_hint
        )
    } else {
        format!(
            "visible page kind {:?} did not normalize to a verified result-list context and supporting evidence was insufficient",
            diagnostic.raw_page_kind_hint
        )
    });
    Err(diagnostic)
}

#[allow(dead_code)]
pub fn plan_next_step_with_history(
    goal: &GoalSpec,
    frame: &SemanticScreenFrame,
    executed_steps: &[PlannerStepExecutionRecord],
    verification_history: &[GoalVerificationRecord],
) -> PlannerStep {
    let input = planner_contract_input(
        goal,
        frame,
        executed_steps,
        verification_history,
        GoalLoopRuntimeConfig::default().retry_budget,
        executed_steps
            .iter()
            .filter(|step| step.status != PlannerStepExecutionStatus::Executed)
            .count(),
    );
    deterministic_planner_contract_decision(&input).proposed_step
}

pub fn planner_contract_input(
    goal: &GoalSpec,
    frame: &SemanticScreenFrame,
    executed_steps: &[PlannerStepExecutionRecord],
    verification_history: &[GoalVerificationRecord],
    retry_budget: usize,
    retries_used: usize,
) -> PlannerContractInput {
    planner_contract_input_with_perception(
        goal,
        frame,
        executed_steps,
        verification_history,
        &[],
        retry_budget,
        retries_used,
        GoalLoopRuntimeConfig::default().max_visible_refinement_passes,
    )
}

pub fn planner_contract_input_with_perception(
    goal: &GoalSpec,
    frame: &SemanticScreenFrame,
    executed_steps: &[PlannerStepExecutionRecord],
    verification_history: &[GoalVerificationRecord],
    perception_requests: &[FocusedPerceptionRequest],
    retry_budget: usize,
    retries_used: usize,
    max_visible_refinement_passes: usize,
) -> PlannerContractInput {
    let visible_refinement_attempts = perception_requests
        .iter()
        .filter(|request| request.mode == PerceptionRequestMode::VisiblePageRefinement)
        .count();
    let draft = PlannerContractInput {
        goal: goal.clone(),
        current_frame: frame.clone(),
        executed_steps: executed_steps.to_vec(),
        verification_history: verification_history.to_vec(),
        perception_requests: perception_requests.to_vec(),
        retry_budget,
        retries_used,
        visible_refinement_attempts,
        max_visible_refinement_passes,
        provider_hint: frame.page_evidence.content_provider_hint.clone(),
        browser_app_hint: frame.page_evidence.browser_app_hint.clone(),
        page_kind_hint: frame.page_evidence.page_kind_hint.clone(),
        visible_actionability: VisibleActionabilityDiagnostic::default(),
    };
    let visible_actionability = visible_actionability_for_goal(&draft);
    PlannerContractInput {
        goal: goal.clone(),
        current_frame: frame.clone(),
        executed_steps: executed_steps.to_vec(),
        verification_history: verification_history.to_vec(),
        perception_requests: perception_requests.to_vec(),
        retry_budget,
        retries_used,
        visible_refinement_attempts,
        max_visible_refinement_passes,
        provider_hint: frame.page_evidence.content_provider_hint.clone(),
        browser_app_hint: frame.page_evidence.browser_app_hint.clone(),
        page_kind_hint: frame.page_evidence.page_kind_hint.clone(),
        visible_actionability,
    }
}

#[derive(Debug, Default)]
struct VisibleSignalSummary {
    result_item_count: usize,
    weak_result_count: usize,
    missing_click_region_count: usize,
    missing_title_count: usize,
    missing_ranking_count: usize,
    entity_signal_count: usize,
    actionable_control_signal_count: usize,
    legacy_candidate_signal_count: usize,
    scene_summary_result_hint: bool,
    result_like_regions: Vec<TargetRegion>,
}

impl VisibleSignalSummary {
    fn has_visible_signals(&self) -> bool {
        self.result_item_count > 0
            || self.entity_signal_count > 0
            || self.actionable_control_signal_count > 0
            || self.legacy_candidate_signal_count > 0
    }

    fn refinement_strategy(&self) -> VisibleRefinementStrategy {
        if self.result_like_regions.is_empty() {
            VisibleRefinementStrategy::FullFrame
        } else {
            VisibleRefinementStrategy::VisibleCluster
        }
    }
}

fn visible_actionability_for_goal(input: &PlannerContractInput) -> VisibleActionabilityDiagnostic {
    let summary = summarize_visible_signals(&input.current_frame);
    match input.goal.goal_type {
        GoalType::OpenListItem | GoalType::OpenMediaResult => {
            visible_actionability_for_open_list_item(input, &summary)
        }
        GoalType::OpenChannel => visible_actionability_for_open_channel(input, &summary),
        GoalType::InspectScreen => base_visible_actionability(input, &summary),
        GoalType::FindBestOffer | GoalType::Unknown => {
            let mut diagnostic = base_visible_actionability(input, &summary);
            diagnostic.relevant_visible_content =
                summary.has_visible_signals() || summary.scene_summary_result_hint;
            diagnostic.status = if diagnostic.relevant_visible_content {
                VisibleActionabilityStatus::VisibleUnderGrounded
            } else {
                VisibleActionabilityStatus::Unknown
            };
            diagnostic
        }
    }
}

fn base_visible_actionability(
    input: &PlannerContractInput,
    summary: &VisibleSignalSummary,
) -> VisibleActionabilityDiagnostic {
    VisibleActionabilityDiagnostic {
        result_item_count: summary.result_item_count,
        weak_result_count: summary.weak_result_count,
        missing_click_region_count: summary.missing_click_region_count,
        missing_title_count: summary.missing_title_count,
        missing_ranking_count: summary.missing_ranking_count,
        entity_signal_count: summary.entity_signal_count,
        actionable_control_signal_count: summary.actionable_control_signal_count,
        legacy_candidate_signal_count: summary.legacy_candidate_signal_count,
        visible_refinement_attempts: input.visible_refinement_attempts,
        safe_fallback_available: summary.actionable_control_signal_count > 0
            || summary.legacy_candidate_signal_count > 0,
        ..VisibleActionabilityDiagnostic::default()
    }
}

fn visible_actionability_for_open_list_item(
    input: &PlannerContractInput,
    summary: &VisibleSignalSummary,
) -> VisibleActionabilityDiagnostic {
    let goal = &input.goal;
    let frame = &input.current_frame;
    let desired_kind = requested_item_kind_label(goal);
    let desired_rank = goal
        .constraints
        .rank_within_kind
        .or(goal.constraints.rank_overall)
        .unwrap_or(1);
    let provider = goal
        .constraints
        .provider
        .as_deref()
        .or(input.provider_hint.as_deref());
    let refinement_remaining =
        input.visible_refinement_attempts < input.max_visible_refinement_passes;
    let mut diagnostic = base_visible_actionability(input, summary);
    let result_context_visible = list_context_visible(frame);

    diagnostic.relevant_visible_content = result_context_visible || summary.has_visible_signals();

    if let Some(item) = find_primary_list_item_match(goal, frame) {
        diagnostic.target_visible_evidence = true;
        if preferred_primary_click_region(item, &["primary", "title", "thumbnail"]).is_some() {
            diagnostic.status = VisibleActionabilityStatus::VisibleExecutable;
            return diagnostic;
        }
        diagnostic.status = VisibleActionabilityStatus::VisibleTargetNeedsClickRegion;
        diagnostic.refinement_eligible =
            primary_item_region(item).is_some() || refinement_remaining;
        diagnostic.refinement_strategy = if primary_item_region(item).is_some() {
            VisibleRefinementStrategy::TargetRegion
        } else {
            summary.refinement_strategy()
        };
        append_visible_gap(
            &mut diagnostic.gaps,
            VisibleGroundingGap::MissingClickRegion,
        );
        return diagnostic;
    }

    if let Some(item) = find_ranked_result_match(goal, frame) {
        diagnostic.target_visible_evidence = true;
        if preferred_click_region(item, &["title", "thumbnail", "primary"]).is_some() {
            diagnostic.status = VisibleActionabilityStatus::VisibleExecutable;
            return diagnostic;
        }
        let target_anchor_available = result_item_region(item).is_some();
        diagnostic.status = VisibleActionabilityStatus::VisibleTargetNeedsClickRegion;
        diagnostic.refinement_eligible = target_anchor_available || refinement_remaining;
        diagnostic.refinement_strategy = if target_anchor_available {
            VisibleRefinementStrategy::TargetRegion
        } else {
            summary.refinement_strategy()
        };
        append_result_item_grounding_gaps(&mut diagnostic, item);
        append_visible_gap(
            &mut diagnostic.gaps,
            VisibleGroundingGap::MissingClickRegion,
        );
        return diagnostic;
    }

    let typed_max_rank = max_visible_rank_for_goal(frame, desired_kind.as_deref(), provider);
    let weak_media_candidate =
        has_weak_open_list_item_candidate(goal, frame, provider, desired_kind.as_deref());
    let partial_target_evidence = has_partial_open_media_target_evidence(goal, frame, provider);
    diagnostic.target_visible_evidence = partial_target_evidence;

    let visible_under_grounded = result_context_visible
        && (summary.result_item_count == 0
            || weak_media_candidate
            || summary.entity_signal_count > 0
            || summary.actionable_control_signal_count > 0
            || summary.legacy_candidate_signal_count > 0
            || partial_target_evidence);

    if visible_under_grounded && refinement_remaining {
        diagnostic.status = VisibleActionabilityStatus::VisibleUnderGrounded;
        diagnostic.refinement_eligible = true;
        diagnostic.refinement_strategy = summary.refinement_strategy();
        diagnostic.offscreen_inference_stage = OffscreenInferenceStage::DeferredPendingRefinement;
        append_summary_grounding_gaps(&mut diagnostic, summary);
        return diagnostic;
    }

    if page_matches_goal_context(input)
        && (typed_max_rank.is_some_and(|max_rank| max_rank < desired_rank)
            || (!visible_under_grounded && result_context_visible)
            || (!refinement_remaining && result_context_visible))
    {
        diagnostic.status = VisibleActionabilityStatus::LikelyOffscreen;
        diagnostic.offscreen_inference_stage =
            offscreen_inference_stage_for_attempts(input.visible_refinement_attempts);
        append_summary_grounding_gaps(&mut diagnostic, summary);
        return diagnostic;
    }

    if result_context_visible && visible_under_grounded {
        diagnostic.status = VisibleActionabilityStatus::LikelyOffscreen;
        diagnostic.offscreen_inference_stage =
            offscreen_inference_stage_for_attempts(input.visible_refinement_attempts);
        append_summary_grounding_gaps(&mut diagnostic, summary);
        return diagnostic;
    }

    diagnostic.status = VisibleActionabilityStatus::NoRelevantVisibleContent;
    diagnostic
}

fn visible_actionability_for_open_channel(
    input: &PlannerContractInput,
    summary: &VisibleSignalSummary,
) -> VisibleActionabilityDiagnostic {
    let goal = &input.goal;
    let frame = &input.current_frame;
    let desired_name = goal.constraints.entity_name.as_deref();
    let refinement_remaining =
        input.visible_refinement_attempts < input.max_visible_refinement_passes;
    let mut diagnostic = base_visible_actionability(input, summary);
    let channel_context_visible = is_search_results_page(frame)
        || is_watch_page(frame)
        || frame.page_evidence.result_list_visible == Some(true)
        || summary.scene_summary_result_hint;

    diagnostic.relevant_visible_content = channel_context_visible || summary.has_visible_signals();

    if let Some(entity) = find_channel_entity(frame, desired_name) {
        diagnostic.target_visible_evidence = true;
        if entity.region.is_some() {
            diagnostic.status = VisibleActionabilityStatus::VisibleExecutable;
            return diagnostic;
        }
        diagnostic.status = VisibleActionabilityStatus::VisibleTargetNeedsClickRegion;
        diagnostic.refinement_eligible = refinement_remaining;
        diagnostic.refinement_strategy = summary.refinement_strategy();
        append_visible_gap(
            &mut diagnostic.gaps,
            VisibleGroundingGap::MissingClickRegion,
        );
        return diagnostic;
    }

    if let Some(item) = find_channel_result_item(frame, desired_name) {
        diagnostic.target_visible_evidence = true;
        if preferred_click_region(item, &["title", "thumbnail", "primary"]).is_some() {
            diagnostic.status = VisibleActionabilityStatus::VisibleExecutable;
            return diagnostic;
        }
        let target_anchor_available = result_item_region(item).is_some();
        diagnostic.status = VisibleActionabilityStatus::VisibleTargetNeedsClickRegion;
        diagnostic.refinement_eligible = target_anchor_available || refinement_remaining;
        diagnostic.refinement_strategy = if target_anchor_available {
            VisibleRefinementStrategy::TargetRegion
        } else {
            summary.refinement_strategy()
        };
        append_result_item_grounding_gaps(&mut diagnostic, item);
        append_visible_gap(
            &mut diagnostic.gaps,
            VisibleGroundingGap::MissingClickRegion,
        );
        return diagnostic;
    }

    let partial_target_evidence = has_partial_channel_target_evidence(frame, desired_name);
    diagnostic.target_visible_evidence = partial_target_evidence;
    let weak_channel_candidate = has_weak_channel_candidate(frame, desired_name);
    let visible_under_grounded = channel_context_visible
        && (summary.result_item_count == 0
            || weak_channel_candidate
            || summary.entity_signal_count > 0
            || summary.actionable_control_signal_count > 0
            || summary.legacy_candidate_signal_count > 0
            || partial_target_evidence);

    if visible_under_grounded && refinement_remaining {
        diagnostic.status = VisibleActionabilityStatus::VisibleUnderGrounded;
        diagnostic.refinement_eligible = true;
        diagnostic.refinement_strategy = summary.refinement_strategy();
        diagnostic.offscreen_inference_stage = OffscreenInferenceStage::DeferredPendingRefinement;
        append_summary_grounding_gaps(&mut diagnostic, summary);
        return diagnostic;
    }

    if channel_context_visible {
        diagnostic.status = VisibleActionabilityStatus::LikelyOffscreen;
        diagnostic.offscreen_inference_stage =
            offscreen_inference_stage_for_attempts(input.visible_refinement_attempts);
        append_summary_grounding_gaps(&mut diagnostic, summary);
        return diagnostic;
    }

    diagnostic.status = VisibleActionabilityStatus::NoRelevantVisibleContent;
    diagnostic
}

fn offscreen_inference_stage_for_attempts(
    visible_refinement_attempts: usize,
) -> OffscreenInferenceStage {
    if visible_refinement_attempts == 0 {
        OffscreenInferenceStage::EligibleAfterRefinement
    } else {
        OffscreenInferenceStage::ConfirmedAfterRefinement
    }
}

fn summarize_visible_signals(frame: &SemanticScreenFrame) -> VisibleSignalSummary {
    let mut summary = VisibleSignalSummary::default();
    summary.scene_summary_result_hint = scene_summary_suggests_results(&frame.scene_summary);

    if let Some(primary_list) = frame.primary_list.as_ref() {
        for item in &primary_list.items {
            summary.result_item_count += 1;
            let missing_title = item
                .title
                .as_deref()
                .map_or(true, |title| title.trim().is_empty());
            let weak_item = item.item_kind.as_deref().map_or(true, |kind| {
                matches!(
                    normalize_item_kind_label(kind).as_str(),
                    "unknown" | "generic"
                )
            }) || missing_title;
            if weak_item {
                summary.weak_result_count += 1;
            }
            if missing_title {
                summary.missing_title_count += 1;
            }
            if item.click_regions.is_empty() {
                summary.missing_click_region_count += 1;
            }
            if let Some(region) = primary_item_region(item) {
                summary.result_like_regions.push(region);
            }
        }
    }

    for item in &frame.visible_result_items {
        summary.result_item_count += 1;
        let missing_title = item
            .title
            .as_deref()
            .map_or(true, |title| title.trim().is_empty());
        let missing_ranking = item.rank_within_kind.is_none() && item.rank_overall.is_none();
        let weak_item = matches!(
            item.kind,
            VisibleResultKind::Unknown | VisibleResultKind::Generic
        ) || missing_title
            || missing_ranking;
        if weak_item {
            summary.weak_result_count += 1;
        }
        if missing_title {
            summary.missing_title_count += 1;
        }
        if missing_ranking {
            summary.missing_ranking_count += 1;
        }
        if item.click_regions.is_empty() {
            summary.missing_click_region_count += 1;
        }
        if let Some(region) = result_item_region(item) {
            summary.result_like_regions.push(region);
        }
    }

    for entity in &frame.visible_entities {
        if !entity_looks_result_like(entity) {
            continue;
        }
        summary.entity_signal_count += 1;
        if let Some(region) = entity.region.clone() {
            summary.result_like_regions.push(region);
        }
    }

    for control in &frame.actionable_controls {
        if !control_looks_result_like(control, frame) {
            continue;
        }
        summary.actionable_control_signal_count += 1;
        if let Some(region) = control.region.clone() {
            summary.result_like_regions.push(region);
        }
    }

    for candidate in &frame.legacy_target_candidates {
        if !matches!(
            candidate.role,
            UITargetRole::RankedResult | UITargetRole::Link
        ) {
            continue;
        }
        summary.legacy_candidate_signal_count += 1;
        if let Some(region) = candidate_region(candidate) {
            summary.result_like_regions.push(region);
        }
    }

    summary
}

fn scene_summary_suggests_results(summary: &str) -> bool {
    let normalized = normalize_label(summary);
    [
        "result",
        "results",
        "search",
        "video",
        "playlist",
        "channel",
        "product",
        "hotel",
        "repository",
        "card",
        "list",
    ]
    .iter()
    .any(|token| normalized.contains(token))
}

fn entity_looks_result_like(entity: &VisibleEntity) -> bool {
    matches!(
        entity.kind,
        VisibleEntityKind::ChannelHeader
            | VisibleEntityKind::ChannelResult
            | VisibleEntityKind::VideoResult
            | VisibleEntityKind::MixResult
            | VisibleEntityKind::PlaylistResult
            | VisibleEntityKind::HotelCard
            | VisibleEntityKind::Avatar
            | VisibleEntityKind::TitleLink
            | VisibleEntityKind::Thumbnail
    )
}

fn control_looks_result_like(control: &ActionableControl, frame: &SemanticScreenFrame) -> bool {
    if control.region.is_none() {
        return false;
    }
    let kind = normalize_label(&control.kind);
    if matches!(
        kind.as_str(),
        "link" | "button" | "result" | "card" | "list_item" | "item" | "thumbnail"
    ) {
        return true;
    }
    frame.page_evidence.result_list_visible == Some(true)
        && control
            .label
            .as_deref()
            .is_some_and(|label| !label.trim().is_empty())
}

fn candidate_region(candidate: &UITargetCandidate) -> Option<TargetRegion> {
    candidate.region.clone().or_else(|| {
        let (x, y) = candidate.center_point()?;
        Some(TargetRegion {
            x: x - 6.0,
            y: y - 6.0,
            width: 12.0,
            height: 12.0,
            coordinate_space: "screen".into(),
        })
    })
}

fn append_summary_grounding_gaps(
    diagnostic: &mut VisibleActionabilityDiagnostic,
    summary: &VisibleSignalSummary,
) {
    if summary.weak_result_count > 0 {
        append_visible_gap(&mut diagnostic.gaps, VisibleGroundingGap::WeakItemTyping);
    }
    if summary.missing_click_region_count > 0 {
        append_visible_gap(
            &mut diagnostic.gaps,
            VisibleGroundingGap::MissingClickRegion,
        );
    }
    if summary.missing_title_count > 0 {
        append_visible_gap(&mut diagnostic.gaps, VisibleGroundingGap::MissingTitle);
    }
    if summary.missing_ranking_count > 0 {
        append_visible_gap(&mut diagnostic.gaps, VisibleGroundingGap::MissingRanking);
    }
    if summary.result_item_count > 1
        || summary.entity_signal_count > 1
        || summary.actionable_control_signal_count > 1
        || summary.legacy_candidate_signal_count > 1
    {
        append_visible_gap(
            &mut diagnostic.gaps,
            VisibleGroundingGap::RepeatedVisibleContent,
        );
    }
    if summary.entity_signal_count > 0
        || summary.actionable_control_signal_count > 0
        || summary.legacy_candidate_signal_count > 0
    {
        append_visible_gap(
            &mut diagnostic.gaps,
            VisibleGroundingGap::PartialSemanticSignals,
        );
    }
}

fn append_result_item_grounding_gaps(
    diagnostic: &mut VisibleActionabilityDiagnostic,
    item: &VisibleResultItem,
) {
    if matches!(
        item.kind,
        VisibleResultKind::Unknown | VisibleResultKind::Generic
    ) {
        append_visible_gap(&mut diagnostic.gaps, VisibleGroundingGap::WeakItemTyping);
    }
    if item
        .title
        .as_deref()
        .map_or(true, |title| title.trim().is_empty())
    {
        append_visible_gap(&mut diagnostic.gaps, VisibleGroundingGap::MissingTitle);
    }
    if item.rank_within_kind.is_none() && item.rank_overall.is_none() {
        append_visible_gap(&mut diagnostic.gaps, VisibleGroundingGap::MissingRanking);
    }
}

fn append_visible_gap(gaps: &mut Vec<VisibleGroundingGap>, gap: VisibleGroundingGap) {
    if !gaps.contains(&gap) {
        gaps.push(gap);
    }
}

fn find_ranked_result_match<'a>(
    goal: &GoalSpec,
    frame: &'a SemanticScreenFrame,
) -> Option<&'a VisibleResultItem> {
    let desired_kind = requested_item_kind_label(goal);
    let desired_rank = goal
        .constraints
        .rank_within_kind
        .or(goal.constraints.rank_overall)
        .unwrap_or(1);
    let provider = goal
        .constraints
        .provider
        .as_deref()
        .or(frame.page_evidence.content_provider_hint.as_deref());
    frame.visible_result_items.iter().find(|item| {
        item_kind_compatibility(desired_kind.as_deref(), Some(result_kind_label(&item.kind)))
            .is_some()
            && provider_matches_item(
                provider,
                frame.page_evidence.content_provider_hint.as_deref(),
                item,
            )
            && effective_rank_for_goal(frame, item, desired_kind.as_deref()) == Some(desired_rank)
    })
}

fn find_primary_list_item_match<'a>(
    goal: &GoalSpec,
    frame: &'a SemanticScreenFrame,
) -> Option<&'a PrimaryListItem> {
    let desired_kind = requested_item_kind_label(goal);
    let desired_rank = goal
        .constraints
        .rank_within_kind
        .or(goal.constraints.rank_overall)
        .unwrap_or(1);
    let list = frame.primary_list.as_ref()?;
    list.items
        .iter()
        .filter(|item| {
            item_kind_compatibility(desired_kind.as_deref(), item.item_kind.as_deref()).is_some()
        })
        .find(|item| item.rank == desired_rank)
        .or_else(|| {
            list.items
                .iter()
                .filter(|item| {
                    item_kind_compatibility(desired_kind.as_deref(), item.item_kind.as_deref())
                        .is_some()
                })
                .nth((desired_rank.saturating_sub(1)) as usize)
        })
}

fn effective_rank_for_goal(
    frame: &SemanticScreenFrame,
    item: &VisibleResultItem,
    requested_kind: Option<&str>,
) -> Option<u32> {
    if requested_kind.is_none() {
        return item.rank_overall.or(item.rank_within_kind);
    }
    item.rank_within_kind.or_else(|| {
        if item_kind_compatibility(requested_kind, Some(result_kind_label(&item.kind))).is_none() {
            return None;
        }
        frame
            .visible_result_items
            .iter()
            .filter(|other| {
                item_kind_compatibility(requested_kind, Some(result_kind_label(&other.kind)))
                    .is_some()
            })
            .position(|other| other.item_id == item.item_id)
            .map(|index| (index + 1) as u32)
    })
}

fn max_visible_rank_for_goal(
    frame: &SemanticScreenFrame,
    requested_kind: Option<&str>,
    provider: Option<&str>,
) -> Option<u32> {
    let primary_rank = frame
        .primary_list
        .as_ref()
        .into_iter()
        .flat_map(|list| list.items.iter())
        .filter(|item| item_kind_compatibility(requested_kind, item.item_kind.as_deref()).is_some())
        .map(|item| item.rank)
        .max();
    let visible_rank = frame
        .visible_result_items
        .iter()
        .filter(|item| {
            item_kind_compatibility(requested_kind, Some(result_kind_label(&item.kind))).is_some()
                && provider_matches_item(
                    provider,
                    frame.page_evidence.content_provider_hint.as_deref(),
                    item,
                )
        })
        .filter_map(|item| {
            effective_rank_for_goal(frame, item, requested_kind).or(item.rank_overall)
        })
        .max();
    match (primary_rank, visible_rank) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(rank), None) | (None, Some(rank)) => Some(rank),
        (None, None) => None,
    }
}

fn has_partial_open_media_target_evidence(
    goal: &GoalSpec,
    frame: &SemanticScreenFrame,
    provider: Option<&str>,
) -> bool {
    let desired_kind = requested_item_kind_label(goal);
    frame.primary_list.as_ref().is_some_and(|list| {
        list.items.iter().any(|item| {
            item_kind_compatibility(desired_kind.as_deref(), item.item_kind.as_deref()).is_some()
                && (item.title.is_some() || primary_item_region(item).is_some())
        })
    }) || frame.visible_result_items.iter().any(|item| {
        provider_matches_item(
            provider,
            frame.page_evidence.content_provider_hint.as_deref(),
            item,
        ) && matches!(
            item.kind,
            VisibleResultKind::Unknown | VisibleResultKind::Generic
        ) && (item.title.is_some() || result_item_region(item).is_some())
    }) || desired_kind.as_deref() == Some("video")
        && (frame.visible_entities.iter().any(|entity| {
            matches!(
                entity.kind,
                VisibleEntityKind::VideoResult
                    | VisibleEntityKind::Thumbnail
                    | VisibleEntityKind::TitleLink
            )
        }) || frame.legacy_target_candidates.iter().any(|candidate| {
            matches!(
                candidate.role,
                UITargetRole::RankedResult | UITargetRole::Link
            )
        }))
}

fn has_weak_open_list_item_candidate(
    _goal: &GoalSpec,
    frame: &SemanticScreenFrame,
    provider: Option<&str>,
    desired_kind: Option<&str>,
) -> bool {
    if frame.primary_list.as_ref().is_some_and(|list| {
        list.items.iter().any(|item| {
            item_kind_compatibility(desired_kind, item.item_kind.as_deref()).is_some()
                && (item.click_regions.is_empty()
                    || item
                        .title
                        .as_deref()
                        .map_or(true, |title| title.trim().is_empty())
                    || item.item_kind.as_deref().map_or(true, |kind| {
                        matches!(
                            normalize_item_kind_label(kind).as_str(),
                            "generic" | "unknown"
                        )
                    }))
        })
    }) {
        return true;
    }

    frame.visible_result_items.iter().any(|item| {
        let missing_title = item
            .title
            .as_deref()
            .map_or(true, |title| title.trim().is_empty());
        let missing_rank = effective_rank_for_goal(frame, item, desired_kind)
            .or(item.rank_overall)
            .is_none();
        let weak_type = matches!(
            item.kind,
            VisibleResultKind::Unknown | VisibleResultKind::Generic
        );
        provider_matches_item(
            provider,
            frame.page_evidence.content_provider_hint.as_deref(),
            item,
        ) && (item_kind_compatibility(desired_kind, Some(result_kind_label(&item.kind))).is_some()
            || weak_type)
            && (item.click_regions.is_empty() || missing_title || missing_rank || weak_type)
    })
}

fn find_channel_entity<'a>(
    frame: &'a SemanticScreenFrame,
    desired_name: Option<&str>,
) -> Option<&'a VisibleEntity> {
    frame.visible_entities.iter().find(|entity| {
        matches!(
            entity.kind,
            VisibleEntityKind::ChannelHeader
                | VisibleEntityKind::ChannelResult
                | VisibleEntityKind::Avatar
        ) && desired_name.map_or(true, |name| {
            entity
                .name
                .as_deref()
                .map_or(false, |observed| loose_text_match(name, observed))
        })
    })
}

fn find_channel_result_item<'a>(
    frame: &'a SemanticScreenFrame,
    desired_name: Option<&str>,
) -> Option<&'a VisibleResultItem> {
    frame.visible_result_items.iter().find(|item| {
        item.kind == VisibleResultKind::Channel
            && desired_name.map_or(true, |name| {
                item.title
                    .as_deref()
                    .or(item.channel_name.as_deref())
                    .map_or(false, |observed| loose_text_match(name, observed))
            })
    })
}

fn has_partial_channel_target_evidence(
    frame: &SemanticScreenFrame,
    desired_name: Option<&str>,
) -> bool {
    frame.visible_entities.iter().any(|entity| {
        matches!(
            entity.kind,
            VisibleEntityKind::TitleLink
                | VisibleEntityKind::Thumbnail
                | VisibleEntityKind::Avatar
                | VisibleEntityKind::Unknown
        ) && desired_name.map_or(true, |name| {
            entity
                .name
                .as_deref()
                .map_or(false, |observed| loose_text_match(name, observed))
        })
    }) || frame.visible_result_items.iter().any(|item| {
        matches!(
            item.kind,
            VisibleResultKind::Unknown | VisibleResultKind::Generic
        ) && desired_name.map_or(true, |name| {
            item.title
                .as_deref()
                .or(item.channel_name.as_deref())
                .map_or(false, |observed| loose_text_match(name, observed))
        })
    })
}

fn has_weak_channel_candidate(frame: &SemanticScreenFrame, desired_name: Option<&str>) -> bool {
    frame.visible_result_items.iter().any(|item| {
        item.kind == VisibleResultKind::Channel
            && desired_name.map_or(true, |name| {
                item.title
                    .as_deref()
                    .or(item.channel_name.as_deref())
                    .map_or(false, |observed| loose_text_match(name, observed))
            })
            && (item.click_regions.is_empty()
                || item
                    .title
                    .as_deref()
                    .map_or(true, |title| title.trim().is_empty())
                || item.rank_within_kind.is_none())
    })
}

pub fn deterministic_planner_contract_decision(
    input: &PlannerContractInput,
) -> PlannerContractDecision {
    let goal = &input.goal;
    let frame = &input.current_frame;
    let executed_steps = &input.executed_steps;
    let verification_history = &input.verification_history;

    if matches!(goal.goal_type, GoalType::OpenChannel)
        && is_watch_page(frame)
        && verification_history
            .last()
            .is_some_and(|record| record.status == GoalVerificationStatus::ReplanRequired)
    {
        let proposed_step = maybe_apply_safe_fallback_step(input, plan_open_channel(goal, frame));
        return rust_decision(input, proposed_step, true);
    }

    if executed_steps
        .last()
        .is_some_and(|record| record.status == PlannerStepExecutionStatus::Failed)
        && verification_history
            .last()
            .is_some_and(|record| record.status == GoalVerificationStatus::PageUnchanged)
    {
        let proposed_step = PlannerStep {
            step_id: Uuid::new_v4().to_string(),
            kind: PlannerStepKind::ReplanAfterPerception,
            confidence: 0.42,
            rationale: "previous governed click failed to change the page; requesting focused perception before retry".into(),
            target_item_id: None,
            target_entity_id: None,
            click_region_key: None,
            executable_candidate: None,
            expected_state: Some(goal.success_condition.clone()),
        };
        return rust_decision_with_strategy(
            input,
            proposed_step,
            "previous_failed_click_focused_replan".into(),
            true,
        );
    }

    let proposed_step = maybe_apply_safe_fallback_step(input, plan_next_step(goal, frame));
    let replan_needed = matches!(
        proposed_step.kind,
        PlannerStepKind::ReplanAfterPerception | PlannerStepKind::RequestClarification
    );
    rust_decision(input, proposed_step, replan_needed)
}

fn maybe_apply_safe_fallback_step(
    input: &PlannerContractInput,
    proposed_step: PlannerStep,
) -> PlannerStep {
    if input.visible_refinement_attempts == 0
        || proposed_step.kind != PlannerStepKind::ReplanAfterPerception
        || !page_matches_goal_context(input)
    {
        return proposed_step;
    }

    match input.goal.goal_type {
        GoalType::OpenListItem | GoalType::OpenMediaResult => {
            safe_fallback_step_for_open_media(input, &proposed_step).unwrap_or(proposed_step)
        }
        GoalType::OpenChannel
        | GoalType::FindBestOffer
        | GoalType::InspectScreen
        | GoalType::Unknown => proposed_step,
    }
}

fn safe_fallback_step_for_open_media(
    input: &PlannerContractInput,
    proposed_step: &PlannerStep,
) -> Option<PlannerStep> {
    let goal = &input.goal;
    let frame = &input.current_frame;
    let desired_kind = requested_item_kind_label(goal);
    let desired_rank = goal
        .constraints
        .rank_within_kind
        .or(goal.constraints.rank_overall)
        .unwrap_or(1);

    let matched_title = proposed_step
        .target_item_id
        .as_deref()
        .and_then(|item_id| {
            frame
                .primary_list
                .as_ref()
                .and_then(|list| list.items.iter().find(|item| item.item_id == item_id))
                .and_then(|item| item.title.clone())
                .or_else(|| {
                    frame
                        .visible_result_items
                        .iter()
                        .find(|item| item.item_id == item_id)
                        .and_then(|item| item.title.clone())
                })
        })
        .or_else(|| {
            find_primary_list_item_match(goal, frame)
                .and_then(|item| item.title.clone())
                .or_else(|| {
                    find_ranked_result_match(goal, frame).and_then(|item| item.title.clone())
                })
        });
    let matched_title = matched_title.as_deref();

    let mut screen_candidates = legacy_result_fallback_candidates(frame);
    screen_candidates.extend(
        frame
            .actionable_controls
            .iter()
            .filter(|control| control_looks_result_like(control, frame))
            .filter_map(|control| {
                candidate_from_actionable_control(
                    control,
                    goal,
                    frame,
                    desired_kind.as_deref(),
                    "planner_actionable_control_fallback",
                    "planner selected actionable control fallback after regionless refinement",
                )
            }),
    );

    if let Some(title) = matched_title {
        screen_candidates.retain(|candidate| {
            candidate.rank == Some(desired_rank)
                || candidate
                    .label
                    .as_deref()
                    .is_some_and(|label| loose_text_match(title, label))
        });
    }

    if screen_candidates.is_empty() {
        return None;
    }

    let request = TargetGroundingRequest {
        requested_role: UITargetRole::RankedResult,
        target: Value::Null,
        selection: Value::Null,
        screen_candidates,
        recent_candidates: Vec::new(),
        app_hint: frame.page_evidence.browser_app_hint.clone(),
        provider_hint: goal
            .constraints
            .provider
            .clone()
            .or_else(|| frame.page_evidence.content_provider_hint.clone()),
        rank_hint: Some(desired_rank),
        result_kind_hint: desired_kind.clone(),
        allow_recent_reuse: false,
        now_ms: Some(frame.captured_at),
        max_recent_age_ms: 0,
    };
    let state = ground_targets_for_request(&request);
    let selection = select_target_candidate(
        &state,
        TargetAction::Click,
        &TargetSelectionPolicy::default(),
    );
    let candidate = selection.selected_candidate?;
    let fallback_source = executable_fallback_source_for_candidate(&candidate)?;
    let source_label = match fallback_source {
        ExecutableFallbackSource::ActionableControl => "actionable_control",
        ExecutableFallbackSource::LegacyTargetCandidate => "legacy_candidate",
    };

    Some(PlannerStep {
        step_id: Uuid::new_v4().to_string(),
        kind: PlannerStepKind::ClickResultRegion,
        confidence: candidate.confidence,
        rationale: format!(
            "bounded {source_label} fallback selected an executable visible result after regionless refinement"
        ),
        target_item_id: proposed_step
            .target_item_id
            .clone()
            .or_else(|| Some(candidate.candidate_id.clone())),
        target_entity_id: None,
        click_region_key: Some(format!("{source_label}_fallback")),
        executable_candidate: Some(candidate),
        expected_state: proposed_step
            .expected_state
            .clone()
            .or_else(|| Some(goal.success_condition.clone())),
    })
}

fn legacy_result_fallback_candidates(frame: &SemanticScreenFrame) -> Vec<UITargetCandidate> {
    frame
        .legacy_target_candidates
        .iter()
        .filter(|candidate| {
            matches!(
                candidate.role,
                UITargetRole::RankedResult | UITargetRole::Link
            )
        })
        .cloned()
        .map(|mut candidate| {
            candidate.observation_source = Some("planner_legacy_candidate_fallback".into());
            candidate.observed_at_ms = candidate.observed_at_ms.or(Some(frame.captured_at));
            candidate
        })
        .collect()
}

fn candidate_from_actionable_control(
    control: &ActionableControl,
    goal: &GoalSpec,
    frame: &SemanticScreenFrame,
    result_kind: Option<&str>,
    observation_source: &str,
    rationale: &str,
) -> Option<UITargetCandidate> {
    let region = control.region.clone()?;
    let control_kind = normalize_label(&control.kind);
    let role = if matches!(control_kind.as_str(), "button" | "play_button") {
        UITargetRole::Button
    } else {
        UITargetRole::Link
    };
    let confidence =
        actionable_control_confidence_diagnostic(goal, frame, control, None).derived_confidence;

    Some(UITargetCandidate {
        candidate_id: format!("planner_control_{}", control.control_id),
        role,
        region: Some(region),
        center_x: None,
        center_y: None,
        app_hint: frame.page_evidence.browser_app_hint.clone(),
        browser_app_hint: frame.page_evidence.browser_app_hint.clone(),
        provider_hint: frame.page_evidence.content_provider_hint.clone(),
        content_provider_hint: frame.page_evidence.content_provider_hint.clone(),
        page_kind_hint: frame.page_evidence.page_kind_hint.clone(),
        capture_backend: frame.page_evidence.capture_backend.clone(),
        observation_source: Some(observation_source.into()),
        result_kind: control_result_kind(control)
            .or_else(|| result_kind.map(ToOwned::to_owned))
            .or_else(|| Some("generic".into())),
        confidence,
        source: TargetGroundingSource::ScreenAnalysis,
        label: control.label.clone(),
        rank: control_rank(control),
        observed_at_ms: Some(frame.captured_at),
        reuse_eligible: true,
        supports_focus: false,
        supports_click: true,
        rationale: rationale.into(),
    })
}

fn control_result_kind(control: &ActionableControl) -> Option<String> {
    control
        .attributes
        .get("result_kind")
        .or_else(|| control.attributes.get("item_kind"))
        .or_else(|| control.attributes.get("content_kind"))
        .and_then(Value::as_str)
        .map(normalize_label)
}

fn control_rank(control: &ActionableControl) -> Option<u32> {
    [
        "rank_within_kind",
        "rank_overall",
        "rank",
        "order",
        "position",
    ]
    .iter()
    .find_map(|key| control.attributes.get(*key).and_then(value_as_u32))
}

fn value_as_u32(value: &Value) -> Option<u32> {
    value.as_u64().map(|value| value as u32).or_else(|| {
        value
            .as_i64()
            .filter(|value| *value >= 0)
            .map(|value| value as u32)
    })
}

fn confidence_signal_state(raw_confidence: Option<f32>) -> ConfidenceSignalState {
    match raw_confidence {
        None => ConfidenceSignalState::Missing,
        Some(confidence) if confidence <= 0.0 => ConfidenceSignalState::ExplicitZero,
        Some(confidence) if confidence < MIN_PLANNER_CLICK_CONFIDENCE => ConfidenceSignalState::Low,
        Some(_) => ConfidenceSignalState::Supported,
    }
}

fn target_execution_context_matches_goal(goal: &GoalSpec, frame: &SemanticScreenFrame) -> bool {
    let provider_matches = goal
        .constraints
        .provider
        .as_deref()
        .map_or(true, |expected| {
            provider_context_compatible(
                Some(expected),
                frame.page_evidence.content_provider_hint.as_deref(),
            )
        });
    if !provider_matches {
        return false;
    }

    match goal.goal_type {
        GoalType::OpenListItem | GoalType::OpenMediaResult => list_context_visible(frame),
        GoalType::OpenChannel => {
            is_search_results_page(frame)
                || frame.page_evidence.result_list_visible == Some(true)
                || is_watch_page(frame)
        }
        GoalType::FindBestOffer | GoalType::InspectScreen | GoalType::Unknown => true,
    }
}

fn derive_structural_semantic_confidence(
    label_present: bool,
    kind_grounded: bool,
    action_anchor_present: bool,
    page_context_verified: bool,
    has_region: bool,
) -> f32 {
    let mut confidence: f32 = 0.46;
    if label_present {
        confidence += 0.16;
    }
    if kind_grounded {
        confidence += 0.10;
    }
    if action_anchor_present {
        confidence += 0.07;
    }
    if page_context_verified {
        confidence += 0.07;
    }
    if has_region {
        confidence += 0.05;
    }
    confidence.clamp(0.0, 0.90)
}

fn compose_executable_target_confidence(
    raw_semantic_confidence: Option<f32>,
    raw_region_confidence: Option<f32>,
    effective_region_confidence: Option<f32>,
    raw_page_confidence: Option<f32>,
    effective_page_confidence: Option<f32>,
    raw_planner_confidence: Option<f32>,
    structural_semantic_confidence: f32,
    has_region: bool,
    page_context_verified: bool,
    fallback_source: Option<ExecutableFallbackSource>,
) -> ExecutableTargetConfidenceDiagnostic {
    let semantic_confidence = raw_semantic_confidence
        .unwrap_or(structural_semantic_confidence)
        .clamp(0.0, 1.0);
    let region_confidence = raw_region_confidence
        .or(effective_region_confidence)
        .or_else(|| has_region.then_some(0.84))
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    let page_confidence = raw_page_confidence
        .or(effective_page_confidence)
        .or_else(|| page_context_verified.then_some(0.78))
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    let mut derived_confidence =
        (semantic_confidence * 0.48) + (region_confidence * 0.37) + (page_confidence * 0.15);
    if let Some(planner_confidence) = raw_planner_confidence {
        derived_confidence = derived_confidence
            .min((derived_confidence * 0.90) + (planner_confidence.clamp(0.0, 1.0) * 0.10));
    }
    derived_confidence = derived_confidence.clamp(0.0, 1.0);

    let accepted =
        has_region && page_context_verified && derived_confidence >= MIN_PLANNER_CLICK_CONFIDENCE;
    let confidence_was_derived = raw_semantic_confidence.is_none()
        || raw_region_confidence.is_none()
        || (raw_page_confidence.is_none() && page_context_verified);
    let reason = if !has_region {
        Some("trusted click region missing for executable target".into())
    } else if !page_context_verified {
        Some("page context is not verified strongly enough for executable target binding".into())
    } else if accepted && confidence_was_derived {
        Some(
            "confidence derived from page, target, and region signals stayed above threshold"
                .into(),
        )
    } else if accepted {
        Some("explicit target confidence signals stayed above execution threshold".into())
    } else if confidence_was_derived {
        Some("derived target confidence stayed below execution threshold".into())
    } else {
        Some("explicit target confidence stayed below execution threshold".into())
    };

    ExecutableTargetConfidenceDiagnostic {
        raw_item_confidence: raw_semantic_confidence,
        raw_region_confidence,
        raw_page_confidence,
        raw_planner_confidence,
        item_confidence_state: confidence_signal_state(raw_semantic_confidence),
        region_confidence_state: confidence_signal_state(raw_region_confidence),
        page_confidence_state: confidence_signal_state(raw_page_confidence),
        planner_confidence_state: confidence_signal_state(raw_planner_confidence),
        structural_semantic_confidence,
        semantic_confidence,
        region_confidence,
        page_confidence,
        derived_confidence,
        required_threshold: MIN_PLANNER_CLICK_CONFIDENCE,
        confidence_was_derived,
        fallback_source,
        accepted,
        reason,
    }
}

fn result_item_confidence_diagnostic(
    goal: &GoalSpec,
    frame: &SemanticScreenFrame,
    item: &VisibleResultItem,
    click_region: Option<&ClickRegion>,
    planner_confidence: Option<f32>,
    fallback_source: Option<ExecutableFallbackSource>,
) -> ExecutableTargetConfidenceDiagnostic {
    let page_context_verified = target_execution_context_matches_goal(goal, frame);
    let has_region = click_region.is_some();
    let label_present = item
        .title
        .as_deref()
        .or(item.channel_name.as_deref())
        .is_some_and(|label| !label.trim().is_empty());
    let kind_grounded = !matches!(
        item.kind,
        VisibleResultKind::Unknown | VisibleResultKind::Generic
    );
    let action_anchor_present = item.rank_within_kind.or(item.rank_overall).is_some();
    let structural_semantic_confidence = derive_structural_semantic_confidence(
        label_present,
        kind_grounded,
        action_anchor_present,
        page_context_verified,
        has_region,
    );
    compose_executable_target_confidence(
        item.raw_confidence,
        click_region.and_then(|region| region.raw_confidence),
        click_region.and_then(|region| (region.confidence > 0.0).then_some(region.confidence)),
        frame.page_evidence.raw_confidence,
        (frame.page_evidence.confidence > 0.0).then_some(frame.page_evidence.confidence),
        planner_confidence,
        structural_semantic_confidence,
        has_region,
        page_context_verified,
        fallback_source,
    )
}

fn entity_confidence_diagnostic(
    goal: &GoalSpec,
    frame: &SemanticScreenFrame,
    entity: &VisibleEntity,
    planner_confidence: Option<f32>,
) -> ExecutableTargetConfidenceDiagnostic {
    let page_context_verified = target_execution_context_matches_goal(goal, frame);
    let has_region = entity.region.is_some();
    let label_present = entity
        .name
        .as_deref()
        .is_some_and(|label| !label.trim().is_empty());
    let kind_grounded = entity.kind != VisibleEntityKind::Unknown;
    let structural_semantic_confidence = derive_structural_semantic_confidence(
        label_present,
        kind_grounded,
        true,
        page_context_verified,
        has_region,
    );
    compose_executable_target_confidence(
        entity.raw_confidence,
        None,
        None,
        frame.page_evidence.raw_confidence,
        (frame.page_evidence.confidence > 0.0).then_some(frame.page_evidence.confidence),
        planner_confidence,
        structural_semantic_confidence,
        has_region,
        page_context_verified,
        None,
    )
}

fn actionable_control_confidence_diagnostic(
    goal: &GoalSpec,
    frame: &SemanticScreenFrame,
    control: &ActionableControl,
    planner_confidence: Option<f32>,
) -> ExecutableTargetConfidenceDiagnostic {
    let page_context_verified = target_execution_context_matches_goal(goal, frame);
    let has_region = control.region.is_some();
    let label_present = control
        .label
        .as_deref()
        .is_some_and(|label| !label.trim().is_empty());
    let kind_grounded = normalize_label(&control.kind) != "unknown";
    let action_anchor_present =
        control_rank(control).is_some() || control_result_kind(control).is_some();
    let structural_semantic_confidence = derive_structural_semantic_confidence(
        label_present,
        kind_grounded,
        action_anchor_present,
        page_context_verified,
        has_region,
    );
    compose_executable_target_confidence(
        control.raw_confidence,
        None,
        None,
        frame.page_evidence.raw_confidence,
        (frame.page_evidence.confidence > 0.0).then_some(frame.page_evidence.confidence),
        planner_confidence,
        structural_semantic_confidence,
        has_region,
        page_context_verified,
        Some(ExecutableFallbackSource::ActionableControl),
    )
}

fn legacy_candidate_confidence_diagnostic(
    goal: &GoalSpec,
    frame: &SemanticScreenFrame,
    candidate: &UITargetCandidate,
    planner_confidence: Option<f32>,
) -> ExecutableTargetConfidenceDiagnostic {
    let page_context_verified = target_execution_context_matches_goal(goal, frame);
    let has_region = candidate.region.is_some() || candidate.center_point().is_some();
    let label_present = candidate
        .label
        .as_deref()
        .is_some_and(|label| !label.trim().is_empty());
    let kind_grounded = candidate
        .result_kind
        .as_deref()
        .is_some_and(|kind| !matches!(normalize_label(kind).as_str(), "" | "unknown" | "generic"));
    let action_anchor_present = candidate.rank.is_some();
    let structural_semantic_confidence = derive_structural_semantic_confidence(
        label_present,
        kind_grounded,
        action_anchor_present,
        page_context_verified,
        has_region,
    );
    compose_executable_target_confidence(
        Some(candidate.confidence),
        None,
        None,
        frame.page_evidence.raw_confidence,
        (frame.page_evidence.confidence > 0.0).then_some(frame.page_evidence.confidence),
        planner_confidence,
        structural_semantic_confidence,
        has_region,
        page_context_verified,
        Some(ExecutableFallbackSource::LegacyTargetCandidate),
    )
}

fn actionable_control_for_candidate<'a>(
    frame: &'a SemanticScreenFrame,
    candidate: &UITargetCandidate,
) -> Option<&'a ActionableControl> {
    let control_id = candidate.candidate_id.strip_prefix("planner_control_")?;
    frame
        .actionable_controls
        .iter()
        .find(|control| control.control_id == control_id)
}

fn legacy_candidate_for_id<'a>(
    frame: &'a SemanticScreenFrame,
    candidate_id: &str,
) -> Option<&'a UITargetCandidate> {
    frame
        .legacy_target_candidates
        .iter()
        .find(|candidate| candidate.candidate_id == candidate_id)
}

fn step_target_confidence_diagnostic(
    input: &PlannerContractInput,
    step: &PlannerStep,
) -> Option<ExecutableTargetConfidenceDiagnostic> {
    if let Some(candidate) = step.executable_candidate.as_ref() {
        match executable_fallback_source_for_candidate(candidate) {
            Some(ExecutableFallbackSource::ActionableControl) => {
                return actionable_control_for_candidate(&input.current_frame, candidate).map(
                    |control| {
                        actionable_control_confidence_diagnostic(
                            &input.goal,
                            &input.current_frame,
                            control,
                            None,
                        )
                    },
                );
            }
            Some(ExecutableFallbackSource::LegacyTargetCandidate) => {
                return legacy_candidate_for_id(&input.current_frame, &candidate.candidate_id).map(
                    |legacy_candidate| {
                        legacy_candidate_confidence_diagnostic(
                            &input.goal,
                            &input.current_frame,
                            legacy_candidate,
                            None,
                        )
                    },
                );
            }
            None => {}
        }
    }

    if let Some(item_id) = step.target_item_id.as_deref() {
        if let Some(item) = input
            .current_frame
            .primary_list
            .as_ref()
            .and_then(|list| list.items.iter().find(|item| item.item_id == item_id))
        {
            let click_region = step
                .click_region_key
                .as_deref()
                .and_then(|key| item.click_regions.get(key));
            let visible_like = visible_result_item_from_primary_list_item(item);
            return Some(result_item_confidence_diagnostic(
                &input.goal,
                &input.current_frame,
                &visible_like,
                click_region,
                None,
                None,
            ));
        }
        if let Some(item) = input
            .current_frame
            .visible_result_items
            .iter()
            .find(|item| item.item_id == item_id)
        {
            let click_region = step
                .click_region_key
                .as_deref()
                .and_then(|key| item.click_regions.get(key));
            return Some(result_item_confidence_diagnostic(
                &input.goal,
                &input.current_frame,
                item,
                click_region,
                None,
                None,
            ));
        }
    }

    if let Some(entity_id) = step.target_entity_id.as_deref() {
        if let Some(entity) = input
            .current_frame
            .visible_entities
            .iter()
            .find(|entity| entity.entity_id == entity_id)
        {
            return Some(entity_confidence_diagnostic(
                &input.goal,
                &input.current_frame,
                entity,
                None,
            ));
        }
    }

    None
}

fn rust_decision(
    input: &PlannerContractInput,
    proposed_step: PlannerStep,
    replan_needed: bool,
) -> PlannerContractDecision {
    rust_decision_with_strategy(
        input,
        proposed_step.clone(),
        strategy_for_step(&input.goal, &proposed_step),
        replan_needed,
    )
}

fn rust_decision_with_strategy(
    input: &PlannerContractInput,
    proposed_step: PlannerStep,
    strategy_rationale: String,
    replan_needed: bool,
) -> PlannerContractDecision {
    let mut visible_actionability = input.visible_actionability.clone();
    normalize_visible_actionability_for_step(&mut visible_actionability, &proposed_step);
    let (visibility_assessment, scroll_intent, rejection_code) =
        planner_diagnostics_for_step(input, &proposed_step);
    let target_confidence = step_target_confidence_diagnostic(input, &proposed_step);
    PlannerContractDecision {
        source: PlannerContractSource::RustDeterministic,
        strategy_rationale,
        focused_perception_needed: proposed_step.kind == PlannerStepKind::ReplanAfterPerception,
        replan_needed,
        expected_verification_target: proposed_step.expected_state.clone(),
        planner_confidence: proposed_step.confidence,
        accepted: true,
        fallback_used: false,
        rejection_reason: None,
        decision_status: PlannerDecisionStatus::Accepted,
        rejection_code,
        visibility_assessment,
        scroll_intent,
        visible_actionability,
        target_confidence,
        normalized: false,
        downgraded: false,
        proposed_step,
    }
}

fn normalize_visible_actionability_for_step(
    visible_actionability: &mut VisibleActionabilityDiagnostic,
    step: &PlannerStep,
) {
    visible_actionability.fallback_source_used = step
        .executable_candidate
        .as_ref()
        .and_then(executable_fallback_source_for_candidate);

    if matches!(
        step.kind,
        PlannerStepKind::ClickResultRegion | PlannerStepKind::ClickEntityRegion
    ) && step.executable_candidate.is_some()
    {
        visible_actionability.status = VisibleActionabilityStatus::VisibleExecutable;
        visible_actionability.relevant_visible_content = true;
        visible_actionability.target_visible_evidence = true;
        visible_actionability.refinement_eligible = false;
        visible_actionability.refinement_strategy = VisibleRefinementStrategy::None;
        visible_actionability.offscreen_inference_stage = OffscreenInferenceStage::NotApplicable;
    }
}

fn executable_fallback_source_for_candidate(
    candidate: &UITargetCandidate,
) -> Option<ExecutableFallbackSource> {
    match candidate.observation_source.as_deref() {
        Some("planner_actionable_control_fallback") => {
            Some(ExecutableFallbackSource::ActionableControl)
        }
        Some("planner_legacy_candidate_fallback") => {
            Some(ExecutableFallbackSource::LegacyTargetCandidate)
        }
        _ => None,
    }
}

fn planner_diagnostics_for_step(
    input: &PlannerContractInput,
    step: &PlannerStep,
) -> (
    PlannerVisibilityAssessment,
    PlannerScrollIntent,
    Option<PlannerRejectionReason>,
) {
    match step.kind {
        PlannerStepKind::ClickResultRegion | PlannerStepKind::ClickEntityRegion => {
            if step.executable_candidate.is_some() {
                (
                    PlannerVisibilityAssessment::VisibleGrounded,
                    PlannerScrollIntent::NotNeeded,
                    None,
                )
            } else {
                (
                    PlannerVisibilityAssessment::VisibleTargetNeedsClickRegion,
                    PlannerScrollIntent::NotNeeded,
                    Some(PlannerRejectionReason::MissingClickRegion),
                )
            }
        }
        PlannerStepKind::ReplanAfterPerception => {
            if step.target_item_id.is_some() || step.target_entity_id.is_some() {
                if target_region_anchor_for_step(&input.current_frame, step).is_some() {
                    (
                        PlannerVisibilityAssessment::VisibleTargetNeedsClickRegion,
                        PlannerScrollIntent::NotNeeded,
                        Some(PlannerRejectionReason::FocusedPerceptionRequested),
                    )
                } else {
                    (
                        PlannerVisibilityAssessment::VisibleTargetNeedsClickRegion,
                        PlannerScrollIntent::NotNeeded,
                        Some(PlannerRejectionReason::VisiblePageRefinementRequested),
                    )
                }
            } else {
                match input.visible_actionability.status {
                    VisibleActionabilityStatus::VisibleUnderGrounded => (
                        PlannerVisibilityAssessment::VisibleUnderGrounded,
                        PlannerScrollIntent::NotNeeded,
                        Some(PlannerRejectionReason::VisiblePageRefinementRequested),
                    ),
                    VisibleActionabilityStatus::LikelyOffscreen => (
                        PlannerVisibilityAssessment::LikelyOffscreen,
                        PlannerScrollIntent::RequiredButUnsupported,
                        Some(PlannerRejectionReason::ScrollRequiredButUnsupported),
                    ),
                    VisibleActionabilityStatus::NoRelevantVisibleContent => (
                        PlannerVisibilityAssessment::NotVisible,
                        PlannerScrollIntent::NotNeeded,
                        Some(PlannerRejectionReason::MissingTarget),
                    ),
                    VisibleActionabilityStatus::VisibleExecutable => (
                        PlannerVisibilityAssessment::VisibleGrounded,
                        PlannerScrollIntent::NotNeeded,
                        None,
                    ),
                    VisibleActionabilityStatus::VisibleTargetNeedsClickRegion => (
                        PlannerVisibilityAssessment::VisibleTargetNeedsClickRegion,
                        PlannerScrollIntent::NotNeeded,
                        Some(PlannerRejectionReason::FocusedPerceptionRequested),
                    ),
                    VisibleActionabilityStatus::Unknown => (
                        PlannerVisibilityAssessment::Unknown,
                        PlannerScrollIntent::NotNeeded,
                        Some(PlannerRejectionReason::MissingTarget),
                    ),
                }
            }
        }
        PlannerStepKind::RequestClarification => (
            PlannerVisibilityAssessment::Unknown,
            PlannerScrollIntent::NotNeeded,
            Some(PlannerRejectionReason::AmbiguousTarget),
        ),
        PlannerStepKind::Refuse => (
            PlannerVisibilityAssessment::Unknown,
            PlannerScrollIntent::NotNeeded,
            Some(PlannerRejectionReason::UnsupportedPrimitive),
        ),
        PlannerStepKind::VerifyGoal | PlannerStepKind::NoOp => (
            PlannerVisibilityAssessment::VisibleGrounded,
            PlannerScrollIntent::NotNeeded,
            None,
        ),
    }
}

fn page_matches_goal_context(input: &PlannerContractInput) -> bool {
    let provider_matches = input
        .goal
        .constraints
        .provider
        .as_deref()
        .map_or(true, |expected| {
            provider_context_compatible(
                Some(expected),
                input
                    .current_frame
                    .page_evidence
                    .content_provider_hint
                    .as_deref()
                    .or(input.provider_hint.as_deref()),
            )
        });
    provider_matches && list_context_visible(&input.current_frame)
}

fn provider_matches_item(
    expected_provider: Option<&str>,
    frame_provider: Option<&str>,
    item: &VisibleResultItem,
) -> bool {
    expected_provider.map_or(true, |expected| {
        item.provider
            .as_deref()
            .or(frame_provider)
            .map_or(true, |observed| labels_match(expected, observed))
    })
}

fn provider_context_compatible(
    expected_provider: Option<&str>,
    observed_provider: Option<&str>,
) -> bool {
    match (expected_provider, observed_provider) {
        (Some(expected), Some(observed)) => labels_match(expected, observed),
        _ => true,
    }
}

fn list_context_visible(frame: &SemanticScreenFrame) -> bool {
    is_search_results_page(frame)
        || frame.page_evidence.result_list_visible == Some(true)
        || structural_list_surface_visible(frame)
        || scene_summary_suggests_results(&frame.scene_summary)
}

fn requested_item_kind_label(goal: &GoalSpec) -> Option<String> {
    goal.constraints
        .item_kind
        .as_deref()
        .map(normalize_item_kind_label)
        .or_else(|| {
            goal.constraints
                .result_kind
                .as_ref()
                .map(result_kind_label)
                .map(normalize_item_kind_label)
        })
        .or_else(|| matches!(goal.goal_type, GoalType::OpenMediaResult).then_some("video".into()))
}

fn item_kind_compatibility(
    requested_kind: Option<&str>,
    observed_kind: Option<&str>,
) -> Option<f32> {
    let Some(requested) = requested_kind.map(normalize_item_kind_label) else {
        return Some(1.0);
    };
    let observed = observed_kind
        .map(normalize_item_kind_label)
        .unwrap_or_else(|| "unknown".into());
    if requested == observed {
        return Some(1.0);
    }
    if matches!(observed.as_str(), "generic" | "result") {
        return Some(0.94);
    }
    if requested == "site" && matches!(observed.as_str(), "article" | "generic" | "result") {
        return Some(0.94);
    }
    None
}

pub fn validate_model_planner_decision(
    input: &PlannerContractInput,
    mut model_decision: PlannerContractDecision,
    deterministic: PlannerContractDecision,
) -> PlannerContractDecision {
    if model_decision.planner_confidence < 0.60 || model_decision.proposed_step.confidence < 0.60 {
        return planner_fallback_decision_with_code(
            deterministic,
            "model planner confidence below safe planning threshold",
            PlannerRejectionReason::LowConfidence,
        );
    }
    model_decision.visible_actionability = input.visible_actionability.clone();

    if matches!(
        model_decision.scroll_intent,
        PlannerScrollIntent::RequiredButUnsupported | PlannerScrollIntent::FuturePrimitiveRequired
    ) && input.visible_actionability.refinement_eligible
        && input.visible_actionability.offscreen_inference_stage
            == OffscreenInferenceStage::DeferredPendingRefinement
    {
        return planner_fallback_decision_with_code(
            deterministic,
            "model planner requested offscreen before visible_page_refinement completed",
            PlannerRejectionReason::VisiblePageRefinementRequested,
        );
    }

    match model_decision.proposed_step.kind {
        PlannerStepKind::ClickResultRegion => validate_model_result_click(input, model_decision)
            .unwrap_or_else(|reason| planner_fallback_decision(deterministic, reason)),
        PlannerStepKind::ClickEntityRegion => validate_model_entity_click(input, model_decision)
            .unwrap_or_else(|reason| planner_fallback_decision(deterministic, reason)),
        PlannerStepKind::ReplanAfterPerception
        | PlannerStepKind::RequestClarification
        | PlannerStepKind::Refuse
        | PlannerStepKind::VerifyGoal
        | PlannerStepKind::NoOp => accepted_model_decision(model_decision),
    }
}

pub fn planner_fallback_decision(
    deterministic: PlannerContractDecision,
    reason: impl Into<String>,
) -> PlannerContractDecision {
    let reason = reason.into();
    let code = planner_rejection_code_from_reason(&reason);
    planner_fallback_decision_with_code(deterministic, reason, code)
}

fn planner_fallback_decision_with_code(
    mut deterministic: PlannerContractDecision,
    reason: impl Into<String>,
    code: PlannerRejectionReason,
) -> PlannerContractDecision {
    deterministic.source = PlannerContractSource::ModelAssistedFallback;
    deterministic.accepted = true;
    deterministic.fallback_used = true;
    deterministic.rejection_reason = Some(reason.into());
    deterministic.rejection_code = Some(code);
    deterministic.decision_status = PlannerDecisionStatus::FallbackUsed;
    deterministic.downgraded = true;
    deterministic
}

fn planner_rejection_code_from_reason(reason: &str) -> PlannerRejectionReason {
    let reason = normalize_label(reason);
    if reason.contains("deterministic_fallback") || reason.contains("advised_deterministic") {
        PlannerRejectionReason::DeterministicFallbackAdvised
    } else if reason.contains("parse_failed")
        || reason.contains("schema_parse")
        || reason.contains("malformed")
    {
        PlannerRejectionReason::MalformedOutput
    } else if reason.contains("unavailable") {
        PlannerRejectionReason::ModelUnavailable
    } else if reason.contains("confidence") {
        PlannerRejectionReason::LowConfidence
    } else if reason.contains("not_present") || reason.contains("not present") {
        PlannerRejectionReason::FabricatedTarget
    } else if reason.contains("omitted") || reason.contains("missing") {
        PlannerRejectionReason::MissingTarget
    } else if reason.contains("provider_mismatch") || reason.contains("provider mismatch") {
        PlannerRejectionReason::ProviderMismatch
    } else if reason.contains("click_region") || reason.contains("click region") {
        PlannerRejectionReason::MissingClickRegion
    } else if reason.contains("visible_page_refinement")
        || reason.contains("visible page refinement")
    {
        PlannerRejectionReason::VisiblePageRefinementRequested
    } else {
        PlannerRejectionReason::Unknown
    }
}

fn accepted_model_decision(mut decision: PlannerContractDecision) -> PlannerContractDecision {
    decision.source = PlannerContractSource::ModelAssisted;
    decision.accepted = true;
    decision.fallback_used = false;
    decision.rejection_reason = None;
    decision.rejection_code = None;
    if decision.decision_status == PlannerDecisionStatus::Rejected
        || decision.decision_status == PlannerDecisionStatus::FallbackUsed
    {
        decision.decision_status = PlannerDecisionStatus::Accepted;
    }
    decision
}

fn validate_model_result_click(
    input: &PlannerContractInput,
    mut decision: PlannerContractDecision,
) -> Result<PlannerContractDecision, String> {
    let step = &decision.proposed_step;
    let item_id = step
        .target_item_id
        .as_deref()
        .ok_or_else(|| "model planner result click omitted target_item_id".to_string())?;
    let item = input
        .current_frame
        .visible_result_items
        .iter()
        .find(|item| item.item_id == item_id)
        .ok_or_else(|| {
            "model planner selected a result item not present in the current semantic frame"
                .to_string()
        })?;
    let region_key = step
        .click_region_key
        .as_deref()
        .unwrap_or("primary")
        .to_string();
    let (region_key, click_region) = item
        .click_regions
        .get(&region_key)
        .map(|region| (region_key, region))
        .or_else(|| preferred_click_region(item, &["title", "thumbnail", "primary"]))
        .ok_or_else(|| {
            "model planner selected a result item without a trusted click region".to_string()
        })?;
    let provider = input.goal.constraints.provider.clone().or_else(|| {
        input
            .current_frame
            .page_evidence
            .content_provider_hint
            .clone()
    });
    if provider.as_deref().is_some_and(|expected| {
        item.provider
            .as_deref()
            .or(input
                .current_frame
                .page_evidence
                .content_provider_hint
                .as_deref())
            .is_some_and(|observed| !labels_match(expected, observed))
    }) {
        return Err("model planner selected result with provider mismatch".into());
    }
    let confidence_diagnostic = result_item_confidence_diagnostic(
        &input.goal,
        &input.current_frame,
        item,
        Some(click_region),
        Some(step.confidence),
        None,
    );
    if !confidence_diagnostic.accepted {
        return Err("model planner selected result below execution confidence threshold".into());
    }
    let confidence = confidence_diagnostic.derived_confidence;
    let requested_rank = input
        .goal
        .constraints
        .rank_within_kind
        .or(item.rank_within_kind)
        .or(item.rank_overall);
    let candidate = candidate_from_result_item(
        item,
        &region_key,
        click_region,
        provider,
        requested_rank,
        confidence,
    );
    decision.proposed_step.confidence = confidence;
    decision.proposed_step.click_region_key = Some(region_key);
    decision.proposed_step.executable_candidate = Some(candidate);
    decision.proposed_step.expected_state = decision
        .proposed_step
        .expected_state
        .clone()
        .or_else(|| Some(input.goal.success_condition.clone()));
    decision.visibility_assessment = PlannerVisibilityAssessment::VisibleGrounded;
    decision.scroll_intent = PlannerScrollIntent::NotNeeded;
    decision.visible_actionability.status = VisibleActionabilityStatus::VisibleExecutable;
    decision.visible_actionability.target_visible_evidence = true;
    decision.visible_actionability.relevant_visible_content = true;
    decision.visible_actionability.refinement_eligible = false;
    decision.visible_actionability.refinement_strategy = VisibleRefinementStrategy::None;
    decision.visible_actionability.offscreen_inference_stage =
        OffscreenInferenceStage::NotApplicable;
    decision.target_confidence = Some(confidence_diagnostic);
    decision.decision_status = PlannerDecisionStatus::Normalized;
    decision.normalized = true;
    Ok(accepted_model_decision(decision))
}

fn validate_model_entity_click(
    input: &PlannerContractInput,
    mut decision: PlannerContractDecision,
) -> Result<PlannerContractDecision, String> {
    let step = &decision.proposed_step;
    let entity_id = step
        .target_entity_id
        .as_deref()
        .ok_or_else(|| "model planner entity click omitted target_entity_id".to_string())?;
    let entity = input
        .current_frame
        .visible_entities
        .iter()
        .find(|entity| entity.entity_id == entity_id)
        .ok_or_else(|| {
            "model planner selected an entity not present in the current semantic frame".to_string()
        })?;
    let confidence_diagnostic = entity_confidence_diagnostic(
        &input.goal,
        &input.current_frame,
        entity,
        Some(step.confidence),
    );
    if !confidence_diagnostic.accepted {
        return Err("model planner selected entity below execution confidence threshold".into());
    }
    let confidence = confidence_diagnostic.derived_confidence;
    let candidate = candidate_from_entity(
        entity,
        &input.current_frame,
        Some(entity_kind_result_label(&entity.kind)),
        "model planner selected visible semantic entity",
        confidence,
    )
    .ok_or_else(|| "model planner selected entity without a trusted region".to_string())?;
    decision.proposed_step.confidence = confidence;
    decision.proposed_step.click_region_key = Some("entity".into());
    decision.proposed_step.executable_candidate = Some(candidate);
    decision.proposed_step.expected_state = decision
        .proposed_step
        .expected_state
        .clone()
        .or_else(|| Some(input.goal.success_condition.clone()));
    decision.visibility_assessment = PlannerVisibilityAssessment::VisibleGrounded;
    decision.scroll_intent = PlannerScrollIntent::NotNeeded;
    decision.visible_actionability.status = VisibleActionabilityStatus::VisibleExecutable;
    decision.visible_actionability.target_visible_evidence = true;
    decision.visible_actionability.relevant_visible_content = true;
    decision.visible_actionability.refinement_eligible = false;
    decision.visible_actionability.refinement_strategy = VisibleRefinementStrategy::None;
    decision.visible_actionability.offscreen_inference_stage =
        OffscreenInferenceStage::NotApplicable;
    decision.target_confidence = Some(confidence_diagnostic);
    decision.decision_status = PlannerDecisionStatus::Normalized;
    decision.normalized = true;
    Ok(accepted_model_decision(decision))
}

#[allow(dead_code)]
pub fn goal_for_open_media_result(
    utterance: impl Into<String>,
    provider: Option<String>,
    result_kind: VisibleResultKind,
    rank_within_kind: u32,
) -> GoalSpec {
    GoalSpec {
        goal_id: Uuid::new_v4().to_string(),
        goal_type: GoalType::OpenMediaResult,
        constraints: GoalConstraints {
            provider,
            item_kind: Some(result_kind_label(&result_kind).into()),
            result_kind: Some(result_kind),
            rank_within_kind: Some(rank_within_kind),
            rank_overall: None,
            entity_name: None,
            attributes: Value::Null,
        },
        success_condition: "video_watch_page_open".into(),
        utterance: utterance.into(),
        confidence: 0.86,
    }
}

pub fn goal_for_open_list_item(
    utterance: impl Into<String>,
    provider: Option<String>,
    item_kind: Option<String>,
    rank_within_kind: Option<u32>,
    rank_overall: Option<u32>,
) -> GoalSpec {
    let normalized_item_kind = item_kind.as_deref().map(normalize_item_kind_label);
    GoalSpec {
        goal_id: Uuid::new_v4().to_string(),
        goal_type: GoalType::OpenListItem,
        constraints: GoalConstraints {
            provider,
            item_kind: normalized_item_kind.clone(),
            result_kind: normalized_item_kind
                .as_deref()
                .map(|kind| result_kind_from_value(Some(kind))),
            rank_within_kind,
            rank_overall,
            entity_name: None,
            attributes: Value::Null,
        },
        success_condition: "list_item_detail_open".into(),
        utterance: utterance.into(),
        confidence: 0.86,
    }
}

pub fn planner_candidate_from_step(step: &PlannerStep) -> Option<UITargetCandidate> {
    step.executable_candidate.clone()
}

fn plan_open_list_item(goal: &GoalSpec, frame: &SemanticScreenFrame) -> PlannerStep {
    let desired_kind = requested_item_kind_label(goal);
    let desired_rank = goal
        .constraints
        .rank_within_kind
        .or(goal.constraints.rank_overall)
        .or(Some(1));
    let provider = goal
        .constraints
        .provider
        .clone()
        .or_else(|| frame.page_evidence.content_provider_hint.clone());

    if let Some(item) = find_primary_list_item_match(goal, frame) {
        let Some((region_key, click_region)) =
            preferred_primary_click_region(item, &["primary", "title", "thumbnail"])
        else {
            let visible_like = visible_result_item_from_primary_list_item(item);
            let replan_confidence =
                result_item_confidence_diagnostic(goal, frame, &visible_like, None, None, None)
                    .semantic_confidence;
            return PlannerStep {
                step_id: Uuid::new_v4().to_string(),
                kind: PlannerStepKind::ReplanAfterPerception,
                confidence: replan_confidence,
                rationale: "matching primary list item is visible but needs focused perception before a click region can be trusted".into(),
                target_item_id: Some(item.item_id.clone()),
                target_entity_id: None,
                click_region_key: None,
                executable_candidate: None,
                expected_state: Some(goal.success_condition.clone()),
            };
        };

        let visible_like = visible_result_item_from_primary_list_item(item);
        let compatibility =
            item_kind_compatibility(desired_kind.as_deref(), item.item_kind.as_deref())
                .unwrap_or(1.0);
        let confidence_diagnostic = result_item_confidence_diagnostic(
            goal,
            frame,
            &visible_like,
            Some(click_region),
            None,
            None,
        );
        let confidence = (confidence_diagnostic.derived_confidence * compatibility).clamp(0.0, 1.0);
        let candidate = candidate_from_primary_list_item(
            item,
            &region_key,
            click_region,
            provider,
            desired_rank,
            confidence,
        );
        return PlannerStep {
            step_id: Uuid::new_v4().to_string(),
            kind: if confidence >= MIN_PLANNER_CLICK_CONFIDENCE {
                PlannerStepKind::ClickResultRegion
            } else {
                PlannerStepKind::Refuse
            },
            confidence,
            rationale: format!(
                "selected primary list item rank {:?} with structural kind {:?} using {region_key} region",
                desired_rank,
                item.item_kind
            ),
            target_item_id: Some(item.item_id.clone()),
            target_entity_id: None,
            click_region_key: Some(region_key),
            executable_candidate: (confidence >= MIN_PLANNER_CLICK_CONFIDENCE).then_some(candidate),
            expected_state: Some(goal.success_condition.clone()),
        };
    }

    let Some(item) = find_ranked_result_match(goal, frame) else {
        return PlannerStep {
            step_id: Uuid::new_v4().to_string(),
            kind: PlannerStepKind::ReplanAfterPerception,
            confidence: 0.35,
            rationale: format!(
                "no visible {:?} list item with rank {:?} is grounded in the semantic frame",
                desired_kind, desired_rank
            ),
            target_item_id: None,
            target_entity_id: None,
            click_region_key: None,
            executable_candidate: None,
            expected_state: Some("more_perception_or_scroll_required".into()),
        };
    };

    let Some((region_key, click_region)) =
        preferred_click_region(item, &["title", "thumbnail", "primary"])
    else {
        let replan_confidence =
            result_item_confidence_diagnostic(goal, frame, item, None, None, None)
                .semantic_confidence;
        return PlannerStep {
            step_id: Uuid::new_v4().to_string(),
            kind: PlannerStepKind::ReplanAfterPerception,
            confidence: replan_confidence,
            rationale: "matching result is visible but needs focused perception before a click region can be trusted".into(),
            target_item_id: Some(item.item_id.clone()),
            target_entity_id: None,
            click_region_key: None,
            executable_candidate: None,
            expected_state: Some(goal.success_condition.clone()),
        };
    };

    let confidence_diagnostic =
        result_item_confidence_diagnostic(goal, frame, item, Some(click_region), None, None);
    let compatibility =
        item_kind_compatibility(desired_kind.as_deref(), Some(result_kind_label(&item.kind)))
            .unwrap_or(1.0);
    let confidence = (confidence_diagnostic.derived_confidence * compatibility).clamp(0.0, 1.0);
    let candidate = candidate_from_result_item(
        item,
        &region_key,
        click_region,
        provider,
        desired_rank,
        confidence,
    );
    PlannerStep {
        step_id: Uuid::new_v4().to_string(),
        kind: if confidence >= MIN_PLANNER_CLICK_CONFIDENCE {
            PlannerStepKind::ClickResultRegion
        } else {
            PlannerStepKind::Refuse
        },
        confidence,
        rationale: format!(
            "selected {:?} result {:?} using {region_key} region",
            item.kind,
            effective_rank_for_goal(frame, item, desired_kind.as_deref()).or(item.rank_overall)
        ),
        target_item_id: Some(item.item_id.clone()),
        target_entity_id: None,
        click_region_key: Some(region_key),
        executable_candidate: (confidence >= MIN_PLANNER_CLICK_CONFIDENCE).then_some(candidate),
        expected_state: Some(goal.success_condition.clone()),
    }
}

fn plan_open_channel(goal: &GoalSpec, frame: &SemanticScreenFrame) -> PlannerStep {
    let desired_name = goal.constraints.entity_name.as_deref();
    if let Some(entity) = frame.visible_entities.iter().find(|entity| {
        matches!(
            entity.kind,
            VisibleEntityKind::ChannelHeader | VisibleEntityKind::ChannelResult
        ) && desired_name.map_or(true, |name| {
            entity
                .name
                .as_deref()
                .map_or(false, |observed| loose_text_match(name, observed))
        })
    }) {
        let confidence = entity_confidence_diagnostic(goal, frame, entity, None).derived_confidence;
        if let Some(candidate) = candidate_from_entity(
            entity,
            frame,
            Some("channel"),
            "planner selected visible channel entity",
            confidence,
        ) {
            return PlannerStep {
                step_id: Uuid::new_v4().to_string(),
                kind: PlannerStepKind::ClickEntityRegion,
                confidence,
                rationale: "direct channel entity is visible".into(),
                target_item_id: None,
                target_entity_id: Some(entity.entity_id.clone()),
                click_region_key: Some("entity".into()),
                executable_candidate: Some(candidate),
                expected_state: Some("channel_page_visible".into()),
            };
        }
    }

    if let Some(item) = frame.visible_result_items.iter().find(|item| {
        item.kind == VisibleResultKind::Channel
            && desired_name.map_or(true, |name| {
                item.title
                    .as_deref()
                    .or(item.channel_name.as_deref())
                    .map_or(false, |observed| loose_text_match(name, observed))
            })
    }) {
        if let Some((region_key, click_region)) =
            preferred_click_region(item, &["title", "thumbnail", "primary"])
        {
            let provider = goal
                .constraints
                .provider
                .clone()
                .or_else(|| frame.page_evidence.content_provider_hint.clone());
            let confidence_diagnostic = result_item_confidence_diagnostic(
                goal,
                frame,
                item,
                Some(click_region),
                None,
                None,
            );
            let confidence = confidence_diagnostic.derived_confidence;
            let candidate = candidate_from_result_item(
                item,
                &region_key,
                click_region,
                provider,
                None,
                confidence,
            );
            return PlannerStep {
                step_id: Uuid::new_v4().to_string(),
                kind: if confidence >= MIN_PLANNER_CLICK_CONFIDENCE {
                    PlannerStepKind::ClickResultRegion
                } else {
                    PlannerStepKind::Refuse
                },
                confidence,
                rationale: format!("selected visible channel result using {region_key} region"),
                target_item_id: Some(item.item_id.clone()),
                target_entity_id: None,
                click_region_key: Some(region_key),
                executable_candidate: (confidence >= MIN_PLANNER_CLICK_CONFIDENCE)
                    .then_some(candidate),
                expected_state: Some("channel_page_visible".into()),
            };
        }

        let replan_confidence =
            result_item_confidence_diagnostic(goal, frame, item, None, None, None)
                .semantic_confidence;
        return PlannerStep {
            step_id: Uuid::new_v4().to_string(),
            kind: PlannerStepKind::ReplanAfterPerception,
            confidence: replan_confidence,
            rationale: "channel result is visible but needs focused perception before its click region can be trusted".into(),
            target_item_id: Some(item.item_id.clone()),
            target_entity_id: None,
            click_region_key: None,
            executable_candidate: None,
            expected_state: Some("channel_page_visible".into()),
        };
    }

    if is_watch_page(frame) {
        if let Some(entity) = frame.visible_entities.iter().find(|entity| {
            matches!(
                entity.kind,
                VisibleEntityKind::Avatar
                    | VisibleEntityKind::ChannelHeader
                    | VisibleEntityKind::ChannelResult
            ) && desired_name.map_or(true, |name| {
                entity
                    .name
                    .as_deref()
                    .map_or(true, |observed| loose_text_match(name, observed))
            })
        }) {
            let confidence =
                entity_confidence_diagnostic(goal, frame, entity, None).derived_confidence;
            if let Some(candidate) = candidate_from_entity(
                entity,
                frame,
                Some("channel"),
                "planner selected channel avatar or link on watch page",
                confidence,
            ) {
                return PlannerStep {
                    step_id: Uuid::new_v4().to_string(),
                    kind: PlannerStepKind::ClickEntityRegion,
                    confidence,
                    rationale: "video page is visible; selecting channel avatar/link to complete channel navigation".into(),
                    target_item_id: None,
                    target_entity_id: Some(entity.entity_id.clone()),
                    click_region_key: Some("entity".into()),
                    executable_candidate: Some(candidate),
                    expected_state: Some("channel_page_visible".into()),
                };
            }
        }

        return PlannerStep {
            step_id: Uuid::new_v4().to_string(),
            kind: PlannerStepKind::ReplanAfterPerception,
            confidence: 0.48,
            rationale: "watch page is visible but channel avatar/link is not grounded yet".into(),
            target_item_id: None,
            target_entity_id: None,
            click_region_key: None,
            executable_candidate: None,
            expected_state: Some("channel_page_visible".into()),
        };
    }

    if let Some(video) = frame.visible_result_items.iter().find(|item| {
        item.kind == VisibleResultKind::Video
            && desired_name.map_or(true, |name| {
                item.channel_name
                    .as_deref()
                    .or(item.title.as_deref())
                    .map_or(true, |observed| loose_text_match(name, observed))
            })
    }) {
        if let Some((region_key, click_region)) =
            preferred_click_region(video, &["title", "thumbnail", "primary"])
        {
            let provider = goal
                .constraints
                .provider
                .clone()
                .or_else(|| frame.page_evidence.content_provider_hint.clone());
            let confidence_diagnostic = result_item_confidence_diagnostic(
                goal,
                frame,
                video,
                Some(click_region),
                None,
                None,
            );
            let confidence = confidence_diagnostic.derived_confidence;
            let candidate = candidate_from_result_item(
                video,
                &region_key,
                click_region,
                provider,
                None,
                confidence,
            );
            return PlannerStep {
                step_id: Uuid::new_v4().to_string(),
                kind: if confidence >= MIN_PLANNER_CLICK_CONFIDENCE {
                    PlannerStepKind::ClickResultRegion
                } else {
                    PlannerStepKind::Refuse
                },
                confidence,
                rationale: format!(
                    "channel is not directly visible; opening relevant video via {region_key} before selecting avatar"
                ),
                target_item_id: Some(video.item_id.clone()),
                target_entity_id: None,
                click_region_key: Some(region_key),
                executable_candidate: (confidence >= MIN_PLANNER_CLICK_CONFIDENCE)
                    .then_some(candidate),
                expected_state: Some("media_watch_page_visible".into()),
            };
        }

        let replan_confidence =
            result_item_confidence_diagnostic(goal, frame, video, None, None, None)
                .semantic_confidence;
        return PlannerStep {
            step_id: Uuid::new_v4().to_string(),
            kind: PlannerStepKind::ReplanAfterPerception,
            confidence: replan_confidence,
            rationale: "fallback video is visible but needs focused perception before its click region can be trusted".into(),
            target_item_id: Some(video.item_id.clone()),
            target_entity_id: None,
            click_region_key: None,
            executable_candidate: None,
            expected_state: Some("media_watch_page_visible".into()),
        };
    }

    PlannerStep {
        step_id: Uuid::new_v4().to_string(),
        kind: PlannerStepKind::ReplanAfterPerception,
        confidence: 0.45,
        rationale:
            "channel target is not directly visible and no safe fallback video is grounded yet"
                .into(),
        target_item_id: None,
        target_entity_id: None,
        click_region_key: None,
        executable_candidate: None,
        expected_state: Some("channel_page_visible".into()),
    }
}

fn candidate_from_result_item(
    item: &VisibleResultItem,
    region_key: &str,
    click_region: &ClickRegion,
    provider: Option<String>,
    requested_rank: Option<u32>,
    confidence: f32,
) -> UITargetCandidate {
    UITargetCandidate {
        candidate_id: format!("planner_result_{}_{}", item.item_id, region_key),
        role: UITargetRole::RankedResult,
        region: Some(click_region.region.clone()),
        center_x: None,
        center_y: None,
        app_hint: None,
        browser_app_hint: None,
        provider_hint: provider.clone().or_else(|| item.provider.clone()),
        content_provider_hint: provider.or_else(|| item.provider.clone()),
        page_kind_hint: Some("result_list".into()),
        capture_backend: None,
        observation_source: Some("planner_semantic_frame".into()),
        result_kind: Some(result_kind_label(&item.kind).into()),
        confidence,
        source: TargetGroundingSource::ScreenAnalysis,
        label: item.title.clone(),
        rank: requested_rank
            .or(item.rank_within_kind)
            .or(item.rank_overall),
        observed_at_ms: None,
        reuse_eligible: true,
        supports_focus: false,
        supports_click: true,
        rationale: format!("planner selected semantic {:?} result", item.kind),
    }
}

fn candidate_from_primary_list_item(
    item: &PrimaryListItem,
    region_key: &str,
    click_region: &ClickRegion,
    provider: Option<String>,
    requested_rank: Option<u32>,
    confidence: f32,
) -> UITargetCandidate {
    UITargetCandidate {
        candidate_id: format!("planner_primary_list_{}_{}", item.item_id, region_key),
        role: UITargetRole::RankedResult,
        region: Some(click_region.region.clone()),
        center_x: None,
        center_y: None,
        app_hint: None,
        browser_app_hint: None,
        provider_hint: provider.clone(),
        content_provider_hint: provider,
        page_kind_hint: Some("result_list".into()),
        capture_backend: None,
        observation_source: Some("planner_primary_list".into()),
        result_kind: item.item_kind.clone().or_else(|| Some("generic".into())),
        confidence,
        source: TargetGroundingSource::ScreenAnalysis,
        label: item.title.clone(),
        rank: requested_rank.or(Some(item.rank)),
        observed_at_ms: None,
        reuse_eligible: true,
        supports_focus: false,
        supports_click: true,
        rationale: format!(
            "planner selected provider-agnostic primary list item rank {}",
            item.rank
        ),
    }
}

fn visible_result_item_from_primary_list_item(item: &PrimaryListItem) -> VisibleResultItem {
    VisibleResultItem {
        item_id: item.item_id.clone(),
        kind: result_kind_from_value(item.item_kind.as_deref()),
        title: item.title.clone(),
        channel_name: None,
        provider: None,
        rank_overall: Some(item.rank),
        rank_within_kind: Some(item.rank),
        click_regions: item.click_regions.clone(),
        raw_confidence: item.raw_confidence,
        confidence: item.confidence,
        rationale: Some("primary_list_structural_item".into()),
        attributes: item.attributes.clone(),
    }
}

fn candidate_from_entity(
    entity: &VisibleEntity,
    frame: &SemanticScreenFrame,
    result_kind: Option<&str>,
    rationale: &str,
    confidence: f32,
) -> Option<UITargetCandidate> {
    Some(UITargetCandidate {
        candidate_id: format!("planner_entity_{}", entity.entity_id),
        role: UITargetRole::Link,
        region: Some(entity.region.clone()?),
        center_x: None,
        center_y: None,
        app_hint: frame.page_evidence.browser_app_hint.clone(),
        browser_app_hint: frame.page_evidence.browser_app_hint.clone(),
        provider_hint: frame.page_evidence.content_provider_hint.clone(),
        content_provider_hint: frame.page_evidence.content_provider_hint.clone(),
        page_kind_hint: frame.page_evidence.page_kind_hint.clone(),
        capture_backend: frame.page_evidence.capture_backend.clone(),
        observation_source: Some("planner_semantic_frame".into()),
        result_kind: result_kind.map(ToOwned::to_owned),
        confidence,
        source: TargetGroundingSource::ScreenAnalysis,
        label: entity.name.clone(),
        rank: None,
        observed_at_ms: Some(frame.captured_at),
        reuse_eligible: true,
        supports_focus: false,
        supports_click: true,
        rationale: rationale.into(),
    })
}

fn preferred_click_region<'a>(
    item: &'a VisibleResultItem,
    keys: &[&str],
) -> Option<(String, &'a ClickRegion)> {
    for key in keys {
        if let Some(region) = item.click_regions.get(*key) {
            return Some(((*key).to_string(), region));
        }
    }
    item.click_regions
        .iter()
        .next()
        .map(|(key, region)| (key.clone(), region))
}

fn preferred_primary_click_region<'a>(
    item: &'a PrimaryListItem,
    keys: &[&str],
) -> Option<(String, &'a ClickRegion)> {
    for key in keys {
        if let Some(region) = item.click_regions.get(*key) {
            return Some(((*key).to_string(), region));
        }
    }
    item.click_regions
        .iter()
        .next()
        .map(|(key, region)| (key.clone(), region))
}

#[allow(dead_code)]
fn step_is_executable(step: &PlannerStep, min_confidence: f32) -> bool {
    matches!(
        step.kind,
        PlannerStepKind::ClickResultRegion | PlannerStepKind::ClickEntityRegion
    ) && step.confidence >= min_confidence
        && step
            .executable_candidate
            .as_ref()
            .is_some_and(|candidate| candidate.supports_click && candidate.center_point().is_some())
}

#[allow(dead_code)]
fn strategy_for_step(goal: &GoalSpec, step: &PlannerStep) -> String {
    match (&goal.goal_type, &step.kind) {
        (
            GoalType::OpenListItem | GoalType::OpenMediaResult,
            PlannerStepKind::ClickResultRegion,
        ) => "open_visible_list_item".into(),
        (GoalType::OpenChannel, PlannerStepKind::ClickEntityRegion)
            if step.expected_state.as_deref() == Some("channel_page_visible") =>
        {
            "youtube_click_visible_channel_entity".into()
        }
        (GoalType::OpenChannel, PlannerStepKind::ClickResultRegion)
            if step.expected_state.as_deref() == Some("media_watch_page_visible") =>
        {
            "youtube_open_relevant_video_then_channel_avatar".into()
        }
        (_, PlannerStepKind::ReplanAfterPerception) => "focused_perception_replan".into(),
        (_, PlannerStepKind::VerifyGoal) => "goal_verification".into(),
        (_, PlannerStepKind::RequestClarification) => "clarification_required".into(),
        (_, PlannerStepKind::Refuse) => "safe_refusal".into(),
        (_, PlannerStepKind::NoOp) => "no_op".into(),
        _ => "semantic_frame_planner".into(),
    }
}

#[allow(dead_code)]
fn fallback_state_for_step(step: &PlannerStep) -> Option<String> {
    match step.kind {
        PlannerStepKind::ReplanAfterPerception => Some("needs_focused_perception".into()),
        PlannerStepKind::ClickResultRegion
            if step.expected_state.as_deref() == Some("media_watch_page_visible") =>
        {
            Some("video_open_pending_before_channel_avatar".into())
        }
        _ => None,
    }
}

fn strategy_for_perception_request(request: &FocusedPerceptionRequest) -> String {
    match request.routing_decision {
        PerceptionRoutingDecision::TargetRegionAnchor => "target_focus_perception".into(),
        PerceptionRoutingDecision::RegionlessTargetVisible => {
            "visible_page_refinement_for_regionless_target".into()
        }
        PerceptionRoutingDecision::VisiblePageUnderGrounded => "visible_page_refinement".into(),
    }
}

fn fallback_state_for_perception_request(request: &FocusedPerceptionRequest) -> Option<String> {
    match request.routing_decision {
        PerceptionRoutingDecision::TargetRegionAnchor => Some("target_region_focus_pending".into()),
        PerceptionRoutingDecision::RegionlessTargetVisible => {
            Some("regionless_target_visible_refinement".into())
        }
        PerceptionRoutingDecision::VisiblePageUnderGrounded => {
            Some("visible_page_refinement_pending".into())
        }
    }
}

#[allow(dead_code)]
fn focused_perception_request_for_step(
    _goal: &GoalSpec,
    frame: &SemanticScreenFrame,
    step: &PlannerStep,
    actionability: &VisibleActionabilityDiagnostic,
    iteration: usize,
) -> Option<FocusedPerceptionRequest> {
    let target_region_anchor = target_region_anchor_for_step(frame, step);

    if step.target_item_id.is_some() || step.target_entity_id.is_some() {
        if let Some(region) = target_region_anchor {
            return Some(FocusedPerceptionRequest {
                request_id: Uuid::new_v4().to_string(),
                iteration,
                reason: step.rationale.clone(),
                mode: PerceptionRequestMode::TargetFocus,
                routing_decision: PerceptionRoutingDecision::TargetRegionAnchor,
                refinement_strategy: VisibleRefinementStrategy::TargetRegion,
                target_item_id: step.target_item_id.clone(),
                target_entity_id: step.target_entity_id.clone(),
                region: Some(region),
                target_region_anchor_present: true,
                verified_surface: None,
            });
        }

        if !actionability.refinement_eligible {
            return None;
        }

        return Some(FocusedPerceptionRequest {
            request_id: Uuid::new_v4().to_string(),
            iteration,
            reason: step.rationale.clone(),
            mode: PerceptionRequestMode::VisiblePageRefinement,
            routing_decision: PerceptionRoutingDecision::RegionlessTargetVisible,
            refinement_strategy: actionability.refinement_strategy.clone(),
            target_item_id: step.target_item_id.clone(),
            target_entity_id: step.target_entity_id.clone(),
            region: visible_cluster_region(frame),
            target_region_anchor_present: false,
            verified_surface: None,
        });
    }

    if !actionability.refinement_eligible {
        return None;
    }

    Some(FocusedPerceptionRequest {
        request_id: Uuid::new_v4().to_string(),
        iteration,
        reason: step.rationale.clone(),
        mode: PerceptionRequestMode::VisiblePageRefinement,
        routing_decision: PerceptionRoutingDecision::VisiblePageUnderGrounded,
        refinement_strategy: actionability.refinement_strategy.clone(),
        target_item_id: None,
        target_entity_id: None,
        region: visible_cluster_region(frame),
        target_region_anchor_present: false,
        verified_surface: None,
    })
}

fn target_region_anchor_for_step(
    frame: &SemanticScreenFrame,
    step: &PlannerStep,
) -> Option<TargetRegion> {
    step.target_item_id
        .as_deref()
        .and_then(|item_id| {
            frame
                .primary_list
                .as_ref()
                .and_then(|list| {
                    list.items
                        .iter()
                        .find(|item| item.item_id == item_id)
                        .and_then(primary_item_region)
                })
                .or_else(|| {
                    frame
                        .visible_result_items
                        .iter()
                        .find(|item| item.item_id == item_id)
                        .and_then(result_item_region)
                })
        })
        .or_else(|| {
            step.target_entity_id.as_deref().and_then(|entity_id| {
                frame
                    .visible_entities
                    .iter()
                    .find(|entity| entity.entity_id == entity_id)
                    .and_then(|entity| entity.region.clone())
            })
        })
}

#[allow(dead_code)]
fn result_item_region(item: &VisibleResultItem) -> Option<TargetRegion> {
    item.click_regions
        .values()
        .next()
        .map(|click_region| click_region.region.clone())
        .or_else(|| {
            item.attributes
                .get("region")
                .or_else(|| item.attributes.get("bounding_region"))
                .or_else(|| item.attributes.get("bounds"))
                .and_then(parse_region)
        })
}

fn primary_item_region(item: &PrimaryListItem) -> Option<TargetRegion> {
    item.click_regions
        .values()
        .next()
        .map(|click_region| click_region.region.clone())
        .or_else(|| {
            item.attributes
                .get("region")
                .or_else(|| item.attributes.get("bounding_region"))
                .or_else(|| item.attributes.get("bounds"))
                .and_then(parse_region)
        })
}

fn visible_cluster_region(frame: &SemanticScreenFrame) -> Option<TargetRegion> {
    let mut regions = Vec::new();

    if let Some(primary_list) = frame.primary_list.as_ref() {
        for item in &primary_list.items {
            if let Some(region) = primary_item_region(item) {
                regions.push(region);
            }
            if regions.len() >= 4 {
                break;
            }
        }
    }
    for item in &frame.visible_result_items {
        if let Some(region) = result_item_region(item) {
            regions.push(region);
        }
        if regions.len() >= 4 {
            break;
        }
    }
    if regions.len() < 4 {
        for entity in &frame.visible_entities {
            if let Some(region) = entity.region.clone() {
                regions.push(region);
            }
            if regions.len() >= 4 {
                break;
            }
        }
    }
    if regions.len() < 4 {
        for control in &frame.actionable_controls {
            if let Some(region) = control.region.clone() {
                regions.push(region);
            }
            if regions.len() >= 4 {
                break;
            }
        }
    }
    if regions.len() < 4 {
        for candidate in &frame.legacy_target_candidates {
            if let Some(region) = candidate_region(candidate) {
                regions.push(region);
            }
            if regions.len() >= 4 {
                break;
            }
        }
    }

    merge_regions(&regions)
}

fn merge_regions(regions: &[TargetRegion]) -> Option<TargetRegion> {
    let first = regions.first()?;
    let mut left = first.x;
    let mut top = first.y;
    let mut right = first.x + first.width;
    let mut bottom = first.y + first.height;

    for region in &regions[1..] {
        left = left.min(region.x);
        top = top.min(region.y);
        right = right.max(region.x + region.width);
        bottom = bottom.max(region.y + region.height);
    }

    Some(TargetRegion {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
        coordinate_space: first.coordinate_space.clone(),
    })
}

fn expanded_region(region: &TargetRegion, padding: f64) -> TargetRegion {
    TargetRegion {
        x: region.x - padding,
        y: region.y - padding,
        width: region.width + (padding * 2.0),
        height: region.height + (padding * 2.0),
        coordinate_space: region.coordinate_space.clone(),
    }
}

fn region_is_valid(region: &TargetRegion) -> bool {
    region.x.is_finite()
        && region.y.is_finite()
        && region.width.is_finite()
        && region.height.is_finite()
        && region.width > 0.0
        && region.height > 0.0
}

fn region_contains_region(outer: &TargetRegion, inner: &TargetRegion) -> bool {
    if !region_is_valid(outer) || !region_is_valid(inner) {
        return false;
    }
    normalize_label(&outer.coordinate_space) == normalize_label(&inner.coordinate_space)
        && inner.x >= outer.x
        && inner.y >= outer.y
        && inner.x + inner.width <= outer.x + outer.width
        && inner.y + inner.height <= outer.y + outer.height
}

fn inferred_browser_surface_bounds(frame: &SemanticScreenFrame) -> Option<TargetRegion> {
    visible_cluster_region(frame).map(|region| expanded_region(&region, 64.0))
}

fn verified_browser_surface_from_frame(
    goal: &GoalSpec,
    frame: &SemanticScreenFrame,
) -> VerifiedInteractionSurface {
    VerifiedInteractionSurface {
        kind: InteractionSurfaceKind::Browser,
        provider_hint: goal
            .constraints
            .provider
            .clone()
            .or_else(|| frame.page_evidence.content_provider_hint.clone()),
        app_hint: frame.page_evidence.browser_app_hint.clone(),
        page_kind_hint: frame.page_evidence.page_kind_hint.clone(),
        bounds: inferred_browser_surface_bounds(frame),
        verified_at_ms: frame.captured_at,
        source_frame_id: frame.frame_id.clone(),
        confidence: Some(frame.page_evidence.confidence),
    }
}

fn browser_surface_evidence_present(goal: &GoalSpec, frame: &SemanticScreenFrame) -> bool {
    if frame.page_evidence.browser_app_hint.is_some() {
        return true;
    }
    if provider_context_compatible(
        goal.constraints.provider.as_deref(),
        frame.page_evidence.content_provider_hint.as_deref(),
    ) && frame.page_evidence.content_provider_hint.is_some()
    {
        return true;
    }
    if structural_list_surface_visible(frame) || structural_detail_surface_visible(frame) {
        return true;
    }
    if browser_page_semantic_kind_for_frame(frame) != BrowserPageSemanticKind::Unknown {
        return true;
    }
    let summary = normalize_label(&frame.scene_summary);
    summary.contains("browser")
        || summary.contains("web_page")
        || summary.contains("result")
        || summary.contains("player")
        || summary.contains("article")
}

fn frame_looks_like_desktop_agent_surface(frame: &SemanticScreenFrame) -> bool {
    [
        frame.page_evidence.browser_app_hint.as_deref(),
        frame.page_evidence.content_provider_hint.as_deref(),
        frame.page_evidence.page_kind_hint.as_deref(),
    ]
    .into_iter()
    .flatten()
    .map(normalize_label)
    .any(|label| {
        label.contains("astra")
            || label.contains("desktop_agent")
            || label.contains("assistant_chat")
            || label.contains("chat_window")
            || label.contains("terminal")
    }) || {
        let summary = normalize_label(&frame.scene_summary);
        summary.contains("astra")
            || summary.contains("assistant_chat")
            || summary.contains("desktop_agent")
            || summary.contains("terminal")
    }
}

fn surface_continuity_diagnostic(
    goal: &GoalSpec,
    surface: &VerifiedInteractionSurface,
    frame: &SemanticScreenFrame,
    iteration: usize,
) -> SurfaceOwnershipDiagnostic {
    let provider_matches = provider_context_compatible(
        surface
            .provider_hint
            .as_deref()
            .or(goal.constraints.provider.as_deref()),
        frame.page_evidence.content_provider_hint.as_deref(),
    );
    let browser_evidence_present = browser_surface_evidence_present(goal, frame);
    let desktop_agent_surface = frame_looks_like_desktop_agent_surface(frame);
    let lost = surface.kind == InteractionSurfaceKind::Browser
        && (desktop_agent_surface
            || (!provider_matches && frame.page_evidence.content_provider_hint.is_some())
            || (!browser_evidence_present
                && verify_goal_against_frame(goal, frame).is_none()
                && !structural_detail_surface_visible(frame)));
    SurfaceOwnershipDiagnostic {
        iteration,
        status: if lost {
            SurfaceOwnershipStatus::Lost
        } else {
            SurfaceOwnershipStatus::Verified
        },
        failure_reason: lost.then_some(FocusedPerceptionFailureReason::SurfaceOwnershipLost),
        surface: Some(surface.clone()),
        observed_frame_id: Some(frame.frame_id.clone()),
        provider_matches: Some(provider_matches),
        browser_evidence_present,
        requested_region: None,
        surface_bounds: surface.bounds.clone(),
        reason: Some(if lost {
            "fresh frame no longer supports the verified browser interaction surface".into()
        } else {
            "fresh frame still supports the verified browser interaction surface".into()
        }),
    }
}

fn surface_verified_diagnostic(
    iteration: usize,
    surface: &VerifiedInteractionSurface,
    status: SurfaceOwnershipStatus,
    reason: impl Into<String>,
) -> SurfaceOwnershipDiagnostic {
    SurfaceOwnershipDiagnostic {
        iteration,
        status,
        failure_reason: None,
        surface: Some(surface.clone()),
        observed_frame_id: Some(surface.source_frame_id.clone()),
        provider_matches: None,
        browser_evidence_present: true,
        requested_region: None,
        surface_bounds: surface.bounds.clone(),
        reason: Some(reason.into()),
    }
}

fn bind_focused_perception_request_to_surface(
    mut request: FocusedPerceptionRequest,
    surface: Option<&VerifiedInteractionSurface>,
    require_verified_surface: bool,
    iteration: usize,
) -> Result<FocusedPerceptionRequest, SurfaceOwnershipDiagnostic> {
    let Some(surface) = surface else {
        if require_verified_surface {
            return Err(SurfaceOwnershipDiagnostic {
                iteration,
                status: SurfaceOwnershipStatus::Refused,
                failure_reason: Some(FocusedPerceptionFailureReason::NoVerifiedSurface),
                surface: None,
                observed_frame_id: None,
                provider_matches: None,
                browser_evidence_present: false,
                requested_region: request.region.clone(),
                surface_bounds: None,
                reason: Some(
                    "browser-owned focused perception requires a verified interaction surface"
                        .into(),
                ),
            });
        }
        return Ok(request);
    };

    if surface.kind != InteractionSurfaceKind::Browser {
        request.verified_surface = Some(surface.clone());
        return Ok(request);
    }

    let Some(bounds) = surface.bounds.as_ref() else {
        return Err(SurfaceOwnershipDiagnostic {
            iteration,
            status: SurfaceOwnershipStatus::Refused,
            failure_reason: Some(FocusedPerceptionFailureReason::SurfaceBoundsUnavailable),
            surface: Some(surface.clone()),
            observed_frame_id: None,
            provider_matches: None,
            browser_evidence_present: true,
            requested_region: request.region.clone(),
            surface_bounds: None,
            reason: Some(
                "browser-owned focused perception has no safe bounded browser region".into(),
            ),
        });
    };

    match request.region.as_ref() {
        Some(region) if !region_contains_region(bounds, region) => {
            Err(SurfaceOwnershipDiagnostic {
                iteration,
                status: SurfaceOwnershipStatus::Refused,
                failure_reason: Some(FocusedPerceptionFailureReason::RequestedRegionOutsideSurface),
                surface: Some(surface.clone()),
                observed_frame_id: None,
                provider_matches: None,
                browser_evidence_present: true,
                requested_region: request.region.clone(),
                surface_bounds: Some(bounds.clone()),
                reason: Some(
                    "focused perception region lies outside the verified browser surface".into(),
                ),
            })
        }
        Some(_) => {
            request.verified_surface = Some(surface.clone());
            Ok(request)
        }
        None => {
            request.region = Some(bounds.clone());
            request.verified_surface = Some(surface.clone());
            Ok(request)
        }
    }
}

fn deterministic_visible_ordinal_fast_path(
    input: &PlannerContractInput,
    step: &PlannerStep,
) -> bool {
    if !matches!(
        input.goal.goal_type,
        GoalType::OpenListItem | GoalType::OpenMediaResult
    ) || step.kind != PlannerStepKind::ClickResultRegion
        || !step_is_executable(step, MIN_PLANNER_CLICK_CONFIDENCE)
    {
        return false;
    }

    let Some(target_item_id) = step.target_item_id.as_deref() else {
        return false;
    };
    find_primary_list_item_match(&input.goal, &input.current_frame)
        .is_some_and(|item| item.item_id == target_item_id)
        || find_ranked_result_match(&input.goal, &input.current_frame)
            .is_some_and(|item| item.item_id == target_item_id)
}

fn candidate_signature(candidate: &UITargetCandidate) -> String {
    let (center_x, center_y) = candidate.center_point().unwrap_or((0.0, 0.0));
    format!(
        "{}|{}|{}|{}|{:.0}|{:.0}",
        candidate.candidate_id,
        candidate.role.as_str(),
        candidate.result_kind.as_deref().unwrap_or("unknown"),
        candidate.rank.unwrap_or(0),
        center_x,
        center_y
    )
}

fn step_target_signature(step: &PlannerStep) -> Option<String> {
    step.executable_candidate
        .as_ref()
        .map(candidate_signature)
        .or_else(|| {
            step.target_item_id.as_ref().map(|item_id| {
                format!(
                    "item:{item_id}:{}",
                    step.click_region_key.as_deref().unwrap_or("unknown")
                )
            })
        })
        .or_else(|| {
            step.target_entity_id
                .as_ref()
                .map(|entity_id| format!("entity:{entity_id}"))
        })
}

fn step_geometry_signature(step: &PlannerStep) -> Option<String> {
    let candidate = step.executable_candidate.as_ref()?;
    if let Some(region) = candidate.region.as_ref() {
        return Some(format!(
            "{:.2}:{:.2}:{:.2}:{:.2}:{}",
            region.x, region.y, region.width, region.height, region.coordinate_space
        ));
    }
    candidate
        .center_point()
        .map(|(x, y)| format!("{:.2}:{:.2}", x, y))
}

fn frame_progress_signature(frame: &SemanticScreenFrame) -> String {
    let results = frame
        .visible_result_items
        .iter()
        .take(5)
        .map(|item| {
            format!(
                "{}:{:?}:{}:{}",
                item.item_id,
                item.kind,
                item.title.as_deref().unwrap_or(""),
                item.rank_overall.unwrap_or(0)
            )
        })
        .collect::<Vec<_>>()
        .join("|");
    let entities = frame
        .visible_entities
        .iter()
        .take(5)
        .map(|entity| {
            format!(
                "{}:{:?}:{}",
                entity.entity_id,
                entity.kind,
                entity.name.as_deref().unwrap_or("")
            )
        })
        .collect::<Vec<_>>()
        .join("|");
    let controls = frame
        .actionable_controls
        .iter()
        .take(5)
        .map(|control| {
            format!(
                "{}:{}:{}",
                control.control_id,
                control.kind,
                control.label.as_deref().unwrap_or("")
            )
        })
        .collect::<Vec<_>>()
        .join("|");
    let primary = frame
        .primary_list
        .as_ref()
        .map(|list| {
            list.items
                .iter()
                .take(5)
                .map(|item| {
                    format!(
                        "{}:{}:{}",
                        item.item_id,
                        item.item_kind.as_deref().unwrap_or("unknown"),
                        item.rank
                    )
                })
                .collect::<Vec<_>>()
                .join("|")
        })
        .unwrap_or_default();
    let page_state = frame
        .page_state
        .as_ref()
        .map(|state| {
            format!(
                "{}:{}:{:?}:{:?}",
                state.kind, state.dominant_content, state.list_visible, state.detail_visible
            )
        })
        .unwrap_or_default();
    format!(
        "{:?}|{:?}|{}|{}|{}|{}|{}",
        frame.page_evidence.content_provider_hint,
        frame.page_evidence.page_kind_hint,
        results,
        entities,
        controls,
        primary,
        page_state
    )
}

fn browser_recovery_reason_for_execution(execution: &PlannerStepExecutionRecord) -> Option<String> {
    let geometry = execution.geometry.as_ref()?;
    (!geometry.validation_passed).then(|| {
        geometry
            .reason
            .clone()
            .unwrap_or_else(|| "click_target_geometry_untrusted".into())
    })
}

fn evaluate_post_click_frame(
    goal: &GoalSpec,
    pending: &PendingPostClickVerification,
    frame: &SemanticScreenFrame,
) -> PostClickVerificationOutcome {
    let structural_transition_achieved = matches!(
        goal.goal_type,
        GoalType::OpenListItem | GoalType::OpenMediaResult
    ) && pending.pre_click_list_surface_visible
        && matches!(
            structural_open_list_item_state(goal, frame),
            OpenListItemVerificationState::Achieved(_)
        );
    let goal_achieved =
        verify_goal_against_frame(goal, frame).is_some() || structural_transition_achieved;
    let frame_unchanged = frame_progress_signature(frame) == pending.pre_click_frame_signature;
    let browser_surface_suspect = !goal_achieved
        && goal.constraints.provider.is_some()
        && verify_browser_handoff_page(goal, frame).is_err();

    PostClickVerificationOutcome {
        goal_achieved,
        progress_observed: goal_achieved || !frame_unchanged,
        frame_unchanged,
        browser_surface_suspect,
    }
}

fn is_watch_page(frame: &SemanticScreenFrame) -> bool {
    browser_page_semantic_kind_for_frame(frame) == BrowserPageSemanticKind::WatchPage
}

#[allow(dead_code)]
fn is_search_results_page(frame: &SemanticScreenFrame) -> bool {
    browser_page_semantic_kind_for_frame(frame) == BrowserPageSemanticKind::SearchResults
}

fn structural_list_surface_visible(frame: &SemanticScreenFrame) -> bool {
    if let Some(state) = frame.page_state.as_ref() {
        return state.list_visible == Some(true)
            || state.kind == "list"
            || state.dominant_content == "result_list";
    }

    frame
        .primary_list
        .as_ref()
        .is_some_and(|list| !list.items.is_empty())
        || frame.page_evidence.result_list_visible == Some(true)
        || !frame.visible_result_items.is_empty()
}

fn structural_detail_surface_visible(frame: &SemanticScreenFrame) -> bool {
    frame.page_state.as_ref().is_some_and(|state| {
        state.detail_visible == Some(true)
            || matches!(state.kind.as_str(), "detail" | "player")
            || matches!(
                state.dominant_content.as_str(),
                "detail_view" | "video_player" | "article" | "product_detail"
            )
    })
}

fn structural_detail_surface_dominant(frame: &SemanticScreenFrame) -> bool {
    frame.page_state.as_ref().is_some_and(|state| {
        matches!(state.kind.as_str(), "detail" | "player")
            || matches!(
                state.dominant_content.as_str(),
                "detail_view" | "video_player" | "article" | "product_detail"
            )
    })
}

fn structural_open_list_item_state(
    goal: &GoalSpec,
    frame: &SemanticScreenFrame,
) -> OpenListItemVerificationState {
    if !provider_context_compatible(
        goal.constraints.provider.as_deref(),
        frame.page_evidence.content_provider_hint.as_deref(),
    ) {
        return OpenListItemVerificationState::NotAchieved;
    }

    if frame.page_state.is_some() {
        let list_visible = structural_list_surface_visible(frame);
        let detail_visible = structural_detail_surface_visible(frame);
        let detail_dominant = structural_detail_surface_dominant(frame);
        if detail_dominant && !list_visible {
            return OpenListItemVerificationState::Achieved(
                "list_to_detail_structural_transition_visible",
            );
        }
        if detail_dominant && detail_visible && list_visible {
            return OpenListItemVerificationState::Ambiguous(
                "detail_surface_visible_but_list_still_visible",
            );
        }
        return OpenListItemVerificationState::NotAchieved;
    }

    if is_watch_page(frame) {
        return OpenListItemVerificationState::Achieved("media_watch_page_visible");
    }

    let requested_kind = requested_item_kind_label(goal);
    let generic_web_target = requested_kind.as_deref().map_or(true, |kind| {
        matches!(kind, "site" | "article" | "product" | "generic" | "result")
    });
    if generic_web_target
        && browser_page_semantic_kind_for_frame(frame) == BrowserPageSemanticKind::WebPage
        && frame.page_evidence.result_list_visible != Some(true)
        && frame.visible_result_items.is_empty()
        && frame
            .primary_list
            .as_ref()
            .map_or(true, |list| list.items.is_empty())
    {
        return OpenListItemVerificationState::Achieved("generic_web_detail_page_visible");
    }

    OpenListItemVerificationState::NotAchieved
}

fn is_channel_page(frame: &SemanticScreenFrame) -> bool {
    frame
        .page_evidence
        .page_kind_hint
        .as_deref()
        .is_some_and(|kind| {
            matches!(
                normalize_label(kind).as_str(),
                "channel_page" | "profile_page" | "youtube_channel"
            )
        })
}

fn browser_page_semantic_kind_for_frame(frame: &SemanticScreenFrame) -> BrowserPageSemanticKind {
    browser_page_semantic_kind(frame.page_evidence.page_kind_hint.as_deref())
}

pub(crate) fn browser_page_semantic_kind(value: Option<&str>) -> BrowserPageSemanticKind {
    match value.map(normalize_label).as_deref() {
        Some(
            "search_results"
            | "search_results_page"
            | "search_result_page"
            | "results_page"
            | "result_page"
            | "youtube_results"
            | "results"
            | "search"
            | "search_page"
            | "result_list"
            | "results_list"
            | "ranked_list"
            | "result"
            | "serp"
            | "listing"
            | "catalog"
            | "list",
        ) => BrowserPageSemanticKind::SearchResults,
        Some(
            "watch_page" | "video_page" | "youtube_watch" | "detail_page" | "watch" | "video"
            | "video_watch" | "player" | "media_page",
        ) => BrowserPageSemanticKind::WatchPage,
        Some("web_page" | "page") => BrowserPageSemanticKind::WebPage,
        _ => BrowserPageSemanticKind::Unknown,
    }
}

fn parse_optional_confidence(value: Option<&Value>) -> Option<f32> {
    value
        .and_then(Value::as_f64)
        .map(|value| value.clamp(0.0, 1.0) as f32)
}

fn page_evidence_from_value(value: Option<&Value>) -> Option<PageSemanticEvidence> {
    let object = value?.as_object()?;
    let provider = object
        .get("content_provider_hint")
        .or_else(|| object.get("provider_hint"))
        .or_else(|| object.get("provider"))
        .and_then(Value::as_str)
        .and_then(normalize_content_provider_hint);
    Some(PageSemanticEvidence {
        browser_app_hint: object
            .get("browser_app_hint")
            .or_else(|| object.get("browser_app"))
            .or_else(|| object.get("app_hint"))
            .and_then(Value::as_str)
            .and_then(normalize_browser_app_hint),
        content_provider_hint: provider,
        page_kind_hint: object
            .get("page_kind_hint")
            .or_else(|| object.get("page_kind"))
            .and_then(Value::as_str)
            .map(normalize_label),
        query_hint: object
            .get("query_hint")
            .or_else(|| object.get("query"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        result_list_visible: object
            .get("result_list_visible")
            .or_else(|| object.get("results_visible"))
            .and_then(Value::as_bool),
        raw_confidence: parse_optional_confidence(object.get("confidence")),
        confidence: parse_optional_confidence(object.get("confidence")).unwrap_or(0.0),
        evidence_sources: vec![PageEvidenceSource::StructuredVision],
        capture_backend: object
            .get("capture_backend")
            .or_else(|| object.get("observation_backend"))
            .and_then(Value::as_str)
            .map(normalize_label),
        observation_source: object
            .get("observation_source")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        uncertainty: string_array(object.get("uncertainty")),
    })
}

fn parse_visible_entities(value: Option<&Value>) -> Vec<VisibleEntity> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(index, value)| {
            let object = value.as_object()?;
            Some(VisibleEntity {
                entity_id: object
                    .get("entity_id")
                    .or_else(|| object.get("id"))
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| format!("entity_{}", index + 1)),
                kind: entity_kind_from_value(
                    object
                        .get("kind")
                        .or_else(|| object.get("entity_kind"))
                        .and_then(Value::as_str),
                ),
                name: object
                    .get("name")
                    .or_else(|| object.get("label"))
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                raw_confidence: parse_optional_confidence(object.get("confidence")),
                confidence: parse_optional_confidence(object.get("confidence")).unwrap_or(0.0),
                region: object.get("region").and_then(parse_region),
                attributes: object.get("attributes").cloned().unwrap_or(Value::Null),
            })
        })
        .collect()
}

fn parse_visible_result_items(value: Option<&Value>) -> Vec<VisibleResultItem> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(index, value)| parse_visible_result_item(value, index))
        .collect()
}

fn parse_primary_list(value: Option<&Value>) -> (Option<PrimaryList>, Vec<FrameUncertainty>) {
    let mut uncertainty = Vec::new();
    let Some(object) = value.and_then(Value::as_object) else {
        return (None, uncertainty);
    };

    let raw_container_kind = object
        .get("container_kind")
        .or_else(|| object.get("kind"))
        .and_then(Value::as_str)
        .unwrap_or("result_list");
    let container_kind = normalize_structural_list_kind(raw_container_kind);
    let raw_confidence = parse_optional_confidence(object.get("confidence"));
    let items_value = object.get("items").and_then(Value::as_array);
    let mut malformed_items = 0usize;
    let items = items_value
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(index, value)| {
            parse_primary_list_item(value, index).or_else(|| {
                malformed_items += 1;
                None
            })
        })
        .collect::<Vec<_>>();

    if malformed_items > 0 {
        uncertainty.push(FrameUncertainty {
            code: "primary_list_malformed_items_discarded".into(),
            message: format!(
                "{malformed_items} malformed primary_list item(s) were discarded; existing result-item fallback remains available"
            ),
            severity: "info".into(),
        });
    }
    if items.is_empty() {
        uncertainty.push(FrameUncertainty {
            code: "primary_list_unusable_fallback_to_visible_result_items".into(),
            message: "primary_list was absent or had no structurally valid items; falling back to visible_result_items and legacy page-kind evidence".into(),
            severity: "info".into(),
        });
        return (None, uncertainty);
    }

    let item_count = object
        .get("item_count")
        .and_then(Value::as_u64)
        .map(|value| value as u32)
        .unwrap_or(items.len() as u32)
        .max(items.len() as u32);

    (
        Some(PrimaryList {
            cluster_id: object
                .get("cluster_id")
                .or_else(|| object.get("id"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| "primary_list".into()),
            container_kind,
            item_count,
            items,
            raw_confidence,
            confidence: raw_confidence.unwrap_or(0.0),
        }),
        uncertainty,
    )
}

fn parse_primary_list_item(value: &Value, index: usize) -> Option<PrimaryListItem> {
    let object = value.as_object()?;
    let rank = object
        .get("rank")
        .or_else(|| object.get("rank_overall"))
        .or_else(|| object.get("position"))
        .and_then(Value::as_u64)
        .filter(|rank| *rank > 0)
        .map(|rank| rank as u32)
        .unwrap_or((index + 1) as u32);
    let raw_confidence = parse_optional_confidence(object.get("confidence"));
    let click_regions_value = object.get("click_regions");
    let click_regions = parse_click_regions(click_regions_value, raw_confidence);
    if click_regions_value.is_some() && click_regions.is_empty() {
        return None;
    }

    Some(PrimaryListItem {
        item_id: object
            .get("item_id")
            .or_else(|| object.get("id"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("primary_item_{}", index + 1)),
        rank,
        title: object
            .get("title")
            .or_else(|| object.get("name"))
            .or_else(|| object.get("label"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        item_kind: object
            .get("item_kind")
            .or_else(|| object.get("kind"))
            .or_else(|| object.get("result_kind"))
            .and_then(Value::as_str)
            .map(normalize_item_kind_label),
        is_sponsored: object.get("is_sponsored").and_then(Value::as_bool),
        raw_confidence,
        confidence: raw_confidence.unwrap_or(0.0),
        click_regions,
        attributes: object.get("attributes").cloned().unwrap_or(Value::Null),
    })
}

fn parse_page_state(value: Option<&Value>) -> Option<PageState> {
    let object = value?.as_object()?;
    Some(PageState {
        kind: normalize_page_state_kind(
            object
                .get("kind")
                .or_else(|| object.get("page_kind"))
                .and_then(Value::as_str)
                .unwrap_or("unknown"),
        ),
        dominant_content: normalize_dominant_content_kind(
            object
                .get("dominant_content")
                .or_else(|| object.get("dominant_surface"))
                .and_then(Value::as_str)
                .unwrap_or("unknown"),
        ),
        list_visible: object.get("list_visible").and_then(Value::as_bool),
        detail_visible: object.get("detail_visible").and_then(Value::as_bool),
        confidence: parse_optional_confidence(object.get("confidence")),
    })
}

fn parse_visible_result_item(value: &Value, index: usize) -> Option<VisibleResultItem> {
    let object = value.as_object()?;
    let raw_confidence = parse_optional_confidence(object.get("confidence"));
    let confidence = raw_confidence.unwrap_or(0.0);
    Some(VisibleResultItem {
        item_id: object
            .get("item_id")
            .or_else(|| object.get("id"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("result_{}", index + 1)),
        kind: result_kind_from_value(
            object
                .get("kind")
                .or_else(|| object.get("result_kind"))
                .and_then(Value::as_str),
        ),
        title: object
            .get("title")
            .or_else(|| object.get("name"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        channel_name: object
            .get("channel_name")
            .or_else(|| object.get("seller_name"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        provider: object
            .get("provider")
            .or_else(|| object.get("content_provider_hint"))
            .and_then(Value::as_str)
            .and_then(normalize_content_provider_hint),
        rank_overall: object
            .get("rank_overall")
            .or_else(|| object.get("rank"))
            .and_then(Value::as_u64)
            .map(|value| value as u32),
        rank_within_kind: object
            .get("rank_within_kind")
            .and_then(Value::as_u64)
            .map(|value| value as u32),
        click_regions: parse_click_regions(object.get("click_regions"), raw_confidence),
        raw_confidence,
        confidence,
        rationale: object
            .get("rationale")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        attributes: object.get("attributes").cloned().unwrap_or(Value::Null),
    })
}

fn parse_click_regions(
    value: Option<&Value>,
    default_confidence: Option<f32>,
) -> HashMap<String, ClickRegion> {
    let mut regions = HashMap::new();
    let Some(object) = value.and_then(Value::as_object) else {
        return regions;
    };
    for (key, value) in object {
        let region_value = value.get("region").unwrap_or(value);
        if let Some(region) = parse_region(region_value) {
            let raw_confidence = parse_optional_confidence(value.get("confidence"));
            regions.insert(
                key.clone(),
                ClickRegion {
                    region,
                    raw_confidence,
                    confidence: raw_confidence.or(default_confidence).unwrap_or(0.0),
                },
            );
        }
    }
    regions
}

fn parse_actionable_controls(value: Option<&Value>) -> Vec<ActionableControl> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(index, value)| {
            let object = value.as_object()?;
            Some(ActionableControl {
                control_id: object
                    .get("control_id")
                    .or_else(|| object.get("id"))
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| format!("control_{}", index + 1)),
                kind: object
                    .get("kind")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string(),
                label: object
                    .get("label")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                region: object
                    .get("region")
                    .or_else(|| object.get("bounds"))
                    .or_else(|| object.get("bounding_region"))
                    .or_else(|| {
                        object.get("attributes").and_then(|attributes| {
                            attributes
                                .get("region")
                                .or_else(|| attributes.get("bounds"))
                                .or_else(|| attributes.get("bounding_region"))
                        })
                    })
                    .and_then(parse_region),
                raw_confidence: parse_optional_confidence(object.get("confidence")),
                confidence: parse_optional_confidence(object.get("confidence")).unwrap_or(0.0),
                attributes: object.get("attributes").cloned().unwrap_or(Value::Null),
            })
        })
        .collect()
}

fn parse_uncertainty(value: Option<&Value>) -> Vec<FrameUncertainty> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(index, value)| {
            if let Some(text) = value.as_str() {
                return FrameUncertainty {
                    code: format!("uncertainty_{}", index + 1),
                    message: text.to_string(),
                    severity: "info".into(),
                };
            }
            FrameUncertainty {
                code: value
                    .get("code")
                    .and_then(Value::as_str)
                    .unwrap_or("uncertainty")
                    .to_string(),
                message: value
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("unspecified frame uncertainty")
                    .to_string(),
                severity: value
                    .get("severity")
                    .and_then(Value::as_str)
                    .unwrap_or("info")
                    .to_string(),
            }
        })
        .collect()
}

fn result_items_from_candidates(candidates: &[UITargetCandidate]) -> Vec<VisibleResultItem> {
    candidates
        .iter()
        .filter(|candidate| candidate.role == UITargetRole::RankedResult)
        .enumerate()
        .filter_map(|(index, candidate)| {
            let region = candidate.region.clone().or_else(|| {
                let (x, y) = candidate.center_point()?;
                Some(TargetRegion {
                    x: x - 4.0,
                    y: y - 4.0,
                    width: 8.0,
                    height: 8.0,
                    coordinate_space: "screen".into(),
                })
            })?;
            let mut click_regions = HashMap::new();
            click_regions.insert(
                "primary".into(),
                ClickRegion {
                    region,
                    raw_confidence: Some(candidate.confidence),
                    confidence: candidate.confidence,
                },
            );
            let kind = candidate
                .result_kind
                .as_deref()
                .map(|value| result_kind_from_value(Some(value)))
                .unwrap_or(VisibleResultKind::Generic);
            Some(VisibleResultItem {
                item_id: candidate.candidate_id.clone(),
                kind,
                title: candidate.label.clone(),
                channel_name: None,
                provider: candidate
                    .content_provider_hint
                    .clone()
                    .or(candidate.provider_hint.clone()),
                rank_overall: candidate.rank.or(Some((index + 1) as u32)),
                rank_within_kind: candidate.rank,
                click_regions,
                raw_confidence: Some(candidate.confidence),
                confidence: candidate.confidence,
                rationale: Some(candidate.rationale.clone()),
                attributes: Value::Null,
            })
        })
        .collect()
}

fn parse_region(value: &Value) -> Option<TargetRegion> {
    Some(TargetRegion {
        x: value
            .get("x")
            .or_else(|| value.get("left"))
            .and_then(Value::as_f64)?,
        y: value
            .get("y")
            .or_else(|| value.get("top"))
            .and_then(Value::as_f64)?,
        width: value.get("width").and_then(Value::as_f64).or_else(|| {
            let left = value
                .get("x")
                .or_else(|| value.get("left"))
                .and_then(Value::as_f64)?;
            let right = value.get("right").and_then(Value::as_f64)?;
            Some(right - left)
        })?,
        height: value.get("height").and_then(Value::as_f64).or_else(|| {
            let top = value
                .get("y")
                .or_else(|| value.get("top"))
                .and_then(Value::as_f64)?;
            let bottom = value.get("bottom").and_then(Value::as_f64)?;
            Some(bottom - top)
        })?,
        coordinate_space: value
            .get("coordinate_space")
            .and_then(Value::as_str)
            .unwrap_or("screen")
            .to_string(),
    })
    .filter(|region| {
        region.x.is_finite()
            && region.y.is_finite()
            && region.width.is_finite()
            && region.height.is_finite()
            && region.x >= 0.0
            && region.y >= 0.0
            && region.width > 0.0
            && region.height > 0.0
    })
}

fn result_kind_from_value(value: Option<&str>) -> VisibleResultKind {
    match value.map(normalize_label).as_deref() {
        Some("video") | Some("video_result") | Some("media") => VisibleResultKind::Video,
        Some("channel") | Some("channel_result") => VisibleResultKind::Channel,
        Some("playlist") | Some("playlist_result") => VisibleResultKind::Playlist,
        Some("mix") | Some("youtube_mix") => VisibleResultKind::Mix,
        Some("hotel") | Some("hotel_card") => VisibleResultKind::Hotel,
        Some("product") => VisibleResultKind::Product,
        Some("repository") | Some("repo") => VisibleResultKind::Repository,
        Some("generic") | Some("result") | Some("search_result") => VisibleResultKind::Generic,
        _ => VisibleResultKind::Unknown,
    }
}

fn normalize_item_kind_label(value: &str) -> String {
    match normalize_label(value).as_str() {
        "video_result" | "media" => "video".into(),
        "article_result" | "article_link" => "article".into(),
        "product_result" | "product_card" => "product".into(),
        "site_result" | "website" | "web_page" | "link" | "url" => "site".into(),
        "channel_result" => "channel".into(),
        "search_result" | "ranked_result" | "result" => "generic".into(),
        "youtube" | "google" | "amazon" | "github" => "generic".into(),
        other if other.is_empty() => "generic".into(),
        other => other.to_string(),
    }
}

fn normalize_structural_list_kind(value: &str) -> String {
    match normalize_label(value).as_str() {
        "result_list"
        | "results_list"
        | "ranked_list"
        | "list"
        | "search_results"
        | "search_results_page"
        | "search_result_page"
        | "results_page"
        | "serp"
        | "listing"
        | "catalog" => "result_list".into(),
        "youtube" | "google" | "amazon" | "github" => "result_list".into(),
        _ => "result_list".into(),
    }
}

fn normalize_page_state_kind(value: &str) -> String {
    match normalize_label(value).as_str() {
        "list"
        | "results"
        | "result_list"
        | "results_list"
        | "ranked_list"
        | "search_results"
        | "search_results_page"
        | "search_result_page"
        | "results_page"
        | "serp"
        | "listing"
        | "catalog"
        | "search" => "list".into(),
        "detail" | "details" | "detail_page" | "web_page" | "article_page" | "product_page" => {
            "detail".into()
        }
        "player" | "watch" | "watch_page" | "video" | "video_page" | "media_page" => {
            "player".into()
        }
        "form" | "input_form" => "form".into(),
        "mixed" | "split" => "mixed".into(),
        "youtube" | "google" | "amazon" | "github" => "unknown".into(),
        _ => "unknown".into(),
    }
}

fn normalize_dominant_content_kind(value: &str) -> String {
    match normalize_label(value).as_str() {
        "result_list"
        | "results"
        | "results_list"
        | "ranked_list"
        | "search_results"
        | "search_results_page"
        | "search_result_page"
        | "results_page"
        | "serp"
        | "listing"
        | "catalog"
        | "search"
        | "list" => "result_list".into(),
        "detail" | "details" | "detail_view" | "web_page" => "detail_view".into(),
        "video_player" | "player" | "watch" | "watch_page" | "video" => "video_player".into(),
        "article" | "article_page" => "article".into(),
        "product" | "product_detail" | "product_page" => "product_detail".into(),
        "generic" => "generic".into(),
        "youtube" | "google" | "amazon" | "github" => "unknown".into(),
        _ => "unknown".into(),
    }
}

fn entity_kind_from_value(value: Option<&str>) -> VisibleEntityKind {
    match value.map(normalize_label).as_deref() {
        Some("channel_header") => VisibleEntityKind::ChannelHeader,
        Some("channel_result") | Some("channel") => VisibleEntityKind::ChannelResult,
        Some("video_result") | Some("video") => VisibleEntityKind::VideoResult,
        Some("mix_result") | Some("mix") => VisibleEntityKind::MixResult,
        Some("playlist_result") | Some("playlist") => VisibleEntityKind::PlaylistResult,
        Some("hotel_card") | Some("hotel") => VisibleEntityKind::HotelCard,
        Some("price_block") | Some("price") => VisibleEntityKind::PriceBlock,
        Some("star_rating") | Some("stars") => VisibleEntityKind::StarRating,
        Some("sort_control") | Some("sort") => VisibleEntityKind::SortControl,
        Some("filter_chip") | Some("filter") => VisibleEntityKind::FilterChip,
        Some("avatar") => VisibleEntityKind::Avatar,
        Some("title_link") | Some("title") => VisibleEntityKind::TitleLink,
        Some("thumbnail") => VisibleEntityKind::Thumbnail,
        _ => VisibleEntityKind::Unknown,
    }
}

fn result_kind_label(kind: &VisibleResultKind) -> &'static str {
    match kind {
        VisibleResultKind::Video => "video",
        VisibleResultKind::Channel => "channel",
        VisibleResultKind::Playlist => "playlist",
        VisibleResultKind::Mix => "mix",
        VisibleResultKind::Hotel => "hotel",
        VisibleResultKind::Product => "product",
        VisibleResultKind::Repository => "repository",
        VisibleResultKind::Generic => "result",
        VisibleResultKind::Unknown => "unknown",
    }
}

fn entity_kind_result_label(kind: &VisibleEntityKind) -> &'static str {
    match kind {
        VisibleEntityKind::ChannelHeader | VisibleEntityKind::ChannelResult => "channel",
        VisibleEntityKind::VideoResult => "video",
        VisibleEntityKind::MixResult => "mix",
        VisibleEntityKind::PlaylistResult => "playlist",
        VisibleEntityKind::HotelCard => "hotel",
        VisibleEntityKind::Avatar => "channel",
        VisibleEntityKind::TitleLink => "link",
        VisibleEntityKind::Thumbnail => "thumbnail",
        VisibleEntityKind::PriceBlock => "price",
        VisibleEntityKind::StarRating => "rating",
        VisibleEntityKind::SortControl => "sort",
        VisibleEntityKind::FilterChip => "filter",
        VisibleEntityKind::Unknown => "entity",
    }
}

fn summary_from_evidence(
    page_evidence: &PageSemanticEvidence,
    candidates: &[UITargetCandidate],
) -> String {
    let provider = page_evidence
        .content_provider_hint
        .as_deref()
        .unwrap_or("unknown provider");
    if page_evidence.result_list_visible == Some(true) {
        format!(
            "{provider} result list visible with {} legacy target candidates",
            candidates.len()
        )
    } else {
        format!(
            "{provider} screen frame with {} legacy target candidates",
            candidates.len()
        )
    }
}

fn string_array(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(ToOwned::to_owned)
        .collect()
}

fn labels_match(left: &str, right: &str) -> bool {
    normalize_label(left) == normalize_label(right)
}

fn loose_text_match(expected: &str, observed: &str) -> bool {
    let expected = normalize_label(expected);
    let observed = normalize_label(observed);
    observed.contains(&expected) || expected.contains(&observed)
}

fn normalize_label(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_")
        .replace(' ', "_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desktop_agent_types::BrowserVisualHandoffRecord;
    use serde_json::json;

    fn open_channel_goal() -> GoalSpec {
        GoalSpec {
            goal_id: "goal_channel".into(),
            goal_type: GoalType::OpenChannel,
            constraints: GoalConstraints {
                provider: Some("youtube".into()),
                item_kind: None,
                result_kind: None,
                rank_within_kind: None,
                rank_overall: None,
                entity_name: Some("Shiva".into()),
                attributes: Value::Null,
            },
            success_condition: "channel_page_visible".into(),
            utterance: "vai sul canale di Shiva".into(),
            confidence: 0.86,
        }
    }

    fn youtube_search_frame(frame_id: &str, captured_at: u64, title: &str) -> SemanticScreenFrame {
        semantic_frame_from_vision_value(
            &json!({
                "frame_id": frame_id,
                "page_evidence": {
                    "content_provider_hint": "youtube",
                    "page_kind_hint": "search_results",
                    "result_list_visible": true,
                    "confidence": 0.94
                },
                "scene_summary": "YouTube results are visible",
                "visible_result_items": [{
                    "item_id": "v1",
                    "kind": "video",
                    "title": title,
                    "rank_within_kind": 1,
                    "click_regions": {
                        "title": {
                            "x": 20,
                            "y": 40,
                            "width": 300,
                            "height": 80,
                            "coordinate_space": "screen"
                        }
                    },
                    "confidence": 0.95
                }]
            }),
            captured_at,
            None,
            None,
            Vec::new(),
        )
        .expect("search frame")
    }

    fn youtube_watch_frame(frame_id: &str, captured_at: u64) -> SemanticScreenFrame {
        semantic_frame_from_vision_value(
            &json!({
                "frame_id": frame_id,
                "page_evidence": {
                    "content_provider_hint": "youtube",
                    "page_kind_hint": "watch_page",
                    "confidence": 0.93
                },
                "scene_summary": "YouTube watch page is visible"
            }),
            captured_at,
            None,
            None,
            Vec::new(),
        )
        .expect("watch frame")
    }

    fn youtube_primary_list_frame(
        frame_id: &str,
        captured_at: u64,
        item_count: u32,
    ) -> SemanticScreenFrame {
        let items = (1..=item_count)
            .map(|rank| {
                json!({
                    "item_id": format!("video_{rank}"),
                    "rank": rank,
                    "title": format!("Video {rank}"),
                    "item_kind": "video",
                    "click_regions": {
                        "primary": {
                            "x": 100,
                            "y": 100 + ((rank - 1) as i64 * 90),
                            "width": 320,
                            "height": 60,
                            "coordinate_space": "screen",
                            "confidence": 0.95
                        }
                    },
                    "confidence": 0.95
                })
            })
            .collect::<Vec<_>>();
        semantic_frame_from_vision_value(
            &json!({
                "frame_id": frame_id,
                "page_evidence": {
                    "browser_app_hint": "chrome",
                    "content_provider_hint": "youtube",
                    "page_kind_hint": "results",
                    "result_list_visible": true,
                    "confidence": 0.94
                },
                "scene_summary": "A browser page shows a ranked list of videos.",
                "page_state": {
                    "kind": "list",
                    "dominant_content": "result_list",
                    "list_visible": true,
                    "detail_visible": false
                },
                "primary_list": {
                    "cluster_id": "main",
                    "container_kind": "result_list",
                    "item_count": item_count,
                    "items": items,
                    "confidence": 0.94
                }
            }),
            captured_at,
            None,
            None,
            Vec::new(),
        )
        .expect("primary list frame")
    }

    fn ambiguous_browser_detail_frame(frame_id: &str, captured_at: u64) -> SemanticScreenFrame {
        semantic_frame_from_vision_value(
            &json!({
                "frame_id": frame_id,
                "page_evidence": {
                    "browser_app_hint": "chrome",
                    "content_provider_hint": "youtube",
                    "page_kind_hint": "detail_page",
                    "result_list_visible": true,
                    "confidence": 0.86
                },
                "scene_summary": "The browser still shows a list surface and a detail/player panel.",
                "page_state": {
                    "kind": "mixed",
                    "dominant_content": "detail_view",
                    "list_visible": true,
                    "detail_visible": true
                },
                "primary_list": {
                    "cluster_id": "main",
                    "container_kind": "result_list",
                    "item_count": 1,
                    "items": [{
                        "item_id": "video_1",
                        "rank": 1,
                        "title": "Video 1",
                        "item_kind": "video",
                        "attributes": {
                            "region": {"x": 100, "y": 100, "width": 320, "height": 60, "coordinate_space": "screen"}
                        },
                        "confidence": 0.9
                    }],
                    "confidence": 0.88
                }
            }),
            captured_at,
            None,
            None,
            Vec::new(),
        )
        .expect("ambiguous browser frame")
    }

    fn astra_chat_frame(frame_id: &str, captured_at: u64) -> SemanticScreenFrame {
        semantic_frame_from_vision_value(
            &json!({
                "frame_id": frame_id,
                "page_evidence": {
                    "browser_app_hint": "astra",
                    "content_provider_hint": "astra",
                    "page_kind_hint": "chat_window",
                    "confidence": 0.92
                },
                "scene_summary": "Astra assistant chat window is visible instead of the browser page."
            }),
            captured_at,
            None,
            None,
            Vec::new(),
        )
        .expect("astra chat frame")
    }

    fn verified_handoff_for_frame(frame: SemanticScreenFrame) -> BrowserVisualHandoffResult {
        BrowserVisualHandoffResult {
            record: BrowserVisualHandoffRecord {
                iteration: 0,
                status: BrowserHandoffStatus::VisuallyVerified,
                activation_status:
                    crate::desktop_agent_types::BrowserHandoffActivationStatus::NotAttempted,
                failure_reason: None,
                app_hint: Some("chrome".into()),
                provider_hint: frame.page_evidence.content_provider_hint.clone(),
                page_kind_hint: frame.page_evidence.page_kind_hint.clone(),
                verification: None,
                activation_attempted: false,
                page_verified: true,
                frame_id: Some(frame.frame_id.clone()),
                confidence: Some(frame.page_evidence.confidence),
                attempts: 1,
                reason: Some("verified test browser surface".into()),
            },
            verified_frame: Some(frame),
        }
    }

    fn open_list_item_goal(
        utterance: &str,
        provider: Option<&str>,
        item_kind: Option<&str>,
        rank_within_kind: Option<u32>,
        rank_overall: Option<u32>,
    ) -> GoalSpec {
        goal_for_open_list_item(
            utterance,
            provider.map(ToOwned::to_owned),
            item_kind.map(ToOwned::to_owned),
            rank_within_kind,
            rank_overall,
        )
    }

    #[test]
    fn open_list_item_grounds_second_video_from_primary_list_without_page_kind_label() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {
                    "content_provider_hint": "youtube",
                    "page_kind_hint": "youtube",
                    "confidence": 0.90
                },
                "scene_summary": "main area shows ranked playable items",
                "page_state": {
                    "kind": "list",
                    "dominant_content": "result_list",
                    "list_visible": true,
                    "detail_visible": false
                },
                "primary_list": {
                    "cluster_id": "main",
                    "container_kind": "result_list",
                    "item_count": 2,
                    "items": [
                        {
                            "item_id": "first",
                            "rank": 1,
                            "title": "First item",
                            "item_kind": "generic",
                            "click_regions": {
                                "primary": {"x": 20, "y": 40, "width": 200, "height": 80, "coordinate_space": "screen"}
                            },
                            "confidence": 0.95
                        },
                        {
                            "item_id": "second",
                            "rank": 2,
                            "title": "Second item",
                            "item_kind": "generic",
                            "click_regions": {
                                "primary": {"x": 20, "y": 140, "width": 200, "height": 80, "coordinate_space": "screen"}
                            },
                            "confidence": 0.95
                        }
                    ]
                }
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");
        let goal = open_list_item_goal(
            "aprimi il secondo video",
            Some("youtube"),
            Some("video"),
            Some(2),
            None,
        );

        let step = plan_next_step(&goal, &frame);

        assert_eq!(step.kind, PlannerStepKind::ClickResultRegion);
        assert_eq!(step.target_item_id.as_deref(), Some("second"));
        assert_eq!(
            step.executable_candidate
                .as_ref()
                .and_then(|candidate| candidate.result_kind.as_deref()),
            Some("generic")
        );
        assert!(step.confidence >= MIN_PLANNER_CLICK_CONFIDENCE);
    }

    #[test]
    fn open_list_item_grounds_first_site_from_generic_results_surface() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"page_kind_hint": "search", "confidence": 0.88},
                "scene_summary": "search results are visible as a ranked list",
                "page_state": {
                    "kind": "list",
                    "dominant_content": "result_list",
                    "list_visible": true,
                    "detail_visible": false
                },
                "primary_list": {
                    "cluster_id": "main",
                    "container_kind": "result_list",
                    "item_count": 1,
                    "items": [{
                        "item_id": "site_1",
                        "rank": 1,
                        "title": "Example site",
                        "item_kind": "site",
                        "click_regions": {
                            "primary": {"x": 100, "y": 120, "width": 360, "height": 42, "coordinate_space": "screen"}
                        },
                        "confidence": 0.96
                    }]
                }
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");
        let goal = open_list_item_goal("aprimi il primo sito", None, Some("site"), Some(1), None);

        let step = plan_next_step(&goal, &frame);

        assert_eq!(step.kind, PlannerStepKind::ClickResultRegion);
        assert_eq!(step.target_item_id.as_deref(), Some("site_1"));
        assert_eq!(
            step.executable_candidate
                .as_ref()
                .and_then(|candidate| candidate.result_kind.as_deref()),
            Some("site")
        );
    }

    #[test]
    fn open_list_item_success_uses_structural_list_to_detail_state() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"confidence": 0.83},
                "scene_summary": "a detail page is visible",
                "page_state": {
                    "kind": "detail",
                    "dominant_content": "detail_view",
                    "list_visible": false,
                    "detail_visible": true
                }
            }),
            2_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");
        let goal = open_list_item_goal("open the first result", None, None, None, Some(1));

        let verification = verify_goal_state(&goal, &frame, 1);

        assert_eq!(verification.status, GoalVerificationStatus::GoalAchieved);
        assert_eq!(
            verification.reason,
            "list_to_detail_structural_transition_visible"
        );
    }

    #[test]
    fn open_list_item_falls_back_to_legacy_visible_result_items_when_primary_list_missing() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"page_kind_hint": "search_results", "result_list_visible": true, "confidence": 0.91},
                "scene_summary": "results visible",
                "visible_result_items": [{
                    "item_id": "legacy_1",
                    "kind": "generic",
                    "title": "Legacy result",
                    "rank_overall": 1,
                    "click_regions": {
                        "title": {"x": 30, "y": 80, "width": 240, "height": 50, "coordinate_space": "screen"}
                    },
                    "confidence": 0.94
                }]
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");
        let goal = open_list_item_goal("open the first result", None, None, None, Some(1));

        let step = plan_next_step(&goal, &frame);

        assert_eq!(step.kind, PlannerStepKind::ClickResultRegion);
        assert_eq!(step.target_item_id.as_deref(), Some("legacy_1"));
    }

    #[test]
    fn browser_handoff_accepts_primary_list_context_without_search_results_label() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "youtube", "confidence": 0.89},
                "scene_summary": "a ranked list of items is visible",
                "page_state": {"kind": "list", "dominant_content": "result_list", "list_visible": true},
                "primary_list": {
                    "cluster_id": "main",
                    "container_kind": "result_list",
                    "item_count": 1,
                    "items": [{
                        "item_id": "item_1",
                        "rank": 1,
                        "title": "Visible item",
                        "item_kind": "generic",
                        "click_regions": {
                            "primary": {"x": 10, "y": 20, "width": 120, "height": 50, "coordinate_space": "screen"}
                        },
                        "confidence": 0.9
                    }]
                }
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");
        let goal = open_list_item_goal(
            "aprimi il primo video",
            Some("youtube"),
            Some("video"),
            Some(1),
            None,
        );

        let diagnostic = verify_browser_handoff_page(&goal, &frame).expect("handoff accepted");

        assert!(diagnostic.accepted);
        assert_eq!(
            diagnostic.decision,
            BrowserHandoffVerificationDecision::SupportingEvidence
        );
        assert_eq!(diagnostic.primary_list_item_count, 1);
        assert!(diagnostic.structural_list_surface_visible);
    }

    #[test]
    fn malformed_primary_list_items_are_discarded_and_valid_items_remain() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"confidence": 0.8},
                "primary_list": {
                    "cluster_id": "main",
                    "container_kind": "result_list",
                    "item_count": 2,
                    "items": [
                        {
                            "item_id": "bad",
                            "rank": 1,
                            "item_kind": "generic",
                            "click_regions": {
                                "primary": {"x": 0, "y": 0, "width": 0, "height": 40, "coordinate_space": "screen"}
                            },
                            "confidence": 0.9
                        },
                        {
                            "item_id": "good",
                            "rank": 2,
                            "item_kind": "generic",
                            "click_regions": {
                                "primary": {"x": 20, "y": 60, "width": 200, "height": 40, "coordinate_space": "screen"}
                            },
                            "confidence": 0.9
                        }
                    ]
                }
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");

        let list = frame.primary_list.expect("primary list");
        assert_eq!(list.items.len(), 1);
        assert_eq!(list.items[0].item_id, "good");
        assert!(frame
            .uncertainty
            .iter()
            .any(|entry| entry.code == "primary_list_malformed_items_discarded"));
    }

    #[test]
    fn low_confidence_primary_list_item_refuses_instead_of_blind_clicking() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"confidence": 0.20},
                "page_state": {"kind": "list", "dominant_content": "result_list", "list_visible": true},
                "primary_list": {
                    "cluster_id": "main",
                    "container_kind": "result_list",
                    "item_count": 1,
                    "items": [{
                        "item_id": "weak",
                        "rank": 1,
                        "item_kind": "video",
                        "click_regions": {
                            "primary": {"x": 20, "y": 60, "width": 200, "height": 40, "coordinate_space": "screen", "confidence": 0.2}
                        },
                        "confidence": 0.2
                    }]
                }
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");
        let goal = open_list_item_goal("aprimi il primo video", None, Some("video"), Some(1), None);

        let step = plan_next_step(&goal, &frame);

        assert_eq!(step.kind, PlannerStepKind::Refuse);
        assert!(step.executable_candidate.is_none());
    }

    struct MockGoalLoopDriver {
        frames: Vec<SemanticScreenFrame>,
        focused_frames: Vec<SemanticScreenFrame>,
        executed_steps: Vec<PlannerStep>,
        focused_requests: Vec<FocusedPerceptionRequest>,
        perceive_fresh_requests: Vec<bool>,
        handoff: Option<BrowserVisualHandoffResult>,
        recovery: Option<BrowserVisualHandoffResult>,
        recovery_attempts: usize,
        model_decision: Option<PlannerContractDecision>,
        plan_calls: usize,
    }

    impl MockGoalLoopDriver {
        fn new(frames: Vec<SemanticScreenFrame>) -> Self {
            Self {
                frames,
                focused_frames: Vec::new(),
                executed_steps: Vec::new(),
                focused_requests: Vec::new(),
                perceive_fresh_requests: Vec::new(),
                handoff: None,
                recovery: None,
                recovery_attempts: 0,
                model_decision: None,
                plan_calls: 0,
            }
        }
    }

    impl GoalLoopDriver for MockGoalLoopDriver {
        fn prepare_visual_handoff<'a>(
            &'a mut self,
            _goal: &'a GoalSpec,
            _iteration: usize,
        ) -> GoalLoopDriverFuture<'a, Result<Option<BrowserVisualHandoffResult>, String>> {
            Box::pin(async move { Ok(self.handoff.take()) })
        }

        fn perceive<'a>(
            &'a mut self,
            _goal: &'a GoalSpec,
            _iteration: usize,
            fresh_capture_required: bool,
        ) -> GoalLoopDriverFuture<'a, Result<SemanticScreenFrame, String>> {
            Box::pin(async move {
                self.perceive_fresh_requests.push(fresh_capture_required);
                if self.frames.is_empty() {
                    return Err("no mock frame available".into());
                }
                Ok(self.frames.remove(0))
            })
        }

        fn execute_planner_step<'a>(
            &'a mut self,
            step: &'a PlannerStep,
        ) -> GoalLoopDriverFuture<'a, PlannerStepExecutionRecord> {
            Box::pin(async move {
                self.executed_steps.push(step.clone());
                PlannerStepExecutionRecord {
                    step_id: step.step_id.clone(),
                    status: PlannerStepExecutionStatus::Executed,
                    primitive: "mock_click".into(),
                    message: "mock governed click executed".into(),
                    selected_target_candidate: step.executable_candidate.clone(),
                    geometry: None,
                    fresh_capture_required: false,
                    fresh_capture_used: false,
                    target_signature: step_target_signature(step),
                }
            })
        }

        fn focused_perception<'a>(
            &'a mut self,
            request: &'a FocusedPerceptionRequest,
        ) -> GoalLoopDriverFuture<'a, Result<Option<SemanticScreenFrame>, String>> {
            Box::pin(async move {
                self.focused_requests.push(request.clone());
                if self.focused_frames.is_empty() {
                    return Ok(None);
                }
                Ok(Some(self.focused_frames.remove(0)))
            })
        }

        fn recover_browser_surface<'a>(
            &'a mut self,
            _goal: &'a GoalSpec,
            _iteration: usize,
            _reason: &'a str,
        ) -> GoalLoopDriverFuture<'a, Result<Option<BrowserVisualHandoffResult>, String>> {
            Box::pin(async move {
                self.recovery_attempts += 1;
                Ok(self.recovery.take())
            })
        }

        fn plan<'a>(
            &'a mut self,
            _input: &'a PlannerContractInput,
        ) -> GoalLoopDriverFuture<'a, Result<Option<PlannerContractDecision>, String>> {
            Box::pin(async move {
                self.plan_calls += 1;
                Ok(self.model_decision.take())
            })
        }
    }

    #[tokio::test]
    async fn visible_ordinal_fast_path_opens_third_video_without_planner_discovery() {
        let list_frame = youtube_primary_list_frame("list", 1_000, 3);
        let watch_frame = youtube_watch_frame("watch", 2_000);
        let goal = open_list_item_goal(
            "aprimi il terzo video",
            Some("youtube"),
            Some("video"),
            Some(3),
            None,
        );
        let runtime = GoalLoopRuntime::new(GoalLoopRuntimeConfig::default());
        let mut driver = MockGoalLoopDriver::new(vec![list_frame, watch_frame]);

        let run = runtime
            .run_goal_loop_until_complete(goal, &mut driver)
            .await;

        assert_eq!(driver.plan_calls, 0);
        assert_eq!(run.status, GoalLoopStatus::GoalAchieved);
        assert_eq!(driver.executed_steps.len(), 1);
        assert_eq!(
            driver.executed_steps[0].target_item_id.as_deref(),
            Some("video_3")
        );
    }

    #[test]
    fn focused_perception_outside_verified_browser_surface_is_typed_refusal() {
        let goal = open_list_item_goal(
            "aprimi il primo video",
            Some("youtube"),
            Some("video"),
            Some(1),
            None,
        );
        let frame = youtube_primary_list_frame("list", 1_000, 1);
        let mut surface = verified_browser_surface_from_frame(&goal, &frame);
        surface.bounds = Some(TargetRegion {
            x: 0.0,
            y: 0.0,
            width: 200.0,
            height: 200.0,
            coordinate_space: "screen".into(),
        });
        let request = FocusedPerceptionRequest {
            request_id: "outside".into(),
            iteration: 1,
            reason: "test outside surface".into(),
            mode: PerceptionRequestMode::TargetFocus,
            routing_decision: PerceptionRoutingDecision::TargetRegionAnchor,
            refinement_strategy: VisibleRefinementStrategy::TargetRegion,
            target_item_id: Some("video_1".into()),
            target_entity_id: None,
            region: Some(TargetRegion {
                x: 500.0,
                y: 500.0,
                width: 80.0,
                height: 40.0,
                coordinate_space: "screen".into(),
            }),
            target_region_anchor_present: true,
            verified_surface: None,
        };

        let diagnostic =
            bind_focused_perception_request_to_surface(request, Some(&surface), true, 1)
                .expect_err("outside region should be refused");

        assert_eq!(diagnostic.status, SurfaceOwnershipStatus::Refused);
        assert_eq!(
            diagnostic.failure_reason,
            Some(FocusedPerceptionFailureReason::RequestedRegionOutsideSurface)
        );
    }

    #[tokio::test]
    async fn ambiguous_post_click_browser_frame_uses_bounded_focused_perception() {
        let list_frame = youtube_primary_list_frame("list", 1_000, 1);
        let ambiguous_frame = ambiguous_browser_detail_frame("mixed", 2_000);
        let goal = open_list_item_goal(
            "aprimi il primo video",
            Some("youtube"),
            Some("video"),
            Some(1),
            None,
        );
        let runtime = GoalLoopRuntime::new(GoalLoopRuntimeConfig::default());
        let mut driver = MockGoalLoopDriver::new(vec![ambiguous_frame]);
        driver.handoff = Some(verified_handoff_for_frame(list_frame));

        let run = runtime
            .run_goal_loop_until_complete(goal, &mut driver)
            .await;

        assert_eq!(run.status, GoalLoopStatus::NeedsPerception);
        assert_eq!(driver.focused_requests.len(), 1);
        let request = &driver.focused_requests[0];
        assert!(request.region.is_some());
        assert!(request.verified_surface.is_some());
        assert_eq!(
            run.focused_perception_failure_reason,
            Some(FocusedPerceptionFailureReason::StructuredPerceptionEmpty)
        );
    }

    #[tokio::test]
    async fn post_click_astra_chat_frame_reports_surface_ownership_lost() {
        let list_frame = youtube_primary_list_frame("list", 1_000, 1);
        let astra_frame = astra_chat_frame("astra", 2_000);
        let goal = open_list_item_goal(
            "aprimi il primo video",
            Some("youtube"),
            Some("video"),
            Some(1),
            None,
        );
        let runtime = GoalLoopRuntime::new(GoalLoopRuntimeConfig::default());
        let mut driver = MockGoalLoopDriver::new(vec![astra_frame]);
        driver.handoff = Some(verified_handoff_for_frame(list_frame));

        let run = runtime
            .run_goal_loop_until_complete(goal, &mut driver)
            .await;

        assert_eq!(run.status, GoalLoopStatus::VerificationFailed);
        assert!(run.surface_ownership_lost);
        assert_eq!(
            run.focused_perception_failure_reason,
            Some(FocusedPerceptionFailureReason::SurfaceOwnershipLost)
        );
        assert_eq!(driver.recovery_attempts, 1);
        assert!(driver.focused_requests.is_empty());
        assert!(run
            .surface_diagnostics
            .iter()
            .any(|diagnostic| diagnostic.status == SurfaceOwnershipStatus::Lost));
    }

    #[tokio::test]
    async fn browser_reacquisition_after_surface_loss_is_bounded_and_rechecked() {
        let list_frame = youtube_primary_list_frame("list", 1_000, 1);
        let astra_frame = astra_chat_frame("astra", 2_000);
        let recovered_frame = youtube_watch_frame("watch", 3_000);
        let goal = open_list_item_goal(
            "aprimi il primo video",
            Some("youtube"),
            Some("video"),
            Some(1),
            None,
        );
        let runtime = GoalLoopRuntime::new(GoalLoopRuntimeConfig::default());
        let mut driver = MockGoalLoopDriver::new(vec![astra_frame]);
        driver.handoff = Some(verified_handoff_for_frame(list_frame));
        driver.recovery = Some(verified_handoff_for_frame(recovered_frame));

        let run = runtime
            .run_goal_loop_until_complete(goal, &mut driver)
            .await;

        assert_eq!(driver.recovery_attempts, 1);
        assert_eq!(run.status, GoalLoopStatus::GoalAchieved);
        assert!(!run.surface_ownership_lost);
        assert!(run
            .surface_diagnostics
            .iter()
            .any(|diagnostic| diagnostic.status == SurfaceOwnershipStatus::Reacquired));
    }

    #[test]
    fn parses_semantic_frame_with_youtube_mix_and_video_results() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "semantic_frame": {
                    "page_evidence": {
                        "content_provider_hint": "youtube",
                        "page_kind_hint": "search_results",
                        "query_hint": "shiva",
                        "result_list_visible": true,
                        "confidence": 0.94
                    },
                    "scene_summary": "YouTube search results are visible.",
                    "visible_result_items": [
                        {
                            "item_id": "r1",
                            "kind": "mix",
                            "title": "Mix - Shiva",
                            "rank_overall": 1,
                            "rank_within_kind": 1,
                            "click_regions": {
                                "thumbnail": {"x": 230, "y": 240, "width": 326, "height": 190, "coordinate_space": "screen"}
                            },
                            "confidence": 0.88
                        },
                        {
                            "item_id": "r2",
                            "kind": "video",
                            "title": "Shiva - Bacio di Giuda",
                            "channel_name": "SHIVA",
                            "rank_overall": 2,
                            "rank_within_kind": 1,
                            "click_regions": {
                                "thumbnail": {"x": 230, "y": 447, "width": 326, "height": 171, "coordinate_space": "screen"},
                                "title": {"x": 563, "y": 444, "width": 455, "height": 48, "coordinate_space": "screen"}
                            },
                            "confidence": 0.95
                        }
                    ]
                }
            }),
            1_000,
            Some("screen.png".into()),
            None,
            Vec::new(),
        )
        .expect("frame");

        assert_eq!(
            frame.page_evidence.content_provider_hint.as_deref(),
            Some("youtube")
        );
        assert_eq!(frame.visible_result_items.len(), 2);
        assert_eq!(frame.visible_result_items[0].kind, VisibleResultKind::Mix);
        assert_eq!(frame.visible_result_items[1].kind, VisibleResultKind::Video);
    }

    #[test]
    fn planner_selects_first_video_not_first_overall_mix() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {
                    "content_provider_hint": "youtube",
                    "page_kind_hint": "search_results",
                    "result_list_visible": true,
                    "confidence": 0.94
                },
                "visible_result_items": [
                    {
                        "item_id": "mix",
                        "kind": "mix",
                        "rank_overall": 1,
                        "rank_within_kind": 1,
                        "click_regions": {
                            "thumbnail": {"x": 100, "y": 100, "width": 300, "height": 180, "coordinate_space": "screen"}
                        },
                        "confidence": 0.95
                    },
                    {
                        "item_id": "video",
                        "kind": "video",
                        "rank_overall": 2,
                        "rank_within_kind": 1,
                        "click_regions": {
                            "title": {"x": 450, "y": 300, "width": 400, "height": 50, "coordinate_space": "screen"}
                        },
                        "confidence": 0.96
                    }
                ]
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");
        let goal = goal_for_open_media_result(
            "aprimi il primo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );
        let step = plan_next_step(&goal, &frame);

        assert_eq!(step.kind, PlannerStepKind::ClickResultRegion);
        assert_eq!(step.target_item_id.as_deref(), Some("video"));
        assert_eq!(step.click_region_key.as_deref(), Some("title"));
        assert_eq!(
            step.executable_candidate
                .as_ref()
                .and_then(|candidate| candidate.result_kind.as_deref()),
            Some("video")
        );
        assert_eq!(
            step.executable_candidate
                .as_ref()
                .and_then(|candidate| candidate.rank),
            Some(1)
        );
    }

    #[test]
    fn loop_run_requests_execution_for_planner_candidate() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "search_results", "confidence": 0.9},
                "visible_result_items": [{
                    "item_id": "video",
                    "kind": "video",
                    "rank_within_kind": 1,
                    "click_regions": {"thumbnail": {"x": 1, "y": 2, "width": 3, "height": 4, "coordinate_space": "screen"}},
                    "confidence": 0.9
                }]
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");
        let goal = goal_for_open_media_result(
            "open first video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );

        let run = run_goal_loop_once(goal, frame);

        assert_eq!(run.status, GoalLoopStatus::NeedsExecution);
        assert_eq!(run.planner_steps.len(), 1);
        assert!(run.planner_steps[0].executable_candidate.is_some());
    }

    #[test]
    fn verifier_marks_youtube_watch_page_as_goal_achieved() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {
                    "content_provider_hint": "youtube",
                    "page_kind_hint": "watch_page",
                    "confidence": 0.9
                }
            }),
            2_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");
        let goal = goal_for_open_media_result(
            "open first video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );

        assert_eq!(
            verify_goal_against_frame(&goal, &frame).as_deref(),
            Some("media_watch_page_visible")
        );
    }

    #[test]
    fn watch_page_aliases_normalize_to_watch_page() {
        for alias in ["watch", "video", "video_watch", "player", "media_page"] {
            assert_eq!(
                browser_page_semantic_kind(Some(alias)),
                BrowserPageSemanticKind::WatchPage,
                "alias `{alias}` should normalize to watch_page"
            );
        }
    }

    #[test]
    fn browser_page_semantic_kind_normalizes_google_serp_variants() {
        for raw in [
            "search_result_page",
            "serp",
            "listing",
            "catalog",
            "results_page",
        ] {
            assert_eq!(
                browser_page_semantic_kind(Some(raw)),
                BrowserPageSemanticKind::SearchResults,
                "expected SearchResults for raw kind {:?}",
                raw
            );
        }
    }

    #[test]
    fn verifier_accepts_watch_alias_through_centralized_normalization() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {
                    "content_provider_hint": "youtube",
                    "page_kind_hint": "player",
                    "confidence": 0.91
                }
            }),
            2_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");
        let goal = goal_for_open_media_result(
            "open first video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );

        assert_eq!(
            verify_goal_against_frame(&goal, &frame).as_deref(),
            Some("media_watch_page_visible")
        );
    }

    #[test]
    fn planner_selects_visible_channel_entity_when_available() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {
                    "content_provider_hint": "youtube",
                    "page_kind_hint": "search_results",
                    "confidence": 0.9
                },
                "visible_entities": [{
                    "entity_id": "channel",
                    "kind": "channel_result",
                    "name": "SHIVA",
                    "region": {"x": 500, "y": 150, "width": 220, "height": 80, "coordinate_space": "screen"},
                    "confidence": 0.91
                }]
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");
        let goal = GoalSpec {
            goal_id: "goal".into(),
            goal_type: GoalType::OpenChannel,
            constraints: GoalConstraints {
                provider: Some("youtube".into()),
                item_kind: None,
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

        let step = plan_next_step(&goal, &frame);

        assert_eq!(step.kind, PlannerStepKind::ClickEntityRegion);
        assert_eq!(step.target_entity_id.as_deref(), Some("channel"));
        assert!(step.executable_candidate.is_some());
    }

    #[tokio::test]
    async fn iterative_goal_loop_executes_first_video_then_verifies_watch_page() {
        let search_frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "search_results", "confidence": 0.94},
                "visible_result_items": [{
                    "item_id": "v1",
                    "kind": "video",
                    "rank_within_kind": 1,
                    "click_regions": {"thumbnail": {"x": 10, "y": 20, "width": 300, "height": 180, "coordinate_space": "screen"}},
                    "confidence": 0.95
                }]
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("search frame");
        let watch_frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "watch_page", "confidence": 0.93}
            }),
            2_000,
            None,
            None,
            Vec::new(),
        )
        .expect("watch frame");
        let goal = goal_for_open_media_result(
            "aprimi il primo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );
        let runtime = GoalLoopRuntime::new(GoalLoopRuntimeConfig::default());
        let mut driver = MockGoalLoopDriver::new(vec![search_frame, watch_frame]);

        let run = runtime
            .run_goal_loop_until_complete(goal, &mut driver)
            .await;

        assert_eq!(run.status, GoalLoopStatus::GoalAchieved);
        assert_eq!(run.executed_steps.len(), 1);
        assert_eq!(run.verification_history.len(), 2);
        assert_eq!(
            run.verification_history.last().map(|record| &record.status),
            Some(&GoalVerificationStatus::GoalAchieved)
        );
        assert_eq!(
            driver.executed_steps[0].kind,
            PlannerStepKind::ClickResultRegion
        );
    }

    #[tokio::test]
    async fn post_click_perception_requires_fresh_capture() {
        let goal = goal_for_open_media_result(
            "aprimi il primo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );
        let runtime = GoalLoopRuntime::new(GoalLoopRuntimeConfig::default());
        let mut driver = MockGoalLoopDriver::new(vec![
            youtube_search_frame("search_1", 1_000, "Shiva"),
            youtube_watch_frame("watch_1", 2_000),
        ]);

        let run = runtime
            .run_goal_loop_until_complete(goal, &mut driver)
            .await;

        assert_eq!(run.status, GoalLoopStatus::GoalAchieved);
        assert_eq!(driver.perceive_fresh_requests, vec![false, true]);
        assert!(run.stale_capture_reuse_prevented);
        assert!(run.executed_steps[0].fresh_capture_required);
        assert!(run.executed_steps[0].fresh_capture_used);
    }

    #[tokio::test]
    async fn watch_alias_after_click_is_verified_before_browser_recovery() {
        let goal = goal_for_open_media_result(
            "aprimi il primo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );
        let runtime = GoalLoopRuntime::new(GoalLoopRuntimeConfig::default());
        let mut driver = MockGoalLoopDriver::new(vec![
            youtube_search_frame("search_1", 1_000, "Shiva"),
            semantic_frame_from_vision_value(
                &json!({
                    "page_evidence": {
                        "content_provider_hint": "youtube",
                        "page_kind_hint": "video_watch",
                        "confidence": 0.93
                    }
                }),
                2_000,
                None,
                None,
                Vec::new(),
            )
            .expect("watch alias frame"),
        ]);
        driver.recovery = Some(BrowserVisualHandoffResult {
            record: BrowserVisualHandoffRecord {
                iteration: 1,
                status: BrowserHandoffStatus::VisuallyVerified,
                activation_status:
                    crate::desktop_agent_types::BrowserHandoffActivationStatus::Executed,
                failure_reason: None,
                app_hint: Some("browser".into()),
                provider_hint: Some("youtube".into()),
                page_kind_hint: Some("watch_page".into()),
                verification: None,
                activation_attempted: true,
                page_verified: true,
                frame_id: Some("recovery_watch".into()),
                confidence: Some(0.92),
                attempts: 1,
                reason: Some("should not be used".into()),
            },
            verified_frame: Some(youtube_watch_frame("recovery_watch", 3_000)),
        });

        let run = runtime
            .run_goal_loop_until_complete(goal, &mut driver)
            .await;

        assert_eq!(run.status, GoalLoopStatus::GoalAchieved);
        assert_eq!(driver.recovery_attempts, 0);
        assert!(!run.browser_recovery_used);
        assert!(run.post_action_progress_observed);
    }

    #[tokio::test]
    async fn suspicious_geometry_is_rejected_after_fresh_no_progress_and_recovery_exhaustion() {
        let goal = goal_for_open_media_result(
            "aprimi il primo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );
        let runtime = GoalLoopRuntime::new(GoalLoopRuntimeConfig {
            max_iterations: 5,
            ..GoalLoopRuntimeConfig::default()
        });
        let mut driver = MockGoalLoopDriver::new(vec![
            youtube_search_frame("search_1", 1_000, "Shiva"),
            youtube_search_frame("search_2", 2_000, "Shiva"),
            youtube_search_frame("search_3", 3_000, "Shiva"),
        ]);

        let run = runtime
            .run_goal_loop_until_complete(goal, &mut driver)
            .await;

        assert_eq!(run.status, GoalLoopStatus::VerificationFailed);
        assert!(!run.repeated_click_protection_triggered);
        assert_eq!(driver.perceive_fresh_requests, vec![false, true]);
        assert_eq!(
            run.verifier_status.as_deref(),
            Some("suspicious_geometry_rejected_after_no_progress")
        );
        assert_eq!(run.executed_steps.len(), 1);
    }

    #[tokio::test]
    async fn browser_recovery_reacquires_verified_surface_after_ineffective_click() {
        let goal = goal_for_open_media_result(
            "aprimi il primo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );
        let runtime = GoalLoopRuntime::new(GoalLoopRuntimeConfig::default());
        let mut driver = MockGoalLoopDriver::new(vec![
            youtube_search_frame("search_1", 1_000, "Shiva"),
            youtube_search_frame("search_2", 2_000, "Shiva"),
        ]);
        let recovered_watch = youtube_watch_frame("watch_recovered", 3_000);
        driver.recovery = Some(BrowserVisualHandoffResult {
            record: BrowserVisualHandoffRecord {
                iteration: 1,
                status: BrowserHandoffStatus::VisuallyVerified,
                activation_status:
                    crate::desktop_agent_types::BrowserHandoffActivationStatus::Executed,
                failure_reason: None,
                app_hint: Some("browser".into()),
                provider_hint: Some("youtube".into()),
                page_kind_hint: Some("watch_page".into()),
                verification: None,
                activation_attempted: true,
                page_verified: true,
                frame_id: Some(recovered_watch.frame_id.clone()),
                confidence: Some(0.93),
                attempts: 1,
                reason: Some("recovered browser surface".into()),
            },
            verified_frame: Some(recovered_watch),
        });

        let run = runtime
            .run_goal_loop_until_complete(goal, &mut driver)
            .await;

        assert_eq!(run.status, GoalLoopStatus::GoalAchieved);
        assert!(run.browser_recovery_used);
        assert_eq!(
            run.browser_recovery_status,
            BrowserRecoveryStatus::Reacquired
        );
        assert_eq!(
            run.browser_handoff_history
                .last()
                .map(|record| &record.status),
            Some(&BrowserHandoffStatus::VisuallyVerified)
        );
    }

    #[tokio::test]
    async fn verified_browser_handoff_frame_is_used_before_planning() {
        let search_frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "search_results", "result_list_visible": true, "confidence": 0.94},
                "visible_result_items": [{
                    "item_id": "v1",
                    "kind": "video",
                    "rank_within_kind": 1,
                    "click_regions": {"title": {"x": 10, "y": 20, "width": 300, "height": 80, "coordinate_space": "screen"}},
                    "confidence": 0.95
                }]
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("search frame");
        let watch_frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "watch_page", "confidence": 0.93}
            }),
            2_000,
            None,
            None,
            Vec::new(),
        )
        .expect("watch frame");
        let goal = goal_for_open_media_result(
            "aprimi il primo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );
        let runtime = GoalLoopRuntime::new(GoalLoopRuntimeConfig::default());
        let mut driver = MockGoalLoopDriver::new(vec![watch_frame]);
        driver.handoff = Some(BrowserVisualHandoffResult {
            record: BrowserVisualHandoffRecord {
                iteration: 0,
                status: BrowserHandoffStatus::VisuallyVerified,
                activation_status:
                    crate::desktop_agent_types::BrowserHandoffActivationStatus::Executed,
                failure_reason: None,
                app_hint: Some("browser".into()),
                provider_hint: Some("youtube".into()),
                page_kind_hint: Some("search_results".into()),
                verification: None,
                activation_attempted: true,
                page_verified: true,
                frame_id: Some(search_frame.frame_id.clone()),
                confidence: Some(0.94),
                attempts: 1,
                reason: Some("verified test handoff".into()),
            },
            verified_frame: Some(search_frame),
        });

        let run = runtime
            .run_goal_loop_until_complete(goal, &mut driver)
            .await;

        assert_eq!(run.status, GoalLoopStatus::GoalAchieved);
        assert_eq!(
            run.browser_handoff.as_ref().map(|record| &record.status),
            Some(&BrowserHandoffStatus::VisuallyVerified)
        );
        assert_eq!(run.executed_steps.len(), 1);
    }

    #[tokio::test]
    async fn browser_handoff_failure_stops_before_planner_selection() {
        let goal = goal_for_open_media_result(
            "aprimi il primo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );
        let runtime = GoalLoopRuntime::new(GoalLoopRuntimeConfig::default());
        let mut driver = MockGoalLoopDriver::new(Vec::new());
        driver.handoff = Some(BrowserVisualHandoffResult {
            record: BrowserVisualHandoffRecord {
                iteration: 0,
                status: BrowserHandoffStatus::ActivationUnsupported,
                activation_status: crate::desktop_agent_types::BrowserHandoffActivationStatus::Unsupported,
                failure_reason: Some(crate::desktop_agent_types::BrowserHandoffFailureReason::BrowserActivationUnsupported),
                app_hint: Some("browser".into()),
                provider_hint: Some("youtube".into()),
                page_kind_hint: None,
                verification: None,
                activation_attempted: true,
                page_verified: false,
                frame_id: None,
                confidence: None,
                attempts: 1,
                reason: Some("activation primitive unsupported".into()),
            },
            verified_frame: None,
        });

        let run = runtime
            .run_goal_loop_until_complete(goal, &mut driver)
            .await;

        assert_eq!(run.status, GoalLoopStatus::BrowserHandoffFailed);
        assert!(run.planner_steps.is_empty());
        assert!(run.executed_steps.is_empty());
        assert_eq!(
            run.browser_handoff.as_ref().map(|record| &record.status),
            Some(&BrowserHandoffStatus::ActivationUnsupported)
        );
    }

    #[test]
    fn browser_handoff_page_verification_rejects_astra_or_unrelated_page() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"content_provider_hint": "astra", "page_kind_hint": "app_panel", "confidence": 0.9},
                "scene_summary": "Astra desktop UI is visible"
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");
        let goal = goal_for_open_media_result(
            "aprimi il primo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );

        assert!(verify_browser_handoff_page(&goal, &frame).is_err());
    }

    #[test]
    fn browser_handoff_page_verification_accepts_result_alias_for_results_page() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {
                    "content_provider_hint": "youtube",
                    "page_kind_hint": "result",
                    "result_list_visible": true,
                    "query_hint": "shiva",
                    "confidence": 0.92
                },
                "scene_summary": "YouTube result page is visible",
                "visible_result_items": [{
                    "item_id": "video_1",
                    "kind": "video",
                    "rank_within_kind": 1,
                    "click_regions": {"title": {"x": 420, "y": 220, "width": 360, "height": 40, "coordinate_space": "screen"}},
                    "confidence": 0.93
                }]
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");
        let goal = goal_for_open_media_result(
            "aprimi il primo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );

        let verification = verify_browser_handoff_page(&goal, &frame).expect("verification");

        assert!(verification.accepted);
        assert_eq!(
            verification.normalized_page_kind,
            BrowserPageSemanticKind::SearchResults
        );
        assert_eq!(
            verification.decision,
            BrowserHandoffVerificationDecision::NormalizedPageKind
        );
    }

    #[test]
    fn browser_handoff_page_verification_accepts_supported_results_evidence_without_exact_label() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {
                    "content_provider_hint": "youtube",
                    "page_kind_hint": "results_surface",
                    "result_list_visible": true,
                    "query_hint": "shiva",
                    "confidence": 0.9
                },
                "scene_summary": "Search results are visible for the requested query"
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");
        let goal = goal_for_open_media_result(
            "aprimi il primo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );

        let verification = verify_browser_handoff_page(&goal, &frame).expect("verification");

        assert!(verification.accepted);
        assert_eq!(
            verification.decision,
            BrowserHandoffVerificationDecision::SupportingEvidence
        );
        assert!(verification.result_list_visible);
        assert!(verification.scene_summary_result_hint);
    }

    #[test]
    fn browser_handoff_page_verification_accepts_generic_provider_label_with_strong_results_evidence(
    ) {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {
                    "content_provider_hint": "youtube",
                    "page_kind_hint": "youtube",
                    "query_hint": "shiva",
                    "confidence": 0.9
                },
                "scene_summary": "YouTube search results are visible for Shiva",
                "visible_result_items": [{
                    "item_id": "video_1",
                    "kind": "video",
                    "title": "Shiva - official video",
                    "rank_within_kind": 1,
                    "click_regions": {"title": {"x": 420, "y": 220, "width": 360, "height": 40, "coordinate_space": "screen"}},
                    "confidence": 0.93
                }]
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");
        let goal = goal_for_open_media_result(
            "aprimi il secondo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            2,
        );

        let verification = verify_browser_handoff_page(&goal, &frame).expect("verification");

        assert!(verification.accepted);
        assert_eq!(
            verification.decision,
            BrowserHandoffVerificationDecision::SupportingEvidence
        );
        assert!(verification.generic_provider_page_kind_hint);
        assert!(verification.goal_expects_results_context);
        assert!(verification
            .reason
            .as_deref()
            .unwrap_or("")
            .contains("generic provider page kind"));
    }

    #[test]
    fn browser_handoff_page_verification_rejects_generic_provider_label_without_strong_results_evidence(
    ) {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {
                    "content_provider_hint": "youtube",
                    "page_kind_hint": "youtube",
                    "confidence": 0.88
                },
                "scene_summary": "YouTube home page is visible"
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");
        let goal = goal_for_open_media_result(
            "aprimi il secondo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            2,
        );

        let verification = verify_browser_handoff_page(&goal, &frame)
            .expect_err("generic provider label should need stronger evidence");

        assert!(verification.generic_provider_page_kind_hint);
        assert!(verification
            .reason
            .as_deref()
            .unwrap_or("")
            .contains("supporting evidence was not strong enough"));
    }

    #[test]
    fn browser_handoff_verification_preserves_provider_backend_separation() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {
                    "content_provider_hint": "youtube",
                    "capture_backend": "powershell_gdi",
                    "page_kind_hint": "result",
                    "result_list_visible": true,
                    "confidence": 0.91
                },
                "scene_summary": "YouTube result page is visible"
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");
        let goal = goal_for_open_media_result(
            "aprimi il primo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );

        let verification = verify_browser_handoff_page(&goal, &frame).expect("verification");

        assert!(verification.accepted);
        assert_eq!(
            frame.page_evidence.capture_backend.as_deref(),
            Some("powershell_gdi")
        );
        assert!(verification.provider_matches);
    }

    #[test]
    fn semantic_frame_parses_nested_click_region_and_control_region_shapes() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "search_results", "confidence": 0.91},
                "visible_result_items": [{
                    "item_id": "video",
                    "kind": "video",
                    "rank_within_kind": 1,
                    "click_regions": {
                        "title": {
                            "region": {
                                "x": 120,
                                "y": 240,
                                "width": 580,
                                "height": 36,
                                "coordinate_space": "screen"
                            },
                            "confidence": 0.88
                        }
                    },
                    "confidence": 0.93
                }],
                "actionable_controls": [{
                    "control_id": "play_button",
                    "kind": "button",
                    "label": "click to play video",
                    "region": {
                        "x": 90,
                        "y": 180,
                        "width": 320,
                        "height": 180,
                        "coordinate_space": "screen"
                    },
                    "confidence": 0.82
                }]
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");

        let title_region = frame.visible_result_items[0]
            .click_regions
            .get("title")
            .expect("title region");
        assert_eq!(title_region.region.x, 120.0);
        assert_eq!(title_region.region.coordinate_space, "screen");
        assert_eq!(title_region.confidence, 0.88);
        assert_eq!(
            frame.actionable_controls[0]
                .region
                .as_ref()
                .expect("control region")
                .width,
            320.0
        );
    }

    #[tokio::test]
    async fn youtube_channel_goal_falls_back_to_video_then_avatar() {
        let results_frame = semantic_frame_from_vision_value(
            &json!({
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
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("results frame");
        let watch_frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "watch_page", "confidence": 0.91},
                "visible_entities": [{
                    "entity_id": "avatar",
                    "kind": "avatar",
                    "name": "SHIVA",
                    "region": {"x": 560, "y": 520, "width": 36, "height": 36, "coordinate_space": "screen"},
                    "confidence": 0.9
                }]
            }),
            2_000,
            None,
            None,
            Vec::new(),
        )
        .expect("watch frame");
        let channel_frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "channel_page", "confidence": 0.94}
            }),
            3_000,
            None,
            None,
            Vec::new(),
        )
        .expect("channel frame");
        let runtime = GoalLoopRuntime::new(GoalLoopRuntimeConfig::default());
        let mut driver = MockGoalLoopDriver::new(vec![results_frame, watch_frame, channel_frame]);

        let run = runtime
            .run_goal_loop_until_complete(open_channel_goal(), &mut driver)
            .await;

        assert_eq!(run.status, GoalLoopStatus::GoalAchieved);
        assert_eq!(run.executed_steps.len(), 2);
        assert_eq!(
            driver.executed_steps[0].expected_state.as_deref(),
            Some("media_watch_page_visible")
        );
        assert_eq!(
            driver.executed_steps[1].expected_state.as_deref(),
            Some("channel_page_visible")
        );
        assert!(run
            .planner_steps
            .iter()
            .any(|step| step.rationale.contains("opening relevant video")));
    }

    #[tokio::test]
    async fn focused_perception_replan_turns_ambiguous_result_region_into_click() {
        let ambiguous_frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "search_results", "confidence": 0.88},
                "visible_result_items": [{
                    "item_id": "video",
                    "kind": "video",
                    "rank_within_kind": 1,
                    "attributes": {"region": {"x": 100, "y": 200, "width": 500, "height": 180, "coordinate_space": "screen"}},
                    "confidence": 0.91
                }]
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("ambiguous frame");
        let focused_frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "search_results", "confidence": 0.91},
                "visible_result_items": [{
                    "item_id": "video",
                    "kind": "video",
                    "rank_within_kind": 1,
                    "click_regions": {"title": {"x": 450, "y": 300, "width": 400, "height": 50, "coordinate_space": "screen"}},
                    "confidence": 0.94
                }]
            }),
            1_500,
            None,
            None,
            Vec::new(),
        )
        .expect("focused frame");
        let watch_frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "watch_page", "confidence": 0.93}
            }),
            2_000,
            None,
            None,
            Vec::new(),
        )
        .expect("watch frame");
        let goal = goal_for_open_media_result(
            "aprimi il primo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );
        let runtime = GoalLoopRuntime::new(GoalLoopRuntimeConfig::default());
        let mut driver = MockGoalLoopDriver::new(vec![ambiguous_frame, watch_frame]);
        driver.focused_frames.push(focused_frame);

        let run = runtime
            .run_goal_loop_until_complete(goal, &mut driver)
            .await;

        assert_eq!(run.status, GoalLoopStatus::GoalAchieved);
        assert!(run.focused_perception_used);
        assert_eq!(run.focused_perception_requests.len(), 1);
        assert_eq!(
            run.focused_perception_requests[0].mode,
            PerceptionRequestMode::TargetFocus
        );
        assert_eq!(
            run.focused_perception_requests[0].routing_decision,
            PerceptionRoutingDecision::TargetRegionAnchor
        );
        assert_eq!(
            run.focused_perception_requests[0].target_item_id.as_deref(),
            Some("video")
        );
        assert_eq!(run.executed_steps.len(), 1);
    }

    #[test]
    fn regionless_target_prefers_visible_page_refinement_over_target_focus() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {
                    "content_provider_hint": "youtube",
                    "page_kind_hint": "search_results",
                    "result_list_visible": true,
                    "confidence": 0.9
                },
                "visible_result_items": [{
                    "item_id": "video",
                    "kind": "video",
                    "rank_within_kind": 1,
                    "title": "Shiva - Intro",
                    "confidence": 0.92
                }],
                "visible_entities": [{
                    "entity_id": "title_link",
                    "kind": "title_link",
                    "name": "Shiva - Intro",
                    "region": {
                        "x": 420,
                        "y": 240,
                        "width": 320,
                        "height": 42,
                        "coordinate_space": "screen"
                    },
                    "confidence": 0.87
                }]
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");
        let goal = goal_for_open_media_result(
            "aprimi il primo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );
        let planner_input = planner_contract_input(&goal, &frame, &[], &[], 3, 0);
        let step = plan_open_list_item(&goal, &frame);

        let request = focused_perception_request_for_step(
            &goal,
            &frame,
            &step,
            &planner_input.visible_actionability,
            0,
        )
        .expect("request");

        assert_eq!(step.kind, PlannerStepKind::ReplanAfterPerception);
        assert_eq!(request.mode, PerceptionRequestMode::VisiblePageRefinement);
        assert_eq!(
            request.routing_decision,
            PerceptionRoutingDecision::RegionlessTargetVisible
        );
        assert!(!request.target_region_anchor_present);
        assert_eq!(request.target_item_id.as_deref(), Some("video"));
        assert!(request.region.is_some());
    }

    #[tokio::test]
    async fn result_alias_frame_reaches_visible_refinement_path() {
        let ambiguous_frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {
                    "content_provider_hint": "youtube",
                    "page_kind_hint": "result",
                    "result_list_visible": true,
                    "query_hint": "shiva",
                    "confidence": 0.88
                },
                "scene_summary": "YouTube result page is visible",
                "visible_result_items": [{
                    "item_id": "video",
                    "kind": "video",
                    "rank_within_kind": 1,
                    "attributes": {"region": {"x": 100, "y": 200, "width": 500, "height": 180, "coordinate_space": "screen"}},
                    "confidence": 0.91
                }]
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("ambiguous frame");
        let focused_frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "result", "result_list_visible": true, "confidence": 0.91},
                "visible_result_items": [{
                    "item_id": "video",
                    "kind": "video",
                    "rank_within_kind": 1,
                    "click_regions": {"title": {"x": 450, "y": 300, "width": 400, "height": 50, "coordinate_space": "screen"}},
                    "confidence": 0.94
                }]
            }),
            1_500,
            None,
            None,
            Vec::new(),
        )
        .expect("focused frame");
        let watch_frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "watch_page", "confidence": 0.93}
            }),
            2_000,
            None,
            None,
            Vec::new(),
        )
        .expect("watch frame");
        let goal = goal_for_open_media_result(
            "aprimi il primo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );
        let runtime = GoalLoopRuntime::new(GoalLoopRuntimeConfig::default());
        let mut driver = MockGoalLoopDriver::new(vec![ambiguous_frame, watch_frame]);
        driver.focused_frames.push(focused_frame);

        let run = runtime
            .run_goal_loop_until_complete(goal, &mut driver)
            .await;

        assert_eq!(run.status, GoalLoopStatus::GoalAchieved);
        assert!(run.focused_perception_used);
        assert_eq!(
            run.planner_diagnostics[0].visibility_assessment,
            PlannerVisibilityAssessment::VisibleTargetNeedsClickRegion
        );
    }

    #[tokio::test]
    async fn visible_page_refinement_runs_before_offscreen_inference() {
        let ambiguous_frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "search_results", "result_list_visible": true, "confidence": 0.89},
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
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("ambiguous frame");
        let refined_frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "search_results", "result_list_visible": true, "confidence": 0.92},
                "visible_result_items": [{
                    "item_id": "mix",
                    "kind": "mix",
                    "rank_overall": 1,
                    "rank_within_kind": 1,
                    "click_regions": {"thumbnail": {"x": 100, "y": 100, "width": 300, "height": 180, "coordinate_space": "screen"}},
                    "confidence": 0.94
                }]
            }),
            1_500,
            None,
            None,
            Vec::new(),
        )
        .expect("refined frame");
        let goal = goal_for_open_media_result(
            "aprimi il primo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );
        let runtime = GoalLoopRuntime::new(GoalLoopRuntimeConfig::default());
        let mut driver = MockGoalLoopDriver::new(vec![ambiguous_frame]);
        driver.focused_frames.push(refined_frame);

        let run = runtime
            .run_goal_loop_until_complete(goal, &mut driver)
            .await;

        assert_eq!(run.status, GoalLoopStatus::ScrollRequiredButUnsupported);
        assert!(run.visible_refinement_used);
        assert_eq!(run.focused_perception_requests.len(), 1);
        assert_eq!(
            run.focused_perception_requests[0].mode,
            PerceptionRequestMode::VisiblePageRefinement
        );
        assert_eq!(
            run.planner_diagnostics[0].visibility_assessment,
            PlannerVisibilityAssessment::VisibleUnderGrounded
        );
        assert_eq!(
            run.planner_diagnostics[0].scroll_intent,
            PlannerScrollIntent::NotNeeded
        );
        assert_eq!(
            run.planner_diagnostics[1].visibility_assessment,
            PlannerVisibilityAssessment::LikelyOffscreen
        );
        assert_eq!(
            run.planner_diagnostics[1].scroll_intent,
            PlannerScrollIntent::RequiredButUnsupported
        );
    }

    #[tokio::test]
    async fn actionable_control_fallback_executes_after_regionless_refinement() {
        let ambiguous_frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {
                    "content_provider_hint": "youtube",
                    "page_kind_hint": "search_results",
                    "result_list_visible": true,
                    "confidence": 0.9
                },
                "visible_result_items": [{
                    "item_id": "video",
                    "kind": "video",
                    "rank_within_kind": 1,
                    "title": "Shiva - Intro",
                    "confidence": 0.92
                }],
                "visible_entities": [{
                    "entity_id": "title_link",
                    "kind": "title_link",
                    "name": "Shiva - Intro",
                    "region": {
                        "x": 420,
                        "y": 240,
                        "width": 320,
                        "height": 42,
                        "coordinate_space": "screen"
                    },
                    "confidence": 0.87
                }]
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("ambiguous frame");
        let refined_frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {
                    "content_provider_hint": "youtube",
                    "page_kind_hint": "search_results",
                    "result_list_visible": true,
                    "confidence": 0.92
                },
                "visible_result_items": [{
                    "item_id": "video",
                    "kind": "video",
                    "rank_within_kind": 1,
                    "title": "Shiva - Intro",
                    "confidence": 0.93
                }],
                "actionable_controls": [{
                    "control_id": "video_link",
                    "kind": "link",
                    "label": "Shiva - Intro",
                    "region": {
                        "x": 430,
                        "y": 242,
                        "width": 300,
                        "height": 40,
                        "coordinate_space": "screen"
                    },
                    "confidence": 0.94,
                    "attributes": {
                        "result_kind": "video",
                        "rank": 1
                    }
                }]
            }),
            1_500,
            None,
            None,
            Vec::new(),
        )
        .expect("refined frame");
        let watch_frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "watch_page", "confidence": 0.94}
            }),
            2_000,
            None,
            None,
            Vec::new(),
        )
        .expect("watch frame");
        let goal = goal_for_open_media_result(
            "aprimi il primo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );
        let runtime = GoalLoopRuntime::new(GoalLoopRuntimeConfig::default());
        let mut driver = MockGoalLoopDriver::new(vec![ambiguous_frame, watch_frame]);
        driver.focused_frames.push(refined_frame);

        let run = runtime
            .run_goal_loop_until_complete(goal, &mut driver)
            .await;

        assert_eq!(run.status, GoalLoopStatus::GoalAchieved);
        assert!(run.visible_refinement_used);
        assert_eq!(run.focused_perception_requests.len(), 1);
        assert_eq!(
            run.focused_perception_requests[0].mode,
            PerceptionRequestMode::VisiblePageRefinement
        );
        assert_eq!(
            run.focused_perception_requests[0].routing_decision,
            PerceptionRoutingDecision::RegionlessTargetVisible
        );
        assert!(!run.focused_perception_requests[0].target_region_anchor_present);
        assert_eq!(run.executed_steps.len(), 1);
        assert_eq!(
            run.selected_target_candidate
                .as_ref()
                .and_then(|candidate| candidate.observation_source.as_deref()),
            Some("planner_actionable_control_fallback")
        );
        assert_eq!(
            run.planner_diagnostics.last().and_then(|diagnostic| {
                diagnostic
                    .visible_actionability
                    .fallback_source_used
                    .clone()
            }),
            Some(ExecutableFallbackSource::ActionableControl)
        );
    }

    #[tokio::test]
    async fn loop_stops_truthfully_when_scroll_would_be_required_but_is_unsupported() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "search_results", "result_list_visible": true, "confidence": 0.92},
                "visible_result_items": [{
                    "item_id": "mix",
                    "kind": "mix",
                    "rank_overall": 1,
                    "rank_within_kind": 1,
                    "click_regions": {"thumbnail": {"x": 100, "y": 100, "width": 300, "height": 180, "coordinate_space": "screen"}},
                    "confidence": 0.94
                }]
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");
        let goal = goal_for_open_media_result(
            "aprimi il primo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );
        let runtime = GoalLoopRuntime::new(GoalLoopRuntimeConfig::default());
        let mut driver = MockGoalLoopDriver::new(vec![frame]);

        let run = runtime
            .run_goal_loop_until_complete(goal, &mut driver)
            .await;

        assert_eq!(run.status, GoalLoopStatus::ScrollRequiredButUnsupported);
        assert!(run.executed_steps.is_empty());
        let diagnostic = run.planner_diagnostics.last().expect("diagnostic");
        assert_eq!(
            diagnostic.visibility_assessment,
            PlannerVisibilityAssessment::LikelyOffscreen
        );
        assert_eq!(
            diagnostic.scroll_intent,
            PlannerScrollIntent::RequiredButUnsupported
        );
        assert_eq!(
            diagnostic.rejection_code,
            Some(PlannerRejectionReason::ScrollRequiredButUnsupported)
        );
    }

    #[test]
    fn provider_backend_separation_keeps_capture_backend_out_of_goal_verification() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {
                    "content_provider_hint": "youtube",
                    "page_kind_hint": "watch_page",
                    "capture_backend": "powershell_gdi",
                    "confidence": 0.93
                }
            }),
            2_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");
        let goal = goal_for_open_media_result(
            "aprimi il primo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );

        let verification = verify_goal_state(&goal, &frame, 1);

        assert_eq!(verification.status, GoalVerificationStatus::GoalAchieved);
        assert_eq!(
            frame.page_evidence.capture_backend.as_deref(),
            Some("powershell_gdi")
        );
    }

    #[test]
    fn deterministic_click_confidence_is_derived_when_item_confidence_is_missing() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {
                    "content_provider_hint": "youtube",
                    "page_kind_hint": "search_results",
                    "result_list_visible": true,
                    "confidence": 0.94
                },
                "visible_result_items": [{
                    "item_id": "video",
                    "kind": "video",
                    "title": "Shiva - Intro",
                    "rank_within_kind": 1,
                    "click_regions": {
                        "title": {
                            "region": {"x": 450, "y": 300, "width": 400, "height": 50, "coordinate_space": "screen"},
                            "confidence": 0.93
                        }
                    }
                }]
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");
        let goal = goal_for_open_media_result(
            "aprimi il primo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );
        let input = planner_contract_input(&goal, &frame, &[], &[], 3, 0);

        let decision = deterministic_planner_contract_decision(&input);

        assert_eq!(
            decision.proposed_step.kind,
            PlannerStepKind::ClickResultRegion
        );
        let confidence = decision
            .target_confidence
            .as_ref()
            .expect("target confidence");
        assert_eq!(confidence.raw_item_confidence, None);
        assert!(confidence.confidence_was_derived);
        assert!(confidence.accepted);
        assert!(decision.proposed_step.confidence >= MIN_PLANNER_CLICK_CONFIDENCE);
    }

    #[test]
    fn deterministic_click_still_refuses_when_multiple_signals_remain_weak() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {
                    "content_provider_hint": "youtube",
                    "page_kind_hint": "search_results",
                    "result_list_visible": true,
                    "confidence": 0.90
                },
                "visible_result_items": [{
                    "item_id": "video",
                    "kind": "video",
                    "rank_within_kind": 1,
                    "click_regions": {
                        "title": {
                            "region": {"x": 450, "y": 300, "width": 400, "height": 50, "coordinate_space": "screen"},
                            "confidence": 0.62
                        }
                    }
                }]
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");
        let goal = goal_for_open_media_result(
            "aprimi il primo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );
        let input = planner_contract_input(&goal, &frame, &[], &[], 3, 0);

        let decision = deterministic_planner_contract_decision(&input);

        assert_eq!(decision.proposed_step.kind, PlannerStepKind::Refuse);
        let confidence = decision
            .target_confidence
            .as_ref()
            .expect("target confidence");
        assert!(!confidence.accepted);
        assert!(decision.proposed_step.confidence < MIN_PLANNER_CLICK_CONFIDENCE);
    }

    #[test]
    fn valid_model_planner_result_click_is_bound_to_semantic_frame_candidate() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "search_results", "confidence": 0.92},
                "visible_result_items": [{
                    "item_id": "video",
                    "kind": "video",
                    "rank_within_kind": 1,
                    "click_regions": {"title": {"x": 450, "y": 300, "width": 400, "height": 50, "coordinate_space": "screen"}},
                    "confidence": 0.95
                }]
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");
        let goal = goal_for_open_media_result(
            "aprimi il primo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );
        let input = planner_contract_input(&goal, &frame, &[], &[], 3, 0);
        let deterministic = deterministic_planner_contract_decision(&input);
        let model_decision = PlannerContractDecision {
            source: PlannerContractSource::ModelAssisted,
            proposed_step: PlannerStep {
                step_id: "model_step".into(),
                kind: PlannerStepKind::ClickResultRegion,
                confidence: 0.91,
                rationale: "model selected visible video".into(),
                target_item_id: Some("video".into()),
                target_entity_id: None,
                click_region_key: Some("title".into()),
                executable_candidate: None,
                expected_state: Some("media_watch_page_visible".into()),
            },
            strategy_rationale: "model_first_video".into(),
            focused_perception_needed: false,
            replan_needed: false,
            expected_verification_target: Some("media_watch_page_visible".into()),
            planner_confidence: 0.91,
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
        };

        let accepted = validate_model_planner_decision(&input, model_decision, deterministic);

        assert_eq!(accepted.source, PlannerContractSource::ModelAssisted);
        assert!(accepted.accepted);
        assert!(!accepted.fallback_used);
        assert!(accepted.proposed_step.executable_candidate.is_some());
        assert_eq!(
            accepted
                .proposed_step
                .executable_candidate
                .as_ref()
                .and_then(|candidate| candidate.result_kind.as_deref()),
            Some("video")
        );
    }

    #[test]
    fn invalid_model_planner_target_falls_back_to_deterministic_decision() {
        let frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {"content_provider_hint": "youtube", "page_kind_hint": "search_results", "confidence": 0.92},
                "visible_result_items": [{
                    "item_id": "video",
                    "kind": "video",
                    "rank_within_kind": 1,
                    "click_regions": {"title": {"x": 450, "y": 300, "width": 400, "height": 50, "coordinate_space": "screen"}},
                    "confidence": 0.95
                }]
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("frame");
        let goal = goal_for_open_media_result(
            "aprimi il primo video",
            Some("youtube".into()),
            VisibleResultKind::Video,
            1,
        );
        let input = planner_contract_input(&goal, &frame, &[], &[], 3, 0);
        let deterministic = deterministic_planner_contract_decision(&input);
        let model_decision = PlannerContractDecision {
            source: PlannerContractSource::ModelAssisted,
            proposed_step: PlannerStep {
                step_id: "model_step".into(),
                kind: PlannerStepKind::ClickResultRegion,
                confidence: 0.93,
                rationale: "model fabricated target".into(),
                target_item_id: Some("not_in_frame".into()),
                target_entity_id: None,
                click_region_key: Some("title".into()),
                executable_candidate: None,
                expected_state: Some("media_watch_page_visible".into()),
            },
            strategy_rationale: "model_first_video".into(),
            focused_perception_needed: false,
            replan_needed: false,
            expected_verification_target: Some("media_watch_page_visible".into()),
            planner_confidence: 0.93,
            accepted: false,
            fallback_used: false,
            rejection_reason: None,
            decision_status: PlannerDecisionStatus::Accepted,
            rejection_code: None,
            visibility_assessment: PlannerVisibilityAssessment::Unknown,
            scroll_intent: PlannerScrollIntent::NotNeeded,
            visible_actionability: VisibleActionabilityDiagnostic::default(),
            target_confidence: None,
            normalized: false,
            downgraded: false,
        };

        let fallback = validate_model_planner_decision(&input, model_decision, deterministic);

        assert_eq!(
            fallback.source,
            PlannerContractSource::ModelAssistedFallback
        );
        assert!(fallback.fallback_used);
        assert!(fallback.rejection_reason.is_some());
        assert_eq!(
            fallback.proposed_step.target_item_id.as_deref(),
            Some("video")
        );
        assert!(fallback.proposed_step.executable_candidate.is_some());
    }
}
