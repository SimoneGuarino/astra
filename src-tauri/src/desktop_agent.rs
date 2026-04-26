use crate::{
    accessibility_layer::{
        synthesize_ranked_uia_result_candidates, validate_accessibility_target_selection,
        AccessibilitySnapshot, AccessibleElement,
    },
    action_policy::DesktopAgentPolicy,
    audit_log::AuditLogStore,
    browser_agent::BrowserAgent,
    capability_manifest::build_capability_manifest,
    contextual_learning::{
        store_verified_continuation, VerifiedContinuationLearningEvent, VerifiedContinuationOutcome,
    },
    desktop_agent_types::{
        BrowserHandoffActivationStatus, BrowserHandoffFailureReason, BrowserHandoffStatus,
        BrowserHandoffVerificationDiagnostic, BrowserVisualHandoffRecord,
        BrowserVisualHandoffResult, CapabilityManifest, ClickRegion, DesktopActionRequest,
        DesktopActionResponse, DesktopActionStatus, DesktopAuditEvent, FrameUncertainty,
        GoalConstraints, GoalLoopRun, GoalLoopStatus, GoalSpec, GoalType, GoalVerificationStatus,
        PendingApproval, PlannerContractDecision, PlannerContractInput, PlannerContractSource,
        PlannerDecisionStatus, PlannerRejectionReason, PlannerScrollIntent, PlannerStep,
        PlannerStepExecutionRecord, PlannerStepExecutionStatus, PlannerStepKind,
        PlannerVisibilityAssessment, PrimaryListItem, ScreenAnalysisRequest, ScreenAnalysisResult,
        ScreenCaptureResult, ScreenObservationStatus, SemanticScreenFrame, VisibleResultItem,
        VisibleResultKind,
    },
    filesystem_service::FilesystemService,
    model_assisted_planner::ModelAssistedPlanner,
    permissions::PermissionProfile,
    screen_workflow::{
        execute_screen_workflow, refresh_screen_workflow_plan, ScreenWorkflow, ScreenWorkflowRun,
        StepSupportStatus, StepVerificationStatus, WorkflowRunStatus, WorkflowStep,
        WorkflowStepKind, WorkflowStepRun,
    },
    semantic_frame::{
        goal_for_open_list_item, planner_candidate_from_step, run_goal_loop_once,
        verify_browser_handoff_page, verify_goal_state, GoalLoopDriver, GoalLoopDriverFuture,
        GoalLoopRuntime, GoalLoopRuntimeConfig,
    },
    structured_vision::StructuredVisionExtraction,
    terminal_runner::TerminalRunner,
    tools_registry::ToolsRegistry,
    ui_control::{
        UIControlRuntime, UIPrimitiveCapabilitySet, UIPrimitiveKind, UIPrimitiveRequest,
        UIPrimitiveResult, UIPrimitiveStatus,
    },
    ui_target_grounding::{
        TargetGroundingSource, TargetSelection, TargetSelectionDiagnostics, TargetSelectionPolicy,
        UITargetCandidate,
    },
    workflow_continuation::{
        build_context_from_action_response, build_context_from_screen_workflow_run,
        build_continuation_verification_result, resolve_workflow_continuation,
        resolve_workflow_continuation_with_model_params, scroll_policy_for_regrounding,
        validate_continuation_page, ContinuationRegroundingDiagnostics, FollowupActionKind,
        RecentWorkflowContext, RegroundingAttemptDiagnostic, ResultListItemKind,
        ScrollContinuationStatus, WorkflowContinuationResolution, MAX_RECENT_SCREEN_AGE_MS,
    },
};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex},
};
use tokio::time::{sleep, Duration};
use uuid::Uuid;

#[derive(Clone)]
pub struct DesktopAgentRuntime {
    policy: DesktopAgentPolicy,
    permissions: PermissionProfile,
    registry: ToolsRegistry,
    audit: AuditLogStore,
    pending: Arc<Mutex<HashMap<String, PendingActionEnvelope>>>,
    filesystem: FilesystemService,
    terminal: TerminalRunner,
    browser: BrowserAgent,
    pending_store: crate::pending_approvals_store::PendingApprovalsStore,
    screen_capture: crate::screen_capture::ScreenCaptureRuntime,
    screen_vision: crate::screen_vision::ScreenVisionRuntime,
    ui_control: UIControlRuntime,
    planner: ModelAssistedPlanner,
    recent_workflow_targets: Arc<Mutex<Vec<UITargetCandidate>>>,
    recent_workflow_context: Arc<Mutex<Option<RecentWorkflowContext>>>,
    recent_goal_loop: Arc<Mutex<Option<GoalLoopRun>>>,
}

#[derive(Clone)]
struct PendingActionEnvelope {
    approval: PendingApproval,
    request: DesktopActionRequest,
}

struct ProductionGoalLoopDriver<'a> {
    runtime: &'a DesktopAgentRuntime,
    capabilities: UIPrimitiveCapabilitySet,
    request_id: String,
    action_id: String,
    last_accessibility_snapshot: Option<AccessibilitySnapshot>,
}

struct VerifiedBrowserPage {
    frame: SemanticScreenFrame,
    verification: BrowserHandoffVerificationDiagnostic,
}

struct BrowserPageVerificationFailure {
    reason: String,
    verification: Option<BrowserHandoffVerificationDiagnostic>,
}

const BROWSER_HANDOFF_VERIFICATION_ATTEMPTS: usize = 2;
const BROWSER_HANDOFF_STABILIZATION_MS: u64 = 650;
const MAX_ACCESSIBILITY_SNAPSHOT_AGE_MS: u64 = 5_000;

impl<'a> ProductionGoalLoopDriver<'a> {
    fn new(
        runtime: &'a DesktopAgentRuntime,
        capabilities: UIPrimitiveCapabilitySet,
        request_id: String,
        action_id: String,
    ) -> Self {
        Self {
            runtime,
            capabilities,
            request_id,
            action_id,
            last_accessibility_snapshot: None,
        }
    }

    async fn verify_current_browser_page(
        &mut self,
        goal: &GoalSpec,
        attempt_label: &str,
    ) -> Result<VerifiedBrowserPage, BrowserPageVerificationFailure> {
        let capture = self
            .runtime
            .screen_capture
            .capture_snapshot()
            .map_err(|reason| BrowserPageVerificationFailure {
                reason,
                verification: None,
            })?;
        let accessibility_snapshot = self.capture_accessibility_snapshot_for_goal_loop();
        let mut frame = self
            .runtime
            .screen_vision
            .perceive_semantic_frame(
                &capture.image_path,
                capture.captured_at,
                &capture.provider,
                Some(&goal.utterance),
                None,
                goal.constraints.provider.as_deref(),
                accessibility_snapshot.as_ref(),
            )
            .await
            .map_err(|reason| BrowserPageVerificationFailure {
                reason,
                verification: None,
            })?;
        if let Some(snapshot) = accessibility_snapshot.as_ref() {
            enrich_frame_with_accessibility(&mut frame, snapshot);
        }
        frame.page_evidence.observation_source = Some(attempt_label.into());
        let verification = verify_browser_handoff_page(goal, &frame).map_err(|verification| {
            BrowserPageVerificationFailure {
                reason: verification
                    .reason
                    .clone()
                    .unwrap_or_else(|| "browser page verification failed".into()),
                verification: Some(verification),
            }
        })?;
        Ok(VerifiedBrowserPage {
            frame,
            verification,
        })
    }

    fn capture_accessibility_snapshot_for_goal_loop(&mut self) -> Option<AccessibilitySnapshot> {
        if !self.runtime.policy.accessibility_snapshot_enabled() {
            return None;
        }
        let snapshot = crate::accessibility_layer::capture_accessibility_snapshot(&[
            "chrome", "msedge", "firefox", "brave", "opera",
        ]);
        if snapshot.elements.is_empty() {
            return None;
        }
        self.last_accessibility_snapshot = Some(snapshot.clone());
        Some(snapshot)
    }

    async fn plan_accessibility_target_selection(
        &self,
        input: &PlannerContractInput,
    ) -> Option<PlannerContractDecision> {
        if !matches!(
            input.goal.goal_type,
            GoalType::OpenListItem | GoalType::OpenMediaResult
        ) {
            return None;
        }
        let snapshot = self.last_accessibility_snapshot.as_ref()?;
        let candidates = synthesize_ranked_uia_result_candidates(snapshot);
        if candidates.is_empty() {
            return None;
        }
        let selector_result = self
            .runtime
            .screen_vision
            .select_accessibility_target_for_goal(
                &input.goal,
                None,
                Some(snapshot),
                Some(&input.current_frame),
                &candidates,
            )
            .await;

        match selector_result {
            Ok(selection) => match validate_accessibility_target_selection(
                &selection,
                snapshot,
                &candidates,
                TargetSelectionPolicy::default().min_click_confidence,
            ) {
                Ok(candidate) => Some(accessibility_selector_click_decision(
                    input,
                    candidate,
                    &selection,
                    candidates.len(),
                    true,
                )),
                Err(reason) => Some(accessibility_selector_suppression_decision(
                    input,
                    candidates.len(),
                    true,
                    selection.selected_element_id.clone(),
                    Some(selection.confidence),
                    format!("rust_element_validation_status=failed: {reason}"),
                    PlannerRejectionReason::FabricatedTarget,
                )),
            },
            Err(error) => {
                if let Some(candidate) = deterministic_uia_candidate_if_unambiguous(&candidates) {
                    return Some(accessibility_selector_click_decision(
                        input,
                        candidate,
                        &crate::accessibility_layer::AccessibilityTargetSelection {
                            selected_element_id: candidates
                                .first()
                                .and_then(|candidate| candidate.element_id.clone()),
                            accessibility_snapshot_id: Some(snapshot.snapshot_id.clone()),
                            selection_kind: Some("deterministic_unambiguous_uia".into()),
                            rank: candidates.first().and_then(|candidate| candidate.rank),
                            confidence: candidates
                                .first()
                                .map(|candidate| candidate.confidence)
                                .unwrap_or(0.0),
                            rationale: Some(
                                "single current UIA candidate available; deterministic UIA fallback selected it after selector failure"
                                    .into(),
                            ),
                        },
                        candidates.len(),
                        false,
                    ));
                }
                Some(accessibility_selector_suppression_decision(
                    input,
                    candidates.len(),
                    true,
                    None,
                    None,
                    format!("llm_uia_selector_unavailable: {error}"),
                    PlannerRejectionReason::ModelUnavailable,
                ))
            }
        }
    }

    fn expected_browser_app(&self, goal: &GoalSpec) -> String {
        self.runtime
            .recent_workflow_context()
            .and_then(|context| context.app)
            .filter(|app| !app.trim().is_empty())
            .unwrap_or_else(|| {
                if goal.constraints.provider.is_some() {
                    "browser".into()
                } else {
                    "chrome".into()
                }
            })
    }

    fn handoff_record(
        &self,
        iteration: usize,
        status: BrowserHandoffStatus,
        activation_status: BrowserHandoffActivationStatus,
        failure_reason: Option<BrowserHandoffFailureReason>,
        goal: &GoalSpec,
        activation_attempted: bool,
        frame: Option<&SemanticScreenFrame>,
        verification: Option<&BrowserHandoffVerificationDiagnostic>,
        attempts: usize,
        reason: impl Into<String>,
    ) -> BrowserVisualHandoffRecord {
        BrowserVisualHandoffRecord {
            iteration,
            status,
            activation_status,
            failure_reason,
            app_hint: Some(self.expected_browser_app(goal)),
            provider_hint: goal.constraints.provider.clone(),
            page_kind_hint: frame
                .and_then(|frame| frame.page_evidence.page_kind_hint.clone())
                .or_else(|| {
                    verification.and_then(|verification| verification.raw_page_kind_hint.clone())
                }),
            verification: verification.cloned(),
            activation_attempted,
            page_verified: frame.is_some(),
            frame_id: frame.map(|frame| frame.frame_id.clone()),
            confidence: frame.map(|frame| frame.page_evidence.confidence),
            attempts,
            reason: Some(reason.into()),
        }
    }
}

impl GoalLoopDriver for ProductionGoalLoopDriver<'_> {
    fn prepare_visual_handoff<'a>(
        &'a mut self,
        goal: &'a GoalSpec,
        iteration: usize,
    ) -> GoalLoopDriverFuture<'a, Result<Option<BrowserVisualHandoffResult>, String>> {
        Box::pin(async move {
            if goal.constraints.provider.is_none()
                && self.runtime.recent_workflow_context().is_none()
            {
                return Ok(None);
            }
            if !self
                .runtime
                .permissions
                .allows(&crate::desktop_agent_types::Permission::DesktopObserve)
                || !self
                    .runtime
                    .policy
                    .permission_enabled(&crate::desktop_agent_types::Permission::DesktopObserve)
            {
                let record = self.handoff_record(
                    iteration,
                    BrowserHandoffStatus::SemanticFrameUnavailable,
                    BrowserHandoffActivationStatus::NotAttempted,
                    Some(BrowserHandoffFailureReason::PermissionDenied),
                    goal,
                    false,
                    None,
                    None,
                    0,
                    "desktop observation permission is required before browser visual handoff",
                );
                return Ok(Some(BrowserVisualHandoffResult {
                    record,
                    verified_frame: None,
                }));
            }

            match self
                .verify_current_browser_page(goal, "browser_handoff_pre_activation_check")
                .await
            {
                Ok(verified) => {
                    let record = self.handoff_record(
                        iteration,
                        BrowserHandoffStatus::VisuallyVerified,
                        BrowserHandoffActivationStatus::NotAttempted,
                        None,
                        goal,
                        false,
                        Some(&verified.frame),
                        Some(&verified.verification),
                        1,
                        "current visible page already matches the expected browser context",
                    );
                    return Ok(Some(BrowserVisualHandoffResult {
                        record,
                        verified_frame: Some(verified.frame),
                    }));
                }
                Err(_) => {}
            }

            let activation_request = UIPrimitiveRequest {
                primitive: UIPrimitiveKind::ActivateWindowOrApp,
                value: None,
                target: json!({
                    "app": self.expected_browser_app(goal),
                    "provider": goal.constraints.provider.clone(),
                    "url": self.runtime.recent_workflow_context().and_then(|context| context.url),
                }),
                reason: Some(
                    "Goal loop requires the expected browser page in the visual foreground before semantic continuation."
                        .into(),
                ),
            };
            self.runtime.audit.append(&DesktopAuditEvent {
                audit_id: Uuid::new_v4().to_string(),
                action_id: self.action_id.clone(),
                request_id: self.request_id.clone(),
                tool_name: "goal_loop.browser_handoff".into(),
                stage: "activation".into(),
                status: "started".into(),
                timestamp: now_ms(),
                risk_level: crate::desktop_agent_types::RiskLevel::Medium,
                details: json!({
                    "goal": goal,
                    "primitive": activation_request.primitive,
                    "target": activation_request.target,
                }),
            });
            let activation = self
                .runtime
                .ui_control
                .execute(&activation_request, &self.capabilities);
            self.runtime.audit.append(&DesktopAuditEvent {
                audit_id: Uuid::new_v4().to_string(),
                action_id: self.action_id.clone(),
                request_id: self.request_id.clone(),
                tool_name: "goal_loop.browser_handoff".into(),
                stage: "activation".into(),
                status: format!("{:?}", activation.status).to_ascii_lowercase(),
                timestamp: now_ms(),
                risk_level: crate::desktop_agent_types::RiskLevel::Medium,
                details: json!({"primitive_result": activation}),
            });

            let activation_status = match activation.status {
                UIPrimitiveStatus::Executed => BrowserHandoffActivationStatus::Executed,
                UIPrimitiveStatus::Unsupported => {
                    let record = self.handoff_record(
                        iteration,
                        BrowserHandoffStatus::ActivationUnsupported,
                        BrowserHandoffActivationStatus::Unsupported,
                        Some(BrowserHandoffFailureReason::BrowserActivationUnsupported),
                        goal,
                        true,
                        None,
                        None,
                        1,
                        activation.message,
                    );
                    return Ok(Some(BrowserVisualHandoffResult {
                        record,
                        verified_frame: None,
                    }));
                }
                UIPrimitiveStatus::Failed => {
                    let record = self.handoff_record(
                        iteration,
                        BrowserHandoffStatus::ActivationFailed,
                        BrowserHandoffActivationStatus::Failed,
                        Some(BrowserHandoffFailureReason::BrowserActivationFailed),
                        goal,
                        true,
                        None,
                        None,
                        1,
                        activation.message,
                    );
                    return Ok(Some(BrowserVisualHandoffResult {
                        record,
                        verified_frame: None,
                    }));
                }
            };

            sleep(Duration::from_millis(BROWSER_HANDOFF_STABILIZATION_MS)).await;
            let mut last_error = None;
            for attempt in 0..BROWSER_HANDOFF_VERIFICATION_ATTEMPTS {
                match self
                    .verify_current_browser_page(goal, "browser_handoff_post_activation_verify")
                    .await
                {
                    Ok(verified) => {
                        let record = self.handoff_record(
                            iteration,
                            BrowserHandoffStatus::VisuallyVerified,
                            activation_status,
                            None,
                            goal,
                            true,
                            Some(&verified.frame),
                            Some(&verified.verification),
                            attempt + 1,
                            "browser foreground activation succeeded and page context was visually verified",
                        );
                        return Ok(Some(BrowserVisualHandoffResult {
                            record,
                            verified_frame: Some(verified.frame),
                        }));
                    }
                    Err(error) => {
                        last_error = Some(error);
                        sleep(Duration::from_millis(250)).await;
                    }
                }
            }

            let verification = last_error
                .as_ref()
                .and_then(|failure| failure.verification.clone());
            let failure_reason = last_error
                .as_ref()
                .map(|failure| failure.reason.clone())
                .unwrap_or_else(|| "browser page did not verify after activation".into());
            let record = self.handoff_record(
                iteration,
                BrowserHandoffStatus::PageNotVerified,
                activation_status,
                Some(BrowserHandoffFailureReason::BrowserPageNotVerified),
                goal,
                true,
                None,
                verification.as_ref(),
                BROWSER_HANDOFF_VERIFICATION_ATTEMPTS,
                failure_reason,
            );
            Ok(Some(BrowserVisualHandoffResult {
                record,
                verified_frame: None,
            }))
        })
    }

    fn perceive<'a>(
        &'a mut self,
        goal: &'a GoalSpec,
        iteration: usize,
        fresh_capture_required: bool,
    ) -> GoalLoopDriverFuture<'a, Result<SemanticScreenFrame, String>> {
        Box::pin(async move {
            if !self
                .runtime
                .permissions
                .allows(&crate::desktop_agent_types::Permission::DesktopObserve)
                || !self
                    .runtime
                    .policy
                    .permission_enabled(&crate::desktop_agent_types::Permission::DesktopObserve)
            {
                return Err("Permission denied for desktop observation".into());
            }
            let capture = self
                .runtime
                .capture_for_structured_grounding_with_policy(fresh_capture_required)?;
            let accessibility_snapshot = self.capture_accessibility_snapshot_for_goal_loop();
            self.runtime
                .screen_vision
                .perceive_semantic_frame(
                    &capture.image_path,
                    capture.captured_at,
                    &capture.provider,
                    Some(&goal.utterance),
                    None,
                    goal.constraints.provider.as_deref(),
                    accessibility_snapshot.as_ref(),
                )
                .await
                .map(|mut frame| {
                    if let Some(snapshot) = accessibility_snapshot.as_ref() {
                        enrich_frame_with_accessibility(&mut frame, snapshot);
                    }
                    frame.page_evidence.observation_source = Some(if fresh_capture_required {
                        format!("goal_loop_perception_iteration_{}_fresh", iteration + 1)
                    } else {
                        format!("goal_loop_perception_iteration_{}", iteration + 1)
                    });
                    frame
                })
        })
    }

    fn execute_planner_step<'a>(
        &'a mut self,
        step: &'a PlannerStep,
    ) -> GoalLoopDriverFuture<'a, PlannerStepExecutionRecord> {
        Box::pin(async move {
            let Some(mut request) = planner_step_primitive_request(step) else {
                return PlannerStepExecutionRecord {
                    step_id: step.step_id.clone(),
                    status: PlannerStepExecutionStatus::Unsupported,
                    primitive: "none".into(),
                    message: "planner step cannot be converted into a governed UI primitive".into(),
                    selected_target_candidate: step.executable_candidate.clone(),
                    geometry: None,
                    fresh_capture_required: false,
                    fresh_capture_used: false,
                    target_signature: None,
                };
            };
            if let Some(target) = request.target.as_object_mut() {
                if step
                    .executable_candidate
                    .as_ref()
                    .is_some_and(candidate_is_accessibility_sourced)
                {
                    target.insert("accessibility_sourced".into(), json!(true));
                    target.insert("observation_source".into(), json!("uia_snapshot"));
                    target.insert("source".into(), json!("accessibility_layer"));
                }
                let browser_app = target
                    .get("browser_app_hint")
                    .and_then(Value::as_str)
                    .filter(|value| !value.trim().is_empty())
                    .map(ToOwned::to_owned)
                    .or_else(|| {
                        self.runtime
                            .recent_workflow_context()
                            .and_then(|context| context.app)
                    })
                    .unwrap_or_else(|| "browser".into());
                target
                    .entry("browser_app_hint")
                    .or_insert_with(|| json!(browser_app.clone()));
                target.entry("app").or_insert_with(|| json!(browser_app));
            }

            self.runtime.audit.append(&DesktopAuditEvent {
                audit_id: Uuid::new_v4().to_string(),
                action_id: self.action_id.clone(),
                request_id: self.request_id.clone(),
                tool_name: "goal_loop.primitive".into(),
                stage: "execution".into(),
                status: "started".into(),
                timestamp: now_ms(),
                risk_level: crate::desktop_agent_types::RiskLevel::Medium,
                details: json!({
                    "planner_step": step,
                    "primitive": request.primitive,
                }),
            });

            let mut result = self
                .runtime
                .ui_control
                .execute(&request, &self.capabilities);
            if step
                .executable_candidate
                .as_ref()
                .is_some_and(candidate_is_accessibility_sourced)
            {
                if let Some(geometry) = result.geometry.as_mut() {
                    geometry.accessibility_sourced = true;
                }
            }
            let status = match result.status {
                UIPrimitiveStatus::Executed => PlannerStepExecutionStatus::Executed,
                UIPrimitiveStatus::Unsupported => PlannerStepExecutionStatus::Unsupported,
                UIPrimitiveStatus::Failed => PlannerStepExecutionStatus::Failed,
            };
            self.runtime.audit.append(&DesktopAuditEvent {
                audit_id: Uuid::new_v4().to_string(),
                action_id: self.action_id.clone(),
                request_id: self.request_id.clone(),
                tool_name: "goal_loop.primitive".into(),
                stage: "execution".into(),
                status: format!("{:?}", result.status).to_ascii_lowercase(),
                timestamp: now_ms(),
                risk_level: crate::desktop_agent_types::RiskLevel::Medium,
                details: json!({
                    "planner_step": step,
                    "primitive_result": result,
                }),
            });

            PlannerStepExecutionRecord {
                step_id: step.step_id.clone(),
                status,
                primitive: format!("{:?}", request.primitive),
                message: result.message,
                selected_target_candidate: step.executable_candidate.clone(),
                geometry: result.geometry,
                fresh_capture_required: false,
                fresh_capture_used: false,
                target_signature: step
                    .executable_candidate
                    .as_ref()
                    .map(|candidate| candidate.candidate_id.clone()),
            }
        })
    }

    fn focused_perception<'a>(
        &'a mut self,
        request: &'a crate::desktop_agent_types::FocusedPerceptionRequest,
    ) -> GoalLoopDriverFuture<'a, Result<Option<SemanticScreenFrame>, String>> {
        Box::pin(async move {
            if !self
                .runtime
                .permissions
                .allows(&crate::desktop_agent_types::Permission::DesktopObserve)
                || !self
                    .runtime
                    .policy
                    .permission_enabled(&crate::desktop_agent_types::Permission::DesktopObserve)
            {
                return Err("Permission denied for desktop observation".into());
            }
            let capture = self.runtime.capture_for_structured_grounding()?;
            let app_hint = request
                .verified_surface
                .as_ref()
                .and_then(|surface| surface.app_hint.as_deref());
            let provider_hint = request
                .verified_surface
                .as_ref()
                .and_then(|surface| surface.provider_hint.as_deref());
            self.runtime
                .screen_vision
                .perceive_focused_region(
                    &capture.image_path,
                    capture.captured_at,
                    &capture.provider,
                    request,
                    Some(&request.reason),
                    app_hint,
                    provider_hint,
                )
                .await
                .map(Some)
        })
    }

    fn recover_browser_surface<'a>(
        &'a mut self,
        goal: &'a GoalSpec,
        iteration: usize,
        reason: &'a str,
    ) -> GoalLoopDriverFuture<'a, Result<Option<BrowserVisualHandoffResult>, String>> {
        Box::pin(async move {
            if goal.constraints.provider.is_none()
                && self.runtime.recent_workflow_context().is_none()
            {
                return Ok(None);
            }
            if !self
                .runtime
                .permissions
                .allows(&crate::desktop_agent_types::Permission::DesktopObserve)
                || !self
                    .runtime
                    .policy
                    .permission_enabled(&crate::desktop_agent_types::Permission::DesktopObserve)
            {
                let record = self.handoff_record(
                    iteration,
                    BrowserHandoffStatus::SemanticFrameUnavailable,
                    BrowserHandoffActivationStatus::NotAttempted,
                    Some(BrowserHandoffFailureReason::PermissionDenied),
                    goal,
                    false,
                    None,
                    None,
                    0,
                    format!(
                        "desktop observation permission is required before browser recovery: {reason}"
                    ),
                );
                return Ok(Some(BrowserVisualHandoffResult {
                    record,
                    verified_frame: None,
                }));
            }

            match self
                .verify_current_browser_page(goal, "browser_recovery_pre_activation_check")
                .await
            {
                Ok(verified) => {
                    let record = self.handoff_record(
                        iteration,
                        BrowserHandoffStatus::VisuallyVerified,
                        BrowserHandoffActivationStatus::NotAttempted,
                        None,
                        goal,
                        false,
                        Some(&verified.frame),
                        Some(&verified.verification),
                        1,
                        format!(
                            "browser recovery found the expected page already visible before foreground activation: {reason}"
                        ),
                    );
                    return Ok(Some(BrowserVisualHandoffResult {
                        record,
                        verified_frame: Some(verified.frame),
                    }));
                }
                Err(_) => {}
            }

            self.runtime.audit.append(&DesktopAuditEvent {
                audit_id: Uuid::new_v4().to_string(),
                action_id: self.action_id.clone(),
                request_id: self.request_id.clone(),
                tool_name: "goal_loop.browser_recovery".into(),
                stage: "activation".into(),
                status: "started".into(),
                timestamp: now_ms(),
                risk_level: crate::desktop_agent_types::RiskLevel::Medium,
                details: json!({
                    "goal": goal,
                    "reason": reason,
                    "expected_browser_app": self.expected_browser_app(goal),
                }),
            });

            let activation_request = UIPrimitiveRequest {
                primitive: UIPrimitiveKind::ActivateWindowOrApp,
                value: None,
                target: json!({
                    "app": self.expected_browser_app(goal),
                    "provider": goal.constraints.provider.clone(),
                    "url": self.runtime.recent_workflow_context().and_then(|context| context.url),
                }),
                reason: Some(format!(
                    "Goal loop browser recovery requested after `{reason}` to restore the expected browser interaction surface."
                )),
            };
            let activation = self
                .runtime
                .ui_control
                .execute(&activation_request, &self.capabilities);
            self.runtime.audit.append(&DesktopAuditEvent {
                audit_id: Uuid::new_v4().to_string(),
                action_id: self.action_id.clone(),
                request_id: self.request_id.clone(),
                tool_name: "goal_loop.browser_recovery".into(),
                stage: "activation".into(),
                status: format!("{:?}", activation.status).to_ascii_lowercase(),
                timestamp: now_ms(),
                risk_level: crate::desktop_agent_types::RiskLevel::Medium,
                details: json!({
                    "reason": reason,
                    "primitive_result": activation,
                }),
            });

            let activation_status = match activation.status {
                UIPrimitiveStatus::Executed => BrowserHandoffActivationStatus::Executed,
                UIPrimitiveStatus::Unsupported => {
                    let record = self.handoff_record(
                        iteration,
                        BrowserHandoffStatus::ActivationUnsupported,
                        BrowserHandoffActivationStatus::Unsupported,
                        Some(BrowserHandoffFailureReason::BrowserActivationUnsupported),
                        goal,
                        true,
                        None,
                        None,
                        1,
                        format!(
                            "browser recovery could not reactivate the expected browser surface: {}",
                            activation.message
                        ),
                    );
                    return Ok(Some(BrowserVisualHandoffResult {
                        record,
                        verified_frame: None,
                    }));
                }
                UIPrimitiveStatus::Failed => {
                    let record = self.handoff_record(
                        iteration,
                        BrowserHandoffStatus::ActivationFailed,
                        BrowserHandoffActivationStatus::Failed,
                        Some(BrowserHandoffFailureReason::BrowserActivationFailed),
                        goal,
                        true,
                        None,
                        None,
                        1,
                        format!(
                            "browser recovery activation failed after `{reason}`: {}",
                            activation.message
                        ),
                    );
                    return Ok(Some(BrowserVisualHandoffResult {
                        record,
                        verified_frame: None,
                    }));
                }
            };

            sleep(Duration::from_millis(BROWSER_HANDOFF_STABILIZATION_MS)).await;
            let mut last_error = None;
            for attempt in 0..BROWSER_HANDOFF_VERIFICATION_ATTEMPTS {
                match self
                    .verify_current_browser_page(goal, "browser_recovery_post_activation_verify")
                    .await
                {
                    Ok(verified) => {
                        let record = self.handoff_record(
                            iteration,
                            BrowserHandoffStatus::VisuallyVerified,
                            activation_status,
                            None,
                            goal,
                            true,
                            Some(&verified.frame),
                            Some(&verified.verification),
                            attempt + 1,
                            format!(
                                "browser recovery succeeded after `{reason}` and the visible page context verified again"
                            ),
                        );
                        return Ok(Some(BrowserVisualHandoffResult {
                            record,
                            verified_frame: Some(verified.frame),
                        }));
                    }
                    Err(error) => {
                        last_error = Some(error);
                        sleep(Duration::from_millis(250)).await;
                    }
                }
            }

            let verification = last_error
                .as_ref()
                .and_then(|failure| failure.verification.clone());
            let failure_reason = last_error
                .as_ref()
                .map(|failure| failure.reason.clone())
                .unwrap_or_else(|| "browser recovery did not verify the expected page".into());
            let record = self.handoff_record(
                iteration,
                BrowserHandoffStatus::PageNotVerified,
                activation_status,
                Some(BrowserHandoffFailureReason::BrowserPageNotVerified),
                goal,
                true,
                None,
                verification.as_ref(),
                BROWSER_HANDOFF_VERIFICATION_ATTEMPTS,
                failure_reason,
            );
            Ok(Some(BrowserVisualHandoffResult {
                record,
                verified_frame: None,
            }))
        })
    }

    fn plan<'a>(
        &'a mut self,
        input: &'a PlannerContractInput,
    ) -> GoalLoopDriverFuture<'a, Result<Option<PlannerContractDecision>, String>> {
        Box::pin(async move {
            if let Some(decision) = self.plan_accessibility_target_selection(input).await {
                return Ok(Some(decision));
            }
            self.runtime.planner.plan(input).await
        })
    }
}

impl DesktopAgentRuntime {
    pub fn new(project_root: PathBuf) -> Self {
        let policy = DesktopAgentPolicy::load_or_default(&project_root);
        let permissions = PermissionProfile::default_local_agent();
        let audit = AuditLogStore::new(&project_root);
        let pending_store =
            crate::pending_approvals_store::PendingApprovalsStore::new(&project_root);
        let pending_entries = pending_store.load();
        let pending = pending_entries
            .into_iter()
            .map(|(action_id, approval)| {
                let request = DesktopActionRequest {
                    tool_name: approval.tool_name.clone(),
                    params: approval.params.clone(),
                    preview_only: false,
                    reason: Some(approval.reason.clone()),
                };
                (action_id, PendingActionEnvelope { approval, request })
            })
            .collect::<HashMap<_, _>>();
        let screen_capture = crate::screen_capture::ScreenCaptureRuntime::new(&project_root);
        let screen_vision = crate::screen_vision::ScreenVisionRuntime::new();
        let ui_control = UIControlRuntime::new();
        Self {
            filesystem: FilesystemService::new(&policy.allowed_roots),
            terminal: TerminalRunner::new(&policy.terminal_allowed_commands, &policy.allowed_roots),
            browser: BrowserAgent,
            registry: ToolsRegistry::new(),
            audit,
            policy,
            permissions,
            pending: Arc::new(Mutex::new(pending)),
            pending_store,
            screen_capture,
            screen_vision,
            ui_control,
            planner: ModelAssistedPlanner::new(),
            recent_workflow_targets: Arc::new(Mutex::new(Vec::new())),
            recent_workflow_context: Arc::new(Mutex::new(None)),
            recent_goal_loop: Arc::new(Mutex::new(None)),
        }
    }

    pub fn list_tools(&self) -> Vec<crate::desktop_agent_types::ToolDescriptor> {
        self.registry.list()
    }

    pub async fn capability_manifest(&self) -> CapabilityManifest {
        let tools = self.registry.list();
        let policy = self.policy.snapshot(&self.permissions);
        let screen_status = self.screen_capture.status();
        let pending_approvals = self.pending_approvals();
        let vision = self.screen_vision.availability().await;
        build_capability_manifest(&tools, &policy, &screen_status, &pending_approvals, &vision)
    }
    pub fn policy_snapshot(&self) -> crate::desktop_agent_types::DesktopPolicySnapshot {
        self.policy.snapshot(&self.permissions)
    }
    pub fn recent_audit_events(&self, limit: usize) -> Vec<DesktopAuditEvent> {
        self.audit.tail(limit)
    }
    pub fn screen_status(&self) -> ScreenObservationStatus {
        self.screen_capture.status()
    }
    pub fn recent_goal_loop(&self) -> Option<GoalLoopRun> {
        self.recent_goal_loop
            .lock()
            .ok()
            .and_then(|memory| memory.clone())
    }
    pub fn set_screen_observation_enabled(&self, enabled: bool) -> ScreenObservationStatus {
        self.screen_capture.set_enabled(enabled)
    }
    pub fn capture_screen_snapshot(&self) -> Result<ScreenCaptureResult, String> {
        self.screen_capture.capture_snapshot()
    }

    pub async fn analyze_screen(
        &self,
        request: ScreenAnalysisRequest,
    ) -> Result<ScreenAnalysisResult, String> {
        if !self
            .permissions
            .allows(&crate::desktop_agent_types::Permission::DesktopObserve)
            || !self
                .policy
                .permission_enabled(&crate::desktop_agent_types::Permission::DesktopObserve)
        {
            return Err("Permission denied for desktop observation".into());
        }

        let capture = if request.capture_fresh
            || self.screen_capture.latest_capture_path().is_none()
        {
            if !self.screen_capture.status().enabled {
                return Err("Screen observation is disabled. Enable observation before capturing a fresh screen snapshot.".into());
            }
            self.screen_capture.capture_snapshot()?
        } else {
            let status = self.screen_capture.status();
            let path = status.last_capture_path.ok_or_else(|| {
                "No screen capture available yet. Capture a snapshot first.".to_string()
            })?;
            let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            ScreenCaptureResult {
                capture_id: Uuid::new_v4().to_string(),
                captured_at: status.last_frame_at.unwrap_or_else(now_ms),
                image_path: path,
                width: None,
                height: None,
                bytes,
                provider: status.provider,
            }
        };

        let question = request.question.clone();
        let action_id = Uuid::new_v4().to_string();
        let request_id = Uuid::new_v4().to_string();
        self.audit.append(&DesktopAuditEvent {
            audit_id: Uuid::new_v4().to_string(),
            action_id: action_id.clone(),
            request_id: request_id.clone(),
            tool_name: "screen.analyze".into(),
            stage: "execution".into(),
            status: "started".into(),
            timestamp: now_ms(),
            risk_level: crate::desktop_agent_types::RiskLevel::Low,
            details: json!({"image_path": capture.image_path, "capture_fresh": request.capture_fresh, "question": question}),
        });

        let result = self
            .screen_vision
            .analyze(
                &capture.image_path,
                capture.captured_at,
                &capture.provider,
                question,
            )
            .await;

        match result {
            Ok(result) => {
                self.audit.append(&DesktopAuditEvent {
                    audit_id: Uuid::new_v4().to_string(),
                    action_id,
                    request_id,
                    tool_name: "screen.analyze".into(),
                    stage: "execution".into(),
                    status: "completed".into(),
                    timestamp: now_ms(),
                    risk_level: crate::desktop_agent_types::RiskLevel::Low,
                    details: json!({
                        "model": result.model,
                        "provider": result.provider,
                        "image_path": result.image_path,
                        "question": result.question,
                        "ui_candidate_count": result.ui_candidates.len(),
                        "structured_candidates_error": result.structured_candidates_error,
                    }),
                });
                Ok(result)
            }
            Err(error) => {
                self.audit.append(&DesktopAuditEvent {
                    audit_id: Uuid::new_v4().to_string(),
                    action_id,
                    request_id,
                    tool_name: "screen.analyze".into(),
                    stage: "execution".into(),
                    status: "failed".into(),
                    timestamp: now_ms(),
                    risk_level: crate::desktop_agent_types::RiskLevel::Low,
                    details: json!({"error": error}),
                });
                Err(error)
            }
        }
    }

    pub fn ui_primitive_capabilities(&self) -> UIPrimitiveCapabilitySet {
        let enabled = self
            .permissions
            .allows(&crate::desktop_agent_types::Permission::DesktopControl)
            && self
                .policy
                .permission_enabled(&crate::desktop_agent_types::Permission::DesktopControl);
        self.ui_control.capabilities(enabled)
    }

    pub async fn execute_screen_workflow(&self, mut workflow: ScreenWorkflow) -> ScreenWorkflowRun {
        let request_id = Uuid::new_v4().to_string();
        let action_id = Uuid::new_v4().to_string();
        let capabilities = self.ui_primitive_capabilities();
        let previous_context = self.recent_workflow_context();
        workflow.grounding.recent_target_candidates = self.recent_workflow_targets();
        if workflow
            .continuation
            .as_ref()
            .is_some_and(|descriptor| !descriptor.policy.recent_candidate_reuse_allowed)
        {
            workflow.grounding.recent_target_candidates.clear();
            workflow.grounding.uncertainty.push(
                "recent_target_candidates_suppressed_for_fresh_continuation_grounding".into(),
            );
        }
        refresh_screen_workflow_plan(&mut workflow);
        let production_goal = goal_for_screen_workflow(&workflow);
        let used_production_goal_loop = production_goal.is_some();
        if production_goal.is_none() {
            self.prepare_continuation_grounding(&mut workflow, &capabilities)
                .await;
        }
        self.audit.append(&DesktopAuditEvent {
            audit_id: Uuid::new_v4().to_string(),
            action_id: action_id.clone(),
            request_id: request_id.clone(),
            tool_name: "screen.workflow".into(),
            stage: "workflow".into(),
            status: "started".into(),
            timestamp: now_ms(),
            risk_level: crate::desktop_agent_types::RiskLevel::Medium,
            details: json!({
                "workflow": workflow.diagnostic_value(),
                "primitive_capabilities": capabilities,
            }),
        });

        let run = if let Some(goal) = production_goal {
            let runtime = GoalLoopRuntime::new(GoalLoopRuntimeConfig::default());
            let mut driver = ProductionGoalLoopDriver::new(
                self,
                capabilities.clone(),
                request_id.clone(),
                action_id.clone(),
            );
            let goal_loop = runtime
                .run_goal_loop_until_complete(goal, &mut driver)
                .await;
            self.store_recent_goal_loop(goal_loop.clone());
            screen_workflow_run_from_goal_loop(workflow, capabilities, goal_loop)
        } else {
            execute_screen_workflow(workflow, &self.ui_control, capabilities)
        };
        self.remember_workflow_targets(&run);
        self.remember_screen_workflow_context(&run, &request_id, &action_id, previous_context);
        if !used_production_goal_loop {
            self.refresh_recent_goal_loop_after_workflow(&run).await;
        }
        if let Some(receipt) =
            verified_continuation_learning_event(&run).map(store_verified_continuation)
        {
            self.audit.append(&DesktopAuditEvent {
                audit_id: Uuid::new_v4().to_string(),
                action_id: action_id.clone(),
                request_id: request_id.clone(),
                tool_name: "screen.workflow".into(),
                stage: "contextual_learning".into(),
                status: if receipt.accepted {
                    "accepted"
                } else {
                    "ignored"
                }
                .into(),
                timestamp: now_ms(),
                risk_level: crate::desktop_agent_types::RiskLevel::Low,
                details: receipt.summary,
            });
        }
        self.audit.append(&DesktopAuditEvent {
            audit_id: Uuid::new_v4().to_string(),
            action_id,
            request_id,
            tool_name: "screen.workflow".into(),
            stage: "workflow".into(),
            status: run.status.as_str().into(),
            timestamp: now_ms(),
            risk_level: crate::desktop_agent_types::RiskLevel::Medium,
            details: run.diagnostic_value(),
        });
        run
    }

    pub fn recent_workflow_context(&self) -> Option<RecentWorkflowContext> {
        let now = now_ms();
        let mut memory = self
            .recent_workflow_context
            .lock()
            .expect("recent workflow context memory poisoned");
        if memory
            .as_ref()
            .is_some_and(|context| now > context.expires_at_ms)
        {
            *memory = None;
        }
        memory.clone()
    }

    pub fn resolve_followup_continuation(
        &self,
        manifest: &CapabilityManifest,
        message: &str,
    ) -> Option<WorkflowContinuationResolution> {
        resolve_workflow_continuation(self.recent_workflow_context(), manifest, message, now_ms())
    }

    pub fn resolve_followup_continuation_with_model_params(
        &self,
        manifest: &CapabilityManifest,
        message: &str,
        model_params: &Value,
        model_confidence: f32,
    ) -> Option<WorkflowContinuationResolution> {
        resolve_workflow_continuation_with_model_params(
            self.recent_workflow_context(),
            manifest,
            message,
            model_params,
            model_confidence,
            now_ms(),
        )
    }

    fn recent_workflow_targets(&self) -> Vec<UITargetCandidate> {
        self.recent_workflow_targets
            .lock()
            .expect("recent workflow target memory poisoned")
            .clone()
    }

    fn remember_workflow_targets(&self, run: &ScreenWorkflowRun) {
        let mut selected = run
            .step_runs
            .iter()
            .filter(|step| {
                matches!(
                    step.status,
                    crate::screen_workflow::WorkflowRunStatus::Completed
                )
            })
            .filter_map(|step| step.target_selection.as_ref())
            .filter_map(|selection| selection.selected_candidate.clone())
            .collect::<Vec<_>>();

        if selected.is_empty() {
            return;
        }

        let now = now_ms();
        for candidate in &mut selected {
            candidate.observed_at_ms = Some(now);
            candidate.reuse_eligible = !candidate_is_accessibility_sourced(candidate);
        }

        let mut memory = self
            .recent_workflow_targets
            .lock()
            .expect("recent workflow target memory poisoned");
        for candidate in selected {
            memory.retain(|existing| existing.candidate_id != candidate.candidate_id);
            memory.push(candidate);
        }
        memory.sort_by(|left, right| right.observed_at_ms.cmp(&left.observed_at_ms));
        memory.truncate(12);
    }

    fn remember_action_workflow_context(
        &self,
        request_id: &str,
        action_id: &str,
        request: &DesktopActionRequest,
        result: &Value,
    ) {
        let Some(context) =
            build_context_from_action_response(request_id, action_id, request, result, now_ms())
        else {
            return;
        };
        self.store_recent_workflow_context(context);
    }

    fn remember_screen_workflow_context(
        &self,
        run: &ScreenWorkflowRun,
        request_id: &str,
        action_id: &str,
        previous_context: Option<RecentWorkflowContext>,
    ) {
        let Some(context) = build_context_from_screen_workflow_run(
            run,
            request_id,
            action_id,
            previous_context,
            now_ms(),
        ) else {
            return;
        };
        self.store_recent_workflow_context(context);
    }

    fn store_recent_workflow_context(&self, context: RecentWorkflowContext) {
        let mut memory = self
            .recent_workflow_context
            .lock()
            .expect("recent workflow context memory poisoned");
        *memory = Some(context);
    }

    fn store_recent_goal_loop(&self, run: GoalLoopRun) {
        let Ok(mut memory) = self.recent_goal_loop.lock() else {
            return;
        };
        *memory = Some(run);
    }

    async fn refresh_recent_goal_loop_after_workflow(&self, run: &ScreenWorkflowRun) {
        let Some(mut goal_loop) = run.workflow.grounding.goal_loop.clone() else {
            return;
        };
        if !matches!(
            run.status,
            crate::screen_workflow::WorkflowRunStatus::Completed
        ) {
            self.store_recent_goal_loop(goal_loop);
            return;
        }
        let has_executed_goal_step = run.step_runs.iter().any(|step| {
            matches!(
                step.status,
                crate::screen_workflow::WorkflowRunStatus::Completed
            ) && step.primitive.is_some()
        });
        if !has_executed_goal_step {
            self.store_recent_goal_loop(goal_loop);
            return;
        }

        let capture = match self.capture_for_structured_grounding_with_policy(true) {
            Ok(capture) => capture,
            Err(error) => {
                goal_loop.status = crate::desktop_agent_types::GoalLoopStatus::VerificationFailed;
                goal_loop.failure_reason = Some(format!(
                    "post-step semantic verification capture failed: {error}"
                ));
                goal_loop.verifier_status = Some("post_step_capture_failed".into());
                self.store_recent_goal_loop(goal_loop);
                return;
            }
        };
        let frame = match self
            .screen_vision
            .perceive_semantic_frame(
                &capture.image_path,
                capture.captured_at,
                &capture.provider,
                Some(&goal_loop.goal.utterance),
                None,
                goal_loop.goal.constraints.provider.as_deref(),
                None,
            )
            .await
        {
            Ok(frame) => frame,
            Err(error) => {
                goal_loop.status = crate::desktop_agent_types::GoalLoopStatus::NeedsPerception;
                goal_loop.failure_reason = Some(format!(
                    "post-step semantic verification perception failed: {error}"
                ));
                goal_loop.verifier_status = Some("post_step_perception_failed".into());
                self.store_recent_goal_loop(goal_loop);
                return;
            }
        };
        goal_loop.stale_capture_reuse_prevented = true;
        if let Some(execution) = goal_loop.executed_steps.last_mut() {
            execution.fresh_capture_used = true;
        }
        let verification =
            verify_goal_state(&goal_loop.goal, &frame, goal_loop.iteration_count + 1);
        let verification_status = verification.status.clone();
        goal_loop.iteration_count += 1;
        goal_loop.frames.push(frame);
        goal_loop.verifier_status = Some(format!("{verification_status:?}"));
        goal_loop.verification_history.push(verification.clone());
        if verification_status == crate::desktop_agent_types::GoalVerificationStatus::GoalAchieved {
            goal_loop.status = crate::desktop_agent_types::GoalLoopStatus::GoalAchieved;
            goal_loop.failure_reason = None;
        } else {
            goal_loop.status = crate::desktop_agent_types::GoalLoopStatus::VerificationFailed;
            goal_loop.failure_reason = Some(format!(
                "post-step semantic verification did not satisfy the goal: {}",
                verification.reason
            ));
        }
        self.store_recent_goal_loop(goal_loop);
    }

    async fn prepare_continuation_grounding(
        &self,
        workflow: &mut ScreenWorkflow,
        capabilities: &UIPrimitiveCapabilitySet,
    ) {
        let max_attempts = if workflow.continuation.is_some() {
            2
        } else {
            1
        };
        let mut attempts = Vec::new();
        if !self
            .permissions
            .allows(&crate::desktop_agent_types::Permission::DesktopObserve)
            || !self
                .policy
                .permission_enabled(&crate::desktop_agent_types::Permission::DesktopObserve)
        {
            workflow
                .grounding
                .uncertainty
                .push("structured_grounding_permission_denied".into());
            return;
        }

        for attempt_index in 0..max_attempts {
            if workflow.continuation.is_none() && !workflow_needs_target_grounding(workflow) {
                break;
            }
            if !workflow_needs_target_grounding(workflow)
                && workflow.grounding.page_validation.is_some()
            {
                break;
            }

            let force_fresh_capture = workflow
                .continuation
                .as_ref()
                .is_some_and(|descriptor| descriptor.policy.fresh_capture_required);
            if force_fresh_capture {
                workflow
                    .grounding
                    .uncertainty
                    .push("fresh_capture_enforced_for_continuation_grounding".into());
            }

            let capture =
                match self.capture_for_structured_grounding_with_policy(force_fresh_capture) {
                    Ok(capture) => capture,
                    Err(error) => {
                        workflow
                            .grounding
                            .uncertainty
                            .push(format!("structured_grounding_capture_unavailable: {error}"));
                        break;
                    }
                };
            let accessibility_snapshot = self.capture_accessibility_snapshot_for_grounding();

            let extraction = self
                .extract_structured_candidates_for_workflow_capture(workflow, &capture)
                .await;
            match extraction {
                Ok(extraction) => {
                    if let Some(page_evidence) = extraction.page_evidence {
                        workflow.grounding.page_evidence.push(page_evidence);
                    }
                    if let Some(frame) = extraction.semantic_frame {
                        let mut frame = frame;
                        if let Some(snapshot) = accessibility_snapshot.as_ref() {
                            enrich_frame_with_accessibility(&mut frame, snapshot);
                            workflow.grounding.uncertainty.push(format!(
                                "uia_snapshot_available:{} candidates={}",
                                snapshot.snapshot_id,
                                frame.legacy_target_candidates.len()
                            ));
                        }
                        if let Some(goal_loop) =
                            goal_loop_for_workflow_frame(workflow, frame.clone())
                        {
                            if let Some(candidate) = goal_loop
                                .planner_steps
                                .first()
                                .and_then(planner_candidate_from_step)
                            {
                                workflow.grounding.visible_target_candidates.push(candidate);
                            }
                            self.store_recent_goal_loop(goal_loop.clone());
                            workflow.grounding.goal_loop = Some(goal_loop);
                        }
                        workflow.grounding.semantic_frame = Some(frame);
                    }
                    if !extraction.candidates.is_empty() {
                        workflow
                            .grounding
                            .visible_target_candidates
                            .extend(extraction.candidates);
                    } else {
                        workflow
                            .grounding
                            .uncertainty
                            .push("structured_vision_returned_no_candidates".into());
                    }
                    if let Some(snapshot) = accessibility_snapshot.as_ref() {
                        let mut uia_candidates = synthesize_ranked_uia_result_candidates(snapshot);
                        uia_candidates.retain(|candidate| {
                            !workflow
                                .grounding
                                .visible_target_candidates
                                .iter()
                                .any(|existing| {
                                    existing.element_id == candidate.element_id
                                        || existing.candidate_id == candidate.candidate_id
                                })
                        });
                        if !uia_candidates.is_empty() {
                            workflow
                                .grounding
                                .visible_target_candidates
                                .extend(uia_candidates);
                        }
                    }
                }
                Err(error) => {
                    workflow
                        .grounding
                        .uncertainty
                        .push(format!("structured_vision_unavailable: {error}"));
                }
            }

            if let Some(descriptor) = workflow.continuation.clone() {
                let page_validation = validate_continuation_page(
                    &descriptor,
                    Some(&capture),
                    &workflow.grounding.page_evidence,
                    &workflow.grounding.visible_target_candidates,
                    now_ms(),
                );
                workflow.grounding.page_validation = Some(page_validation.clone());
                refresh_screen_workflow_plan(workflow);
                let selected = workflow.step_plans.iter().any(|plan| {
                    plan.target_selection
                        .as_ref()
                        .is_some_and(|selection| selection.selected_candidate.is_some())
                });
                let scroll_decision = scroll_policy_for_regrounding(
                    &descriptor,
                    capabilities,
                    attempt_index,
                    max_attempts,
                    &page_validation,
                    workflow.grounding.visible_target_candidates.len(),
                    selected,
                );
                attempts.push(RegroundingAttemptDiagnostic {
                    attempt_index,
                    page_validation,
                    visible_candidate_count: workflow.grounding.visible_target_candidates.len(),
                    selected_candidate_id: workflow
                        .step_plans
                        .iter()
                        .find_map(|plan| plan.target_selection.as_ref())
                        .and_then(|selection| selection.selected_candidate.as_ref())
                        .map(|candidate| candidate.candidate_id.clone()),
                    target_selection_status: workflow
                        .step_plans
                        .iter()
                        .find_map(|plan| plan.target_selection.as_ref())
                        .map(|selection| format!("{:?}", selection.status)),
                    scroll_decision: scroll_decision.clone(),
                    notes: workflow.grounding.uncertainty.clone(),
                });

                if selected
                    || matches!(
                        scroll_decision.status,
                        ScrollContinuationStatus::Unsupported
                            | ScrollContinuationStatus::RetryBudgetExhausted
                            | ScrollContinuationStatus::PageMismatch
                            | ScrollContinuationStatus::NotNeeded
                            | ScrollContinuationStatus::NotApplicable
                    )
                {
                    break;
                }

                workflow.grounding.uncertainty.push(
                    "scroll_regrounding_requested_but_scroll_execution_is_not_enabled".into(),
                );
                break;
            } else {
                refresh_screen_workflow_plan(workflow);
                break;
            }
        }

        if let Some(descriptor) = workflow.continuation.as_mut() {
            if let Some(page_validation) = workflow.grounding.page_validation.clone() {
                descriptor.page_validation = Some(page_validation);
            }
            if !attempts.is_empty() {
                let final_status = attempts
                    .last()
                    .map(|attempt| attempt.scroll_decision.status.clone())
                    .unwrap_or(ScrollContinuationStatus::NotApplicable);
                let final_reason = attempts
                    .last()
                    .map(|attempt| attempt.scroll_decision.reason.clone())
                    .unwrap_or_else(|| "no re-grounding attempts were recorded".into());
                let diagnostics = ContinuationRegroundingDiagnostics {
                    max_attempts,
                    attempts,
                    final_status,
                    final_reason,
                };
                descriptor.regrounding = Some(diagnostics.clone());
                workflow.grounding.regrounding = Some(diagnostics);
            }
        }

        refresh_screen_workflow_plan(workflow);
    }

    async fn extract_structured_candidates_for_workflow_capture(
        &self,
        workflow: &ScreenWorkflow,
        capture: &ScreenCaptureResult,
    ) -> Result<StructuredVisionExtraction, String> {
        if !self
            .permissions
            .allows(&crate::desktop_agent_types::Permission::DesktopObserve)
            || !self
                .policy
                .permission_enabled(&crate::desktop_agent_types::Permission::DesktopObserve)
        {
            return Err("Permission denied for desktop observation".into());
        }

        let requested_roles = requested_roles_for_workflow(workflow);
        let app_hint = workflow
            .steps
            .iter()
            .find_map(|step| value_str(&step.target, "app"));
        let provider_hint = workflow.steps.iter().find_map(|step| {
            value_str(&step.target, "provider").or_else(|| value_str(&step.selection, "provider"))
        });
        let extraction = self
            .screen_vision
            .extract_ui_candidates(
                &capture.image_path,
                capture.captured_at,
                &capture.provider,
                &requested_roles,
                app_hint,
                provider_hint,
            )
            .await?;

        Ok(extraction)
    }

    fn capture_accessibility_snapshot_for_grounding(&self) -> Option<AccessibilitySnapshot> {
        if !self.policy.accessibility_snapshot_enabled() {
            return None;
        }
        let snapshot = crate::accessibility_layer::capture_accessibility_snapshot(&[
            "chrome", "msedge", "firefox", "brave", "opera",
        ]);
        (!snapshot.elements.is_empty()).then_some(snapshot)
    }

    fn capture_for_structured_grounding(&self) -> Result<ScreenCaptureResult, String> {
        self.capture_for_structured_grounding_with_policy(false)
    }

    fn capture_for_structured_grounding_with_policy(
        &self,
        force_fresh_capture: bool,
    ) -> Result<ScreenCaptureResult, String> {
        if force_fresh_capture {
            let status = self.screen_capture.status();
            if !status.enabled {
                return Err(
                    "fresh screen capture is required but screen observation is disabled".into(),
                );
            }
            return self.screen_capture.capture_snapshot();
        }

        let status = self.screen_capture.status();
        if let Some(path) = status.last_capture_path.clone() {
            let capture_age_ms = status
                .last_frame_at
                .map(|captured_at| now_ms().saturating_sub(captured_at));
            if status.enabled
                && capture_age_ms
                    .map(|age| age > MAX_RECENT_SCREEN_AGE_MS)
                    .unwrap_or(false)
            {
                return self.screen_capture.capture_snapshot();
            }
            let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            return Ok(ScreenCaptureResult {
                capture_id: Uuid::new_v4().to_string(),
                captured_at: status.last_frame_at.unwrap_or_else(now_ms),
                image_path: path,
                width: None,
                height: None,
                bytes,
                provider: status.provider,
            });
        }

        if !status.enabled {
            return Err(
                "No recent screen capture is available and screen observation is disabled.".into(),
            );
        }

        self.screen_capture.capture_snapshot()
    }

    pub fn pending_approvals(&self) -> Vec<PendingApproval> {
        self.pending
            .lock()
            .expect("pending approvals mutex poisoned")
            .values()
            .map(|e| e.approval.clone())
            .collect()
    }

    pub fn reject_pending(&self, action_id: &str, note: Option<String>) -> Result<(), String> {
        let mut pending = self
            .pending
            .lock()
            .expect("pending approvals mutex poisoned");
        let Some(envelope) = pending.remove(action_id) else {
            return Err(format!("Pending approval not found: {action_id}"));
        };
        self.persist_pending_locked(&pending)?;
        self.audit.append(&DesktopAuditEvent {
            audit_id: Uuid::new_v4().to_string(),
            action_id: envelope.approval.action_id.clone(),
            request_id: envelope.approval.request_id.clone(),
            tool_name: envelope.approval.tool_name.clone(),
            stage: "approval".into(),
            status: "rejected".into(),
            timestamp: now_ms(),
            risk_level: envelope.approval.risk_level.clone(),
            details: json!({"note": note}),
        });
        Ok(())
    }

    pub fn approve_pending(
        &self,
        action_id: &str,
        note: Option<String>,
    ) -> Result<DesktopActionResponse, String> {
        let envelope = {
            let mut pending = self
                .pending
                .lock()
                .expect("pending approvals mutex poisoned");
            let envelope = pending
                .remove(action_id)
                .ok_or_else(|| format!("Pending approval not found: {action_id}"))?;
            self.persist_pending_locked(&pending)?;
            envelope
        };
        self.audit.append(&DesktopAuditEvent {
            audit_id: Uuid::new_v4().to_string(),
            action_id: envelope.approval.action_id.clone(),
            request_id: envelope.approval.request_id.clone(),
            tool_name: envelope.approval.tool_name.clone(),
            stage: "approval".into(),
            status: "approved".into(),
            timestamp: now_ms(),
            risk_level: envelope.approval.risk_level.clone(),
            details: json!({"note": note}),
        });
        self.execute_internal(
            envelope.approval.request_id,
            envelope.approval.action_id,
            envelope.request,
            true,
        )
    }

    pub fn submit_action(
        &self,
        request_id: String,
        request: DesktopActionRequest,
    ) -> Result<DesktopActionResponse, String> {
        let descriptor = self
            .registry
            .get(&request.tool_name)
            .ok_or_else(|| format!("Unknown tool: {}", request.tool_name))?;
        for permission in &descriptor.required_permissions {
            if !self.permissions.allows(permission) || !self.policy.permission_enabled(permission) {
                return Err(format!("Permission denied for {:?}", permission));
            }
        }

        let action_id = Uuid::new_v4().to_string();
        if request.preview_only {
            return Ok(DesktopActionResponse {
                action_id,
                request_id,
                tool_name: descriptor.tool_name,
                status: DesktopActionStatus::Executed,
                message: Some("Preview ready".into()),
                result: Some(json!({"preview": true, "params": request.params})),
                risk_level: Some(descriptor.default_risk),
            });
        }

        if descriptor.requires_confirmation
            || self.policy.requires_approval(&descriptor.default_risk)
        {
            let approval = PendingApproval {
                action_id: action_id.clone(),
                request_id: request_id.clone(),
                tool_name: descriptor.tool_name.clone(),
                params: request.params.clone(),
                risk_level: descriptor.default_risk.clone(),
                reason: request
                    .reason
                    .clone()
                    .unwrap_or_else(|| "Confirmation required by policy".into()),
                requested_at: now_ms(),
            };
            {
                let mut pending = self
                    .pending
                    .lock()
                    .expect("pending approvals mutex poisoned");
                pending.insert(
                    action_id.clone(),
                    PendingActionEnvelope {
                        approval: approval.clone(),
                        request,
                    },
                );
                self.persist_pending_locked(&pending)?;
            }
            self.audit.append(&DesktopAuditEvent {
                audit_id: Uuid::new_v4().to_string(),
                action_id: action_id.clone(),
                request_id: request_id.clone(),
                tool_name: approval.tool_name.clone(),
                stage: "policy".into(),
                status: "approval_required".into(),
                timestamp: now_ms(),
                risk_level: approval.risk_level.clone(),
                details: json!({"params": approval.params, "reason": approval.reason}),
            });
            return Ok(DesktopActionResponse {
                action_id,
                request_id,
                tool_name: descriptor.tool_name,
                status: DesktopActionStatus::ApprovalRequired,
                message: Some("Approval required".into()),
                result: None,
                risk_level: Some(descriptor.default_risk),
            });
        }

        self.execute_internal(request_id, action_id, request, false)
    }

    fn execute_internal(
        &self,
        request_id: String,
        action_id: String,
        request: DesktopActionRequest,
        was_approved: bool,
    ) -> Result<DesktopActionResponse, String> {
        let descriptor = self
            .registry
            .get(&request.tool_name)
            .ok_or_else(|| format!("Unknown tool: {}", request.tool_name))?;
        self.audit.append(&DesktopAuditEvent {
            audit_id: Uuid::new_v4().to_string(),
            action_id: action_id.clone(),
            request_id: request_id.clone(),
            tool_name: request.tool_name.clone(),
            stage: "execution".into(),
            status: "started".into(),
            timestamp: now_ms(),
            risk_level: descriptor.default_risk.clone(),
            details: json!({"params": request.params, "approved": was_approved}),
        });

        let result = self.dispatch(&request);
        match result {
            Ok(result) => {
                self.remember_action_workflow_context(&request_id, &action_id, &request, &result);
                self.audit.append(&DesktopAuditEvent {
                    audit_id: Uuid::new_v4().to_string(),
                    action_id: action_id.clone(),
                    request_id: request_id.clone(),
                    tool_name: request.tool_name.clone(),
                    stage: "execution".into(),
                    status: "completed".into(),
                    timestamp: now_ms(),
                    risk_level: descriptor.default_risk.clone(),
                    details: json!({"result": result.clone()}),
                });
                Ok(DesktopActionResponse {
                    action_id,
                    request_id,
                    tool_name: request.tool_name,
                    status: DesktopActionStatus::Executed,
                    message: Some(if was_approved {
                        "Approved action executed".into()
                    } else {
                        "Action executed".into()
                    }),
                    result: Some(result),
                    risk_level: Some(descriptor.default_risk),
                })
            }
            Err(error) => {
                self.audit.append(&DesktopAuditEvent {
                    audit_id: Uuid::new_v4().to_string(),
                    action_id: action_id.clone(),
                    request_id: request_id.clone(),
                    tool_name: request.tool_name.clone(),
                    stage: "execution".into(),
                    status: "failed".into(),
                    timestamp: now_ms(),
                    risk_level: descriptor.default_risk.clone(),
                    details: json!({"error": error}),
                });
                Err(error)
            }
        }
    }

    fn persist_pending_locked(
        &self,
        pending: &HashMap<String, PendingActionEnvelope>,
    ) -> Result<(), String> {
        self.pending_store
            .save(pending.values().map(|entry| entry.approval.clone()))
    }

    fn dispatch(&self, request: &DesktopActionRequest) -> Result<Value, String> {
        match request.tool_name.as_str() {
            "filesystem.read_text" => {
                let path = request
                    .params
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "filesystem.read_text requires params.path".to_string())?;
                let mut result = self.filesystem.read_text(path)?;
                if let Some(post_processing) = request.params.get("post_processing") {
                    result["post_processing"] = post_processing.clone();
                }
                if let Some(operation) = request.params.get("operation") {
                    result["operation"] = operation.clone();
                }
                Ok(result)
            }
            "filesystem.write_text" => {
                let path = request
                    .params
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "filesystem.write_text requires params.path".to_string())?;
                let create_empty = request
                    .params
                    .get("create_empty")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let content = request
                    .params
                    .get("content")
                    .and_then(|v| v.as_str())
                    .or_else(|| create_empty.then_some(""))
                    .ok_or_else(|| "filesystem.write_text requires params.content".to_string())?;
                let mode = request
                    .params
                    .get("mode")
                    .and_then(|v| v.as_str())
                    .unwrap_or("overwrite");
                self.filesystem.write_text(path, content, mode)
            }
            "filesystem.search" => {
                let root = request.params.get("root").and_then(|v| v.as_str());
                let pattern = request
                    .params
                    .get("pattern")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "filesystem.search requires params.pattern".to_string())?;
                let max_results = request
                    .params
                    .get("max_results")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(25) as usize;
                self.filesystem.search(root, pattern, max_results)
            }
            "terminal.run" => {
                let command = request
                    .params
                    .get("command")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "terminal.run requires params.command".to_string())?;
                let args = request
                    .params
                    .get("args")
                    .and_then(|v| v.as_array())
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(|item| item.as_str().map(ToString::to_string))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let cwd = request.params.get("cwd").and_then(|v| v.as_str());
                self.terminal.run(command, &args, cwd)
            }
            "browser.open" => {
                let url = request
                    .params
                    .get("url")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "browser.open requires params.url".to_string())?;
                self.browser.open(url)
            }
            "browser.search" => {
                let query = request
                    .params
                    .get("query")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "browser.search requires params.query".to_string())?;
                self.browser.search(query)
            }
            "desktop.launch_app" => {
                let requested =
                    string_param(&request.params, &["path", "app", "app_name", "application"])
                        .ok_or_else(|| {
                            "desktop.launch_app requires params.path or params.app_name".to_string()
                        })?;
                let path = resolve_launch_target(&requested);
                let args = string_array_param(&request.params, &["args", "arguments"]);
                launch_app(&path, &args)
            }
            _ => Err(format!("Unsupported tool: {}", request.tool_name)),
        }
    }
}

fn string_param(params: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        params
            .get(key)
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    })
}

fn string_array_param(params: &Value, keys: &[&str]) -> Vec<String> {
    keys.iter()
        .find_map(|key| {
            params
                .get(key)
                .and_then(|value| value.as_array())
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.as_str().map(ToOwned::to_owned))
                        .collect::<Vec<_>>()
                })
        })
        .unwrap_or_default()
}

fn resolve_launch_target(requested: &str) -> String {
    let normalized = requested
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .to_ascii_lowercase()
        .replace('-', " ");
    let is_chrome = matches!(
        normalized.as_str(),
        "browser" | "chrome" | "google chrome" | "googlechrome"
    );

    if !is_chrome {
        return requested.to_string();
    }

    if cfg!(target_os = "windows") {
        let candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ];
        if let Some(path) = candidates
            .iter()
            .find(|path| std::path::Path::new(path).exists())
        {
            return (*path).to_string();
        }
        return "chrome.exe".into();
    }

    "google-chrome".into()
}

fn launch_app(path: &str, args: &[String]) -> Result<Value, String> {
    std::process::Command::new(path)
        .args(args)
        .spawn()
        .map_err(|e| e.to_string())?;
    Ok(json!({"path": path, "args": args, "launched": true}))
}

fn workflow_needs_target_grounding(workflow: &ScreenWorkflow) -> bool {
    workflow
        .step_plans
        .iter()
        .any(|plan| plan.support == StepSupportStatus::NeedsTargetGrounding)
}

fn planner_step_primitive_request(step: &PlannerStep) -> Option<UIPrimitiveRequest> {
    if !matches!(
        step.kind,
        PlannerStepKind::ClickResultRegion | PlannerStepKind::ClickEntityRegion
    ) {
        return None;
    }
    let candidate = step.executable_candidate.as_ref()?;
    Some(UIPrimitiveRequest {
        primitive: UIPrimitiveKind::ClickTargetCandidate,
        value: None,
        target: candidate.execution_payload(),
        reason: Some(format!(
            "Goal loop requested governed click: {}",
            step.rationale
        )),
    })
}

fn deterministic_uia_candidate_if_unambiguous(
    candidates: &[UITargetCandidate],
) -> Option<UITargetCandidate> {
    let [candidate] = candidates else {
        return None;
    };
    if !candidate.supports_click
        || candidate.confidence < TargetSelectionPolicy::default().min_click_confidence
        || candidate.element_id.as_deref().is_none()
        || candidate.accessibility_snapshot_id.as_deref().is_none()
    {
        return None;
    }
    Some(candidate.clone())
}

fn accessibility_selector_click_decision(
    input: &PlannerContractInput,
    candidate: UITargetCandidate,
    selection: &crate::accessibility_layer::AccessibilityTargetSelection,
    candidate_count: usize,
    selector_used: bool,
) -> PlannerContractDecision {
    let element_id = candidate
        .element_id
        .clone()
        .unwrap_or_else(|| candidate.candidate_id.clone());
    let selection_confidence = selection.confidence;
    PlannerContractDecision {
        source: if selector_used {
            PlannerContractSource::ModelAssisted
        } else {
            PlannerContractSource::RustDeterministic
        },
        proposed_step: PlannerStep {
            step_id: Uuid::new_v4().to_string(),
            kind: PlannerStepKind::ClickResultRegion,
            confidence: candidate.confidence,
            rationale: selection.rationale.clone().unwrap_or_else(|| {
                "accessibility target selector selected a current-snapshot element_id".into()
            }),
            target_item_id: Some(element_id.clone()),
            target_entity_id: None,
            click_region_key: Some("primary".into()),
            executable_candidate: Some(candidate.clone()),
            expected_state: Some(input.goal.success_condition.clone()),
        },
        strategy_rationale: format!(
            "uia_selector_required=true; uia_candidate_count={candidate_count}; llm_uia_selector_used={selector_used}; llm_selected_element_id={}; llm_selection_confidence={selection_confidence:.2}; rust_element_validation_status=passed; vision_only_candidate_suppressed=true; vision_only_fallback_used=false",
            selection
                .selected_element_id
                .as_deref()
                .unwrap_or(element_id.as_str())
        ),
        focused_perception_needed: false,
        replan_needed: false,
        expected_verification_target: Some(input.goal.success_condition.clone()),
        planner_confidence: candidate.confidence,
        accepted: true,
        fallback_used: false,
        rejection_reason: None,
        decision_status: PlannerDecisionStatus::Accepted,
        rejection_code: None,
        visibility_assessment: PlannerVisibilityAssessment::VisibleGrounded,
        scroll_intent: PlannerScrollIntent::NotNeeded,
        visible_actionability: input.visible_actionability.clone(),
        target_confidence: None,
        normalized: false,
        downgraded: false,
    }
}

fn accessibility_selector_suppression_decision(
    input: &PlannerContractInput,
    candidate_count: usize,
    selector_used: bool,
    selected_element_id: Option<String>,
    selection_confidence: Option<f32>,
    reason: String,
    rejection_code: PlannerRejectionReason,
) -> PlannerContractDecision {
    let planning_confidence = TargetSelectionPolicy::default().min_click_confidence;
    PlannerContractDecision {
        source: PlannerContractSource::ModelAssisted,
        proposed_step: PlannerStep {
            step_id: Uuid::new_v4().to_string(),
            kind: PlannerStepKind::Refuse,
            confidence: planning_confidence,
            rationale: format!(
                "current UIA candidates are available, but no validated UIA element_id was selected; suppressing vision-only click geometry: {reason}"
            ),
            target_item_id: None,
            target_entity_id: None,
            click_region_key: None,
            executable_candidate: None,
            expected_state: Some(input.goal.success_condition.clone()),
        },
        strategy_rationale: format!(
            "uia_selector_required=true; uia_candidate_count={candidate_count}; llm_uia_selector_used={selector_used}; llm_selected_element_id={}; llm_selection_confidence={}; rust_element_validation_status=failed; vision_only_candidate_suppressed=true; vision_only_fallback_used=false; {reason}",
            selected_element_id.as_deref().unwrap_or("none"),
            selection_confidence
                .map(|confidence| format!("{confidence:.2}"))
                .unwrap_or_else(|| "none".into())
        ),
        focused_perception_needed: false,
        replan_needed: false,
        expected_verification_target: Some(input.goal.success_condition.clone()),
        planner_confidence: planning_confidence,
        accepted: true,
        fallback_used: false,
        rejection_reason: Some(reason),
        decision_status: PlannerDecisionStatus::Accepted,
        rejection_code: Some(rejection_code),
        visibility_assessment: PlannerVisibilityAssessment::Unknown,
        scroll_intent: PlannerScrollIntent::NotNeeded,
        visible_actionability: input.visible_actionability.clone(),
        target_confidence: None,
        normalized: false,
        downgraded: false,
    }
}

fn enrich_frame_with_accessibility(
    frame: &mut SemanticScreenFrame,
    snapshot: &AccessibilitySnapshot,
) {
    if snapshot
        .captured_at_ms
        .saturating_add(MAX_ACCESSIBILITY_SNAPSHOT_AGE_MS)
        < frame.captured_at
    {
        frame.uncertainty.push(FrameUncertainty {
            code: "stale_accessibility_snapshot".into(),
            message: format!(
                "accessibility snapshot {} captured at {} was older than frame capture {}",
                snapshot.snapshot_id, snapshot.captured_at_ms, frame.captured_at
            ),
            severity: "medium".into(),
        });
        return;
    }

    let mut matched_element_ids = Vec::new();
    if let Some(primary_list) = frame.primary_list.as_mut() {
        for item in &mut primary_list.items {
            if let Some(element_id) = primary_list_item_element_id(item).map(ToOwned::to_owned) {
                if let Some(element) = find_accessible_element(snapshot, &element_id) {
                    if populate_accessibility_click_region(&mut item.click_regions, element) {
                        item.element_id = Some(element_id.clone());
                        item.attributes = accessibility_attributes(
                            item.attributes.clone(),
                            &element_id,
                            &snapshot.snapshot_id,
                        );
                        matched_element_ids.push(element_id);
                    }
                }
            }
        }
    }
    for item in &mut frame.visible_result_items {
        if let Some(element_id) = visible_result_item_element_id(item).map(ToOwned::to_owned) {
            if let Some(element) = find_accessible_element(snapshot, &element_id) {
                if populate_accessibility_click_region(&mut item.click_regions, element) {
                    item.element_id = Some(element_id.clone());
                    item.attributes = accessibility_attributes(
                        item.attributes.clone(),
                        &element_id,
                        &snapshot.snapshot_id,
                    );
                    matched_element_ids.push(element_id);
                }
            }
        }
    }

    let synthesized_candidates = synthesize_ranked_uia_result_candidates(snapshot);

    for candidate in synthesized_candidates {
        let element_id = candidate
            .element_id
            .as_deref()
            .unwrap_or(candidate.candidate_id.as_str())
            .to_string();
        if !frame.visible_result_items.iter().any(|item| {
            visible_result_item_element_id(item) == Some(element_id.as_str())
                || item.item_id == element_id
        }) {
            if let Some(item) = visible_result_item_from_uia_candidate(&candidate) {
                frame.visible_result_items.push(item);
            }
        }
        if matched_element_ids
            .iter()
            .any(|existing| existing == &element_id)
            || frame
                .legacy_target_candidates
                .iter()
                .any(|existing| existing.element_id.as_deref() == Some(element_id.as_str()))
        {
            continue;
        }
        frame.legacy_target_candidates.push(candidate);
    }
}

fn populate_accessibility_click_region(
    click_regions: &mut HashMap<String, ClickRegion>,
    element: &AccessibleElement,
) -> bool {
    let Some(region) = element.bounding_rect.clone() else {
        return false;
    };
    if element.is_offscreen || !element.is_enabled {
        return false;
    }
    click_regions.insert(
        "primary".into(),
        ClickRegion {
            region,
            raw_confidence: Some(0.97),
            confidence: 0.97,
        },
    );
    true
}

fn find_accessible_element<'a>(
    snapshot: &'a AccessibilitySnapshot,
    element_id: &str,
) -> Option<&'a AccessibleElement> {
    snapshot
        .elements
        .iter()
        .find(|element| element.element_id == element_id)
}

fn primary_list_item_element_id(item: &PrimaryListItem) -> Option<&str> {
    item.element_id
        .as_deref()
        .or_else(|| item.attributes.get("element_id").and_then(Value::as_str))
        .or_else(|| {
            item.item_id
                .starts_with("a11y_")
                .then_some(item.item_id.as_str())
        })
}

fn visible_result_item_element_id(item: &VisibleResultItem) -> Option<&str> {
    item.element_id
        .as_deref()
        .or_else(|| item.attributes.get("element_id").and_then(Value::as_str))
        .or_else(|| {
            item.item_id
                .starts_with("a11y_")
                .then_some(item.item_id.as_str())
        })
}

fn accessibility_attributes(attributes: Value, element_id: &str, snapshot_id: &str) -> Value {
    let mut object = attributes.as_object().cloned().unwrap_or_default();
    object.insert("element_id".into(), json!(element_id));
    object.insert("accessibility_snapshot_id".into(), json!(snapshot_id));
    object.insert("accessibility_sourced".into(), json!(true));
    Value::Object(object)
}

fn visible_result_item_from_uia_candidate(
    candidate: &UITargetCandidate,
) -> Option<VisibleResultItem> {
    let region = candidate.region.clone()?;
    let mut click_regions = HashMap::new();
    click_regions.insert(
        "primary".into(),
        ClickRegion {
            region,
            raw_confidence: Some(candidate.confidence),
            confidence: candidate.confidence,
        },
    );
    Some(VisibleResultItem {
        item_id: candidate
            .element_id
            .clone()
            .unwrap_or_else(|| candidate.candidate_id.clone()),
        element_id: candidate.element_id.clone(),
        kind: VisibleResultKind::Generic,
        title: candidate.label.clone(),
        channel_name: None,
        provider: candidate
            .content_provider_hint
            .clone()
            .or(candidate.provider_hint.clone()),
        rank_overall: candidate.rank,
        rank_within_kind: candidate.rank,
        click_regions,
        raw_confidence: Some(candidate.confidence),
        confidence: candidate.confidence,
        rationale: Some("provider_agnostic_uia_candidate".into()),
        attributes: accessibility_attributes(
            Value::Null,
            candidate
                .element_id
                .as_deref()
                .unwrap_or(candidate.candidate_id.as_str()),
            candidate
                .accessibility_snapshot_id
                .as_deref()
                .unwrap_or_default(),
        ),
    })
}

fn candidate_is_accessibility_sourced(candidate: &UITargetCandidate) -> bool {
    candidate.source == TargetGroundingSource::AccessibilityLayer
        || candidate.observation_source.as_deref() == Some("uia_snapshot")
        || candidate
            .element_id
            .as_deref()
            .is_some_and(|id| id.starts_with("a11y_"))
        || candidate.candidate_id.contains("a11y_")
}

fn requested_roles_for_workflow(workflow: &ScreenWorkflow) -> Vec<String> {
    let mut roles = Vec::new();
    for step in &workflow.steps {
        let role = match step.step_kind {
            WorkflowStepKind::FocusSearchInput => Some("search_input"),
            WorkflowStepKind::OpenRankedResult => Some("ranked_result"),
            WorkflowStepKind::ClickVisibleElement => Some("button"),
            WorkflowStepKind::EnterText => Some("search_input"),
            _ => None,
        };
        if let Some(role) = role {
            if !roles.iter().any(|existing| existing == role) {
                roles.push(role.to_string());
            }
        }
    }
    roles
}

fn goal_loop_for_workflow_frame(
    workflow: &ScreenWorkflow,
    frame: SemanticScreenFrame,
) -> Option<crate::desktop_agent_types::GoalLoopRun> {
    let goal = goal_for_screen_workflow(workflow)?;
    Some(run_goal_loop_once(goal, frame))
}

fn goal_for_screen_workflow(workflow: &ScreenWorkflow) -> Option<GoalSpec> {
    let continuation = workflow.continuation.as_ref()?;
    let utterance = continuation
        .followup
        .interpretation
        .as_ref()
        .map(|interpretation| interpretation.utterance.clone())
        .unwrap_or_else(|| "screen-guided continuation".into());
    let provider = continuation
        .policy
        .provider
        .clone()
        .or_else(|| continuation.source_context.provider.clone());

    if provider.as_deref() == Some("youtube") && looks_like_channel_goal(&utterance) {
        let entity_name = continuation
            .followup
            .interpretation
            .as_ref()
            .and_then(|interpretation| interpretation.query_hint.clone())
            .or_else(|| continuation.source_context.query.clone());
        return Some(GoalSpec {
            goal_id: Uuid::new_v4().to_string(),
            goal_type: GoalType::OpenChannel,
            constraints: GoalConstraints {
                provider,
                item_kind: None,
                result_kind: None,
                rank_within_kind: None,
                rank_overall: None,
                entity_name,
                attributes: Value::Null,
            },
            success_condition: "channel_page_visible".into(),
            utterance,
            confidence: 0.86,
        });
    }

    let reference = continuation.followup.result_reference.as_ref()?;
    if !matches!(
        continuation.followup.action_kind,
        FollowupActionKind::OpenResult | FollowupActionKind::ClickResult
    ) {
        return None;
    }
    let item_kind = match reference.item_kind {
        ResultListItemKind::Video => Some("video".into()),
        ResultListItemKind::Link => Some("site".into()),
        ResultListItemKind::Result | ResultListItemKind::Unknown => None,
    };
    let rank = reference.rank.unwrap_or(1);
    let has_item_kind = item_kind.is_some();
    Some(goal_for_open_list_item(
        utterance,
        provider,
        item_kind,
        if has_item_kind { Some(rank) } else { None },
        if has_item_kind { None } else { Some(rank) },
    ))
}

fn looks_like_channel_goal(utterance: &str) -> bool {
    utterance
        .to_ascii_lowercase()
        .split(|ch: char| !ch.is_alphanumeric())
        .any(|token| matches!(token, "canale" | "channel"))
}

fn screen_workflow_run_from_goal_loop(
    mut workflow: ScreenWorkflow,
    primitive_capabilities: UIPrimitiveCapabilitySet,
    mut goal_loop: GoalLoopRun,
) -> ScreenWorkflowRun {
    if goal_loop_has_verified_success(&goal_loop) && goal_loop.status != GoalLoopStatus::GoalAchieved
    {
        goal_loop.status = GoalLoopStatus::GoalAchieved;
        goal_loop.failure_reason = None;
        goal_loop.verifier_status = Some("GoalAchieved".into());
        goal_loop.completion_diagnostics.status_downgrade_prevented = true;
        goal_loop.completion_diagnostics.goal_verifier_status =
            Some(GoalVerificationStatus::GoalAchieved);
        goal_loop
            .completion_diagnostics
            .goal_achieved_reason
            .get_or_insert_with(|| "verified success preserved during final aggregation".into());
    }
    if let Some(candidate) = goal_loop.selected_target_candidate.clone() {
        remember_goal_loop_candidate(&mut workflow, candidate);
    }
    for candidate in goal_loop
        .executed_steps
        .iter()
        .filter_map(|step| step.selected_target_candidate.clone())
    {
        remember_goal_loop_candidate(&mut workflow, candidate);
    }

    let step_runs = goal_loop
        .executed_steps
        .iter()
        .enumerate()
        .map(goal_loop_execution_to_step_run)
        .collect::<Vec<_>>();
    let planned_steps = workflow.steps.len().max(1);
    let status = workflow_status_for_goal_loop(&goal_loop);
    let executed_attempts = goal_loop
        .executed_steps
        .iter()
        .filter(|step| step.status == PlannerStepExecutionStatus::Executed)
        .count();
    let completed_steps = logical_completed_steps_for_goal_loop(
        &goal_loop,
        &status,
        planned_steps,
        executed_attempts,
    );
    goal_loop.completion_diagnostics.planned_steps = planned_steps;
    goal_loop.completion_diagnostics.logical_steps_completed = completed_steps;
    goal_loop.completion_diagnostics.executed_attempts = goal_loop.executed_steps.len();
    let stopped_reason = if status == WorkflowRunStatus::Completed {
        None
    } else {
        goal_loop.failure_reason.clone().or_else(|| {
            goal_loop
                .verification_history
                .last()
                .map(|verification| verification.reason.clone())
        })
    };

    workflow.grounding.goal_loop = Some(goal_loop);
    let continuation_verification = workflow.continuation.as_ref().map(|descriptor| {
        build_continuation_verification_result(
            descriptor,
            &status,
            completed_steps,
            planned_steps,
            stopped_reason.as_deref(),
            workflow.grounding.goal_loop.as_ref(),
        )
    });

    ScreenWorkflowRun {
        run_id: Uuid::new_v4().to_string(),
        status,
        workflow,
        primitive_capabilities,
        step_runs,
        completed_steps,
        stopped_reason,
        continuation_verification,
    }
}

fn remember_goal_loop_candidate(workflow: &mut ScreenWorkflow, candidate: UITargetCandidate) {
    if workflow
        .grounding
        .visible_target_candidates
        .iter()
        .any(|existing| existing.candidate_id == candidate.candidate_id)
    {
        return;
    }
    workflow.grounding.visible_target_candidates.push(candidate);
}

fn goal_loop_has_executed_action(goal_loop: &GoalLoopRun) -> bool {
    goal_loop
        .executed_steps
        .iter()
        .any(|step| step.status == PlannerStepExecutionStatus::Executed)
}

fn goal_loop_has_verified_success(goal_loop: &GoalLoopRun) -> bool {
    goal_loop.status == GoalLoopStatus::GoalAchieved
        || goal_loop
            .verification_history
            .iter()
            .any(|verification| verification.status == GoalVerificationStatus::GoalAchieved)
        || goal_loop.completion_diagnostics.goal_verifier_status
            == Some(GoalVerificationStatus::GoalAchieved)
}

fn logical_completed_steps_for_goal_loop(
    goal_loop: &GoalLoopRun,
    status: &WorkflowRunStatus,
    planned_steps: usize,
    executed_attempts: usize,
) -> usize {
    let logical_completed = if matches!(status, WorkflowRunStatus::Completed) {
        planned_steps
    } else if executed_attempts > 0 || goal_loop_has_executed_action(goal_loop) {
        planned_steps.min(1)
    } else {
        0
    };
    logical_completed.min(planned_steps)
}

fn goal_loop_has_ambiguous_final_verification(goal_loop: &GoalLoopRun) -> bool {
    goal_loop
        .verification_history
        .last()
        .is_some_and(|verification| {
            matches!(
                verification.status,
                GoalVerificationStatus::Ambiguous | GoalVerificationStatus::PageChangedWrongOutcome
            )
        })
}

fn goal_loop_should_project_partial_completion(goal_loop: &GoalLoopRun) -> bool {
    goal_loop_has_executed_action(goal_loop)
        && (goal_loop.post_action_progress_observed
            || goal_loop_has_ambiguous_final_verification(goal_loop)
            || goal_loop.surface_ownership_lost
            || matches!(
                goal_loop.browser_recovery_status,
                crate::desktop_agent_types::BrowserRecoveryStatus::Failed
            ))
}

fn workflow_status_for_goal_loop(goal_loop: &GoalLoopRun) -> WorkflowRunStatus {
    match goal_loop.status {
        GoalLoopStatus::GoalAchieved => WorkflowRunStatus::Completed,
        GoalLoopStatus::NeedsPerception => WorkflowRunStatus::NeedsScreenContext,
        GoalLoopStatus::NeedsExecution | GoalLoopStatus::Running => {
            WorkflowRunStatus::PartiallyCompleted
        }
        GoalLoopStatus::NeedsClarification | GoalLoopStatus::Refused => {
            if goal_loop_should_project_partial_completion(goal_loop) {
                WorkflowRunStatus::PartiallyCompleted
            } else {
                WorkflowRunStatus::NeedsTargetGrounding
            }
        }
        GoalLoopStatus::BudgetExhausted | GoalLoopStatus::VerificationFailed
            if goal_loop_should_project_partial_completion(goal_loop) =>
        {
            WorkflowRunStatus::PartiallyCompleted
        }
        GoalLoopStatus::BrowserHandoffFailed
        | GoalLoopStatus::ScrollRequiredButUnsupported
        | GoalLoopStatus::BudgetExhausted
        | GoalLoopStatus::VerificationFailed => WorkflowRunStatus::StepFailed,
    }
}

fn goal_loop_execution_to_step_run(
    (index, execution): (usize, &PlannerStepExecutionRecord),
) -> WorkflowStepRun {
    let primitive = if execution.primitive.contains("ClickTargetCandidate") {
        Some(UIPrimitiveKind::ClickTargetCandidate)
    } else {
        None
    };
    let step_kind = execution
        .selected_target_candidate
        .as_ref()
        .and_then(|candidate| candidate.result_kind.as_deref())
        .map(|kind| {
            if kind == "channel" {
                WorkflowStepKind::ClickVisibleElement
            } else {
                WorkflowStepKind::OpenRankedResult
            }
        })
        .unwrap_or(WorkflowStepKind::ClickVisibleElement);
    let status = match execution.status {
        PlannerStepExecutionStatus::Executed => WorkflowRunStatus::Completed,
        PlannerStepExecutionStatus::Unsupported => WorkflowRunStatus::StepUnsupported,
        PlannerStepExecutionStatus::Failed => WorkflowRunStatus::StepFailed,
        PlannerStepExecutionStatus::Skipped => WorkflowRunStatus::Aborted,
    };
    let verification = match execution.status {
        PlannerStepExecutionStatus::Executed => StepVerificationStatus::PartiallySatisfied,
        PlannerStepExecutionStatus::Unsupported => StepVerificationStatus::Unsupported,
        PlannerStepExecutionStatus::Failed => StepVerificationStatus::Failed,
        PlannerStepExecutionStatus::Skipped => StepVerificationStatus::NotRun,
    };
    let target_selection = execution
        .selected_target_candidate
        .clone()
        .map(goal_loop_target_selection);

    WorkflowStepRun {
        step: WorkflowStep {
            step_id: format!("goal_loop_step_{}", index + 1),
            step_kind,
            target: json!({}),
            value: None,
            selection: json!({}),
            expected_outcome: None,
        },
        primitive: primitive.clone(),
        status,
        verification,
        primitive_result: primitive.map(|primitive| UIPrimitiveResult {
            primitive,
            status: match execution.status {
                PlannerStepExecutionStatus::Executed => UIPrimitiveStatus::Executed,
                PlannerStepExecutionStatus::Unsupported => UIPrimitiveStatus::Unsupported,
                PlannerStepExecutionStatus::Failed | PlannerStepExecutionStatus::Skipped => {
                    UIPrimitiveStatus::Failed
                }
            },
            message: execution.message.clone(),
            geometry: execution.geometry.clone(),
        }),
        target_selection,
        note: Some(execution.message.clone()),
    }
}

fn goal_loop_target_selection(candidate: UITargetCandidate) -> TargetSelection {
    TargetSelection::selected(
        candidate.clone(),
        vec![candidate],
        TargetSelectionPolicy::default().min_click_confidence,
        TargetSelectionDiagnostics::default(),
    )
}

fn verified_continuation_learning_event(
    run: &ScreenWorkflowRun,
) -> Option<VerifiedContinuationLearningEvent> {
    if !matches!(
        run.status,
        crate::screen_workflow::WorkflowRunStatus::Completed
    ) {
        return None;
    }
    let continuation = run.workflow.continuation.as_ref()?;
    let interpretation = continuation.followup.interpretation.clone()?;
    let phrase = interpretation.utterance.clone();
    Some(VerifiedContinuationLearningEvent {
        phrase,
        interpretation,
        merge: continuation.followup.merge_diagnostic.clone(),
        page_validation: continuation.page_validation.clone(),
        regrounding: continuation.regrounding.clone(),
        outcome: VerifiedContinuationOutcome {
            run_id: run.run_id.clone(),
            status: run.status.as_str().into(),
            completed_steps: run.completed_steps,
            verifier_status: run
                .continuation_verification
                .as_ref()
                .map(|verification| format!("{:?}", verification.status)),
        },
    })
}

fn value_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::{
        accessibility_selector_click_decision, enrich_frame_with_accessibility, DesktopAgentRuntime,
    };
    use crate::{
        accessibility_layer::{
            synthesize_ranked_uia_result_candidates, validate_accessibility_target_selection,
            AccessibilitySnapshot, AccessibilityTargetSelection, AccessibleElement,
        },
        action_resolution::{ActionOperation, ResolutionSource},
        desktop_agent_types::{
            DesktopActionRequest, DesktopActionStatus, GoalConstraints, GoalLoopRun,
            GoalLoopStatus, GoalSpec, GoalType, GoalVerificationStatus, PendingApproval,
            PlannerContractInput, PlannerStepExecutionRecord, PlannerStepExecutionStatus,
            VisibleActionabilityDiagnostic, VisibleResultKind,
        },
        screen_workflow::{
            ScreenFreshness, ScreenGroundingState, ScreenWorkflow, ScreenWorkflowDomain,
            WorkflowSupportSummary,
        },
        semantic_frame::{run_goal_loop_once, semantic_frame_from_vision_value},
        ui_control::UIPrimitiveCapabilitySet,
        ui_target_grounding::{
            TargetGroundingSource, TargetRegion, UITargetCandidate, UITargetRole,
        },
    };
    use serde_json::json;
    use uuid::Uuid;

    fn accessibility_snapshot(elements: Vec<AccessibleElement>) -> AccessibilitySnapshot {
        AccessibilitySnapshot {
            snapshot_id: "uia_test_snapshot".into(),
            element_count: elements.len(),
            elements,
            browser_url: None,
            browser_window_bounds: None,
            captured_at_ms: 1_000,
            capture_backend: "powershell_uia".into(),
            window_is_foreground: true,
            window_pid: Some(1234),
            window_process_name: Some("chrome".into()),
            error: None,
        }
    }

    fn accessible_element(
        element_id: &str,
        bounding_rect: Option<TargetRegion>,
    ) -> AccessibleElement {
        AccessibleElement {
            element_id: element_id.into(),
            automation_id: None,
            runtime_id: None,
            role: "hyperlink".into(),
            name: Some("Example result".into()),
            value: None,
            bounding_rect,
            is_enabled: true,
            is_offscreen: false,
            depth: 3,
            parent_id: None,
            children: Vec::new(),
        }
    }

    #[test]
    fn filesystem_write_creates_persisted_pending_approval() {
        let root = std::env::temp_dir().join(format!("astra_desktop_agent_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&root).expect("temp root");
        let target_path = root.join("test.txt");
        let runtime = DesktopAgentRuntime::new(root.clone());

        let response = runtime
            .submit_action(
                "test-request".into(),
                DesktopActionRequest {
                    tool_name: "filesystem.write_text".into(),
                    params: json!({
                        "path": target_path.display().to_string(),
                        "content": "",
                        "mode": "overwrite",
                        "create_empty": true,
                    }),
                    preview_only: false,
                    reason: Some("test create empty file".into()),
                },
            )
            .expect("submit action");

        assert!(matches!(
            response.status,
            DesktopActionStatus::ApprovalRequired
        ));
        let pending_path = root
            .join(".astra")
            .join("state")
            .join("pending_approvals.json");
        assert!(pending_path.exists());
        let approvals = serde_json::from_str::<Vec<PendingApproval>>(
            &std::fs::read_to_string(&pending_path).expect("pending approvals file"),
        )
        .expect("pending approval json");
        assert_eq!(approvals.len(), 1);
        assert_eq!(approvals[0].tool_name, "filesystem.write_text");

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn launch_target_resolves_chrome_alias_without_losing_args_contract() {
        let target = super::resolve_launch_target("google-chrome");
        assert!(target.to_ascii_lowercase().contains("chrome"));
    }

    #[test]
    fn enrich_frame_with_accessibility_populates_click_regions_from_os_bounds() {
        let mut frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {
                    "content_provider_hint": "google",
                    "page_kind_hint": "search_results",
                    "confidence": 0.9
                },
                "primary_list": {
                    "container_kind": "result_list",
                    "items": [{
                        "item_id": "a11y_7",
                        "element_id": "a11y_7",
                        "rank": 1,
                        "title": "Example result",
                        "item_kind": "site",
                        "confidence": 0.9
                    }]
                }
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("semantic frame");
        let snapshot = accessibility_snapshot(vec![accessible_element(
            "a11y_7",
            Some(TargetRegion {
                x: 120.0,
                y: 240.0,
                width: 500.0,
                height: 42.0,
                coordinate_space: "screen".into(),
            }),
        )]);

        enrich_frame_with_accessibility(&mut frame, &snapshot);

        let item = &frame.primary_list.as_ref().expect("primary list").items[0];
        let region = &item
            .click_regions
            .get("primary")
            .expect("primary region")
            .region;
        assert_eq!(region.x, 120.0);
        assert_eq!(region.coordinate_space, "screen");
        assert_eq!(
            item.attributes
                .get("accessibility_sourced")
                .and_then(serde_json::Value::as_bool),
            Some(true)
        );
        assert_eq!(
            item.attributes
                .get("accessibility_snapshot_id")
                .and_then(serde_json::Value::as_str),
            Some("uia_test_snapshot")
        );
    }

    #[test]
    fn enrich_frame_with_accessibility_skips_elements_without_bounding_rect() {
        let mut frame = semantic_frame_from_vision_value(
            &json!({
                "visible_result_items": [{
                    "item_id": "a11y_3",
                    "element_id": "a11y_3",
                    "kind": "generic",
                    "title": "No bounds",
                    "rank_overall": 1,
                    "confidence": 0.9
                }]
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("semantic frame");
        let snapshot = accessibility_snapshot(vec![accessible_element("a11y_3", None)]);

        enrich_frame_with_accessibility(&mut frame, &snapshot);

        assert!(frame.visible_result_items[0].click_regions.is_empty());
    }

    #[test]
    fn perceive_continues_when_accessibility_snapshot_is_empty() {
        let mut frame = semantic_frame_from_vision_value(
            &json!({
                "visible_result_items": [{
                    "item_id": "result_1",
                    "kind": "generic",
                    "title": "Vision result",
                    "rank_overall": 1,
                    "confidence": 0.9
                }]
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("semantic frame");
        let snapshot = AccessibilitySnapshot {
            snapshot_id: "uia_unavailable".into(),
            elements: Vec::new(),
            browser_url: None,
            browser_window_bounds: None,
            captured_at_ms: 1_000,
            capture_backend: "unavailable".into(),
            element_count: 0,
            window_is_foreground: false,
            window_pid: None,
            window_process_name: None,
            error: Some("uia unavailable".into()),
        };

        enrich_frame_with_accessibility(&mut frame, &snapshot);

        assert_eq!(frame.visible_result_items.len(), 1);
        assert!(frame.legacy_target_candidates.is_empty());
    }

    #[test]
    fn open_list_item_uses_uia_candidates_when_semantic_candidates_empty() {
        let mut frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {
                    "content_provider_hint": "example",
                    "page_kind_hint": "browser",
                    "query_hint": "docs",
                    "confidence": 0.88
                },
                "scene_summary": "A result list is visible",
                "page_state": {
                    "kind": "list",
                    "dominant_content": "result_list",
                    "list_visible": true
                }
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("semantic frame");
        let snapshot = accessibility_snapshot(vec![accessible_element(
            "a11y_1",
            Some(TargetRegion {
                x: 100.0,
                y: 160.0,
                width: 480.0,
                height: 44.0,
                coordinate_space: "screen".into(),
            }),
        )]);

        enrich_frame_with_accessibility(&mut frame, &snapshot);

        assert_eq!(frame.visible_result_items.len(), 1);
        assert_eq!(
            frame.visible_result_items[0].element_id.as_deref(),
            Some("a11y_1")
        );
        let goal = GoalSpec {
            goal_id: "goal".into(),
            goal_type: GoalType::OpenListItem,
            constraints: GoalConstraints {
                provider: Some("example".into()),
                item_kind: Some("result".into()),
                result_kind: Some(VisibleResultKind::Generic),
                rank_within_kind: Some(1),
                rank_overall: None,
                entity_name: None,
                attributes: serde_json::Value::Null,
            },
            success_condition: "list_item_detail_open".into(),
            utterance: "open the first result".into(),
            confidence: 0.9,
        };

        let run = run_goal_loop_once(goal, frame);
        let candidate = run.planner_steps[0]
            .executable_candidate
            .as_ref()
            .expect("uia candidate selected");

        assert_eq!(run.status, GoalLoopStatus::NeedsExecution);
        assert_eq!(candidate.element_id.as_deref(), Some("a11y_1"));
        assert_eq!(
            candidate.accessibility_snapshot_id.as_deref(),
            Some("uia_test_snapshot")
        );
        assert!(!candidate.reuse_eligible);
    }

    #[test]
    fn browser_open_list_item_prefers_llm_uia_selection_over_vision_candidate() {
        let mut frame = semantic_frame_from_vision_value(
            &json!({
                "page_evidence": {
                    "content_provider_hint": "example",
                    "page_kind_hint": "browser",
                    "result_list_visible": true,
                    "confidence": 0.88
                },
                "scene_summary": "A result list is visible",
                "visible_result_items": [{
                    "item_id": "vision_1",
                    "kind": "video",
                    "title": "Misleading sidebar item",
                    "rank_within_kind": 1,
                    "click_regions": {
                        "title": {"x": 1600, "y": 400, "width": 250, "height": 250, "coordinate_space": "screen"}
                    },
                    "confidence": 0.95
                }]
            }),
            1_000,
            None,
            None,
            Vec::new(),
        )
        .expect("semantic frame");
        let mut sidebar = accessible_element(
            "a11y_1",
            Some(TargetRegion {
                x: 1500.0,
                y: 240.0,
                width: 280.0,
                height: 80.0,
                coordinate_space: "screen".into(),
            }),
        );
        sidebar.name = Some("Sidebar channel".into());
        let mut main_result = accessible_element(
            "a11y_2",
            Some(TargetRegion {
                x: 300.0,
                y: 320.0,
                width: 620.0,
                height: 92.0,
                coordinate_space: "screen".into(),
            }),
        );
        main_result.name = Some("First main result".into());
        let snapshot = accessibility_snapshot(vec![sidebar, main_result]);
        enrich_frame_with_accessibility(&mut frame, &snapshot);
        let candidates = synthesize_ranked_uia_result_candidates(&snapshot);
        let selection = AccessibilityTargetSelection {
            selected_element_id: Some("a11y_2".into()),
            accessibility_snapshot_id: Some(snapshot.snapshot_id.clone()),
            selection_kind: Some("ranked_result".into()),
            rank: Some(1),
            confidence: 0.93,
            rationale: Some("The selected element is the main content result.".into()),
        };
        let selected_candidate = validate_accessibility_target_selection(
            &selection,
            &snapshot,
            &candidates,
            crate::ui_target_grounding::TargetSelectionPolicy::default().min_click_confidence,
        )
        .expect("validated UIA selection");
        let goal = GoalSpec {
            goal_id: "goal".into(),
            goal_type: GoalType::OpenListItem,
            constraints: GoalConstraints {
                provider: Some("example".into()),
                item_kind: Some("video".into()),
                result_kind: Some(VisibleResultKind::Video),
                rank_within_kind: Some(1),
                rank_overall: None,
                entity_name: None,
                attributes: serde_json::Value::Null,
            },
            success_condition: "list_item_detail_open".into(),
            utterance: "open the first video".into(),
            confidence: 0.9,
        };
        let input = PlannerContractInput {
            goal,
            current_frame: frame,
            executed_steps: Vec::new(),
            verification_history: Vec::new(),
            perception_requests: Vec::new(),
            retry_budget: 3,
            retries_used: 0,
            visible_refinement_attempts: 0,
            max_visible_refinement_passes: 1,
            provider_hint: Some("example".into()),
            browser_app_hint: Some("browser".into()),
            page_kind_hint: Some("browser".into()),
            visible_actionability: VisibleActionabilityDiagnostic::default(),
        };

        let decision = accessibility_selector_click_decision(
            &input,
            selected_candidate,
            &selection,
            candidates.len(),
            true,
        );

        assert_eq!(
            decision.proposed_step.target_item_id.as_deref(),
            Some("a11y_2")
        );
        let candidate = decision
            .proposed_step
            .executable_candidate
            .as_ref()
            .expect("UIA candidate");
        assert_eq!(candidate.element_id.as_deref(), Some("a11y_2"));
        assert_eq!(candidate.source, TargetGroundingSource::AccessibilityLayer);
        assert!(decision
            .strategy_rationale
            .contains("vision_only_candidate_suppressed=true"));
        assert!(decision
            .strategy_rationale
            .contains("llm_uia_selector_used=true"));
    }

    #[test]
    fn stale_accessibility_snapshot_is_not_used_for_enrichment() {
        let mut frame = semantic_frame_from_vision_value(
            &json!({
                "visible_result_items": [{
                    "item_id": "a11y_9",
                    "kind": "generic",
                    "title": "Stale result",
                    "rank_overall": 1,
                    "confidence": 0.9
                }]
            }),
            7_000,
            None,
            None,
            Vec::new(),
        )
        .expect("semantic frame");
        let mut snapshot = accessibility_snapshot(vec![accessible_element(
            "a11y_9",
            Some(TargetRegion {
                x: 120.0,
                y: 240.0,
                width: 500.0,
                height: 42.0,
                coordinate_space: "screen".into(),
            }),
        )]);
        snapshot.captured_at_ms = 1_000;

        enrich_frame_with_accessibility(&mut frame, &snapshot);

        let item = &frame.visible_result_items[0];
        assert!(item.click_regions.is_empty());
        assert!(item.element_id.is_none());
        assert!(item.attributes.get("accessibility_sourced").is_none());
        assert!(frame
            .uncertainty
            .iter()
            .any(|uncertainty| uncertainty.code == "stale_accessibility_snapshot"));
    }

    #[test]
    fn goal_loop_run_is_projected_back_into_screen_workflow_diagnostics() {
        let candidate = UITargetCandidate {
            candidate_id: "video_candidate".into(),
            element_id: None,
            accessibility_snapshot_id: None,
            role: UITargetRole::RankedResult,
            region: Some(TargetRegion {
                x: 10.0,
                y: 20.0,
                width: 200.0,
                height: 100.0,
                coordinate_space: "screen".into(),
            }),
            center_x: None,
            center_y: None,
            app_hint: None,
            browser_app_hint: Some("chrome".into()),
            provider_hint: Some("youtube".into()),
            content_provider_hint: Some("youtube".into()),
            page_kind_hint: Some("search_results".into()),
            capture_backend: Some("powershell_gdi".into()),
            observation_source: Some("test".into()),
            result_kind: Some("video".into()),
            confidence: 0.95,
            source: TargetGroundingSource::ScreenAnalysis,
            label: Some("Shiva video".into()),
            rank: Some(1),
            observed_at_ms: Some(1_000),
            reuse_eligible: true,
            supports_focus: false,
            supports_click: true,
            rationale: "test candidate".into(),
        };
        let workflow = ScreenWorkflow {
            operation: ActionOperation::ScreenGuidedFollowupAction,
            domain: ScreenWorkflowDomain::BrowserScreenInteraction,
            requires_screen_context: true,
            depends_on_recent_screen_context: true,
            continuation: None,
            grounding: ScreenGroundingState {
                observation_supported: true,
                observation_enabled: true,
                capture_available: true,
                analysis_available: true,
                recent_capture_available: true,
                recent_capture_age_ms: Some(100),
                last_capture_path: Some("screen.png".into()),
                freshness: ScreenFreshness::RecentAvailable,
                fresh_capture_required: false,
                sufficient_for_workflow: true,
                visible_target_candidates: Vec::new(),
                page_evidence: Vec::new(),
                semantic_frame: None,
                goal_loop: None,
                recent_target_candidates: Vec::new(),
                generated_at_ms: Some(1_000),
                page_validation: None,
                regrounding: None,
                uncertainty: Vec::new(),
            },
            steps: Vec::new(),
            step_plans: Vec::new(),
            support: WorkflowSupportSummary {
                executable: true,
                requires_screen_context: true,
                unsupported_steps: Vec::new(),
                reason: "test".into(),
            },
            confidence: 0.9,
            source: ResolutionSource::RustNormalizer,
            rationale: Some("test".into()),
        };
        let goal_loop = GoalLoopRun {
            run_id: "loop".into(),
            goal: GoalSpec {
                goal_id: "goal".into(),
                goal_type: GoalType::OpenMediaResult,
                constraints: GoalConstraints {
                    provider: Some("youtube".into()),
                    item_kind: Some("video".into()),
                    result_kind: Some(VisibleResultKind::Video),
                    rank_within_kind: Some(1),
                    rank_overall: None,
                    entity_name: None,
                    attributes: serde_json::Value::Null,
                },
                success_condition: "video_watch_page_open".into(),
                utterance: "aprimi il primo video".into(),
                confidence: 0.9,
            },
            status: GoalLoopStatus::GoalAchieved,
            iteration_count: 2,
            retry_budget: 3,
            retries_used: 0,
            current_strategy: Some("youtube_open_first_visible_media_result".into()),
            fallback_strategy_state: None,
            frames: Vec::new(),
            planner_steps: Vec::new(),
            planner_diagnostics: Vec::new(),
            executed_steps: vec![PlannerStepExecutionRecord {
                step_id: "step".into(),
                status: PlannerStepExecutionStatus::Executed,
                primitive: "ClickTargetCandidate".into(),
                message: "clicked".into(),
                selected_target_candidate: Some(candidate.clone()),
                geometry: None,
                fresh_capture_required: true,
                fresh_capture_used: true,
                target_signature: Some("planner_result_v1_title".into()),
            }],
            verification_history: Vec::new(),
            focused_perception_requests: Vec::new(),
            browser_handoff_history: Vec::new(),
            browser_handoff: None,
            verified_surface: None,
            surface_diagnostics: Vec::new(),
            focused_perception_used: false,
            visible_refinement_used: false,
            stale_capture_reuse_prevented: true,
            browser_recovery_used: false,
            browser_recovery_status: crate::desktop_agent_types::BrowserRecoveryStatus::NotNeeded,
            post_action_progress_observed: true,
            surface_ownership_lost: false,
            focused_perception_failure_reason: None,
            repeated_click_protection_triggered: false,
            selected_target_candidate: Some(candidate),
            verifier_status: Some("GoalAchieved".into()),
            failure_reason: None,
        };

        let run = super::screen_workflow_run_from_goal_loop(
            workflow,
            UIPrimitiveCapabilitySet::for_runtime(true),
            goal_loop,
        );

        assert_eq!(
            run.status,
            crate::screen_workflow::WorkflowRunStatus::Completed
        );
        assert_eq!(run.completed_steps, 1);
        assert!(run.workflow.grounding.goal_loop.is_some());
        assert_eq!(run.workflow.grounding.visible_target_candidates.len(), 1);
        assert!(run.step_runs[0]
            .target_selection
            .as_ref()
            .and_then(|selection| selection.selected_candidate.as_ref())
            .is_some());
    }

    #[test]
    fn workflow_status_projects_executed_ambiguous_goal_loop_as_partial() {
        let goal_loop = GoalLoopRun {
            run_id: "loop".into(),
            goal: GoalSpec {
                goal_id: "goal".into(),
                goal_type: GoalType::OpenMediaResult,
                constraints: GoalConstraints {
                    provider: Some("youtube".into()),
                    item_kind: Some("video".into()),
                    result_kind: Some(VisibleResultKind::Video),
                    rank_within_kind: Some(1),
                    rank_overall: None,
                    entity_name: None,
                    attributes: serde_json::Value::Null,
                },
                success_condition: "video_watch_page_open".into(),
                utterance: "aprimi il primo video".into(),
                confidence: 0.9,
            },
            status: GoalLoopStatus::Refused,
            iteration_count: 2,
            retry_budget: 3,
            retries_used: 0,
            current_strategy: Some("youtube_open_first_visible_media_result".into()),
            fallback_strategy_state: None,
            frames: Vec::new(),
            planner_steps: Vec::new(),
            planner_diagnostics: Vec::new(),
            executed_steps: vec![PlannerStepExecutionRecord {
                step_id: "step".into(),
                status: PlannerStepExecutionStatus::Executed,
                primitive: "ClickTargetCandidate".into(),
                message: "clicked".into(),
                selected_target_candidate: None,
                geometry: None,
                fresh_capture_required: true,
                fresh_capture_used: true,
                target_signature: Some("planner_result_v1_title".into()),
            }],
            verification_history: vec![crate::desktop_agent_types::GoalVerificationRecord {
                iteration: 1,
                status: GoalVerificationStatus::Ambiguous,
                confidence: 0.71,
                reason: "page changed but final confirmation remained uncertain".into(),
                frame_id: Some("frame_2".into()),
            }],
            focused_perception_requests: Vec::new(),
            browser_handoff_history: Vec::new(),
            browser_handoff: None,
            verified_surface: None,
            surface_diagnostics: Vec::new(),
            focused_perception_used: false,
            visible_refinement_used: false,
            stale_capture_reuse_prevented: true,
            browser_recovery_used: false,
            browser_recovery_status: crate::desktop_agent_types::BrowserRecoveryStatus::NotNeeded,
            post_action_progress_observed: true,
            surface_ownership_lost: false,
            focused_perception_failure_reason: None,
            repeated_click_protection_triggered: false,
            selected_target_candidate: None,
            verifier_status: Some("Ambiguous".into()),
            failure_reason: Some("final watch-page confirmation remained uncertain".into()),
        };

        assert_eq!(
            super::workflow_status_for_goal_loop(&goal_loop),
            crate::screen_workflow::WorkflowRunStatus::PartiallyCompleted
        );
    }

    #[test]
    fn screen_workflow_run_from_goal_loop_hydrates_goal_aware_continuation_verification() {
        let workflow = ScreenWorkflow {
            operation: ActionOperation::ScreenGuidedFollowupAction,
            domain: ScreenWorkflowDomain::BrowserScreenInteraction,
            requires_screen_context: true,
            depends_on_recent_screen_context: true,
            continuation: Some(crate::workflow_continuation::WorkflowContinuationDescriptor {
                followup: crate::workflow_continuation::FollowupActionResolution {
                    action_kind: crate::workflow_continuation::FollowupActionKind::OpenResult,
                    result_reference: Some(crate::workflow_continuation::ResultListReference {
                        rank: Some(1),
                        ordinal_label: "primo".into(),
                        item_kind: crate::workflow_continuation::ResultListItemKind::Video,
                    }),
                    text_value: None,
                    provider_hint: Some("youtube".into()),
                    query_hint: Some("shiva".into()),
                    browser_hint: Some("chrome".into()),
                    app_hint: Some("chrome".into()),
                    page_context_hint: Some(
                        crate::workflow_continuation::PageContextHint::RecentResultsPage,
                    ),
                    requires_result_list: true,
                    requires_recent_focus_target: false,
                    confidence: 0.91,
                    source: crate::workflow_continuation::FollowupResolutionSource::RustFollowupResolver,
                    rationale: "test".into(),
                    interpretation: None,
                    merge_diagnostic: None,
                },
                policy: crate::workflow_continuation::ContinuationPolicyDecision {
                    status: crate::workflow_continuation::ContinuationPolicyStatus::SafeToAttempt,
                    executable: true,
                    reason: "test".into(),
                    context_age_ms: Some(100),
                    screen_age_ms: Some(100),
                    required_rank: Some(1),
                    provider: Some("youtube".into()),
                    query: Some("shiva".into()),
                    fresh_capture_required: false,
                    recent_candidate_reuse_allowed: true,
                    max_reusable_screen_age_ms: Some(60_000),
                    freshness_reason: None,
                },
                source_context: crate::workflow_continuation::RecentWorkflowContextSummary {
                    context_id: "ctx".into(),
                    kind: crate::workflow_continuation::RecentWorkflowContextKind::BrowserSearchResults,
                    page_kind: crate::workflow_continuation::BrowserPageKind::SearchResults,
                    provider: Some("youtube".into()),
                    app: Some("chrome".into()),
                    url: Some("https://www.youtube.com/results?search_query=shiva".into()),
                    query: Some("shiva".into()),
                    has_result_list: true,
                    has_recent_focused_target: false,
                    has_recent_selected_target: false,
                    last_run_status: Some("completed".into()),
                    resumable: false,
                    continuation_allowed: true,
                    updated_at_ms: 1_000,
                    expires_at_ms: 2_000,
                },
                verifier: crate::workflow_continuation::ContinuationVerifierExpectation {
                    verifier_kind: crate::workflow_continuation::ContinuationVerifierKind::ResultNavigationExpected,
                    expected_state_change: "watch page visible".into(),
                    requires_post_step_screen_check: true,
                },
                page_validation: None,
                regrounding: None,
            }),
            grounding: ScreenGroundingState {
                observation_supported: true,
                observation_enabled: true,
                capture_available: true,
                analysis_available: true,
                recent_capture_available: true,
                recent_capture_age_ms: Some(100),
                last_capture_path: Some("screen.png".into()),
                freshness: ScreenFreshness::RecentAvailable,
                fresh_capture_required: false,
                sufficient_for_workflow: true,
                visible_target_candidates: Vec::new(),
                page_evidence: Vec::new(),
                semantic_frame: None,
                goal_loop: None,
                recent_target_candidates: Vec::new(),
                generated_at_ms: Some(1_000),
                page_validation: None,
                regrounding: None,
                uncertainty: Vec::new(),
            },
            steps: Vec::new(),
            step_plans: Vec::new(),
            support: WorkflowSupportSummary {
                executable: true,
                requires_screen_context: true,
                unsupported_steps: Vec::new(),
                reason: "test".into(),
            },
            confidence: 0.9,
            source: ResolutionSource::RustNormalizer,
            rationale: Some("test".into()),
        };
        let goal_loop = GoalLoopRun {
            run_id: "loop".into(),
            goal: GoalSpec {
                goal_id: "goal".into(),
                goal_type: GoalType::OpenMediaResult,
                constraints: GoalConstraints {
                    provider: Some("youtube".into()),
                    item_kind: Some("video".into()),
                    result_kind: Some(VisibleResultKind::Video),
                    rank_within_kind: Some(1),
                    rank_overall: None,
                    entity_name: None,
                    attributes: serde_json::Value::Null,
                },
                success_condition: "video_watch_page_open".into(),
                utterance: "aprimi il primo video".into(),
                confidence: 0.9,
            },
            status: GoalLoopStatus::GoalAchieved,
            iteration_count: 2,
            retry_budget: 3,
            retries_used: 0,
            current_strategy: Some("youtube_open_first_visible_media_result".into()),
            fallback_strategy_state: None,
            frames: Vec::new(),
            planner_steps: Vec::new(),
            planner_diagnostics: Vec::new(),
            executed_steps: vec![PlannerStepExecutionRecord {
                step_id: "step".into(),
                status: PlannerStepExecutionStatus::Executed,
                primitive: "ClickTargetCandidate".into(),
                message: "clicked".into(),
                selected_target_candidate: None,
                geometry: None,
                fresh_capture_required: true,
                fresh_capture_used: true,
                target_signature: Some("planner_result_v1_title".into()),
            }],
            verification_history: vec![crate::desktop_agent_types::GoalVerificationRecord {
                iteration: 1,
                status: GoalVerificationStatus::GoalAchieved,
                confidence: 0.93,
                reason: "watch page visible".into(),
                frame_id: Some("frame_1".into()),
            }],
            focused_perception_requests: Vec::new(),
            browser_handoff_history: Vec::new(),
            browser_handoff: None,
            verified_surface: None,
            surface_diagnostics: Vec::new(),
            focused_perception_used: false,
            visible_refinement_used: false,
            stale_capture_reuse_prevented: true,
            browser_recovery_used: false,
            browser_recovery_status: crate::desktop_agent_types::BrowserRecoveryStatus::NotNeeded,
            post_action_progress_observed: true,
            surface_ownership_lost: false,
            focused_perception_failure_reason: None,
            repeated_click_protection_triggered: false,
            selected_target_candidate: None,
            verifier_status: Some("GoalAchieved".into()),
            failure_reason: None,
        };

        let run = super::screen_workflow_run_from_goal_loop(
            workflow,
            UIPrimitiveCapabilitySet::for_runtime(true),
            goal_loop,
        );

        assert_eq!(
            run.status,
            crate::screen_workflow::WorkflowRunStatus::Completed
        );
        assert_eq!(
            run.continuation_verification
                .as_ref()
                .map(|verification| &verification.status),
            Some(&crate::workflow_continuation::ContinuationVerificationStatus::GoalAchieved)
        );
    }

    #[test]
    fn workflow_status_keeps_unexecuted_refusal_as_needs_target_grounding() {
        let goal_loop = GoalLoopRun {
            run_id: "loop".into(),
            goal: GoalSpec {
                goal_id: "goal".into(),
                goal_type: GoalType::OpenMediaResult,
                constraints: GoalConstraints {
                    provider: Some("youtube".into()),
                    item_kind: Some("video".into()),
                    result_kind: Some(VisibleResultKind::Video),
                    rank_within_kind: Some(1),
                    rank_overall: None,
                    entity_name: None,
                    attributes: serde_json::Value::Null,
                },
                success_condition: "video_watch_page_open".into(),
                utterance: "aprimi il primo video".into(),
                confidence: 0.9,
            },
            status: GoalLoopStatus::Refused,
            iteration_count: 1,
            retry_budget: 3,
            retries_used: 0,
            current_strategy: Some("youtube_open_first_visible_media_result".into()),
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
            browser_recovery_status: crate::desktop_agent_types::BrowserRecoveryStatus::NotNeeded,
            post_action_progress_observed: false,
            surface_ownership_lost: false,
            focused_perception_failure_reason: None,
            repeated_click_protection_triggered: false,
            selected_target_candidate: None,
            verifier_status: Some("Refused".into()),
            failure_reason: Some("no sufficiently grounded candidate was found".into()),
        };

        assert_eq!(
            super::workflow_status_for_goal_loop(&goal_loop),
            crate::screen_workflow::WorkflowRunStatus::NeedsTargetGrounding
        );
    }
}
