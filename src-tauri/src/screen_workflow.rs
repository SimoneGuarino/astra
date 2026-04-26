use crate::{
    action_resolution::{ActionOperation, ActionResolution, ResolutionSource},
    desktop_agent_types::{
        CapabilityManifest, GoalLoopRun, GoalLoopStatus, PageSemanticEvidence,
        PlannerStepExecutionStatus, SemanticScreenFrame,
    },
    ui_control::{
        UIControlRuntime, UIPrimitiveCapabilitySet, UIPrimitiveKind, UIPrimitiveRequest,
        UIPrimitiveResult, UIPrimitiveStatus,
    },
    ui_target_grounding::{
        ground_targets_for_request, select_target_candidate, structured_candidates_from_value,
        TargetAction, TargetGroundingRequest, TargetGroundingSource, TargetSelection,
        TargetSelectionPolicy, TargetSelectionStatus, UITargetCandidate, UITargetRole,
    },
    workflow_continuation::{
        build_continuation_verification_result, ContinuationRegroundingDiagnostics,
        ContinuationVerificationResult, ContinuationVerificationStatus,
        SemanticPageValidationResult, SemanticPageValidationStatus, WorkflowContinuationDescriptor,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScreenWorkflowDomain {
    BrowserScreenInteraction,
    ScreenInteraction,
    ScreenNavigation,
}

impl ScreenWorkflowDomain {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::BrowserScreenInteraction => "browser_screen_interaction",
            Self::ScreenInteraction => "screen_interaction",
            Self::ScreenNavigation => "screen_navigation",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowStepKind {
    LocateBrowserContext,
    FocusSearchInput,
    EnterText,
    SubmitSearch,
    OpenRankedResult,
    ClickVisibleElement,
    NavigateBack,
    VerifyScreenState,
}

impl WorkflowStepKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::LocateBrowserContext => "locate_browser_context",
            Self::FocusSearchInput => "focus_search_input",
            Self::EnterText => "enter_text",
            Self::SubmitSearch => "submit_search",
            Self::OpenRankedResult => "open_ranked_result",
            Self::ClickVisibleElement => "click_visible_element",
            Self::NavigateBack => "navigate_back",
            Self::VerifyScreenState => "verify_screen_state",
        }
    }

    fn from_str(value: &str) -> Option<Self> {
        match normalize_label(value).as_str() {
            "locate_browser_context" | "locate_existing_browser_tab" => {
                Some(Self::LocateBrowserContext)
            }
            "focus_search_input" => Some(Self::FocusSearchInput),
            "enter_text" | "enter_query" | "type_text" => Some(Self::EnterText),
            "submit_search" => Some(Self::SubmitSearch),
            "open_ranked_result" | "open_first_result" => Some(Self::OpenRankedResult),
            "click_visible_element" | "click_element" => Some(Self::ClickVisibleElement),
            "navigate_back" | "go_back" => Some(Self::NavigateBack),
            "verify_screen_state" | "verify" => Some(Self::VerifyScreenState),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStep {
    pub step_id: String,
    pub step_kind: WorkflowStepKind,
    #[serde(default)]
    pub target: Value,
    #[serde(default)]
    pub value: Option<String>,
    #[serde(default)]
    pub selection: Value,
    #[serde(default)]
    pub expected_outcome: Option<String>,
}

impl WorkflowStep {
    fn new(index: usize, step_kind: WorkflowStepKind) -> Self {
        Self {
            step_id: format!("step_{}", index + 1),
            step_kind,
            target: json!({}),
            value: None,
            selection: json!({}),
            expected_outcome: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScreenFreshness {
    FreshAvailable,
    RecentAvailable,
    Unavailable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenGroundingState {
    pub observation_supported: bool,
    pub observation_enabled: bool,
    pub capture_available: bool,
    pub analysis_available: bool,
    pub recent_capture_available: bool,
    pub recent_capture_age_ms: Option<u64>,
    pub last_capture_path: Option<String>,
    pub freshness: ScreenFreshness,
    pub fresh_capture_required: bool,
    pub sufficient_for_workflow: bool,
    #[serde(default)]
    pub visible_target_candidates: Vec<UITargetCandidate>,
    #[serde(default)]
    pub page_evidence: Vec<PageSemanticEvidence>,
    #[serde(default)]
    pub semantic_frame: Option<SemanticScreenFrame>,
    #[serde(default)]
    pub goal_loop: Option<GoalLoopRun>,
    #[serde(default)]
    pub recent_target_candidates: Vec<UITargetCandidate>,
    #[serde(default)]
    pub generated_at_ms: Option<u64>,
    #[serde(default)]
    pub page_validation: Option<SemanticPageValidationResult>,
    #[serde(default)]
    pub regrounding: Option<ContinuationRegroundingDiagnostics>,
    #[serde(default)]
    pub uncertainty: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StepSupportStatus {
    SupportedByScreenGrounding,
    NeedsScreenContext,
    NeedsTargetGrounding,
    UnsupportedPrimitive,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StepVerificationStatus {
    NotRun,
    Satisfied,
    PartiallySatisfied,
    Failed,
    Unsupported,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStepPlan {
    pub step: WorkflowStep,
    pub support: StepSupportStatus,
    pub verification: StepVerificationStatus,
    #[serde(default)]
    pub target_selection: Option<TargetSelection>,
    #[serde(default)]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowSupportSummary {
    pub executable: bool,
    pub requires_screen_context: bool,
    #[serde(default)]
    pub unsupported_steps: Vec<String>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenWorkflow {
    pub operation: ActionOperation,
    pub domain: ScreenWorkflowDomain,
    pub requires_screen_context: bool,
    pub depends_on_recent_screen_context: bool,
    #[serde(default)]
    pub continuation: Option<WorkflowContinuationDescriptor>,
    pub grounding: ScreenGroundingState,
    pub steps: Vec<WorkflowStep>,
    pub step_plans: Vec<WorkflowStepPlan>,
    pub support: WorkflowSupportSummary,
    pub confidence: f32,
    pub source: ResolutionSource,
    #[serde(default)]
    pub rationale: Option<String>,
}

impl ScreenWorkflow {
    pub fn diagnostic_value(&self) -> Value {
        json!({
            "operation": self.operation.as_str(),
            "domain": self.domain.as_str(),
            "requires_screen_context": self.requires_screen_context,
            "depends_on_recent_screen_context": self.depends_on_recent_screen_context,
            "continuation": self.continuation,
            "grounding": self.grounding,
            "steps": self.steps,
            "step_plans": self.step_plans,
            "support": self.support,
            "confidence": self.confidence,
            "resolution_source": self.source.as_str(),
            "rationale": self.rationale,
        })
    }
}

pub fn refresh_screen_workflow_plan(workflow: &mut ScreenWorkflow) {
    let (step_plans, support) = build_step_plans_and_support(
        &workflow.steps,
        &workflow.grounding,
        workflow.depends_on_recent_screen_context,
    );
    workflow.step_plans = step_plans;
    workflow.support = support;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowRunStatus {
    Planned,
    Executing,
    Completed,
    PartiallyCompleted,
    StepUnsupported,
    StepFailed,
    NeedsScreenContext,
    NeedsTargetGrounding,
    Aborted,
}

impl WorkflowRunStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Planned => "planned",
            Self::Executing => "executing",
            Self::Completed => "completed",
            Self::PartiallyCompleted => "partially_completed",
            Self::StepUnsupported => "step_unsupported",
            Self::StepFailed => "step_failed",
            Self::NeedsScreenContext => "needs_screen_context",
            Self::NeedsTargetGrounding => "needs_target_grounding",
            Self::Aborted => "aborted",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStepRun {
    pub step: WorkflowStep,
    #[serde(default)]
    pub primitive: Option<UIPrimitiveKind>,
    pub status: WorkflowRunStatus,
    pub verification: StepVerificationStatus,
    #[serde(default)]
    pub primitive_result: Option<UIPrimitiveResult>,
    #[serde(default)]
    pub target_selection: Option<TargetSelection>,
    #[serde(default)]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenWorkflowRun {
    pub run_id: String,
    pub status: WorkflowRunStatus,
    pub workflow: ScreenWorkflow,
    pub primitive_capabilities: UIPrimitiveCapabilitySet,
    pub step_runs: Vec<WorkflowStepRun>,
    pub completed_steps: usize,
    #[serde(default)]
    pub stopped_reason: Option<String>,
    #[serde(default)]
    pub continuation_verification: Option<ContinuationVerificationResult>,
}

impl ScreenWorkflowRun {
    pub fn diagnostic_value(&self) -> Value {
        json!({
            "run_id": self.run_id,
            "status": self.status.as_str(),
            "workflow": self.workflow.diagnostic_value(),
            "primitive_capabilities": self.primitive_capabilities,
            "step_runs": self.step_runs,
            "completed_steps": self.completed_steps,
            "stopped_reason": self.stopped_reason,
            "continuation_verification": self.continuation_verification,
        })
    }
}

pub fn execute_screen_workflow(
    workflow: ScreenWorkflow,
    ui_control: &UIControlRuntime,
    primitive_capabilities: UIPrimitiveCapabilitySet,
) -> ScreenWorkflowRun {
    let run_id = uuid::Uuid::new_v4().to_string();
    let mut step_runs = Vec::new();
    let mut completed_steps = 0usize;
    let mut status = WorkflowRunStatus::Completed;
    let mut stopped_reason = None;

    if let Some(validation) = workflow
        .grounding
        .page_validation
        .as_ref()
        .filter(|validation| {
            matches!(
                validation.status,
                SemanticPageValidationStatus::Mismatched
                    | SemanticPageValidationStatus::NeedsFreshCapture
                    | SemanticPageValidationStatus::Unsupported
            )
        })
    {
        status = WorkflowRunStatus::NeedsScreenContext;
        stopped_reason = Some(format!(
            "Semantic page validation did not confirm the expected continuation context: {}",
            validation
                .mismatch_reason
                .as_deref()
                .unwrap_or("page state is not trustworthy enough")
        ));
    }

    for plan in &workflow.step_plans {
        if stopped_reason.is_some() {
            break;
        }
        let step = plan.step.clone();
        let target_selection = target_selection_for_step(
            &step,
            &workflow.grounding,
            workflow.depends_on_recent_screen_context || step_requires_recent_focus_target(&step),
        );

        if plan.support == StepSupportStatus::NeedsScreenContext {
            status = WorkflowRunStatus::NeedsScreenContext;
            stopped_reason = Some("Step needs a fresh or recent screen context.".into());
            step_runs.push(WorkflowStepRun {
                step,
                primitive: None,
                status: WorkflowRunStatus::NeedsScreenContext,
                verification: StepVerificationStatus::PartiallySatisfied,
                primitive_result: None,
                target_selection,
                note: stopped_reason.clone(),
            });
            break;
        }

        if matches!(plan.support, StepSupportStatus::NeedsTargetGrounding)
            && !target_selection
                .as_ref()
                .is_some_and(|selection| selection.status == TargetSelectionStatus::Selected)
        {
            status = WorkflowRunStatus::NeedsTargetGrounding;
            let mut reason = target_selection
                .as_ref()
                .map(|selection| selection.reason.clone())
                .unwrap_or_else(|| {
                    "No high-confidence grounded target candidate is available for this step."
                        .into()
                });
            if let Some(regrounding_reason) = workflow
                .grounding
                .regrounding
                .as_ref()
                .map(|regrounding| regrounding.final_reason.as_str())
                .filter(|reason| !reason.is_empty())
            {
                reason.push_str(" Re-grounding diagnostic: ");
                reason.push_str(regrounding_reason);
            }
            stopped_reason = Some(reason);
            step_runs.push(WorkflowStepRun {
                step,
                primitive: None,
                status: WorkflowRunStatus::NeedsTargetGrounding,
                verification: StepVerificationStatus::Unsupported,
                primitive_result: None,
                target_selection,
                note: stopped_reason.clone(),
            });
            break;
        }

        let Some(request) = primitive_request_for_step(&step, target_selection.as_ref()) else {
            if plan.support == StepSupportStatus::SupportedByScreenGrounding {
                completed_steps += 1;
                step_runs.push(WorkflowStepRun {
                    step,
                    primitive: None,
                    status: WorkflowRunStatus::Completed,
                    verification: StepVerificationStatus::Satisfied,
                    primitive_result: None,
                    target_selection,
                    note: Some("Step satisfied by available screen grounding.".into()),
                });
                continue;
            }

            status = WorkflowRunStatus::StepUnsupported;
            stopped_reason = Some("No executable primitive exists for this semantic step.".into());
            step_runs.push(WorkflowStepRun {
                step,
                primitive: None,
                status: WorkflowRunStatus::StepUnsupported,
                verification: StepVerificationStatus::Unsupported,
                primitive_result: None,
                target_selection,
                note: stopped_reason.clone(),
            });
            break;
        };

        let primitive = request.primitive.clone();
        let result = ui_control.execute(&request, &primitive_capabilities);
        match result.status {
            UIPrimitiveStatus::Executed => {
                completed_steps += 1;
                step_runs.push(WorkflowStepRun {
                    step,
                    primitive: Some(primitive),
                    status: WorkflowRunStatus::Completed,
                    verification: verify_executed_step(&request, &result),
                    primitive_result: Some(result),
                    target_selection,
                    note: Some(
                        "Primitive executed; verification is currently primitive-level.".into(),
                    ),
                });
            }
            UIPrimitiveStatus::Unsupported => {
                status = WorkflowRunStatus::StepUnsupported;
                stopped_reason = Some(result.message.clone());
                step_runs.push(WorkflowStepRun {
                    step,
                    primitive: Some(primitive),
                    status: WorkflowRunStatus::StepUnsupported,
                    verification: StepVerificationStatus::Unsupported,
                    primitive_result: Some(result),
                    target_selection,
                    note: stopped_reason.clone(),
                });
                break;
            }
            UIPrimitiveStatus::Failed => {
                status = WorkflowRunStatus::StepFailed;
                stopped_reason = Some(result.message.clone());
                step_runs.push(WorkflowStepRun {
                    step,
                    primitive: Some(primitive),
                    status: WorkflowRunStatus::StepFailed,
                    verification: StepVerificationStatus::Failed,
                    primitive_result: Some(result),
                    target_selection,
                    note: stopped_reason.clone(),
                });
                break;
            }
        }
    }

    if status == WorkflowRunStatus::Completed && completed_steps < workflow.steps.len() {
        status = WorkflowRunStatus::PartiallyCompleted;
    }

    let continuation_verification = workflow.continuation.as_ref().map(|descriptor| {
        build_continuation_verification_result(
            descriptor,
            &status,
            completed_steps,
            workflow.steps.len(),
            stopped_reason.as_deref(),
            workflow.grounding.goal_loop.as_ref(),
        )
    });

    ScreenWorkflowRun {
        run_id,
        status,
        workflow,
        primitive_capabilities,
        step_runs,
        completed_steps,
        stopped_reason,
        continuation_verification,
    }
}

pub fn resolve_screen_workflow(
    resolution: &ActionResolution,
    manifest: &CapabilityManifest,
    original_message: &str,
) -> Option<ScreenWorkflow> {
    if !matches!(
        resolution.operation,
        ActionOperation::ScreenGuidedBrowserWorkflow
            | ActionOperation::ScreenGuidedFollowupAction
            | ActionOperation::ScreenGuidedNavigationWorkflow
    ) {
        return None;
    }

    let mut grounding = screen_grounding_state(manifest);
    apply_screen_derived_candidates(&mut grounding, resolution);
    let steps = workflow_steps_from_resolution(resolution, original_message);
    let depends_on_recent_screen_context =
        depends_on_recent_context(&resolution.operation, original_message);
    let (step_plans, support) =
        build_step_plans_and_support(&steps, &grounding, depends_on_recent_screen_context);

    Some(ScreenWorkflow {
        operation: resolution.operation.clone(),
        domain: domain_for_operation(&resolution.operation),
        requires_screen_context: true,
        depends_on_recent_screen_context,
        continuation: None,
        grounding,
        steps,
        step_plans,
        support,
        confidence: resolution.confidence,
        source: resolution.source.clone(),
        rationale: resolution.rationale.clone(),
    })
}

pub fn render_screen_workflow_run_response(run: &ScreenWorkflowRun, italian: bool) -> String {
    if let Some(continuation) = run.workflow.continuation.as_ref() {
        return render_continuation_workflow_run_response(run, continuation, italian);
    }

    let planned = run.workflow.steps.len();
    let completed = run.completed_steps;
    let stopped = run.stopped_reason.as_deref().unwrap_or("");

    match run.status {
        WorkflowRunStatus::Completed => {
            if let Some(message) = render_goal_loop_completion_response(
                run.workflow.grounding.goal_loop.as_ref(),
                completed,
                planned,
                italian,
            ) {
                return message;
            }
            if italian {
                format!(
                    "Ho eseguito il workflow screen-grounded: {completed}/{planned} passaggi completati e verificati a livello primitiva."
                )
            } else {
                format!(
                    "I executed the screen-grounded workflow: {completed}/{planned} steps completed and verified at primitive level."
                )
            }
        }
        WorkflowRunStatus::PartiallyCompleted => {
            if let Some(message) = render_goal_loop_partial_completion_response(
                run.workflow.grounding.goal_loop.as_ref(),
                completed,
                planned,
                stopped,
                italian,
            ) {
                return message;
            }
            if italian {
                format!(
                    "Ho eseguito una parte del workflow screen-grounded: {completed}/{planned} passaggi completati. Mi sono fermata perche': {stopped}"
                )
            } else {
                format!(
                    "I executed part of the screen-grounded workflow: {completed}/{planned} steps completed. I stopped because: {stopped}"
                )
            }
        }
        WorkflowRunStatus::StepUnsupported => {
            if completed == 0 {
                if italian {
                    format!(
                        "Ho capito il workflow screen-grounded, ma non lo eseguo perche' il primo step operativo non e' ancora supportato in modo sicuro: {stopped}"
                    )
                } else {
                    format!(
                        "I understood the screen-grounded workflow, but I am not executing it because the first operational step is not safely supported yet: {stopped}"
                    )
                }
            } else if italian {
                format!(
                    "Ho eseguito {completed}/{planned} passaggi del workflow. Mi sono fermata su uno step non supportato: {stopped}"
                )
            } else {
                format!(
                    "I executed {completed}/{planned} workflow steps. I stopped at an unsupported step: {stopped}"
                )
            }
        }
        WorkflowRunStatus::NeedsScreenContext => {
            if italian {
                "Ho capito il workflow, ma serve prima un contesto schermo verificabile: abilita l'osservazione o crea una cattura recente.".into()
            } else {
                "I understood the workflow, but it needs verifiable screen context first: enable observation or create a recent capture.".into()
            }
        }
        WorkflowRunStatus::NeedsTargetGrounding => {
            if let Some(message) =
                render_goal_loop_no_click_response(run.workflow.grounding.goal_loop.as_ref(), italian)
            {
                return message;
            }
            if italian {
                format!(
                    "Ho capito quale azione vuoi fare sullo schermo, ma non ho un target abbastanza sicuro per cliccare o mettere il focus: {stopped}"
                )
            } else {
                format!(
                    "I understood the screen action, but I do not have a safe enough grounded target to click or focus: {stopped}"
                )
            }
        }
        WorkflowRunStatus::StepFailed => {
            if italian {
                format!(
                    "Ho iniziato il workflow ma uno step e' fallito dopo {completed}/{planned} passaggi completati: {stopped}"
                )
            } else {
                format!(
                    "I started the workflow, but one step failed after {completed}/{planned} completed steps: {stopped}"
                )
            }
        }
        WorkflowRunStatus::Planned | WorkflowRunStatus::Executing | WorkflowRunStatus::Aborted => {
            if italian {
                format!("Il workflow e' stato preparato ma non completato: {stopped}")
            } else {
                format!("The workflow was prepared but not completed: {stopped}")
            }
        }
    }
}

fn render_continuation_workflow_run_response(
    run: &ScreenWorkflowRun,
    continuation: &WorkflowContinuationDescriptor,
    italian: bool,
) -> String {
    let planned = run.workflow.steps.len();
    let completed = run.completed_steps;
    let stopped = run.stopped_reason.as_deref().unwrap_or("");
    let provider = continuation
        .source_context
        .provider
        .as_deref()
        .unwrap_or("screen");
    let query = continuation
        .source_context
        .query
        .as_deref()
        .unwrap_or("contesto recente");
    let rank = continuation
        .followup
        .result_reference
        .as_ref()
        .and_then(|reference| reference.rank)
        .map(|rank| rank.to_string())
        .unwrap_or_else(|| "referenced".into());
    let continuation_verification = run.continuation_verification.as_ref();
    let verification = continuation_verification
        .map(|verification| format!("{:?}", verification.status))
        .unwrap_or_else(|| "not_recorded".into());

    match run.status {
        WorkflowRunStatus::Completed => {
            if let Some(message) = render_goal_loop_completion_response(
                run.workflow.grounding.goal_loop.as_ref(),
                completed,
                planned,
                italian,
            ) {
                return message;
            }
            if continuation_verification.is_some_and(|verification| {
                verification.status == ContinuationVerificationStatus::GoalAchieved
            }) {
                if italian {
                    format!(
                        "Ho continuato il workflow recente ({provider}, {query}) ed eseguito {completed}/{planned} passaggi. Ho aperto correttamente il risultato {rank}; la verifica finale ha confermato il risultato richiesto."
                    )
                } else {
                    format!(
                        "I continued the recent workflow ({provider}, {query}) and executed {completed}/{planned} steps. I opened result {rank} successfully; final verification confirmed the requested outcome."
                    )
                }
            } else if italian {
                if continuation.verifier.requires_post_step_screen_check {
                    format!(
                        "Ho continuato il workflow recente ({provider}, {query}) e ho eseguito {completed}/{planned} passaggi. Ho cliccato il risultato {rank}; la verifica e' a livello primitiva e la prossima cattura dovra' confermare la navigazione."
                    )
                } else {
                    format!(
                        "Ho continuato il workflow recente ({provider}, {query}) e ho eseguito {completed}/{planned} passaggi con verifica: {verification}."
                    )
                }
            } else if continuation.verifier.requires_post_step_screen_check {
                format!(
                    "I continued the recent workflow ({provider}, {query}) and executed {completed}/{planned} steps. I clicked result {rank}; verification is primitive-level and the next capture should confirm navigation."
                )
            } else {
                format!(
                    "I continued the recent workflow ({provider}, {query}) and executed {completed}/{planned} steps with verification: {verification}."
                )
            }
        }
        WorkflowRunStatus::NeedsTargetGrounding => {
            if let Some(message) =
                render_goal_loop_no_click_response(run.workflow.grounding.goal_loop.as_ref(), italian)
            {
                return message;
            }
            if italian {
                format!(
                    "Ho capito la richiesta come continuazione del workflow recente ({provider}, {query}), ma non ho identificato un candidato abbastanza sicuro per il passaggio richiesto: {stopped}"
                )
            } else {
                format!(
                    "I understood this as a continuation of the recent workflow ({provider}, {query}), but I could not identify a safe enough candidate for the requested step: {stopped}"
                )
            }
        }
        WorkflowRunStatus::NeedsScreenContext => {
            if italian {
                format!(
                    "Ho capito la continuazione del workflow recente ({provider}, {query}), ma serve una schermata verificabile prima di scegliere il target: {stopped}"
                )
            } else {
                format!(
                    "I understood the continuation of the recent workflow ({provider}, {query}), but I need verifiable screen context before choosing the target: {stopped}"
                )
            }
        }
        WorkflowRunStatus::StepUnsupported => {
            if italian {
                format!(
                    "Ho capito la continuazione del workflow recente ({provider}, {query}), ma il runtime non espone ancora una primitiva sicura per completarla: {stopped}"
                )
            } else {
                format!(
                    "I understood the continuation of the recent workflow ({provider}, {query}), but this runtime does not yet expose a safe primitive to complete it: {stopped}"
                )
            }
        }
        WorkflowRunStatus::StepFailed => {
            if italian {
                format!(
                    "Ho iniziato la continuazione del workflow recente ({provider}, {query}), ma uno step e' fallito dopo {completed}/{planned} passaggi: {stopped}"
                )
            } else {
                format!(
                    "I started continuing the recent workflow ({provider}, {query}), but one step failed after {completed}/{planned} steps: {stopped}"
                )
            }
        }
        WorkflowRunStatus::PartiallyCompleted => {
            if let Some(message) = render_goal_loop_continuation_partial_completion_response(
                run.workflow.grounding.goal_loop.as_ref(),
                provider,
                query,
                &rank,
                completed,
                planned,
                stopped,
                italian,
            ) {
                return message;
            }
            if italian {
                format!(
                    "La continuazione del workflow recente ({provider}, {query}) non e' stata completata: {completed}/{planned} passaggi. Motivo: {stopped}"
                )
            } else {
                format!(
                    "The continuation of the recent workflow ({provider}, {query}) was not completed: {completed}/{planned} steps. Reason: {stopped}"
                )
            }
        }
        WorkflowRunStatus::Planned | WorkflowRunStatus::Executing | WorkflowRunStatus::Aborted => {
            if italian {
                format!(
                    "La continuazione del workflow recente ({provider}, {query}) non e' stata completata: {completed}/{planned} passaggi. Motivo: {stopped}"
                )
            } else {
                format!(
                    "The continuation of the recent workflow ({provider}, {query}) was not completed: {completed}/{planned} steps. Reason: {stopped}"
                )
            }
        }
    }
}

fn goal_loop_executed_but_final_confirmation_uncertain(goal_loop: &GoalLoopRun) -> bool {
    goal_loop
        .executed_steps
        .iter()
        .any(|step| step.status == PlannerStepExecutionStatus::Executed)
        && goal_loop.status != GoalLoopStatus::GoalAchieved
}

fn render_goal_loop_completion_response(
    goal_loop: Option<&GoalLoopRun>,
    _completed: usize,
    _planned: usize,
    italian: bool,
) -> Option<String> {
    goal_loop.filter(|goal_loop| goal_loop.status == GoalLoopStatus::GoalAchieved)?;

    Some(if italian {
        "Ho aperto il video/risultato richiesto.".into()
    } else {
        "I opened the requested video/result.".into()
    })
}

fn render_goal_loop_no_click_response(
    goal_loop: Option<&GoalLoopRun>,
    italian: bool,
) -> Option<String> {
    let _goal_loop = goal_loop.filter(|goal_loop| {
        goal_loop.status != GoalLoopStatus::GoalAchieved
            && !goal_loop
                .executed_steps
                .iter()
                .any(|step| step.status == PlannerStepExecutionStatus::Executed)
    })?;
    Some(if italian {
        "Non ho eseguito il click perché non ho trovato un target sufficientemente sicuro.".into()
    } else {
        "I did not click because I did not find a sufficiently safe target.".into()
    })
}

fn goal_loop_partial_completion_reason_suffix(stopped: &str, italian: bool) -> String {
    if stopped.trim().is_empty() {
        String::new()
    } else if italian {
        format!(" Motivo finale: {stopped}")
    } else {
        format!(" Final reason: {stopped}")
    }
}

fn render_goal_loop_partial_completion_response(
    goal_loop: Option<&GoalLoopRun>,
    completed: usize,
    planned: usize,
    stopped: &str,
    italian: bool,
) -> Option<String> {
    let goal_loop = goal_loop.filter(|goal_loop| {
        goal_loop_executed_but_final_confirmation_uncertain(goal_loop)
            && matches!(
                goal_loop.status,
                GoalLoopStatus::Refused
                    | GoalLoopStatus::NeedsClarification
                    | GoalLoopStatus::BudgetExhausted
                    | GoalLoopStatus::VerificationFailed
            )
    })?;
    let suffix = goal_loop_partial_completion_reason_suffix(stopped, italian);

    Some(if italian {
        if goal_loop.post_action_progress_observed {
            format!("Ho eseguito il click sul candidato selezionato, ma la verifica finale non conferma con certezza che sia stato aperto il contenuto richiesto.{suffix}")
        } else {
            format!(
                "Ho eseguito il click richiesto nel workflow screen-grounded: {completed}/{planned} passaggi completati. L'azione e' stata eseguita, ma la verifica visiva finale resta incerta.{suffix}"
            )
        }
    } else if goal_loop.post_action_progress_observed {
        format!("I clicked the selected candidate, but final verification does not confirm with certainty that the requested content opened.{suffix}")
    } else {
        format!(
            "I executed the requested click in the screen-grounded workflow: {completed}/{planned} steps completed. The action ran, but final visual verification is still uncertain.{suffix}"
        )
    })
}

fn render_goal_loop_continuation_partial_completion_response(
    goal_loop: Option<&GoalLoopRun>,
    provider: &str,
    query: &str,
    rank: &str,
    completed: usize,
    planned: usize,
    stopped: &str,
    italian: bool,
) -> Option<String> {
    let goal_loop = goal_loop.filter(|goal_loop| {
        goal_loop_executed_but_final_confirmation_uncertain(goal_loop)
            && matches!(
                goal_loop.status,
                GoalLoopStatus::Refused
                    | GoalLoopStatus::NeedsClarification
                    | GoalLoopStatus::BudgetExhausted
                    | GoalLoopStatus::VerificationFailed
            )
    })?;
    let suffix = goal_loop_partial_completion_reason_suffix(stopped, italian);

    Some(if italian {
        if goal_loop.post_action_progress_observed {
            format!("Ho eseguito il click sul candidato selezionato, ma la verifica finale non conferma con certezza che sia stato aperto il contenuto richiesto.{suffix}")
        } else {
            format!(
                "Ho continuato il workflow recente ({provider}, {query}) ed eseguito {completed}/{planned} passaggi. Ho cliccato il risultato {rank}, ma la verifica visiva finale resta incerta.{suffix}"
            )
        }
    } else if goal_loop.post_action_progress_observed {
        format!("I clicked the selected candidate, but final verification does not confirm with certainty that the requested content opened.{suffix}")
    } else {
        format!(
            "I continued the recent workflow ({provider}, {query}) and executed {completed}/{planned} steps. I clicked result {rank}, but final visual verification is still uncertain.{suffix}"
        )
    })
}

pub fn screen_grounding_state(manifest: &CapabilityManifest) -> ScreenGroundingState {
    let freshness =
        if manifest.screen.observation_enabled && manifest.screen.fresh_capture_available {
            ScreenFreshness::FreshAvailable
        } else if manifest.screen.recent_capture_available {
            ScreenFreshness::RecentAvailable
        } else {
            ScreenFreshness::Unavailable
        };

    let mut uncertainty = Vec::new();
    if !manifest.screen.observation_supported {
        uncertainty.push("screen_observation_not_supported".into());
    }
    if !manifest.screen.observation_enabled {
        uncertainty.push("screen_observation_disabled".into());
    }
    if !manifest.screen.analysis_available {
        uncertainty.push("screen_analysis_unavailable".into());
    }
    if !manifest.screen.recent_capture_available {
        uncertainty.push("no_recent_capture".into());
    }

    ScreenGroundingState {
        observation_supported: manifest.screen.observation_supported,
        observation_enabled: manifest.screen.observation_enabled,
        capture_available: manifest.screen.capture_available,
        analysis_available: manifest.screen.analysis_available,
        recent_capture_available: manifest.screen.recent_capture_available,
        recent_capture_age_ms: manifest.screen.recent_capture_age_ms,
        last_capture_path: manifest.screen.last_capture_path.clone(),
        freshness,
        fresh_capture_required: !manifest.screen.recent_capture_available,
        sufficient_for_workflow: manifest.screen.analysis_available
            && (manifest.screen.observation_enabled || manifest.screen.recent_capture_available),
        visible_target_candidates: Vec::new(),
        page_evidence: Vec::new(),
        semantic_frame: None,
        goal_loop: None,
        recent_target_candidates: Vec::new(),
        generated_at_ms: Some(now_ms()),
        page_validation: None,
        regrounding: None,
        uncertainty,
    }
}

fn build_step_plans_and_support(
    steps: &[WorkflowStep],
    grounding: &ScreenGroundingState,
    allow_recent_reuse: bool,
) -> (Vec<WorkflowStepPlan>, WorkflowSupportSummary) {
    let step_plans = steps
        .iter()
        .cloned()
        .map(|step| plan_step(step, grounding, allow_recent_reuse))
        .collect::<Vec<_>>();
    let unsupported_steps = step_plans
        .iter()
        .filter(|plan| {
            matches!(
                plan.support,
                StepSupportStatus::UnsupportedPrimitive | StepSupportStatus::NeedsTargetGrounding
            )
        })
        .map(|plan| plan.step.step_kind.as_str().to_string())
        .collect::<Vec<_>>();
    let needs_context = step_plans
        .iter()
        .any(|plan| plan.support == StepSupportStatus::NeedsScreenContext);
    let needs_target_grounding = step_plans
        .iter()
        .any(|plan| plan.support == StepSupportStatus::NeedsTargetGrounding);
    let executable = unsupported_steps.is_empty() && !needs_context;
    let reason = if needs_target_grounding {
        "The workflow is understood, but one or more target-dependent steps do not have a high-confidence structured screen target yet.".into()
    } else if !unsupported_steps.is_empty() {
        "The workflow is understood, but this runtime does not yet expose reliable low-level UI control primitives for one or more steps.".into()
    } else if needs_context {
        "The workflow needs screen context before any step can be verified or executed.".into()
    } else {
        "The workflow can be grounded against the current screen state.".into()
    };

    (
        step_plans,
        WorkflowSupportSummary {
            executable,
            requires_screen_context: true,
            unsupported_steps,
            reason,
        },
    )
}

fn apply_screen_derived_candidates(
    grounding: &mut ScreenGroundingState,
    resolution: &ActionResolution,
) {
    let mut candidates = Vec::new();
    for key in [
        "screen_candidates",
        "visible_candidates",
        "target_candidates",
        "ui_candidates",
    ] {
        if let Some(value) = resolution.entities.get(key) {
            candidates.extend(structured_candidates_from_value(
                value,
                &UITargetRole::Unknown,
                value_str(&resolution.entities, "app"),
                value_str(&resolution.entities, "provider").or(resolution.provider.as_deref()),
                TargetGroundingSource::ScreenAnalysis,
            ));
        }
    }

    if !candidates.is_empty() {
        grounding.visible_target_candidates = candidates;
    }
}

fn workflow_steps_from_resolution(
    resolution: &ActionResolution,
    original_message: &str,
) -> Vec<WorkflowStep> {
    if !resolution.workflow_steps.is_empty() {
        let mut steps = resolution
            .workflow_steps
            .iter()
            .enumerate()
            .filter_map(|(index, step)| {
                WorkflowStepKind::from_str(step).map(|kind| WorkflowStep::new(index, kind))
            })
            .collect::<Vec<_>>();
        enrich_steps(&mut steps, resolution);
        return steps;
    }

    let mut steps = match resolution.operation {
        ActionOperation::ScreenGuidedBrowserWorkflow => vec![
            WorkflowStep::new(0, WorkflowStepKind::LocateBrowserContext),
            WorkflowStep::new(1, WorkflowStepKind::FocusSearchInput),
            WorkflowStep::new(2, WorkflowStepKind::EnterText),
            WorkflowStep::new(3, WorkflowStepKind::SubmitSearch),
            WorkflowStep::new(4, WorkflowStepKind::OpenRankedResult),
        ],
        ActionOperation::ScreenGuidedFollowupAction => vec![WorkflowStep::new(
            0,
            if original_message.to_lowercase().contains("primo") {
                WorkflowStepKind::OpenRankedResult
            } else {
                WorkflowStepKind::ClickVisibleElement
            },
        )],
        ActionOperation::ScreenGuidedNavigationWorkflow => {
            vec![WorkflowStep::new(0, WorkflowStepKind::NavigateBack)]
        }
        _ => Vec::new(),
    };
    enrich_steps(&mut steps, resolution);
    steps
}

fn enrich_steps(steps: &mut [WorkflowStep], resolution: &ActionResolution) {
    let provider = value_str(&resolution.entities, "provider")
        .or(resolution.provider.as_deref())
        .unwrap_or("unknown");
    let browser_app = value_str(&resolution.entities, "browser_app")
        .or_else(|| value_str(&resolution.entities, "app"))
        .unwrap_or("chrome");
    let query = value_str(&resolution.entities, "query_candidate")
        .or_else(|| value_str(&resolution.entities, "query"));

    for step in steps {
        match step.step_kind {
            WorkflowStepKind::LocateBrowserContext => {
                step.target = json!({
                    "app": browser_app,
                    "provider": provider,
                });
                step.expected_outcome = Some("visible browser context located".into());
            }
            WorkflowStepKind::FocusSearchInput => {
                let mut target = json!({
                    "element_role": "search_input",
                    "provider": provider,
                    "app": browser_app,
                });
                if let Some(candidate) = target_candidate_from_entities(
                    &resolution.entities,
                    &["search_input_candidate", "target_candidate"],
                ) {
                    target["candidate"] = candidate.clone();
                }
                step.target = target;
                step.expected_outcome = Some("search input focused".into());
            }
            WorkflowStepKind::EnterText => {
                step.value = query.map(ToOwned::to_owned);
                if resolution.operation == ActionOperation::ScreenGuidedFollowupAction
                    || value_bool(&resolution.entities, "requires_recent_focus_target")
                        .unwrap_or(false)
                {
                    step.target = json!({
                        "element_role": "search_input",
                        "provider": provider,
                        "app": browser_app,
                        "requires_recent_focus_target": true,
                    });
                }
                step.expected_outcome = Some("query text entered".into());
            }
            WorkflowStepKind::SubmitSearch => {
                step.expected_outcome = Some("search results visible".into());
            }
            WorkflowStepKind::OpenRankedResult => {
                let rank = value_u32(&resolution.entities, "rank").unwrap_or(1);
                let mut selection = json!({
                    "strategy": value_str(&resolution.entities, "selection_strategy")
                        .unwrap_or("ranked_result"),
                    "rank": rank,
                    "provider": provider,
                    "app": browser_app,
                    "query": query,
                    "result_kind": value_str(&resolution.entities, "result_kind")
                        .unwrap_or("result"),
                });
                if let Some(candidate) = target_candidate_from_entities(
                    &resolution.entities,
                    &[
                        "first_result_candidate",
                        "result_candidate",
                        "target_candidate",
                    ],
                ) {
                    selection["candidate"] = candidate.clone();
                }
                step.selection = selection;
                step.expected_outcome = Some("selected result opened".into());
            }
            WorkflowStepKind::ClickVisibleElement => {
                let mut selection = json!({"strategy": "referenced_visible_element"});
                if let Some(candidate) =
                    target_candidate_from_entities(&resolution.entities, &["target_candidate"])
                {
                    selection["candidate"] = candidate.clone();
                }
                step.selection = selection;
                step.expected_outcome = Some("referenced visible element activated".into());
            }
            WorkflowStepKind::NavigateBack => {
                step.expected_outcome =
                    Some("previous screen state restored or browser navigated back".into());
            }
            WorkflowStepKind::VerifyScreenState => {
                step.expected_outcome = Some("screen state matches expected result".into());
            }
        }
    }
}

fn plan_step(
    step: WorkflowStep,
    grounding: &ScreenGroundingState,
    allow_recent_reuse: bool,
) -> WorkflowStepPlan {
    let target_selection = target_selection_for_step(&step, grounding, allow_recent_reuse);
    let support = match step.step_kind {
        WorkflowStepKind::LocateBrowserContext | WorkflowStepKind::VerifyScreenState => {
            if grounding.sufficient_for_workflow {
                StepSupportStatus::SupportedByScreenGrounding
            } else {
                StepSupportStatus::NeedsScreenContext
            }
        }
        WorkflowStepKind::FocusSearchInput
        | WorkflowStepKind::OpenRankedResult
        | WorkflowStepKind::ClickVisibleElement => {
            if !grounding.sufficient_for_workflow {
                StepSupportStatus::NeedsScreenContext
            } else if target_selection
                .as_ref()
                .is_some_and(|selection| selection.status == TargetSelectionStatus::Selected)
            {
                StepSupportStatus::UnsupportedPrimitive
            } else {
                StepSupportStatus::NeedsTargetGrounding
            }
        }
        WorkflowStepKind::EnterText if step_requires_recent_focus_target(&step) => {
            if target_selection
                .as_ref()
                .is_some_and(|selection| selection.status == TargetSelectionStatus::Selected)
            {
                StepSupportStatus::UnsupportedPrimitive
            } else {
                StepSupportStatus::NeedsTargetGrounding
            }
        }
        WorkflowStepKind::EnterText
        | WorkflowStepKind::SubmitSearch
        | WorkflowStepKind::NavigateBack => StepSupportStatus::UnsupportedPrimitive,
    };
    let verification = match support {
        StepSupportStatus::SupportedByScreenGrounding => StepVerificationStatus::NotRun,
        StepSupportStatus::NeedsScreenContext => StepVerificationStatus::PartiallySatisfied,
        StepSupportStatus::NeedsTargetGrounding => StepVerificationStatus::Unsupported,
        StepSupportStatus::UnsupportedPrimitive => StepVerificationStatus::Unsupported,
    };
    let note = Some(match support {
        StepSupportStatus::SupportedByScreenGrounding => {
            "Can be grounded with the current screen analysis capability.".into()
        }
        StepSupportStatus::NeedsScreenContext => {
            "Needs a fresh or recent screen capture before verification.".into()
        }
        StepSupportStatus::NeedsTargetGrounding => target_selection
            .as_ref()
            .map(|selection| selection.reason.clone())
            .unwrap_or_else(|| {
                "Needs a high-confidence target candidate before focus/click is safe.".into()
            }),
        StepSupportStatus::UnsupportedPrimitive => {
            "Requires UI-control primitives that are not exposed by this runtime yet.".into()
        }
    });

    WorkflowStepPlan {
        step,
        support,
        verification,
        target_selection,
        note,
    }
}

fn primitive_request_for_step(
    step: &WorkflowStep,
    target_selection: Option<&TargetSelection>,
) -> Option<UIPrimitiveRequest> {
    match step.step_kind {
        WorkflowStepKind::LocateBrowserContext | WorkflowStepKind::VerifyScreenState => None,
        WorkflowStepKind::FocusSearchInput => Some(UIPrimitiveRequest {
            primitive: UIPrimitiveKind::FocusCurrentInput,
            value: None,
            target: target_execution_payload(target_selection?),
            reason: Some("Workflow requested focusing a search input.".into()),
        }),
        WorkflowStepKind::EnterText => Some(UIPrimitiveRequest {
            primitive: UIPrimitiveKind::TypeText,
            value: step.value.clone(),
            target: target_selection
                .map(target_execution_payload)
                .unwrap_or_else(|| step.target.clone()),
            reason: Some("Workflow requested entering text.".into()),
        }),
        WorkflowStepKind::SubmitSearch => Some(UIPrimitiveRequest {
            primitive: UIPrimitiveKind::PressEnter,
            value: None,
            target: step.target.clone(),
            reason: Some("Workflow requested submitting the current input.".into()),
        }),
        WorkflowStepKind::OpenRankedResult | WorkflowStepKind::ClickVisibleElement => {
            Some(UIPrimitiveRequest {
                primitive: UIPrimitiveKind::ClickTargetCandidate,
                value: None,
                target: target_execution_payload(target_selection?),
                reason: Some("Workflow requested clicking a grounded target candidate.".into()),
            })
        }
        WorkflowStepKind::NavigateBack => Some(UIPrimitiveRequest {
            primitive: UIPrimitiveKind::NavigateBack,
            value: None,
            target: step.target.clone(),
            reason: Some("Workflow requested navigating back.".into()),
        }),
    }
}

fn target_selection_for_step(
    step: &WorkflowStep,
    grounding: &ScreenGroundingState,
    allow_recent_reuse: bool,
) -> Option<TargetSelection> {
    if !matches!(
        step.step_kind,
        WorkflowStepKind::FocusSearchInput
            | WorkflowStepKind::OpenRankedResult
            | WorkflowStepKind::ClickVisibleElement
            | WorkflowStepKind::EnterText
    ) {
        return None;
    }
    if step.step_kind == WorkflowStepKind::EnterText && !step_requires_recent_focus_target(step) {
        return None;
    }

    let request = TargetGroundingRequest {
        requested_role: target_role_for_step(step),
        target: step.target.clone(),
        selection: step.selection.clone(),
        screen_candidates: grounding.visible_target_candidates.clone(),
        recent_candidates: grounding.recent_target_candidates.clone(),
        app_hint: value_str(&step.target, "app")
            .or_else(|| value_str(&step.selection, "app"))
            .or_else(|| value_str(&step.selection, "browser_app"))
            .map(ToOwned::to_owned),
        provider_hint: value_str(&step.target, "provider")
            .or_else(|| value_str(&step.selection, "provider"))
            .map(ToOwned::to_owned),
        rank_hint: rank_hint_for_step(step),
        result_kind_hint: value_str(&step.selection, "result_kind")
            .or_else(|| value_str(&step.target, "result_kind"))
            .map(ToOwned::to_owned),
        allow_recent_reuse,
        now_ms: Some(now_ms()),
        max_recent_age_ms: 120_000,
    };
    let state = ground_targets_for_request(&request);
    if !grounding.sufficient_for_workflow && state.candidates.is_empty() {
        return Some(TargetSelection::rejected(
            TargetSelectionStatus::NoCandidatesPresent,
            Vec::new(),
            TargetSelectionPolicy::default().min_click_confidence,
            "Screen context is not sufficient and no structured target candidates are available.",
        ));
    }
    Some(select_target_candidate(
        &state,
        target_action_for_step(step),
        &TargetSelectionPolicy::default(),
    ))
}

fn target_execution_payload(target_selection: &TargetSelection) -> Value {
    json!({
        "selection_status": target_selection.status,
        "required_confidence": target_selection.required_confidence,
        "reason": target_selection.reason,
        "candidate": target_selection
            .selected_candidate
            .as_ref()
            .map(|candidate| candidate.execution_payload()),
    })
}

fn target_role_for_step(step: &WorkflowStep) -> UITargetRole {
    match step.step_kind {
        WorkflowStepKind::FocusSearchInput => UITargetRole::SearchInput,
        WorkflowStepKind::OpenRankedResult => UITargetRole::RankedResult,
        WorkflowStepKind::EnterText if step_requires_recent_focus_target(step) => {
            UITargetRole::SearchInput
        }
        WorkflowStepKind::ClickVisibleElement => UITargetRole::Unknown,
        _ => UITargetRole::Unknown,
    }
}

fn target_action_for_step(step: &WorkflowStep) -> TargetAction {
    match step.step_kind {
        WorkflowStepKind::FocusSearchInput | WorkflowStepKind::EnterText => TargetAction::Focus,
        WorkflowStepKind::OpenRankedResult | WorkflowStepKind::ClickVisibleElement => {
            TargetAction::Click
        }
        _ => TargetAction::Click,
    }
}

fn rank_hint_for_step(step: &WorkflowStep) -> Option<u32> {
    if step.step_kind == WorkflowStepKind::OpenRankedResult {
        step.selection
            .get("rank")
            .and_then(Value::as_u64)
            .map(|value| value as u32)
            .or(Some(1))
    } else {
        step.selection
            .get("rank")
            .and_then(Value::as_u64)
            .map(|value| value as u32)
    }
}

fn step_requires_recent_focus_target(step: &WorkflowStep) -> bool {
    step.target
        .get("requires_recent_focus_target")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn verify_executed_step(
    request: &UIPrimitiveRequest,
    result: &UIPrimitiveResult,
) -> StepVerificationStatus {
    if result.status != UIPrimitiveStatus::Executed {
        return StepVerificationStatus::Failed;
    }

    match request.primitive {
        UIPrimitiveKind::TypeText | UIPrimitiveKind::PressEnter | UIPrimitiveKind::NavigateBack => {
            StepVerificationStatus::Satisfied
        }
        UIPrimitiveKind::ScrollViewport => StepVerificationStatus::PartiallySatisfied,
        UIPrimitiveKind::ActivateWindowOrApp
        | UIPrimitiveKind::FocusCurrentInput
        | UIPrimitiveKind::ClickTargetCandidate => StepVerificationStatus::PartiallySatisfied,
    }
}

fn domain_for_operation(operation: &ActionOperation) -> ScreenWorkflowDomain {
    match operation {
        ActionOperation::ScreenGuidedFollowupAction => ScreenWorkflowDomain::ScreenInteraction,
        ActionOperation::ScreenGuidedNavigationWorkflow => ScreenWorkflowDomain::ScreenNavigation,
        _ => ScreenWorkflowDomain::BrowserScreenInteraction,
    }
}

fn depends_on_recent_context(operation: &ActionOperation, original_message: &str) -> bool {
    matches!(
        operation,
        ActionOperation::ScreenGuidedFollowupAction
            | ActionOperation::ScreenGuidedNavigationWorkflow
    ) || {
        let lower = original_message.to_lowercase();
        lower.contains("quello")
            || lower.contains("quella")
            || lower.contains("prima")
            || lower.contains("that")
            || lower.contains("previous")
    }
}

fn value_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn value_bool(value: &Value, key: &str) -> Option<bool> {
    value.get(key).and_then(Value::as_bool)
}

fn value_u32(value: &Value, key: &str) -> Option<u32> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|value| value as u32)
}

fn target_candidate_from_entities<'a>(entities: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    for key in keys {
        if let Some(candidate) = entities.get(*key).filter(|value| value.is_object()) {
            return Some(candidate);
        }
    }

    None
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
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
    use crate::desktop_agent_types::{
        CapabilityApprovalState, CapabilityPermissionState, CapabilityRuntimeState,
        CapabilityScreenState, CapabilityToolAvailability, GoalConstraints, GoalLoopStatus,
        GoalSpec, GoalType, GoalVerificationRecord, GoalVerificationStatus, Permission,
        PlannerStepExecutionRecord, PlannerStepExecutionStatus, VisibleResultKind,
    };
    use crate::ui_control::{UIPrimitiveCapability, UIPrimitiveKind};
    use crate::ui_target_grounding::{TargetGroundingSource, TargetRegion, UITargetCandidate};

    #[test]
    fn workflow_marks_ui_control_steps_as_unsupported() {
        let mut resolution = ActionResolution::new(
            ActionOperation::ScreenGuidedBrowserWorkflow,
            crate::action_resolution::ActionDomain::BrowserScreenInteraction,
            0.7,
            ResolutionSource::RustNormalizer,
        );
        resolution.provider = Some("youtube".into());
        resolution.entities = json!({"provider": "youtube", "query_candidate": "Shiva canzone"});
        let workflow =
            resolve_screen_workflow(&resolution, &manifest_with_screen(false), "cercami shiva")
                .expect("workflow");

        assert_eq!(workflow.steps.len(), 5);
        assert!(!workflow.support.executable);
        assert!(workflow
            .support
            .unsupported_steps
            .contains(&"focus_search_input".to_string()));
    }

    #[test]
    fn followup_workflow_depends_on_recent_context() {
        let resolution = ActionResolution::new(
            ActionOperation::ScreenGuidedFollowupAction,
            crate::action_resolution::ActionDomain::Screen,
            0.6,
            ResolutionSource::RustNormalizer,
        );
        let workflow = resolve_screen_workflow(
            &resolution,
            &manifest_with_screen(true),
            "clicca il primo risultato",
        )
        .expect("workflow");

        assert!(workflow.depends_on_recent_screen_context);
        assert_eq!(
            workflow.steps[0].step_kind,
            WorkflowStepKind::OpenRankedResult
        );
        assert!(!workflow.support.executable);
    }

    #[test]
    fn supported_keyboard_subset_executes_and_verifies() {
        let mut resolution = ActionResolution::new(
            ActionOperation::ScreenGuidedBrowserWorkflow,
            crate::action_resolution::ActionDomain::BrowserScreenInteraction,
            0.7,
            ResolutionSource::RustNormalizer,
        );
        resolution.entities = json!({"query_candidate": "Shiva canzone"});
        resolution.workflow_steps = vec!["enter_text".into(), "submit_search".into()];
        let workflow =
            resolve_screen_workflow(&resolution, &manifest_with_screen(true), "scrivi e invia")
                .expect("workflow");
        let run = execute_screen_workflow(
            workflow,
            &UIControlRuntime::dry_run(),
            keyboard_test_capabilities(),
        );

        assert_eq!(run.status, WorkflowRunStatus::Completed);
        assert_eq!(run.completed_steps, 2);
        assert!(run
            .step_runs
            .iter()
            .all(|step| step.verification == StepVerificationStatus::Satisfied));
    }

    #[test]
    fn high_confidence_focus_target_executes_in_dry_run() {
        let mut resolution = ActionResolution::new(
            ActionOperation::ScreenGuidedBrowserWorkflow,
            crate::action_resolution::ActionDomain::BrowserScreenInteraction,
            0.8,
            ResolutionSource::RustNormalizer,
        );
        resolution.workflow_steps = vec!["focus_search_input".into()];
        resolution.entities = json!({
            "provider": "youtube",
            "search_input_candidate": {
                "role": "search_input",
                "center_x": 500,
                "center_y": 120,
                "confidence": 0.92,
                "supports_focus": true
            }
        });

        let workflow = resolve_screen_workflow(
            &resolution,
            &manifest_with_screen(true),
            "metti il focus nella barra youtube",
        )
        .expect("workflow");
        let run = execute_screen_workflow(
            workflow,
            &UIControlRuntime::dry_run(),
            pointer_test_capabilities(UIPrimitiveKind::FocusCurrentInput),
        );

        assert_eq!(run.status, WorkflowRunStatus::Completed);
        assert_eq!(run.completed_steps, 1);
        assert!(run.step_runs[0].target_selection.is_some());
        assert_eq!(
            run.step_runs[0].verification,
            StepVerificationStatus::PartiallySatisfied
        );
    }

    #[test]
    fn automatic_screen_derived_search_input_candidate_executes() {
        let mut resolution = ActionResolution::new(
            ActionOperation::ScreenGuidedBrowserWorkflow,
            crate::action_resolution::ActionDomain::BrowserScreenInteraction,
            0.8,
            ResolutionSource::RustNormalizer,
        );
        resolution.workflow_steps = vec!["focus_search_input".into()];
        resolution.entities = json!({
            "provider": "youtube",
            "screen_candidates": [
                {
                    "role": "search_input",
                    "region": {"x": 100, "y": 60, "width": 600, "height": 42},
                    "confidence": 0.91,
                    "provider": "youtube",
                    "app": "chrome",
                    "supports_focus": true
                }
            ]
        });

        let workflow = resolve_screen_workflow(
            &resolution,
            &manifest_with_screen(true),
            "metti il focus nella barra di youtube",
        )
        .expect("workflow");
        let run = execute_screen_workflow(
            workflow,
            &UIControlRuntime::dry_run(),
            pointer_test_capabilities(UIPrimitiveKind::FocusCurrentInput),
        );

        assert_eq!(run.status, WorkflowRunStatus::Completed);
        assert_eq!(
            run.step_runs[0]
                .target_selection
                .as_ref()
                .and_then(|selection| selection.selected_candidate.as_ref())
                .map(|candidate| candidate.source.clone()),
            Some(TargetGroundingSource::ScreenAnalysis)
        );
    }

    #[test]
    fn automatic_screen_derived_ranked_result_candidate_executes() {
        let mut resolution = ActionResolution::new(
            ActionOperation::ScreenGuidedFollowupAction,
            crate::action_resolution::ActionDomain::Screen,
            0.8,
            ResolutionSource::RustNormalizer,
        );
        resolution.workflow_steps = vec!["open_ranked_result".into()];
        resolution.entities = json!({
            "provider": "youtube",
            "screen_candidates": [
                {
                    "role": "ranked_result",
                    "center_x": 640,
                    "center_y": 360,
                    "confidence": 0.91,
                    "provider": "youtube",
                    "rank": 1,
                    "supports_click": true
                }
            ]
        });

        let workflow =
            resolve_screen_workflow(&resolution, &manifest_with_screen(true), "apri il primo")
                .expect("workflow");
        let run = execute_screen_workflow(
            workflow,
            &UIControlRuntime::dry_run(),
            pointer_test_capabilities(UIPrimitiveKind::ClickTargetCandidate),
        );

        assert_eq!(run.status, WorkflowRunStatus::Completed);
        assert_eq!(run.completed_steps, 1);
        assert_eq!(
            run.step_runs
                .first()
                .and_then(|step| step.target_selection.as_ref())
                .map(|selection| selection.status.clone()),
            Some(TargetSelectionStatus::Selected)
        );
    }

    #[test]
    fn low_confidence_click_target_is_not_executed() {
        let mut resolution = ActionResolution::new(
            ActionOperation::ScreenGuidedFollowupAction,
            crate::action_resolution::ActionDomain::Screen,
            0.8,
            ResolutionSource::RustNormalizer,
        );
        resolution.workflow_steps = vec!["open_ranked_result".into()];
        resolution.entities = json!({
            "first_result_candidate": {
                "role": "ranked_result",
                "center_x": 700,
                "center_y": 360,
                "confidence": 0.52,
                "supports_click": true,
                "rank": 1
            }
        });

        let workflow =
            resolve_screen_workflow(&resolution, &manifest_with_screen(true), "apri il primo")
                .expect("workflow");
        let run = execute_screen_workflow(
            workflow,
            &UIControlRuntime::dry_run(),
            pointer_test_capabilities(UIPrimitiveKind::ClickTargetCandidate),
        );

        assert_eq!(run.status, WorkflowRunStatus::NeedsTargetGrounding);
        assert_eq!(run.completed_steps, 0);
        assert!(run
            .stopped_reason
            .as_deref()
            .unwrap_or("")
            .contains("below"));
    }

    #[test]
    fn recent_focus_target_allows_typing_followup() {
        let mut resolution = ActionResolution::new(
            ActionOperation::ScreenGuidedFollowupAction,
            crate::action_resolution::ActionDomain::Screen,
            0.8,
            ResolutionSource::RustNormalizer,
        );
        resolution.workflow_steps = vec!["enter_text".into()];
        resolution.entities = json!({
            "query_candidate": "Shiva",
            "provider": "youtube",
            "requires_recent_focus_target": true
        });

        let mut workflow =
            resolve_screen_workflow(&resolution, &manifest_with_screen(true), "ora scrivi Shiva")
                .expect("workflow");
        workflow.grounding.recent_target_candidates = vec![recent_search_candidate()];
        workflow.depends_on_recent_screen_context = true;
        refresh_screen_workflow_plan(&mut workflow);
        let run = execute_screen_workflow(
            workflow,
            &UIControlRuntime::dry_run(),
            keyboard_test_capabilities(),
        );

        assert_eq!(run.status, WorkflowRunStatus::Completed);
        assert_eq!(run.completed_steps, 1);
        assert!(run.step_runs[0].target_selection.is_some());
    }

    #[test]
    fn typing_followup_without_recent_focus_refuses() {
        let mut resolution = ActionResolution::new(
            ActionOperation::ScreenGuidedFollowupAction,
            crate::action_resolution::ActionDomain::Screen,
            0.8,
            ResolutionSource::RustNormalizer,
        );
        resolution.workflow_steps = vec!["enter_text".into()];
        resolution.entities = json!({
            "query_candidate": "Shiva",
            "requires_recent_focus_target": true
        });

        let workflow =
            resolve_screen_workflow(&resolution, &manifest_with_screen(true), "ora scrivi Shiva")
                .expect("workflow");
        let run = execute_screen_workflow(
            workflow,
            &UIControlRuntime::dry_run(),
            keyboard_test_capabilities(),
        );

        assert_eq!(run.status, WorkflowRunStatus::NeedsTargetGrounding);
    }

    #[test]
    fn render_partial_completion_response_describes_executed_but_unverified_goal_loop() {
        let response = render_screen_workflow_run_response(
            &ScreenWorkflowRun {
                run_id: "run".into(),
                status: WorkflowRunStatus::PartiallyCompleted,
                workflow: ScreenWorkflow {
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
                        goal_loop: Some(GoalLoopRun {
                            run_id: "goal_loop".into(),
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
                                    attributes: Value::Null,
                                },
                                success_condition: "media_watch_page_visible".into(),
                                utterance: "aprimi il primo video".into(),
                                confidence: 0.9,
                            },
                            status: GoalLoopStatus::VerificationFailed,
                            iteration_count: 2,
                            retry_budget: 3,
                            retries_used: 0,
                            current_strategy: None,
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
                                target_signature: Some("video_1".into()),
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
                            browser_recovery_status:
                                crate::desktop_agent_types::BrowserRecoveryStatus::NotNeeded,
                            post_action_progress_observed: true,
                            surface_ownership_lost: false,
                            focused_perception_failure_reason: None,
                            repeated_click_protection_triggered: false,
                            selected_target_candidate: None,
                            verifier_status: Some("Ambiguous".into()),
                            failure_reason: Some(
                                "final watch-page confirmation remained uncertain".into(),
                            ),
                        }),
                        recent_target_candidates: Vec::new(),
                        generated_at_ms: Some(1_000),
                        page_validation: None,
                        regrounding: None,
                        uncertainty: Vec::new(),
                    },
                    steps: vec![WorkflowStep {
                        step_id: "step".into(),
                        step_kind: WorkflowStepKind::OpenRankedResult,
                        target: json!({}),
                        value: None,
                        selection: json!({}),
                        expected_outcome: None,
                    }],
                    step_plans: Vec::new(),
                    support: WorkflowSupportSummary {
                        executable: true,
                        requires_screen_context: true,
                        unsupported_steps: Vec::new(),
                        reason: "test".into(),
                    },
                    confidence: 0.9,
                    source: ResolutionSource::RustNormalizer,
                    rationale: None,
                },
                primitive_capabilities: pointer_test_capabilities(
                    UIPrimitiveKind::ClickTargetCandidate,
                ),
                step_runs: Vec::new(),
                completed_steps: 1,
                stopped_reason: Some("final watch-page confirmation remained uncertain".into()),
                continuation_verification: None,
            },
            true,
        );

        assert!(response.contains("Ho eseguito il click sul candidato selezionato"));
        assert!(response.contains("verifica finale non conferma"));
        assert!(!response.contains("cambiata correttamente"));
        assert!(!response.contains("non ho un target abbastanza sicuro"));
    }

    #[test]
    fn post_click_message_does_not_claim_success_when_goal_not_achieved() {
        let response = render_screen_workflow_run_response(
            &ScreenWorkflowRun {
                run_id: "run".into(),
                status: WorkflowRunStatus::PartiallyCompleted,
                workflow: ScreenWorkflow {
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
                        goal_loop: Some(GoalLoopRun {
                            run_id: "goal_loop".into(),
                            goal: GoalSpec {
                                goal_id: "goal".into(),
                                goal_type: GoalType::OpenMediaResult,
                                constraints: GoalConstraints {
                                    provider: None,
                                    item_kind: Some("video".into()),
                                    result_kind: Some(VisibleResultKind::Video),
                                    rank_within_kind: Some(1),
                                    rank_overall: None,
                                    entity_name: None,
                                    attributes: Value::Null,
                                },
                                success_condition: "media_watch_page_visible".into(),
                                utterance: "aprimi il primo video".into(),
                                confidence: 0.9,
                            },
                            status: GoalLoopStatus::VerificationFailed,
                            iteration_count: 2,
                            retry_budget: 3,
                            retries_used: 0,
                            current_strategy: None,
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
                                target_signature: Some("video_1".into()),
                            }],
                            verification_history: vec![GoalVerificationRecord {
                                iteration: 1,
                                status: GoalVerificationStatus::GoalNotAchieved,
                                confidence: 0.40,
                                reason: "not achieved".into(),
                                frame_id: Some("frame".into()),
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
                            browser_recovery_status:
                                crate::desktop_agent_types::BrowserRecoveryStatus::NotNeeded,
                            post_action_progress_observed: true,
                            surface_ownership_lost: false,
                            focused_perception_failure_reason: None,
                            repeated_click_protection_triggered: false,
                            selected_target_candidate: None,
                            verifier_status: Some("GoalNotAchieved".into()),
                            failure_reason: Some("goal was not confirmed".into()),
                        }),
                        recent_target_candidates: Vec::new(),
                        generated_at_ms: Some(1_000),
                        page_validation: None,
                        regrounding: None,
                        uncertainty: Vec::new(),
                    },
                    steps: vec![WorkflowStep {
                        step_id: "step".into(),
                        step_kind: WorkflowStepKind::OpenRankedResult,
                        target: json!({}),
                        value: None,
                        selection: json!({}),
                        expected_outcome: None,
                    }],
                    step_plans: Vec::new(),
                    support: WorkflowSupportSummary {
                        executable: true,
                        requires_screen_context: true,
                        unsupported_steps: Vec::new(),
                        reason: "test".into(),
                    },
                    confidence: 0.9,
                    source: ResolutionSource::RustNormalizer,
                    rationale: None,
                },
                primitive_capabilities: pointer_test_capabilities(
                    UIPrimitiveKind::ClickTargetCandidate,
                ),
                step_runs: Vec::new(),
                completed_steps: 1,
                stopped_reason: Some("goal was not confirmed".into()),
                continuation_verification: None,
            },
            true,
        );

        assert!(response.contains("verifica finale non conferma"));
        assert!(!response.contains("cambiata correttamente"));
    }

    #[test]
    fn render_completed_continuation_response_uses_goal_achieved_verification() {
        let response = render_screen_workflow_run_response(
            &ScreenWorkflowRun {
                run_id: "run".into(),
                status: WorkflowRunStatus::Completed,
                workflow: ScreenWorkflow {
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
                            confidence: 0.92,
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
                        goal_loop: Some(GoalLoopRun {
                            run_id: "goal_loop".into(),
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
                                    attributes: Value::Null,
                                },
                                success_condition: "media_watch_page_visible".into(),
                                utterance: "aprimi il primo video".into(),
                                confidence: 0.9,
                            },
                            status: GoalLoopStatus::GoalAchieved,
                            iteration_count: 2,
                            retry_budget: 3,
                            retries_used: 0,
                            current_strategy: None,
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
                                target_signature: Some("video_1".into()),
                            }],
                            verification_history: vec![crate::desktop_agent_types::GoalVerificationRecord {
                                iteration: 1,
                                status: crate::desktop_agent_types::GoalVerificationStatus::GoalAchieved,
                                confidence: 0.94,
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
                            browser_recovery_status:
                                crate::desktop_agent_types::BrowserRecoveryStatus::NotNeeded,
                            post_action_progress_observed: true,
                            surface_ownership_lost: false,
                            focused_perception_failure_reason: None,
                            repeated_click_protection_triggered: false,
                            selected_target_candidate: None,
                            verifier_status: Some("GoalAchieved".into()),
                            failure_reason: None,
                        }),
                        recent_target_candidates: Vec::new(),
                        generated_at_ms: Some(1_000),
                        page_validation: None,
                        regrounding: None,
                        uncertainty: Vec::new(),
                    },
                    steps: vec![WorkflowStep {
                        step_id: "step".into(),
                        step_kind: WorkflowStepKind::OpenRankedResult,
                        target: json!({}),
                        value: None,
                        selection: json!({"rank": 1}),
                        expected_outcome: Some("watch page visible".into()),
                    }],
                    step_plans: Vec::new(),
                    support: WorkflowSupportSummary {
                        executable: true,
                        requires_screen_context: true,
                        unsupported_steps: Vec::new(),
                        reason: "test".into(),
                    },
                    confidence: 0.9,
                    source: ResolutionSource::RustNormalizer,
                    rationale: None,
                },
                primitive_capabilities: pointer_test_capabilities(
                    UIPrimitiveKind::ClickTargetCandidate,
                ),
                step_runs: Vec::new(),
                completed_steps: 1,
                stopped_reason: None,
                continuation_verification: Some(
                    crate::workflow_continuation::ContinuationVerificationResult {
                        status: crate::workflow_continuation::ContinuationVerificationStatus::GoalAchieved,
                        expected_state_change: "watch page visible".into(),
                        observed_evidence: vec!["goal_loop_status=GoalAchieved".into()],
                        requires_post_step_screen_check: true,
                        next_resumable: false,
                        goal_loop_status: Some(GoalLoopStatus::GoalAchieved),
                        goal_verifier_status: Some(
                            crate::desktop_agent_types::GoalVerificationStatus::GoalAchieved,
                        ),
                        post_action_progress_observed: true,
                    },
                ),
            },
            true,
        );

        assert!(response.contains("verifica finale ha confermato"));
        assert!(!response.contains("prossima cattura"));
    }

    fn manifest_with_screen(recent_capture: bool) -> CapabilityManifest {
        let ready = CapabilityToolAvailability {
            available: true,
            enabled: true,
            requires_approval: false,
            state: CapabilityRuntimeState::Ready,
            disabled_reason: None,
        };
        CapabilityManifest {
            version: "test".into(),
            generated_at: 0,
            tool_names: Vec::new(),
            enabled_tool_names: Vec::new(),
            disabled_tool_names: Vec::new(),
            tools: Vec::new(),
            filesystem_read: ready.clone(),
            filesystem_write: ready.clone(),
            filesystem_search: ready.clone(),
            terminal: ready.clone(),
            browser_open: ready.clone(),
            browser_search: ready.clone(),
            desktop_launch: ready,
            screen: CapabilityScreenState {
                observation_supported: true,
                observation_enabled: true,
                capture_available: true,
                analysis_available: true,
                vision_model_available: true,
                vision_model_name: Some("test-vision".into()),
                recent_capture_available: recent_capture,
                recent_capture_age_ms: recent_capture.then_some(500),
                fresh_capture_available: true,
                fresh_capture_requires_observation_enabled: true,
                accessibility_snapshot_enabled: cfg!(target_os = "windows"),
                last_capture_path: recent_capture.then(|| "screen.png".into()),
                last_frame_at: recent_capture.then_some(1),
                provider: "test".into(),
                note: "test".into(),
            },
            approvals: CapabilityApprovalState {
                pending_count: 0,
                approval_required_for_high_risk: true,
                pending_actions: Vec::new(),
            },
            permissions: CapabilityPermissionState {
                allowed_permissions: vec![Permission::DesktopObserve],
                browser_enabled: true,
                desktop_control_enabled: false,
                allowed_roots: Vec::new(),
                terminal_allowed_commands: Vec::new(),
            },
        }
    }

    fn keyboard_test_capabilities() -> UIPrimitiveCapabilitySet {
        UIPrimitiveCapabilitySet {
            platform: "test".into(),
            desktop_control_enabled: true,
            primitives: vec![
                UIPrimitiveCapability {
                    primitive: UIPrimitiveKind::TypeText,
                    available: true,
                    enabled: true,
                    requires_screen_context: false,
                    requires_high_confidence_target: false,
                    requires_approval: false,
                    platform_note: "test".into(),
                },
                UIPrimitiveCapability {
                    primitive: UIPrimitiveKind::PressEnter,
                    available: true,
                    enabled: true,
                    requires_screen_context: false,
                    requires_high_confidence_target: false,
                    requires_approval: false,
                    platform_note: "test".into(),
                },
            ],
        }
    }

    fn pointer_test_capabilities(primitive: UIPrimitiveKind) -> UIPrimitiveCapabilitySet {
        UIPrimitiveCapabilitySet {
            platform: "test".into(),
            desktop_control_enabled: true,
            primitives: vec![UIPrimitiveCapability {
                primitive,
                available: true,
                enabled: true,
                requires_screen_context: true,
                requires_high_confidence_target: true,
                requires_approval: false,
                platform_note: "test".into(),
            }],
        }
    }

    fn recent_search_candidate() -> UITargetCandidate {
        UITargetCandidate {
            candidate_id: "recent_search".into(),
            element_id: None,
            accessibility_snapshot_id: None,
            role: UITargetRole::SearchInput,
            region: Some(TargetRegion {
                x: 100.0,
                y: 60.0,
                width: 600.0,
                height: 42.0,
                coordinate_space: "screen".into(),
            }),
            center_x: None,
            center_y: None,
            app_hint: Some("chrome".into()),
            browser_app_hint: Some("chrome".into()),
            provider_hint: Some("youtube".into()),
            content_provider_hint: Some("youtube".into()),
            page_kind_hint: None,
            capture_backend: None,
            observation_source: None,
            result_kind: None,
            confidence: 0.88,
            source: TargetGroundingSource::RecentContext,
            label: Some("YouTube search".into()),
            rank: None,
            observed_at_ms: Some(now_ms()),
            reuse_eligible: true,
            supports_focus: true,
            supports_click: true,
            rationale: "recent successful focus target".into(),
        }
    }
}
