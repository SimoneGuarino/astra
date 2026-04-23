use crate::{
    action_resolution::{ActionDomain, ActionOperation, ActionResolution, ResolutionSource},
    desktop_agent_types::{
        CapabilityManifest, DesktopActionRequest, PageSemanticEvidence, ScreenCaptureResult,
    },
    screen_workflow::{
        resolve_screen_workflow, ScreenWorkflow, ScreenWorkflowRun, WorkflowRunStatus,
        WorkflowStepKind,
    },
    ui_control::{UIPrimitiveCapabilitySet, UIPrimitiveKind},
    ui_target_grounding::{
        candidate_browser_app_hint, candidate_content_provider_hint, is_technical_capture_backend,
        normalize_content_provider_hint, UITargetCandidate, UITargetRole,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

pub const RECENT_WORKFLOW_CONTEXT_TTL_MS: u64 = 5 * 60 * 1_000;
pub const RECENT_RESULT_LIST_TTL_MS: u64 = 3 * 60 * 1_000;
pub const MAX_RECENT_SCREEN_AGE_MS: u64 = 3 * 60 * 1_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RecentWorkflowContextKind {
    BrowserSearchResults,
    BrowserPage,
    ScreenWorkflow,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BrowserPageKind {
    SearchResults,
    WatchPage,
    WebPage,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResultListAvailability {
    ExpectedOnScreen,
    ObservedCandidates,
    OpenedFromList,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResultListItemKind {
    Result,
    Video,
    Link,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultListContext {
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub query: Option<String>,
    pub item_kind: ResultListItemKind,
    pub availability: ResultListAvailability,
    #[serde(default)]
    pub observed_candidates: Vec<UITargetCandidate>,
    pub expected_visible: bool,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    pub expires_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentWorkflowContext {
    pub context_id: String,
    pub request_id: String,
    #[serde(default)]
    pub action_id: Option<String>,
    #[serde(default)]
    pub run_id: Option<String>,
    pub kind: RecentWorkflowContextKind,
    pub page_kind: BrowserPageKind,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub app: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub result_list: Option<ResultListContext>,
    #[serde(default)]
    pub recent_focused_target: Option<UITargetCandidate>,
    #[serde(default)]
    pub recent_selected_target: Option<UITargetCandidate>,
    #[serde(default)]
    pub last_run_status: Option<String>,
    #[serde(default)]
    pub last_followup: Option<FollowupActionResolution>,
    pub resumable: bool,
    pub continuation_allowed: bool,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    pub expires_at_ms: u64,
}

impl RecentWorkflowContext {
    pub fn summary(&self) -> RecentWorkflowContextSummary {
        RecentWorkflowContextSummary {
            context_id: self.context_id.clone(),
            kind: self.kind.clone(),
            page_kind: self.page_kind.clone(),
            provider: self.provider.clone(),
            app: self.app.clone(),
            url: self.url.clone(),
            query: self.query.clone(),
            has_result_list: self.result_list.is_some(),
            has_recent_focused_target: self.recent_focused_target.is_some(),
            has_recent_selected_target: self.recent_selected_target.is_some(),
            last_run_status: self.last_run_status.clone(),
            resumable: self.resumable,
            continuation_allowed: self.continuation_allowed,
            updated_at_ms: self.updated_at_ms,
            expires_at_ms: self.expires_at_ms,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentWorkflowContextSummary {
    pub context_id: String,
    pub kind: RecentWorkflowContextKind,
    pub page_kind: BrowserPageKind,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub app: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub query: Option<String>,
    pub has_result_list: bool,
    pub has_recent_focused_target: bool,
    pub has_recent_selected_target: bool,
    #[serde(default)]
    pub last_run_status: Option<String>,
    pub resumable: bool,
    pub continuation_allowed: bool,
    pub updated_at_ms: u64,
    pub expires_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FollowupActionKind {
    OpenResult,
    ClickResult,
    ClickReferencedTarget,
    TypeText,
    NavigateBack,
    NavigateBackThenOpenResult,
    ContinueWorkflow,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FollowupResolutionSource {
    RustFollowupResolver,
    ModelAssistedClassifier,
}

impl FollowupResolutionSource {
    pub fn as_resolution_source(&self) -> ResolutionSource {
        match self {
            Self::RustFollowupResolver => ResolutionSource::RustNormalizer,
            Self::ModelAssistedClassifier => ResolutionSource::ModelAssisted,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContextualFollowupKind {
    OpenResult,
    ClickResult,
    ClickReferencedTarget,
    TypeIntoRecentFocus,
    NavigateBack,
    NavigateBackThenOpenResult,
    ResumeRecentWorkflow,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EntityRoleKind {
    BrowserApp,
    AppContainer,
    ContentProvider,
    PageContext,
    QueryText,
    ResultReference,
    ContinuationControl,
    TextValue,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PageContextHint {
    RecentResultsPage,
    MatchingBrowserTab,
    CurrentPage,
    PreviousPage,
    RecentFocus,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRoleAssignment {
    pub text: String,
    pub role: EntityRoleKind,
    #[serde(default)]
    pub normalized_value: Option<String>,
    pub confidence: f32,
    pub source: FollowupResolutionSource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualFollowupInterpretation {
    pub utterance: String,
    pub continuation_kind: ContextualFollowupKind,
    #[serde(default)]
    pub browser_hint: Option<String>,
    #[serde(default)]
    pub app_hint: Option<String>,
    #[serde(default)]
    pub provider_hint: Option<String>,
    #[serde(default)]
    pub page_context_hint: Option<PageContextHint>,
    #[serde(default)]
    pub query_hint: Option<String>,
    #[serde(default)]
    pub text_value: Option<String>,
    #[serde(default)]
    pub result_reference: Option<ResultListReference>,
    #[serde(default)]
    pub ordinal_reference: Option<u32>,
    pub requires_recent_focus_target: bool,
    pub requires_recent_results_context: bool,
    pub requires_resumable_workflow: bool,
    pub confidence: f32,
    pub source: FollowupResolutionSource,
    pub rationale: String,
    #[serde(default)]
    pub entity_roles: Vec<EntityRoleAssignment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualMergeDiagnostic {
    #[serde(default)]
    pub effective_provider: Option<String>,
    #[serde(default)]
    pub provider_source: Option<String>,
    #[serde(default)]
    pub effective_query: Option<String>,
    #[serde(default)]
    pub query_source: Option<String>,
    #[serde(default)]
    pub effective_browser_app: Option<String>,
    #[serde(default)]
    pub browser_app_source: Option<String>,
    pub reused_recent_context: bool,
    #[serde(default)]
    pub conflicts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultListReference {
    #[serde(default)]
    pub rank: Option<u32>,
    pub ordinal_label: String,
    pub item_kind: ResultListItemKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FollowupActionResolution {
    pub action_kind: FollowupActionKind,
    #[serde(default)]
    pub result_reference: Option<ResultListReference>,
    #[serde(default)]
    pub text_value: Option<String>,
    #[serde(default)]
    pub provider_hint: Option<String>,
    #[serde(default)]
    pub query_hint: Option<String>,
    #[serde(default)]
    pub browser_hint: Option<String>,
    #[serde(default)]
    pub app_hint: Option<String>,
    #[serde(default)]
    pub page_context_hint: Option<PageContextHint>,
    pub requires_result_list: bool,
    pub requires_recent_focus_target: bool,
    pub confidence: f32,
    pub source: FollowupResolutionSource,
    pub rationale: String,
    #[serde(default)]
    pub interpretation: Option<ContextualFollowupInterpretation>,
    #[serde(default)]
    pub merge_diagnostic: Option<ContextualMergeDiagnostic>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContinuationPolicyStatus {
    SafeToAttempt,
    NotAContinuation,
    NoRecentWorkflowContext,
    StaleWorkflowContext,
    ProviderMismatch,
    QueryMismatch,
    ResultListUnavailable,
    ResultListStale,
    ScreenContextUnavailable,
    TargetContextUnavailable,
    NoResumableWorkflow,
    UnsupportedMultiStepRegrounding,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SemanticPageValidationStatus {
    Matched,
    LikelyMatched,
    Unknown,
    NeedsFreshCapture,
    Mismatched,
    Unsupported,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticPageValidationResult {
    pub status: SemanticPageValidationStatus,
    pub expected_page_kind: BrowserPageKind,
    pub observed_page_kind: BrowserPageKind,
    #[serde(default)]
    pub expected_provider: Option<String>,
    #[serde(default)]
    pub observed_provider: Option<String>,
    #[serde(default)]
    pub observed_browser_app: Option<String>,
    #[serde(default)]
    pub expected_query: Option<String>,
    #[serde(default)]
    pub observed_query: Option<String>,
    #[serde(default)]
    pub capture_backend: Option<String>,
    #[serde(default)]
    pub observation_source: Option<String>,
    pub confidence: f32,
    pub needs_fresh_capture: bool,
    #[serde(default)]
    pub mismatch_reason: Option<String>,
    #[serde(default)]
    pub evidence: Vec<String>,
    #[serde(default)]
    pub captured_at_ms: Option<u64>,
    #[serde(default)]
    pub capture_age_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScrollContinuationStatus {
    NotNeeded,
    SafeToAttempt,
    Unsupported,
    RetryBudgetExhausted,
    PageMismatch,
    NotApplicable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrollContinuationDecision {
    pub status: ScrollContinuationStatus,
    pub attempt_index: usize,
    pub max_attempts: usize,
    pub scroll_supported: bool,
    #[serde(default)]
    pub requested_rank: Option<u32>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegroundingAttemptDiagnostic {
    pub attempt_index: usize,
    pub page_validation: SemanticPageValidationResult,
    pub visible_candidate_count: usize,
    #[serde(default)]
    pub selected_candidate_id: Option<String>,
    #[serde(default)]
    pub target_selection_status: Option<String>,
    pub scroll_decision: ScrollContinuationDecision,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuationRegroundingDiagnostics {
    pub max_attempts: usize,
    #[serde(default)]
    pub attempts: Vec<RegroundingAttemptDiagnostic>,
    pub final_status: ScrollContinuationStatus,
    pub final_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuationPolicyDecision {
    pub status: ContinuationPolicyStatus,
    pub executable: bool,
    pub reason: String,
    #[serde(default)]
    pub context_age_ms: Option<u64>,
    #[serde(default)]
    pub screen_age_ms: Option<u64>,
    #[serde(default)]
    pub required_rank: Option<u32>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub query: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContinuationVerifierKind {
    PrimitiveOnly,
    ResultNavigationExpected,
    FocusPreservationExpected,
    BrowserBackExpected,
    RequiresNextScreenCheck,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuationVerifierExpectation {
    pub verifier_kind: ContinuationVerifierKind,
    pub expected_state_change: String,
    pub requires_post_step_screen_check: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowContinuationDescriptor {
    pub followup: FollowupActionResolution,
    pub policy: ContinuationPolicyDecision,
    pub source_context: RecentWorkflowContextSummary,
    pub verifier: ContinuationVerifierExpectation,
    #[serde(default)]
    pub page_validation: Option<SemanticPageValidationResult>,
    #[serde(default)]
    pub regrounding: Option<ContinuationRegroundingDiagnostics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContinuationVerificationStatus {
    SatisfiedAtPrimitiveLevel,
    NeedsPostStepScreenCheck,
    PageMismatch,
    RegroundingRequired,
    ScrollUnsupported,
    RegroundingExhausted,
    Failed,
    Unsupported,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuationVerificationResult {
    pub status: ContinuationVerificationStatus,
    pub expected_state_change: String,
    #[serde(default)]
    pub observed_evidence: Vec<String>,
    pub requires_post_step_screen_check: bool,
    pub next_resumable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuationRefusal {
    pub followup: FollowupActionResolution,
    #[serde(default)]
    pub source_context: Option<RecentWorkflowContextSummary>,
    pub policy: ContinuationPolicyDecision,
}

impl ContinuationRefusal {
    pub fn diagnostic_value(&self) -> Value {
        json!({
            "followup": self.followup,
            "source_context": self.source_context,
            "policy": self.policy,
        })
    }
}

pub enum WorkflowContinuationResolution {
    Workflow(ScreenWorkflow),
    Refusal(ContinuationRefusal),
}

pub fn is_contextual_followup_message(message: &str) -> bool {
    interpret_contextual_followup(message, None, None).is_some()
}

pub fn resolve_workflow_continuation(
    context: Option<RecentWorkflowContext>,
    manifest: &CapabilityManifest,
    message: &str,
    now_ms: u64,
) -> Option<WorkflowContinuationResolution> {
    let interpretation = interpret_contextual_followup(message, None, None)?;
    resolve_workflow_continuation_from_interpretation(
        context,
        manifest,
        message,
        now_ms,
        interpretation,
    )
}

pub fn resolve_workflow_continuation_with_model_params(
    context: Option<RecentWorkflowContext>,
    manifest: &CapabilityManifest,
    message: &str,
    model_params: &Value,
    model_confidence: f32,
    now_ms: u64,
) -> Option<WorkflowContinuationResolution> {
    let interpretation =
        interpret_contextual_followup(message, Some(model_params), Some(model_confidence))?;
    resolve_workflow_continuation_from_interpretation(
        context,
        manifest,
        message,
        now_ms,
        interpretation,
    )
}

fn resolve_workflow_continuation_from_interpretation(
    context: Option<RecentWorkflowContext>,
    manifest: &CapabilityManifest,
    message: &str,
    now_ms: u64,
    interpretation: ContextualFollowupInterpretation,
) -> Option<WorkflowContinuationResolution> {
    let mut followup = followup_from_interpretation(interpretation);
    let context = match context {
        Some(context) => context,
        None => {
            let policy = ContinuationPolicyDecision {
                status: ContinuationPolicyStatus::NoRecentWorkflowContext,
                executable: false,
                reason: "The message looks like a workflow follow-up, but no recent workflow context is active.".into(),
                context_age_ms: None,
                screen_age_ms: manifest.screen.recent_capture_age_ms,
                required_rank: followup
                    .result_reference
                    .as_ref()
                    .and_then(|reference| reference.rank),
                provider: followup.provider_hint.clone(),
                query: followup.query_hint.clone(),
            };
            return Some(WorkflowContinuationResolution::Refusal(
                ContinuationRefusal {
                    followup,
                    source_context: None,
                    policy,
                },
            ));
        }
    };

    followup = merge_followup_with_recent_context(&context, followup);

    if matches!(followup.action_kind, FollowupActionKind::ContinueWorkflow) {
        if let Some(previous_followup) = context.last_followup.clone() {
            followup = previous_followup;
        }
    }

    let policy = continuation_policy(&context, &followup, manifest, now_ms);
    if !policy.executable {
        return Some(WorkflowContinuationResolution::Refusal(
            ContinuationRefusal {
                followup,
                source_context: Some(context.summary()),
                policy,
            },
        ));
    }

    let workflow = build_continuation_workflow(&context, &followup, &policy, manifest, message)?;
    Some(WorkflowContinuationResolution::Workflow(workflow))
}

pub fn build_context_from_action_response(
    request_id: &str,
    action_id: &str,
    request: &DesktopActionRequest,
    result: &Value,
    now_ms: u64,
) -> Option<RecentWorkflowContext> {
    let (url, provider, query, app) = match request.tool_name.as_str() {
        "browser.search" => {
            let query = value_str(&request.params, "query")
                .or_else(|| value_str(result, "query"))
                .map(ToOwned::to_owned);
            let url = value_str(result, "url").map(ToOwned::to_owned);
            let provider = value_str(&request.params, "provider")
                .map(ToOwned::to_owned)
                .or_else(|| Some("google".into()));
            (url, provider, query, Some("browser".into()))
        }
        "browser.open" => {
            let url = value_str(&request.params, "url")
                .or_else(|| value_str(result, "url"))
                .map(ToOwned::to_owned)?;
            let provider = value_str(&request.params, "provider")
                .map(ToOwned::to_owned)
                .or_else(|| infer_provider_from_url(&url));
            let query = value_str(&request.params, "query")
                .map(ToOwned::to_owned)
                .or_else(|| infer_query_from_url(&url));
            (Some(url), provider, query, Some("browser".into()))
        }
        "desktop.launch_app" => {
            let url = request
                .params
                .get("args")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .find(|arg| is_http_url(arg))
                .map(ToOwned::to_owned)?;
            let provider = infer_provider_from_url(&url);
            let query = infer_query_from_url(&url);
            (Some(url), provider, query, Some("chrome".into()))
        }
        _ => return None,
    };

    let page_kind = url
        .as_deref()
        .map(page_kind_from_url)
        .unwrap_or(BrowserPageKind::SearchResults);
    let result_list = if matches!(page_kind, BrowserPageKind::SearchResults) || query.is_some() {
        Some(ResultListContext {
            provider: provider.clone(),
            query: query.clone(),
            item_kind: if provider
                .as_deref()
                .is_some_and(|value| labels_match(value, "youtube"))
            {
                ResultListItemKind::Video
            } else {
                ResultListItemKind::Result
            },
            availability: ResultListAvailability::ExpectedOnScreen,
            observed_candidates: Vec::new(),
            expected_visible: true,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
            expires_at_ms: now_ms.saturating_add(RECENT_RESULT_LIST_TTL_MS),
        })
    } else {
        None
    };

    Some(RecentWorkflowContext {
        context_id: Uuid::new_v4().to_string(),
        request_id: request_id.to_string(),
        action_id: Some(action_id.to_string()),
        run_id: None,
        kind: if result_list.is_some() {
            RecentWorkflowContextKind::BrowserSearchResults
        } else {
            RecentWorkflowContextKind::BrowserPage
        },
        page_kind,
        provider,
        app,
        url,
        query,
        result_list,
        recent_focused_target: None,
        recent_selected_target: None,
        last_run_status: None,
        last_followup: None,
        resumable: false,
        continuation_allowed: true,
        created_at_ms: now_ms,
        updated_at_ms: now_ms,
        expires_at_ms: now_ms.saturating_add(RECENT_WORKFLOW_CONTEXT_TTL_MS),
    })
}

pub fn build_context_from_screen_workflow_run(
    run: &ScreenWorkflowRun,
    request_id: &str,
    action_id: &str,
    previous_context: Option<RecentWorkflowContext>,
    now_ms: u64,
) -> Option<RecentWorkflowContext> {
    let provider = run
        .workflow
        .continuation
        .as_ref()
        .and_then(|continuation| continuation.source_context.provider.clone())
        .or_else(|| workflow_provider(&run.workflow))
        .or_else(|| {
            previous_context
                .as_ref()
                .and_then(|context| context.provider.clone())
        });
    let query = run
        .workflow
        .continuation
        .as_ref()
        .and_then(|continuation| continuation.source_context.query.clone())
        .or_else(|| workflow_query(&run.workflow))
        .or_else(|| {
            previous_context
                .as_ref()
                .and_then(|context| context.query.clone())
        });
    let url = previous_context
        .as_ref()
        .and_then(|context| context.url.clone());
    let page_kind = previous_context
        .as_ref()
        .map(|context| context.page_kind.clone())
        .unwrap_or(BrowserPageKind::Unknown);

    let recent_focused_target = run
        .step_runs
        .iter()
        .rev()
        .find(|step| {
            matches!(
                step.step.step_kind,
                WorkflowStepKind::FocusSearchInput | WorkflowStepKind::EnterText
            ) && matches!(step.status, WorkflowRunStatus::Completed)
        })
        .and_then(|step| {
            step.target_selection
                .as_ref()
                .and_then(|selection| selection.selected_candidate.clone())
        })
        .or_else(|| {
            previous_context
                .as_ref()
                .and_then(|context| context.recent_focused_target.clone())
        });

    let recent_selected_target = run
        .step_runs
        .iter()
        .rev()
        .find(|step| {
            matches!(
                step.step.step_kind,
                WorkflowStepKind::OpenRankedResult | WorkflowStepKind::ClickVisibleElement
            ) && matches!(step.status, WorkflowRunStatus::Completed)
        })
        .and_then(|step| {
            step.target_selection
                .as_ref()
                .and_then(|selection| selection.selected_candidate.clone())
        })
        .or_else(|| {
            previous_context
                .as_ref()
                .and_then(|context| context.recent_selected_target.clone())
        });

    let observed_result_candidates = run
        .workflow
        .grounding
        .visible_target_candidates
        .iter()
        .chain(run.workflow.grounding.recent_target_candidates.iter())
        .filter(|candidate| candidate.role == UITargetRole::RankedResult)
        .cloned()
        .collect::<Vec<_>>();
    let result_list = if !observed_result_candidates.is_empty()
        || workflow_mentions_result_list(&run.workflow)
        || previous_context
            .as_ref()
            .and_then(|context| context.result_list.as_ref())
            .is_some()
    {
        Some(ResultListContext {
            provider: provider.clone(),
            query: query.clone(),
            item_kind: provider_result_kind(provider.as_deref()),
            availability: if !observed_result_candidates.is_empty() {
                ResultListAvailability::ObservedCandidates
            } else if run
                .step_runs
                .iter()
                .any(|step| step.step.step_kind == WorkflowStepKind::OpenRankedResult)
            {
                ResultListAvailability::OpenedFromList
            } else {
                ResultListAvailability::ExpectedOnScreen
            },
            observed_candidates: observed_result_candidates,
            expected_visible: true,
            created_at_ms: previous_context
                .as_ref()
                .and_then(|context| context.result_list.as_ref())
                .map(|result_list| result_list.created_at_ms)
                .unwrap_or(now_ms),
            updated_at_ms: now_ms,
            expires_at_ms: now_ms.saturating_add(RECENT_RESULT_LIST_TTL_MS),
        })
    } else {
        None
    };

    let resumable = matches!(
        run.status,
        WorkflowRunStatus::PartiallyCompleted
            | WorkflowRunStatus::NeedsScreenContext
            | WorkflowRunStatus::NeedsTargetGrounding
            | WorkflowRunStatus::StepFailed
            | WorkflowRunStatus::StepUnsupported
    );

    Some(RecentWorkflowContext {
        context_id: previous_context
            .as_ref()
            .map(|context| context.context_id.clone())
            .unwrap_or_else(|| Uuid::new_v4().to_string()),
        request_id: request_id.to_string(),
        action_id: Some(action_id.to_string()),
        run_id: Some(run.run_id.clone()),
        kind: RecentWorkflowContextKind::ScreenWorkflow,
        page_kind,
        provider,
        app: Some("chrome".into()),
        url,
        query,
        result_list,
        recent_focused_target,
        recent_selected_target,
        last_run_status: Some(run.status.as_str().into()),
        last_followup: run
            .workflow
            .continuation
            .as_ref()
            .map(|continuation| continuation.followup.clone()),
        resumable,
        continuation_allowed: true,
        created_at_ms: previous_context
            .as_ref()
            .map(|context| context.created_at_ms)
            .unwrap_or(now_ms),
        updated_at_ms: now_ms,
        expires_at_ms: now_ms.saturating_add(RECENT_WORKFLOW_CONTEXT_TTL_MS),
    })
}

pub fn build_continuation_verification_result(
    descriptor: &WorkflowContinuationDescriptor,
    run_status: &WorkflowRunStatus,
    completed_steps: usize,
    planned_steps: usize,
    stopped_reason: Option<&str>,
) -> ContinuationVerificationResult {
    let completed = matches!(run_status, WorkflowRunStatus::Completed)
        && completed_steps == planned_steps
        && planned_steps > 0;
    let unsupported = matches!(
        run_status,
        WorkflowRunStatus::StepUnsupported
            | WorkflowRunStatus::NeedsScreenContext
            | WorkflowRunStatus::NeedsTargetGrounding
    );
    let failed = matches!(
        run_status,
        WorkflowRunStatus::StepFailed | WorkflowRunStatus::Aborted
    );
    let status = if failed {
        ContinuationVerificationStatus::Failed
    } else if descriptor
        .page_validation
        .as_ref()
        .is_some_and(|validation| validation.status == SemanticPageValidationStatus::Mismatched)
    {
        ContinuationVerificationStatus::PageMismatch
    } else if descriptor.regrounding.as_ref().is_some_and(|regrounding| {
        regrounding.final_status == ScrollContinuationStatus::Unsupported
    }) {
        ContinuationVerificationStatus::ScrollUnsupported
    } else if descriptor.regrounding.as_ref().is_some_and(|regrounding| {
        regrounding.final_status == ScrollContinuationStatus::RetryBudgetExhausted
    }) {
        ContinuationVerificationStatus::RegroundingExhausted
    } else if unsupported {
        if matches!(run_status, WorkflowRunStatus::NeedsTargetGrounding) {
            ContinuationVerificationStatus::RegroundingRequired
        } else {
            ContinuationVerificationStatus::Unsupported
        }
    } else if completed && descriptor.verifier.requires_post_step_screen_check {
        ContinuationVerificationStatus::NeedsPostStepScreenCheck
    } else if completed {
        ContinuationVerificationStatus::SatisfiedAtPrimitiveLevel
    } else {
        ContinuationVerificationStatus::Failed
    };

    let mut observed_evidence = Vec::new();
    observed_evidence.push(format!("workflow_status={}", run_status.as_str()));
    observed_evidence.push(format!("completed_steps={completed_steps}/{planned_steps}"));
    if let Some(reason) = stopped_reason.filter(|value| !value.trim().is_empty()) {
        observed_evidence.push(format!("stopped_reason={reason}"));
    }
    if let Some(validation) = descriptor.page_validation.as_ref() {
        observed_evidence.push(format!("page_validation={:?}", validation.status));
        if let Some(reason) = validation.mismatch_reason.as_deref() {
            observed_evidence.push(format!("page_validation_reason={reason}"));
        }
    }
    if let Some(regrounding) = descriptor.regrounding.as_ref() {
        observed_evidence.push(format!("regrounding_status={:?}", regrounding.final_status));
        observed_evidence.push(format!("regrounding_reason={}", regrounding.final_reason));
        observed_evidence.push(format!(
            "regrounding_attempts={}",
            regrounding.attempts.len()
        ));
    }

    ContinuationVerificationResult {
        status,
        expected_state_change: descriptor.verifier.expected_state_change.clone(),
        observed_evidence,
        requires_post_step_screen_check: descriptor.verifier.requires_post_step_screen_check,
        next_resumable: !completed,
    }
}

pub fn render_continuation_refusal(refusal: &ContinuationRefusal, italian: bool) -> String {
    let target = continuation_target_label(&refusal.followup, refusal.source_context.as_ref());
    let reason = refusal.policy.reason.as_str();
    if italian {
        match refusal.policy.status {
            ContinuationPolicyStatus::NoRecentWorkflowContext => format!(
                "Ho capito la richiesta come follow-up ({target}), ma non ho un workflow recente attivo a cui collegarla."
            ),
            ContinuationPolicyStatus::StaleWorkflowContext => format!(
                "Ho capito la richiesta come continuazione di {target}, ma il contesto recente e' scaduto. Riporta la pagina corretta sullo schermo o ripeti la richiesta completa."
            ),
            ContinuationPolicyStatus::ResultListUnavailable => format!(
                "Ho capito che vuoi usare la lista risultati per {target}, ma nel runtime non c'e' una lista risultati recente e riutilizzabile."
            ),
            ContinuationPolicyStatus::ResultListStale => format!(
                "Ho capito che vuoi usare la lista risultati per {target}, ma quella lista e' troppo vecchia per cliccarla in modo sicuro."
            ),
            ContinuationPolicyStatus::ScreenContextUnavailable => format!(
                "Ho capito la continuazione ({target}), ma serve una schermata verificabile prima di scegliere un candidato da cliccare: {reason}"
            ),
            ContinuationPolicyStatus::ProviderMismatch
            | ContinuationPolicyStatus::QueryMismatch => format!(
                "Ho capito la continuazione ({target}), ma non combacia abbastanza con il workflow recente: {reason}"
            ),
            ContinuationPolicyStatus::TargetContextUnavailable => format!(
                "Ho capito la continuazione ({target}), ma non ho un target recente o visibile abbastanza sicuro: {reason}"
            ),
            ContinuationPolicyStatus::NoResumableWorkflow => {
                "Ho capito 'continua', ma non c'e' un workflow screen-grounded fermo in uno stato riprendibile.".into()
            }
            ContinuationPolicyStatus::UnsupportedMultiStepRegrounding => format!(
                "Ho capito la richiesta multi-step ({target}), ma non la eseguo ancora: dopo il torna indietro devo ricatturare e ri-groundare la lista risultati prima di cliccare."
            ),
            ContinuationPolicyStatus::NotAContinuation | ContinuationPolicyStatus::SafeToAttempt => {
                format!("Non posso continuare quel workflow in modo sicuro: {reason}")
            }
        }
    } else {
        match refusal.policy.status {
            ContinuationPolicyStatus::NoRecentWorkflowContext => format!(
                "I understood this as a follow-up ({target}), but there is no active recent workflow context to attach it to."
            ),
            ContinuationPolicyStatus::StaleWorkflowContext => format!(
                "I understood this as a continuation of {target}, but the recent context is stale. Put the relevant page back on screen or repeat the full request."
            ),
            ContinuationPolicyStatus::ResultListUnavailable => format!(
                "I understood that you want to use the result list for {target}, but the runtime has no recent reusable result-list context."
            ),
            ContinuationPolicyStatus::ResultListStale => format!(
                "I understood that you want to use the result list for {target}, but that result list is too old to click safely."
            ),
            ContinuationPolicyStatus::ScreenContextUnavailable => format!(
                "I understood the continuation ({target}), but I need verifiable screen context before selecting a click candidate: {reason}"
            ),
            ContinuationPolicyStatus::ProviderMismatch
            | ContinuationPolicyStatus::QueryMismatch => format!(
                "I understood the continuation ({target}), but it does not match the recent workflow closely enough: {reason}"
            ),
            ContinuationPolicyStatus::TargetContextUnavailable => format!(
                "I understood the continuation ({target}), but I do not have a recent or visible target safe enough to use: {reason}"
            ),
            ContinuationPolicyStatus::NoResumableWorkflow => {
                "I understood 'continue', but there is no screen-grounded workflow stopped in a resumable state.".into()
            }
            ContinuationPolicyStatus::UnsupportedMultiStepRegrounding => format!(
                "I understood the multi-step request ({target}), but I am not executing it yet: after going back I need to recapture and re-ground the result list before clicking."
            ),
            ContinuationPolicyStatus::NotAContinuation | ContinuationPolicyStatus::SafeToAttempt => {
                format!("I cannot continue that workflow safely: {reason}")
            }
        }
    }
}

pub fn validate_continuation_page(
    descriptor: &WorkflowContinuationDescriptor,
    capture: Option<&ScreenCaptureResult>,
    page_evidence: &[PageSemanticEvidence],
    candidates: &[UITargetCandidate],
    now_ms: u64,
) -> SemanticPageValidationResult {
    let expected_provider = descriptor
        .policy
        .provider
        .clone()
        .or_else(|| descriptor.source_context.provider.clone());
    let expected_query = descriptor
        .policy
        .query
        .clone()
        .or_else(|| descriptor.source_context.query.clone());
    let expected_page_kind =
        expected_page_kind_for_followup(&descriptor.followup, &descriptor.source_context);
    let observed_page_kind = observed_page_kind_from_evidence(page_evidence)
        .unwrap_or_else(|| observed_page_kind_from_candidates(candidates));
    let observed_provider = observed_provider_from_page_evidence(page_evidence)
        .or_else(|| observed_provider_from_candidates(candidates));
    let observed_browser_app = observed_browser_app_from_page_evidence(page_evidence)
        .or_else(|| observed_browser_app_from_candidates(candidates));
    let observed_query = observed_query_from_page_evidence(page_evidence)
        .or_else(|| observed_query_from_candidates(candidates));
    let capture_age_ms = capture.map(|capture| now_ms.saturating_sub(capture.captured_at));
    let capture_backend = capture.map(|capture| normalize_label(&capture.provider));
    let observation_source = page_evidence
        .iter()
        .find_map(|evidence| evidence.observation_source.clone());
    let mut evidence = Vec::new();
    evidence.push(format!("candidate_count={}", candidates.len()));
    evidence.push(format!("page_evidence_count={}", page_evidence.len()));
    if let Some(provider) = observed_provider.as_deref() {
        evidence.push(format!("observed_content_provider={provider}"));
    }
    if let Some(browser) = observed_browser_app.as_deref() {
        evidence.push(format!("observed_browser_app={browser}"));
    }
    if let Some(query) = observed_query.as_deref() {
        evidence.push(format!("observed_query={query}"));
    }
    if let Some(capture) = capture {
        evidence.push(format!("capture_id={}", capture.capture_id));
        evidence.push(format!("capture_backend={}", capture.provider));
    }
    for page in page_evidence {
        if let Some(backend) = page.capture_backend.as_deref() {
            evidence.push(format!("page_evidence_capture_backend={backend}"));
        }
        if let Some(provider) = page.content_provider_hint.as_deref() {
            evidence.push(format!("page_evidence_content_provider={provider}"));
        }
        if page
            .capture_backend
            .as_deref()
            .is_some_and(is_technical_capture_backend)
        {
            evidence.push("technical_backend_not_used_as_content_provider".into());
        }
    }

    if capture.is_none() {
        return SemanticPageValidationResult {
            status: SemanticPageValidationStatus::Unsupported,
            expected_page_kind,
            observed_page_kind,
            expected_provider,
            observed_provider,
            observed_browser_app,
            expected_query,
            observed_query,
            capture_backend,
            observation_source,
            confidence: 0.0,
            needs_fresh_capture: true,
            mismatch_reason: Some("no screen capture was available for page validation".into()),
            evidence,
            captured_at_ms: None,
            capture_age_ms: None,
        };
    }

    if capture_age_ms.is_some_and(|age| age > MAX_RECENT_SCREEN_AGE_MS) {
        return SemanticPageValidationResult {
            status: SemanticPageValidationStatus::NeedsFreshCapture,
            expected_page_kind,
            observed_page_kind,
            expected_provider,
            observed_provider,
            observed_browser_app,
            expected_query,
            observed_query,
            capture_backend,
            observation_source,
            confidence: 0.30,
            needs_fresh_capture: true,
            mismatch_reason: Some(
                "screen capture is too old for semantic continuation validation".into(),
            ),
            evidence,
            captured_at_ms: capture.map(|capture| capture.captured_at),
            capture_age_ms,
        };
    }

    if let (Some(expected), Some(observed)) =
        (expected_provider.as_deref(), observed_provider.as_deref())
    {
        if !labels_match(expected, observed) {
            let mismatch_reason = format!(
                "visible candidates indicate provider {observed}, but continuation expects {expected}"
            );
            return SemanticPageValidationResult {
                status: SemanticPageValidationStatus::Mismatched,
                expected_page_kind,
                observed_page_kind,
                expected_provider,
                observed_provider,
                observed_browser_app,
                expected_query,
                observed_query,
                capture_backend,
                observation_source,
                confidence: 0.86,
                needs_fresh_capture: false,
                mismatch_reason: Some(mismatch_reason),
                evidence,
                captured_at_ms: capture.map(|capture| capture.captured_at),
                capture_age_ms,
            };
        }
    }

    if let (Some(expected), Some(observed)) = (expected_query.as_deref(), observed_query.as_deref())
    {
        if !query_matches(expected, observed) {
            let mismatch_reason = format!(
                "visible candidates indicate query {observed}, but continuation expects {expected}"
            );
            return SemanticPageValidationResult {
                status: SemanticPageValidationStatus::Mismatched,
                expected_page_kind,
                observed_page_kind,
                expected_provider,
                observed_provider,
                observed_browser_app,
                expected_query,
                observed_query,
                capture_backend,
                observation_source,
                confidence: 0.78,
                needs_fresh_capture: false,
                mismatch_reason: Some(mismatch_reason),
                evidence,
                captured_at_ms: capture.map(|capture| capture.captured_at),
                capture_age_ms,
            };
        }
    }

    let provider_confirmed_or_not_required =
        expected_provider.as_deref().map_or(true, |expected| {
            observed_provider
                .as_deref()
                .is_some_and(|observed| labels_match(expected, observed))
        });
    let status = if observed_page_kind == expected_page_kind
        && observed_page_kind != BrowserPageKind::Unknown
        && provider_confirmed_or_not_required
    {
        SemanticPageValidationStatus::Matched
    } else if !candidates.is_empty()
        && observed_provider.as_deref().map_or(true, |observed| {
            expected_provider
                .as_deref()
                .map_or(true, |expected| labels_match(expected, observed))
        })
    {
        SemanticPageValidationStatus::LikelyMatched
    } else {
        SemanticPageValidationStatus::Unknown
    };
    let confidence = match status {
        SemanticPageValidationStatus::Matched => 0.86,
        SemanticPageValidationStatus::LikelyMatched => 0.66,
        _ => 0.42,
    };

    SemanticPageValidationResult {
        status,
        expected_page_kind,
        observed_page_kind,
        expected_provider,
        observed_provider,
        observed_browser_app,
        expected_query,
        observed_query,
        capture_backend,
        observation_source,
        confidence,
        needs_fresh_capture: false,
        mismatch_reason: None,
        evidence,
        captured_at_ms: capture.map(|capture| capture.captured_at),
        capture_age_ms,
    }
}

pub fn scroll_policy_for_regrounding(
    descriptor: &WorkflowContinuationDescriptor,
    capabilities: &UIPrimitiveCapabilitySet,
    attempt_index: usize,
    max_attempts: usize,
    page_validation: &SemanticPageValidationResult,
    _candidate_count: usize,
    target_selected: bool,
) -> ScrollContinuationDecision {
    let requested_rank = descriptor
        .followup
        .result_reference
        .as_ref()
        .and_then(|reference| reference.rank);
    let scroll_supported = capabilities
        .get(&UIPrimitiveKind::ScrollViewport)
        .is_some_and(|capability| capability.available && capability.enabled);

    if page_validation.status == SemanticPageValidationStatus::Mismatched {
        return ScrollContinuationDecision {
            status: ScrollContinuationStatus::PageMismatch,
            attempt_index,
            max_attempts,
            scroll_supported,
            requested_rank,
            reason: "page validation failed; scrolling would act on the wrong context".into(),
        };
    }
    if target_selected {
        return ScrollContinuationDecision {
            status: ScrollContinuationStatus::NotNeeded,
            attempt_index,
            max_attempts,
            scroll_supported,
            requested_rank,
            reason: "target is already selected; scrolling is not needed".into(),
        };
    }
    if !descriptor.followup.requires_result_list || requested_rank.is_none() {
        return ScrollContinuationDecision {
            status: ScrollContinuationStatus::NotApplicable,
            attempt_index,
            max_attempts,
            scroll_supported,
            requested_rank,
            reason: "follow-up does not reference a scrollable ranked result list".into(),
        };
    }
    if attempt_index + 1 >= max_attempts {
        return ScrollContinuationDecision {
            status: ScrollContinuationStatus::RetryBudgetExhausted,
            attempt_index,
            max_attempts,
            scroll_supported,
            requested_rank,
            reason: "re-grounding retry budget is exhausted".into(),
        };
    }
    if !scroll_supported {
        return ScrollContinuationDecision {
            status: ScrollContinuationStatus::Unsupported,
            attempt_index,
            max_attempts,
            scroll_supported,
            requested_rank,
            reason: "scrolling could help expose the ranked result, but no safe scroll primitive is enabled".into(),
        };
    }

    ScrollContinuationDecision {
        status: ScrollContinuationStatus::SafeToAttempt,
        attempt_index,
        max_attempts,
        scroll_supported,
        requested_rank,
        reason: format!(
            "ranked result {:?} is not safely grounded yet; scroll then re-ground is allowed",
            requested_rank
        ),
    }
}

fn build_continuation_workflow(
    context: &RecentWorkflowContext,
    followup: &FollowupActionResolution,
    policy: &ContinuationPolicyDecision,
    manifest: &CapabilityManifest,
    message: &str,
) -> Option<ScreenWorkflow> {
    let effective_followup = if matches!(followup.action_kind, FollowupActionKind::ContinueWorkflow)
    {
        context.last_followup.as_ref().unwrap_or(followup)
    } else {
        followup
    };

    let mut resolution = ActionResolution::new(
        operation_for_followup(effective_followup),
        domain_for_followup(effective_followup),
        effective_followup.confidence,
        effective_followup.source.as_resolution_source(),
    );
    resolution.provider = effective_followup
        .provider_hint
        .clone()
        .or_else(|| context.provider.clone());
    resolution.requires_screen_context = true;
    resolution.entities = entities_for_followup(context, effective_followup);
    resolution.workflow_steps = workflow_steps_for_followup(context, effective_followup);
    resolution.rationale = Some("recent workflow continuation resolver".into());

    let mut workflow = resolve_screen_workflow(&resolution, manifest, message)?;
    workflow.depends_on_recent_screen_context = true;
    workflow.continuation = Some(WorkflowContinuationDescriptor {
        followup: effective_followup.clone(),
        policy: policy.clone(),
        source_context: context.summary(),
        verifier: verifier_for_followup(effective_followup),
        page_validation: None,
        regrounding: None,
    });
    Some(workflow)
}

fn merge_followup_with_recent_context(
    context: &RecentWorkflowContext,
    mut followup: FollowupActionResolution,
) -> FollowupActionResolution {
    let effective_provider = followup
        .provider_hint
        .clone()
        .or_else(|| context.provider.clone());
    let effective_query = followup
        .query_hint
        .clone()
        .or_else(|| context.query.clone());
    let effective_browser_app = followup
        .browser_hint
        .clone()
        .or_else(|| followup.app_hint.clone())
        .or_else(|| context.app.clone());

    let mut conflicts = Vec::new();
    if let (Some(explicit), Some(recent)) = (
        followup.provider_hint.as_deref(),
        context.provider.as_deref(),
    ) {
        if !labels_match(explicit, recent) {
            conflicts.push(format!("provider:{explicit}!={recent}"));
        }
    }
    if let (Some(explicit), Some(recent)) =
        (followup.query_hint.as_deref(), context.query.as_deref())
    {
        if !query_matches(recent, explicit) {
            conflicts.push(format!("query:{explicit}!={recent}"));
        }
    }

    followup.merge_diagnostic = Some(ContextualMergeDiagnostic {
        effective_provider,
        provider_source: if followup.provider_hint.is_some() {
            Some("followup_explicit".into())
        } else if context.provider.is_some() {
            Some("recent_context".into())
        } else {
            None
        },
        effective_query,
        query_source: if followup.query_hint.is_some() {
            Some("followup_explicit".into())
        } else if context.query.is_some() {
            Some("recent_context".into())
        } else {
            None
        },
        effective_browser_app,
        browser_app_source: if followup.browser_hint.is_some() || followup.app_hint.is_some() {
            Some("followup_explicit".into())
        } else if context.app.is_some() {
            Some("recent_context".into())
        } else {
            None
        },
        reused_recent_context: followup.provider_hint.is_none() || followup.query_hint.is_none(),
        conflicts,
    });
    followup
}

fn continuation_policy(
    context: &RecentWorkflowContext,
    followup: &FollowupActionResolution,
    manifest: &CapabilityManifest,
    now_ms: u64,
) -> ContinuationPolicyDecision {
    let context_age_ms = Some(now_ms.saturating_sub(context.updated_at_ms));
    let screen_age_ms = manifest.screen.recent_capture_age_ms;
    let required_rank = followup
        .result_reference
        .as_ref()
        .and_then(|reference| reference.rank);
    let provider = followup
        .provider_hint
        .clone()
        .or_else(|| context.provider.clone());
    let query = followup
        .query_hint
        .clone()
        .or_else(|| context.query.clone());

    let policy = |status, executable, reason: String| ContinuationPolicyDecision {
        status,
        executable,
        reason,
        context_age_ms,
        screen_age_ms,
        required_rank,
        provider: provider.clone(),
        query: query.clone(),
    };

    if now_ms > context.expires_at_ms {
        return policy(
            ContinuationPolicyStatus::StaleWorkflowContext,
            false,
            "recent workflow context exceeded its continuation freshness window".into(),
        );
    }
    if !context.continuation_allowed {
        return policy(
            ContinuationPolicyStatus::NoRecentWorkflowContext,
            false,
            "recent context is not marked continuation-eligible".into(),
        );
    }
    if matches!(
        followup.action_kind,
        FollowupActionKind::NavigateBackThenOpenResult
    ) {
        return policy(
            ContinuationPolicyStatus::UnsupportedMultiStepRegrounding,
            false,
            "navigate-back-plus-click needs a post-navigation screen recapture before target selection".into(),
        );
    }
    if matches!(followup.action_kind, FollowupActionKind::ContinueWorkflow)
        && (!context.resumable || context.last_followup.is_none())
    {
        return policy(
            ContinuationPolicyStatus::NoResumableWorkflow,
            false,
            "the recent workflow is not stopped at a resumable follow-up step".into(),
        );
    }
    if let Some(provider_hint) = followup.provider_hint.as_deref() {
        if context
            .provider
            .as_deref()
            .is_some_and(|provider| !labels_match(provider, provider_hint))
        {
            return policy(
                ContinuationPolicyStatus::ProviderMismatch,
                false,
                format!(
                    "follow-up provider {provider_hint} does not match recent provider {}",
                    context.provider.as_deref().unwrap_or("unknown")
                ),
            );
        }
    }
    if let Some(query_hint) = followup.query_hint.as_deref() {
        if context
            .query
            .as_deref()
            .is_some_and(|query| !query_matches(query, query_hint))
        {
            return policy(
                ContinuationPolicyStatus::QueryMismatch,
                false,
                format!(
                    "follow-up query {query_hint} does not match recent query {}",
                    context.query.as_deref().unwrap_or("unknown")
                ),
            );
        }
    }

    if followup.requires_result_list {
        let Some(result_list) = context.result_list.as_ref() else {
            return policy(
                ContinuationPolicyStatus::ResultListUnavailable,
                false,
                "the recent workflow did not record a reusable result-list expectation".into(),
            );
        };
        if now_ms > result_list.expires_at_ms {
            return policy(
                ContinuationPolicyStatus::ResultListStale,
                false,
                "the recorded result-list context exceeded its freshness window".into(),
            );
        }
        if !screen_grounding_possible(manifest) {
            return policy(
                ContinuationPolicyStatus::ScreenContextUnavailable,
                false,
                "screen observation or recent screen analysis is required for candidate-aware result selection".into(),
            );
        }
        if manifest
            .screen
            .recent_capture_age_ms
            .is_some_and(|age| age > MAX_RECENT_SCREEN_AGE_MS)
            && !manifest.screen.observation_enabled
        {
            return policy(
                ContinuationPolicyStatus::ScreenContextUnavailable,
                false,
                "the available screen capture is stale and fresh observation is disabled".into(),
            );
        }
    }

    if matches!(
        followup.action_kind,
        FollowupActionKind::ClickReferencedTarget
    ) && context.recent_selected_target.is_none()
        && context.recent_focused_target.is_none()
    {
        return policy(
            ContinuationPolicyStatus::TargetContextUnavailable,
            false,
            "the follow-up refers to 'that', but no recent selected or focused target is recorded"
                .into(),
        );
    }

    if matches!(followup.action_kind, FollowupActionKind::TypeText)
        && context.recent_focused_target.is_none()
        && !screen_grounding_possible(manifest)
    {
        return policy(
            ContinuationPolicyStatus::TargetContextUnavailable,
            false,
            "typing follow-up has no recent focused input and cannot re-ground a visible input target".into(),
        );
    }

    policy(
        ContinuationPolicyStatus::SafeToAttempt,
        true,
        "recent context is fresh enough and the next step can be delegated to target grounding"
            .into(),
    )
}

fn interpret_contextual_followup(
    message: &str,
    model_params: Option<&Value>,
    model_confidence: Option<f32>,
) -> Option<ContextualFollowupInterpretation> {
    let lower = message.to_lowercase();
    let trimmed = lower.trim();
    if trimmed.is_empty() {
        return None;
    }

    let mut roles =
        entity_role_assignments(message, FollowupResolutionSource::RustFollowupResolver);
    let browser_hint = browser_app_hint(&lower);
    let app_hint = browser_hint.clone();
    let provider_hint = provider_hint(&lower);
    let page_context_hint = page_context_hint(&lower, browser_hint.as_deref());
    let model = model_params.and_then(|params| {
        contextual_interpretation_from_model_params(
            message,
            params,
            model_confidence.unwrap_or(0.62),
        )
    });

    if let Some(mut model) = model {
        model.browser_hint = model.browser_hint.or(browser_hint);
        model.app_hint = model.app_hint.or(app_hint);
        model.provider_hint = normalize_provider_hint(model.provider_hint).or(provider_hint);
        model.page_context_hint = model.page_context_hint.or(page_context_hint);
        if model.entity_roles.is_empty() {
            roles.extend(entity_role_assignments(
                message,
                FollowupResolutionSource::ModelAssistedClassifier,
            ));
            model.entity_roles = roles;
        }
        return Some(model);
    }

    if matches!(
        trimmed,
        "continua" | "continue" | "prosegui" | "vai avanti" | "go on"
    ) {
        return Some(ContextualFollowupInterpretation {
            utterance: message.to_string(),
            continuation_kind: ContextualFollowupKind::ResumeRecentWorkflow,
            browser_hint,
            app_hint,
            provider_hint,
            page_context_hint: Some(PageContextHint::CurrentPage),
            query_hint: None,
            text_value: None,
            result_reference: None,
            ordinal_reference: None,
            requires_recent_focus_target: false,
            requires_recent_results_context: false,
            requires_resumable_workflow: true,
            confidence: 0.74,
            source: FollowupResolutionSource::RustFollowupResolver,
            rationale: "short continuation command".into(),
            entity_roles: roles,
        });
    }

    if looks_like_typing_followup(&lower) {
        let text = extract_typing_value(message);
        return Some(ContextualFollowupInterpretation {
            utterance: message.to_string(),
            continuation_kind: ContextualFollowupKind::TypeIntoRecentFocus,
            browser_hint,
            app_hint,
            provider_hint,
            page_context_hint: Some(PageContextHint::RecentFocus),
            query_hint: None,
            text_value: text,
            result_reference: None,
            ordinal_reference: None,
            requires_recent_focus_target: true,
            requires_recent_results_context: false,
            requires_resumable_workflow: false,
            confidence: 0.72,
            source: FollowupResolutionSource::RustFollowupResolver,
            rationale: "typing follow-up references the current or recent input focus".into(),
            entity_roles: roles,
        });
    }

    let rank = ordinal_rank(&lower);
    let navigation = looks_like_back_navigation(&lower);
    let click_or_open = looks_like_click_or_open(&lower);
    let result_reference = rank.map(|rank| ResultListReference {
        rank: Some(rank),
        ordinal_label: ordinal_label(rank).into(),
        item_kind: result_item_kind(&lower),
    });

    if navigation && click_or_open && result_reference.is_some() {
        return Some(ContextualFollowupInterpretation {
            utterance: message.to_string(),
            continuation_kind: ContextualFollowupKind::NavigateBackThenOpenResult,
            browser_hint,
            app_hint,
            provider_hint,
            page_context_hint: Some(PageContextHint::PreviousPage),
            query_hint: extract_followup_query_hint(message),
            text_value: None,
            result_reference,
            ordinal_reference: rank,
            requires_recent_focus_target: false,
            requires_recent_results_context: true,
            requires_resumable_workflow: false,
            confidence: 0.68,
            source: FollowupResolutionSource::RustFollowupResolver,
            rationale: "multi-step navigation plus ordinal result follow-up".into(),
            entity_roles: roles,
        });
    }

    if navigation {
        return Some(ContextualFollowupInterpretation {
            utterance: message.to_string(),
            continuation_kind: ContextualFollowupKind::NavigateBack,
            browser_hint,
            app_hint,
            provider_hint,
            page_context_hint: Some(PageContextHint::PreviousPage),
            query_hint: None,
            text_value: None,
            result_reference: None,
            ordinal_reference: None,
            requires_recent_focus_target: false,
            requires_recent_results_context: false,
            requires_resumable_workflow: false,
            confidence: 0.70,
            source: FollowupResolutionSource::RustFollowupResolver,
            rationale: "browser or screen back-navigation follow-up".into(),
            entity_roles: roles,
        });
    }

    if click_or_open && (result_reference.is_some() || mentions_result_list_item(&lower)) {
        let result_reference = Some(result_reference.unwrap_or(ResultListReference {
            rank: Some(1),
            ordinal_label: "first".into(),
            item_kind: result_item_kind(&lower),
        }));
        return Some(ContextualFollowupInterpretation {
            utterance: message.to_string(),
            continuation_kind: if lower.contains("clicca")
                || lower.contains("click")
                || lower.contains("premi")
            {
                ContextualFollowupKind::ClickResult
            } else {
                ContextualFollowupKind::OpenResult
            },
            browser_hint,
            app_hint,
            provider_hint,
            page_context_hint: page_context_hint.or(Some(PageContextHint::RecentResultsPage)),
            query_hint: extract_followup_query_hint(message),
            text_value: None,
            ordinal_reference: rank.or(Some(1)),
            result_reference,
            requires_recent_focus_target: false,
            requires_recent_results_context: true,
            requires_resumable_workflow: false,
            confidence: 0.76,
            source: FollowupResolutionSource::RustFollowupResolver,
            rationale: "ordinal result-list follow-up".into(),
            entity_roles: roles,
        });
    }

    if click_or_open && refers_to_previous_target(&lower) {
        return Some(ContextualFollowupInterpretation {
            utterance: message.to_string(),
            continuation_kind: ContextualFollowupKind::ClickReferencedTarget,
            browser_hint,
            app_hint,
            provider_hint,
            page_context_hint,
            query_hint: None,
            text_value: None,
            result_reference: None,
            ordinal_reference: None,
            requires_recent_focus_target: false,
            requires_recent_results_context: false,
            requires_resumable_workflow: false,
            confidence: 0.62,
            source: FollowupResolutionSource::RustFollowupResolver,
            rationale: "deictic follow-up refers to a recent selected or focused target".into(),
            entity_roles: roles,
        });
    }

    None
}

fn followup_from_interpretation(
    interpretation: ContextualFollowupInterpretation,
) -> FollowupActionResolution {
    let action_kind = match interpretation.continuation_kind {
        ContextualFollowupKind::OpenResult => FollowupActionKind::OpenResult,
        ContextualFollowupKind::ClickResult => FollowupActionKind::ClickResult,
        ContextualFollowupKind::ClickReferencedTarget => FollowupActionKind::ClickReferencedTarget,
        ContextualFollowupKind::TypeIntoRecentFocus => FollowupActionKind::TypeText,
        ContextualFollowupKind::NavigateBack => FollowupActionKind::NavigateBack,
        ContextualFollowupKind::NavigateBackThenOpenResult => {
            FollowupActionKind::NavigateBackThenOpenResult
        }
        ContextualFollowupKind::ResumeRecentWorkflow => FollowupActionKind::ContinueWorkflow,
        ContextualFollowupKind::Unknown => FollowupActionKind::ClickReferencedTarget,
    };

    FollowupActionResolution {
        action_kind,
        result_reference: interpretation.result_reference.clone(),
        text_value: interpretation.text_value.clone(),
        provider_hint: interpretation.provider_hint.clone(),
        query_hint: interpretation.query_hint.clone(),
        browser_hint: interpretation.browser_hint.clone(),
        app_hint: interpretation.app_hint.clone(),
        page_context_hint: interpretation.page_context_hint.clone(),
        requires_result_list: interpretation.requires_recent_results_context,
        requires_recent_focus_target: interpretation.requires_recent_focus_target,
        confidence: interpretation.confidence,
        source: interpretation.source.clone(),
        rationale: interpretation.rationale.clone(),
        interpretation: Some(interpretation),
        merge_diagnostic: None,
    }
}

fn entities_for_followup(
    context: &RecentWorkflowContext,
    followup: &FollowupActionResolution,
) -> Value {
    let provider = followup
        .provider_hint
        .clone()
        .or_else(|| context.provider.clone())
        .unwrap_or_else(|| "unknown".into());
    let query = followup
        .query_hint
        .clone()
        .or_else(|| context.query.clone());
    let browser_app = followup
        .browser_hint
        .clone()
        .or_else(|| followup.app_hint.clone())
        .or_else(|| context.app.clone());
    let mut entities = json!({
        "provider": provider,
        "query_candidate": query,
        "browser_app": browser_app,
        "continuation": {
            "context_id": context.context_id.clone(),
            "kind": context.kind.clone(),
            "page_kind": context.page_kind.clone(),
            "result_list_available": context.result_list.is_some(),
            "contextual_interpretation": followup.interpretation.clone(),
            "contextual_merge": followup.merge_diagnostic.clone(),
        },
    });

    match followup.action_kind {
        FollowupActionKind::OpenResult | FollowupActionKind::ClickResult => {
            if let Some(reference) = followup.result_reference.as_ref() {
                entities["selection_strategy"] = json!("ranked_result");
                entities["rank"] = json!(reference.rank);
                entities["result_kind"] = json!(reference.item_kind);
            }
        }
        FollowupActionKind::ClickReferencedTarget => {
            if let Some(candidate) = context
                .recent_selected_target
                .as_ref()
                .or(context.recent_focused_target.as_ref())
            {
                entities["target_candidate"] = candidate.execution_payload();
            }
        }
        FollowupActionKind::TypeText => {
            entities["query_candidate"] = json!(followup.text_value);
            if context.recent_focused_target.is_some() {
                entities["requires_recent_focus_target"] = json!(true);
            }
        }
        FollowupActionKind::NavigateBack
        | FollowupActionKind::NavigateBackThenOpenResult
        | FollowupActionKind::ContinueWorkflow => {}
    }

    entities
}

fn workflow_steps_for_followup(
    context: &RecentWorkflowContext,
    followup: &FollowupActionResolution,
) -> Vec<String> {
    match followup.action_kind {
        FollowupActionKind::OpenResult | FollowupActionKind::ClickResult => {
            vec!["open_ranked_result".into()]
        }
        FollowupActionKind::ClickReferencedTarget => vec!["click_visible_element".into()],
        FollowupActionKind::TypeText => {
            if context.recent_focused_target.is_some() {
                vec!["enter_text".into()]
            } else {
                vec!["focus_search_input".into(), "enter_text".into()]
            }
        }
        FollowupActionKind::NavigateBack => vec!["navigate_back".into()],
        FollowupActionKind::NavigateBackThenOpenResult => {
            vec!["navigate_back".into(), "open_ranked_result".into()]
        }
        FollowupActionKind::ContinueWorkflow => Vec::new(),
    }
}

fn operation_for_followup(followup: &FollowupActionResolution) -> ActionOperation {
    match followup.action_kind {
        FollowupActionKind::NavigateBack | FollowupActionKind::NavigateBackThenOpenResult => {
            ActionOperation::ScreenGuidedNavigationWorkflow
        }
        _ => ActionOperation::ScreenGuidedFollowupAction,
    }
}

fn domain_for_followup(followup: &FollowupActionResolution) -> ActionDomain {
    match followup.action_kind {
        FollowupActionKind::OpenResult | FollowupActionKind::ClickResult => {
            ActionDomain::BrowserScreenInteraction
        }
        FollowupActionKind::NavigateBack | FollowupActionKind::NavigateBackThenOpenResult => {
            ActionDomain::Screen
        }
        _ => ActionDomain::Screen,
    }
}

fn verifier_for_followup(followup: &FollowupActionResolution) -> ContinuationVerifierExpectation {
    match followup.action_kind {
        FollowupActionKind::OpenResult | FollowupActionKind::ClickResult => {
            ContinuationVerifierExpectation {
                verifier_kind: ContinuationVerifierKind::ResultNavigationExpected,
                expected_state_change: "ranked result click executed; result page or navigation should be confirmed by the next screen capture".into(),
                requires_post_step_screen_check: true,
            }
        }
        FollowupActionKind::TypeText => ContinuationVerifierExpectation {
            verifier_kind: ContinuationVerifierKind::FocusPreservationExpected,
            expected_state_change:
                "text entry primitive executed while input focus was reused or grounded".into(),
            requires_post_step_screen_check: false,
        },
        FollowupActionKind::NavigateBack => ContinuationVerifierExpectation {
            verifier_kind: ContinuationVerifierKind::BrowserBackExpected,
            expected_state_change:
                "browser back primitive executed; previous page should be confirmed by screen state"
                    .into(),
            requires_post_step_screen_check: true,
        },
        FollowupActionKind::ClickReferencedTarget => ContinuationVerifierExpectation {
            verifier_kind: ContinuationVerifierKind::RequiresNextScreenCheck,
            expected_state_change:
                "referenced target click executed; resulting screen state requires confirmation"
                    .into(),
            requires_post_step_screen_check: true,
        },
        FollowupActionKind::NavigateBackThenOpenResult | FollowupActionKind::ContinueWorkflow => {
            ContinuationVerifierExpectation {
                verifier_kind: ContinuationVerifierKind::PrimitiveOnly,
                expected_state_change: "continuation step prepared".into(),
                requires_post_step_screen_check: true,
            }
        }
    }
}

fn workflow_provider(workflow: &ScreenWorkflow) -> Option<String> {
    workflow.steps.iter().find_map(|step| {
        value_str(&step.target, "provider")
            .or_else(|| value_str(&step.selection, "provider"))
            .map(ToOwned::to_owned)
    })
}

fn workflow_query(workflow: &ScreenWorkflow) -> Option<String> {
    workflow
        .steps
        .iter()
        .find_map(|step| step.value.clone())
        .or_else(|| {
            workflow.steps.iter().find_map(|step| {
                value_str(&step.target, "query_candidate")
                    .or_else(|| value_str(&step.selection, "query_candidate"))
                    .map(ToOwned::to_owned)
            })
        })
}

fn workflow_mentions_result_list(workflow: &ScreenWorkflow) -> bool {
    workflow.steps.iter().any(|step| {
        matches!(
            step.step_kind,
            WorkflowStepKind::SubmitSearch | WorkflowStepKind::OpenRankedResult
        )
    })
}

fn screen_grounding_possible(manifest: &CapabilityManifest) -> bool {
    manifest.screen.analysis_available
        && (manifest.screen.observation_enabled || manifest.screen.recent_capture_available)
}

fn looks_like_typing_followup(lower: &str) -> bool {
    [
        "ora scrivi",
        "adesso scrivi",
        "scrivi qui",
        "scrivi nel campo",
        "now type",
        "type here",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn extract_typing_value(original: &str) -> Option<String> {
    [
        "ora scrivi",
        "adesso scrivi",
        "scrivi qui",
        "scrivi nel campo",
        "now type",
        "type here",
    ]
    .iter()
    .find_map(|marker| extract_after_marker(original, marker))
    .map(trim_query_edges)
    .filter(|value| !value.is_empty())
}

fn looks_like_back_navigation(lower: &str) -> bool {
    lower.contains("torna indietro")
        || lower.contains("torna alla pagina")
        || lower.contains("vai indietro")
        || lower.contains("go back")
        || lower.contains("back")
}

fn looks_like_click_or_open(lower: &str) -> bool {
    lower.contains("clicca")
        || lower.contains("click")
        || lower.contains("premi")
        || lower.contains("apri")
        || lower.contains("aprimi")
        || lower.contains("open")
}

fn mentions_result_list_item(lower: &str) -> bool {
    lower.contains("risultato")
        || lower.contains("result")
        || lower.contains("video")
        || lower.contains("link")
}

fn refers_to_previous_target(lower: &str) -> bool {
    lower.contains("quello") || lower.contains("quella") || lower.contains("that")
}

fn ordinal_rank(lower: &str) -> Option<u32> {
    if lower.contains("primo")
        || lower.contains("prima")
        || lower.contains("first")
        || lower.contains("top result")
        || lower.contains("in alto")
    {
        return Some(1);
    }
    if lower.contains("secondo") || lower.contains("seconda") || lower.contains("second") {
        return Some(2);
    }
    if lower.contains("terzo") || lower.contains("terza") || lower.contains("third") {
        return Some(3);
    }
    None
}

fn ordinal_label(rank: u32) -> &'static str {
    match rank {
        1 => "first",
        2 => "second",
        3 => "third",
        _ => "ranked",
    }
}

fn result_item_kind(lower: &str) -> ResultListItemKind {
    if lower.contains("video") {
        ResultListItemKind::Video
    } else if lower.contains("link") {
        ResultListItemKind::Link
    } else {
        ResultListItemKind::Result
    }
}

fn provider_result_kind(provider: Option<&str>) -> ResultListItemKind {
    if provider.is_some_and(|provider| labels_match(provider, "youtube")) {
        ResultListItemKind::Video
    } else {
        ResultListItemKind::Result
    }
}

fn expected_page_kind_for_followup(
    followup: &FollowupActionResolution,
    context: &RecentWorkflowContextSummary,
) -> BrowserPageKind {
    match followup.action_kind {
        FollowupActionKind::OpenResult
        | FollowupActionKind::ClickResult
        | FollowupActionKind::NavigateBackThenOpenResult => BrowserPageKind::SearchResults,
        FollowupActionKind::NavigateBack => BrowserPageKind::Unknown,
        _ => context.page_kind.clone(),
    }
}

fn observed_page_kind_from_candidates(candidates: &[UITargetCandidate]) -> BrowserPageKind {
    if let Some(kind) = candidates.iter().find_map(|candidate| {
        candidate
            .page_kind_hint
            .as_deref()
            .and_then(browser_page_kind_from_hint)
    }) {
        return kind;
    }
    if candidates
        .iter()
        .any(|candidate| candidate.role == UITargetRole::RankedResult)
    {
        BrowserPageKind::SearchResults
    } else {
        BrowserPageKind::Unknown
    }
}

fn observed_page_kind_from_evidence(
    page_evidence: &[PageSemanticEvidence],
) -> Option<BrowserPageKind> {
    page_evidence
        .iter()
        .filter(|evidence| evidence.confidence >= 0.35)
        .find_map(|evidence| {
            evidence
                .page_kind_hint
                .as_deref()
                .and_then(browser_page_kind_from_hint)
        })
        .or_else(|| {
            page_evidence
                .iter()
                .any(|evidence| {
                    evidence.result_list_visible == Some(true) && evidence.confidence >= 0.35
                })
                .then_some(BrowserPageKind::SearchResults)
        })
}

fn observed_provider_from_candidates(candidates: &[UITargetCandidate]) -> Option<String> {
    most_common_candidate_field(
        candidates
            .iter()
            .filter_map(candidate_content_provider_hint),
    )
    .or_else(|| {
        candidates
            .iter()
            .flat_map(candidate_text_signals)
            .find_map(|text| infer_content_provider_from_text(&text))
    })
}

fn observed_provider_from_page_evidence(page_evidence: &[PageSemanticEvidence]) -> Option<String> {
    most_common_candidate_field(
        page_evidence
            .iter()
            .filter(|evidence| evidence.confidence >= 0.35)
            .filter_map(|evidence| {
                evidence
                    .content_provider_hint
                    .as_deref()
                    .and_then(normalize_content_provider_hint)
            }),
    )
}

fn observed_browser_app_from_candidates(candidates: &[UITargetCandidate]) -> Option<String> {
    most_common_candidate_field(candidates.iter().filter_map(candidate_browser_app_hint))
}

fn observed_browser_app_from_page_evidence(
    page_evidence: &[PageSemanticEvidence],
) -> Option<String> {
    most_common_candidate_field(page_evidence.iter().filter_map(|evidence| {
        (evidence.confidence >= 0.35)
            .then(|| {
                evidence
                    .browser_app_hint
                    .as_deref()
                    .map(|value| normalize_label(value))
            })
            .flatten()
    }))
}

fn observed_query_from_candidates(candidates: &[UITargetCandidate]) -> Option<String> {
    candidates
        .iter()
        .filter_map(|candidate| candidate.label.as_deref())
        .find_map(extract_query_from_candidate_label)
}

fn observed_query_from_page_evidence(page_evidence: &[PageSemanticEvidence]) -> Option<String> {
    page_evidence
        .iter()
        .filter(|evidence| evidence.confidence >= 0.35)
        .filter_map(|evidence| evidence.query_hint.as_deref())
        .find_map(normalize_followup_query_hint)
}

fn most_common_candidate_field(values: impl Iterator<Item = String>) -> Option<String> {
    let mut counts: Vec<(String, usize)> = Vec::new();
    for value in values.filter(|value| !value.is_empty()) {
        if let Some((_, count)) = counts.iter_mut().find(|(existing, _)| existing == &value) {
            *count += 1;
        } else {
            counts.push((value, 1));
        }
    }
    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(value, _)| value)
}

fn browser_page_kind_from_hint(value: &str) -> Option<BrowserPageKind> {
    match normalize_label(value).as_str() {
        "search_results" | "results" | "result_list" | "youtube_results" => {
            Some(BrowserPageKind::SearchResults)
        }
        "watch_page" | "video_page" | "youtube_watch" | "detail_page" => {
            Some(BrowserPageKind::WatchPage)
        }
        "web_page" | "page" => Some(BrowserPageKind::WebPage),
        _ => None,
    }
}

fn candidate_text_signals(candidate: &UITargetCandidate) -> Vec<String> {
    [
        candidate.label.as_deref(),
        Some(candidate.candidate_id.as_str()),
        Some(candidate.rationale.as_str()),
    ]
    .into_iter()
    .flatten()
    .map(ToOwned::to_owned)
    .collect()
}

fn infer_content_provider_from_text(value: &str) -> Option<String> {
    let normalized = normalize_label(value);
    if normalized.contains("youtube") || normalized.contains("youtu_be") {
        Some("youtube".into())
    } else if normalized.contains("google") {
        Some("google".into())
    } else if normalized.contains("github") {
        Some("github".into())
    } else if normalized.contains("amazon") {
        Some("amazon".into())
    } else {
        None
    }
}

fn extract_query_from_candidate_label(label: &str) -> Option<String> {
    for marker in ["search_query=", "q=", "results for", "risultati per"] {
        if let Some(value) = extract_after_marker(label, marker) {
            return normalize_followup_query_hint(value);
        }
    }
    None
}

fn provider_hint(lower: &str) -> Option<String> {
    let provider_scope = without_browser_app_phrases(lower);
    if provider_scope.contains("youtube") || provider_scope.contains("you tube") {
        Some("youtube".into())
    } else if provider_scope.contains("google") {
        Some("google".into())
    } else {
        None
    }
}

fn normalize_provider_hint(value: Option<String>) -> Option<String> {
    let value = value?;
    let normalized = normalize_label(&value);
    match normalized.as_str() {
        "youtube" | "you_tube" => Some("youtube".into()),
        "google" | "web" => Some(normalized),
        "chrome" | "google_chrome" | "firefox" | "safari" | "edge" | "microsoft_edge" => None,
        _ => Some(value.trim().to_ascii_lowercase()),
    }
}

fn browser_app_hint(lower: &str) -> Option<String> {
    if contains_phrase(lower, "google chrome") || contains_token(lower, "chrome") {
        Some("chrome".into())
    } else if contains_token(lower, "firefox") {
        Some("firefox".into())
    } else if contains_token(lower, "safari") {
        Some("safari".into())
    } else if contains_phrase(lower, "microsoft edge")
        || (contains_token(lower, "edge") && has_browser_app_context(lower))
    {
        Some("edge".into())
    } else {
        None
    }
}

fn page_context_hint(lower: &str, browser_hint: Option<&str>) -> Option<PageContextHint> {
    if looks_like_back_navigation(lower) {
        Some(PageContextHint::PreviousPage)
    } else if lower.contains("scheda") || lower.contains("tab") {
        if browser_hint.is_some() {
            Some(PageContextHint::MatchingBrowserTab)
        } else {
            Some(PageContextHint::CurrentPage)
        }
    } else if mentions_result_list_item(lower) || ordinal_rank(lower).is_some() {
        Some(PageContextHint::RecentResultsPage)
    } else if looks_like_typing_followup(lower) {
        Some(PageContextHint::RecentFocus)
    } else {
        None
    }
}

fn without_browser_app_phrases(value: &str) -> String {
    let mut stripped = [
        "google chrome",
        "chrome",
        "mozilla firefox",
        "firefox",
        "microsoft edge",
        "safari",
    ]
    .iter()
    .fold(value.to_string(), |acc, marker| acc.replace(marker, " "));
    if browser_app_hint(value).as_deref() == Some("edge") {
        stripped = stripped.replace("edge", " ");
    }
    stripped
}

fn has_browser_app_context(lower: &str) -> bool {
    [
        "microsoft edge",
        "browser edge",
        "edge browser",
        "app edge",
        "apri edge",
        "aprimi edge",
        "open edge",
        "in edge",
        "su edge",
        "scheda edge",
        "tab edge",
        "finestra edge",
        "window edge",
    ]
    .iter()
    .any(|marker| contains_phrase(lower, marker))
}

fn contains_phrase(value: &str, phrase: &str) -> bool {
    value
        .split(|ch: char| !ch.is_alphanumeric())
        .collect::<Vec<_>>()
        .windows(phrase.split_whitespace().count())
        .any(|window| window.join(" ") == phrase)
}

fn contains_token(value: &str, token: &str) -> bool {
    value
        .split(|ch: char| !ch.is_alphanumeric())
        .any(|part| part == token)
}

fn entity_role_assignments(
    message: &str,
    source: FollowupResolutionSource,
) -> Vec<EntityRoleAssignment> {
    let lower = message.to_lowercase();
    let mut roles = Vec::new();
    if let Some(browser) = browser_app_hint(&lower) {
        roles.push(EntityRoleAssignment {
            text: if lower.contains("google chrome") {
                "google chrome".into()
            } else {
                browser.clone()
            },
            role: EntityRoleKind::BrowserApp,
            normalized_value: Some(browser),
            confidence: 0.94,
            source: source.clone(),
        });
    }
    if let Some(provider) = provider_hint(&lower) {
        roles.push(EntityRoleAssignment {
            text: provider.clone(),
            role: EntityRoleKind::ContentProvider,
            normalized_value: Some(provider),
            confidence: 0.86,
            source: source.clone(),
        });
    }
    if let Some(rank) = ordinal_rank(&lower) {
        roles.push(EntityRoleAssignment {
            text: ordinal_label(rank).into(),
            role: EntityRoleKind::ResultReference,
            normalized_value: Some(rank.to_string()),
            confidence: 0.90,
            source: source.clone(),
        });
    }
    if let Some(query) = extract_followup_query_hint(message) {
        roles.push(EntityRoleAssignment {
            text: query.clone(),
            role: EntityRoleKind::QueryText,
            normalized_value: Some(query),
            confidence: 0.70,
            source: source.clone(),
        });
    }
    if let Some(page) = page_context_hint(&lower, browser_app_hint(&lower).as_deref()) {
        roles.push(EntityRoleAssignment {
            text: match page {
                PageContextHint::MatchingBrowserTab => "matching browser tab",
                PageContextHint::RecentResultsPage => "recent results page",
                PageContextHint::PreviousPage => "previous page",
                PageContextHint::RecentFocus => "recent focus",
                PageContextHint::CurrentPage => "current page",
                PageContextHint::Unknown => "unknown page context",
            }
            .into(),
            role: EntityRoleKind::PageContext,
            normalized_value: Some(format!("{page:?}").to_ascii_lowercase()),
            confidence: 0.74,
            source,
        });
    }
    roles
}

fn contextual_interpretation_from_model_params(
    message: &str,
    params: &Value,
    confidence: f32,
) -> Option<ContextualFollowupInterpretation> {
    let operation = value_str(params, "operation").or_else(|| {
        params
            .get("entities")
            .and_then(|entities| value_str(entities, "operation"))
    });
    let entities = params.get("entities").unwrap_or(params);
    let operation_is_followup = operation.is_some_and(|operation| {
        matches!(
            normalize_label(operation).as_str(),
            "screen_guided_followup_action" | "screen_guided_navigation_workflow"
        )
    });
    if !operation_is_followup && !looks_like_click_or_open(&message.to_lowercase()) {
        return None;
    }

    let lower = message.to_lowercase();
    let browser_hint = value_str(params, "browser_app")
        .or_else(|| value_str(entities, "browser_app"))
        .or_else(|| value_str(entities, "app"))
        .map(ToOwned::to_owned)
        .or_else(|| browser_app_hint(&lower));
    let provider_hint = normalize_provider_hint(
        value_str(params, "provider")
            .or_else(|| value_str(entities, "provider"))
            .map(ToOwned::to_owned),
    )
    .or_else(|| provider_hint(&lower));
    let rank = value_u32(entities, "rank").or_else(|| ordinal_rank(&lower));
    let result_reference = rank.map(|rank| ResultListReference {
        rank: Some(rank),
        ordinal_label: ordinal_label(rank).into(),
        item_kind: value_str(entities, "result_kind")
            .map(result_item_kind_from_value)
            .unwrap_or_else(|| result_item_kind(&lower)),
    });
    let text_value = value_str(entities, "text_value")
        .or_else(|| value_str(entities, "query_candidate"))
        .map(ToOwned::to_owned)
        .or_else(|| {
            looks_like_typing_followup(&lower)
                .then(|| extract_typing_value(message))
                .flatten()
        });
    let continuation_kind = if looks_like_typing_followup(&lower) || text_value.is_some() {
        ContextualFollowupKind::TypeIntoRecentFocus
    } else if looks_like_back_navigation(&lower) && result_reference.is_some() {
        ContextualFollowupKind::NavigateBackThenOpenResult
    } else if looks_like_back_navigation(&lower) {
        ContextualFollowupKind::NavigateBack
    } else if result_reference.is_some() || mentions_result_list_item(&lower) {
        if lower.contains("clicca") || lower.contains("click") || lower.contains("premi") {
            ContextualFollowupKind::ClickResult
        } else {
            ContextualFollowupKind::OpenResult
        }
    } else if refers_to_previous_target(&lower) {
        ContextualFollowupKind::ClickReferencedTarget
    } else {
        ContextualFollowupKind::Unknown
    };
    if matches!(continuation_kind, ContextualFollowupKind::Unknown) {
        return None;
    }
    let requires_recent_results_context = matches!(
        continuation_kind,
        ContextualFollowupKind::OpenResult
            | ContextualFollowupKind::ClickResult
            | ContextualFollowupKind::NavigateBackThenOpenResult
    );
    let requires_resumable_workflow = matches!(
        continuation_kind,
        ContextualFollowupKind::ResumeRecentWorkflow
    );

    Some(ContextualFollowupInterpretation {
        utterance: message.to_string(),
        continuation_kind,
        browser_hint: browser_hint.clone(),
        app_hint: browser_hint,
        provider_hint,
        page_context_hint: page_context_hint(&lower, browser_app_hint(&lower).as_deref()),
        query_hint: value_str(entities, "query_hint")
            .or_else(|| value_str(entities, "query"))
            .and_then(|value| normalize_followup_query_hint(value))
            .or_else(|| extract_followup_query_hint(message)),
        text_value,
        result_reference,
        ordinal_reference: rank,
        requires_recent_focus_target: value_bool(entities, "requires_recent_focus_target")
            .unwrap_or_else(|| looks_like_typing_followup(&lower)),
        requires_recent_results_context,
        requires_resumable_workflow,
        confidence: confidence.clamp(0.0, 1.0),
        source: FollowupResolutionSource::ModelAssistedClassifier,
        rationale: "model-assisted contextual follow-up interpretation normalized by Rust".into(),
        entity_roles: entity_role_assignments(
            message,
            FollowupResolutionSource::ModelAssistedClassifier,
        ),
    })
}

fn extract_followup_query_hint(original: &str) -> Option<String> {
    for marker in [
        " video di ",
        " risultato di ",
        " risultati di ",
        " result for ",
        " video by ",
        " di ",
        " for ",
    ] {
        if let Some(value) = extract_after_marker(original, marker) {
            let clean = trim_query_edges(value);
            if let Some(clean) = normalize_followup_query_hint(&clean) {
                return Some(clean);
            }
        }
    }
    None
}

fn normalize_followup_query_hint(value: &str) -> Option<String> {
    let clean = trim_query_edges(value);
    if clean.is_empty() {
        return None;
    }
    let normalized = normalize_label(&clean);
    if matches!(
        normalized.as_str(),
        "youtube"
            | "you_tube"
            | "google"
            | "chrome"
            | "google_chrome"
            | "browser"
            | "scheda"
            | "tab"
    ) {
        return None;
    }
    if browser_app_hint(&clean.to_lowercase()).is_some()
        && normalize_query(&without_browser_app_phrases(&clean.to_lowercase())).is_empty()
    {
        return None;
    }
    Some(clean)
}

fn extract_after_marker<'a>(value: &'a str, marker: &str) -> Option<&'a str> {
    let lower = value.to_lowercase();
    let marker = marker.to_lowercase();
    let start = lower.find(&marker)? + marker.len();
    value.get(start..)
}

fn trim_query_edges(value: &str) -> String {
    value
        .trim()
        .trim_matches(|ch: char| matches!(ch, ',' | '.' | ':' | ';' | '?' | '!' | '"' | '\''))
        .trim()
        .to_string()
}

fn continuation_target_label(
    followup: &FollowupActionResolution,
    context: Option<&RecentWorkflowContextSummary>,
) -> String {
    let provider = context
        .and_then(|context| context.provider.as_deref())
        .or(followup.provider_hint.as_deref())
        .unwrap_or("screen");
    let query = context
        .and_then(|context| context.query.as_deref())
        .or(followup.query_hint.as_deref());
    let action = match followup.action_kind {
        FollowupActionKind::OpenResult | FollowupActionKind::ClickResult => {
            let rank = followup
                .result_reference
                .as_ref()
                .and_then(|reference| reference.rank)
                .map(ordinal_label)
                .unwrap_or("ranked");
            format!("{rank} result")
        }
        FollowupActionKind::TypeText => "typing".into(),
        FollowupActionKind::NavigateBack => "back navigation".into(),
        FollowupActionKind::NavigateBackThenOpenResult => {
            "back navigation plus ranked result".into()
        }
        FollowupActionKind::ClickReferencedTarget => "referenced target".into(),
        FollowupActionKind::ContinueWorkflow => "continue".into(),
    };

    if let Some(query) = query {
        format!("{provider} / {query} / {action}")
    } else {
        format!("{provider} / {action}")
    }
}

fn infer_provider_from_url(url: &str) -> Option<String> {
    let lower = url.to_ascii_lowercase();
    if lower.contains("youtube.com") || lower.contains("youtu.be") {
        Some("youtube".into())
    } else if lower.contains("google.") {
        Some("google".into())
    } else {
        None
    }
}

fn infer_query_from_url(url: &str) -> Option<String> {
    let lower = url.to_ascii_lowercase();
    let key = if lower.contains("youtube.com/results") {
        "search_query="
    } else if lower.contains("google.") && lower.contains("/search") {
        "q="
    } else {
        return None;
    };
    let start = lower.find(key)? + key.len();
    let raw = url.get(start..)?;
    let end = raw.find('&').unwrap_or(raw.len());
    let value = &raw[..end];
    Some(url_decode_query(value))
        .map(|value| trim_query_edges(&value))
        .filter(|value| !value.is_empty())
}

fn page_kind_from_url(url: &str) -> BrowserPageKind {
    let lower = url.to_ascii_lowercase();
    if lower.contains("youtube.com/results")
        || (lower.contains("google.") && lower.contains("/search"))
    {
        BrowserPageKind::SearchResults
    } else if lower.contains("youtube.com/watch") || lower.contains("youtu.be/") {
        BrowserPageKind::WatchPage
    } else if lower.starts_with("http://") || lower.starts_with("https://") {
        BrowserPageKind::WebPage
    } else {
        BrowserPageKind::Unknown
    }
}

fn is_http_url(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

fn url_decode_query(value: &str) -> String {
    let mut output = String::new();
    let bytes = value.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'+' => {
                output.push(' ');
                index += 1;
            }
            b'%' if index + 2 < bytes.len() => {
                let hex = &value[index + 1..index + 3];
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    output.push(byte as char);
                    index += 3;
                } else {
                    output.push('%');
                    index += 1;
                }
            }
            byte => {
                output.push(byte as char);
                index += 1;
            }
        }
    }
    output
}

fn query_matches(context_query: &str, hint: &str) -> bool {
    let context = normalize_query(context_query);
    let hint = normalize_query(hint);
    !context.is_empty() && !hint.is_empty() && (context.contains(&hint) || hint.contains(&context))
}

fn normalize_query(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn labels_match(left: &str, right: &str) -> bool {
    normalize_label(left) == normalize_label(right)
}

fn normalize_label(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_")
        .replace(' ', "_")
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

fn result_item_kind_from_value(value: &str) -> ResultListItemKind {
    match normalize_label(value).as_str() {
        "video" | "first_video" | "youtube_video" => ResultListItemKind::Video,
        "link" | "url" => ResultListItemKind::Link,
        "result" | "ranked_result" | "search_result" => ResultListItemKind::Result,
        _ => ResultListItemKind::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desktop_agent_types::{
        CapabilityApprovalState, CapabilityPermissionState, CapabilityRuntimeState,
        CapabilityScreenState, CapabilityToolAvailability, PageEvidenceSource, ScreenCaptureResult,
    };
    use crate::ui_control::{UIPrimitiveCapability, UIPrimitiveCapabilitySet, UIPrimitiveKind};

    #[test]
    fn remembers_youtube_search_as_result_list_context() {
        let request = DesktopActionRequest {
            tool_name: "browser.open".into(),
            params: json!({
                "url": "https://www.youtube.com/results?search_query=shiva",
                "query": "shiva",
                "provider": "youtube"
            }),
            preview_only: false,
            reason: None,
        };
        let context = build_context_from_action_response(
            "request",
            "action",
            &request,
            &json!({"url": "https://www.youtube.com/results?search_query=shiva"}),
            1_000,
        )
        .expect("context");

        assert_eq!(context.provider.as_deref(), Some("youtube"));
        assert_eq!(context.query.as_deref(), Some("shiva"));
        assert!(context.result_list.is_some());
        assert_eq!(
            context.kind,
            RecentWorkflowContextKind::BrowserSearchResults
        );
    }

    #[test]
    fn resolves_first_video_followup_against_recent_youtube_results() {
        let context = youtube_context();
        let resolution = resolve_workflow_continuation(
            Some(context),
            &manifest_with_screen(true),
            "aprimi il primo video di shiva",
            2_000,
        );

        match resolution {
            Some(WorkflowContinuationResolution::Workflow(workflow)) => {
                let continuation = workflow.continuation.expect("continuation");
                assert_eq!(
                    continuation.followup.action_kind,
                    FollowupActionKind::OpenResult
                );
                assert_eq!(
                    continuation
                        .followup
                        .result_reference
                        .and_then(|reference| reference.rank),
                    Some(1)
                );
                assert_eq!(
                    workflow.steps[0].step_kind,
                    WorkflowStepKind::OpenRankedResult
                );
            }
            _ => panic!("expected workflow continuation"),
        }
    }

    #[test]
    fn chrome_tab_phrase_keeps_recent_youtube_provider_and_query() {
        let resolution = resolve_workflow_continuation(
            Some(youtube_context()),
            &manifest_with_screen(true),
            "Aprimi il primo video nella scheda corrispettiva di google chrome",
            2_000,
        );

        match resolution {
            Some(WorkflowContinuationResolution::Workflow(workflow)) => {
                let continuation = workflow.continuation.as_ref().expect("continuation");
                assert_eq!(
                    continuation.followup.action_kind,
                    FollowupActionKind::OpenResult
                );
                assert_eq!(continuation.followup.provider_hint.as_deref(), None);
                assert_eq!(continuation.followup.query_hint.as_deref(), None);
                assert_eq!(
                    continuation.followup.browser_hint.as_deref(),
                    Some("chrome")
                );
                let merge = continuation
                    .followup
                    .merge_diagnostic
                    .as_ref()
                    .expect("merge diagnostic");
                assert_eq!(merge.effective_provider.as_deref(), Some("youtube"));
                assert_eq!(merge.effective_query.as_deref(), Some("shiva"));
                assert_eq!(merge.effective_browser_app.as_deref(), Some("chrome"));
                assert!(merge.conflicts.is_empty());
                assert_eq!(
                    workflow.steps[0]
                        .selection
                        .get("provider")
                        .and_then(Value::as_str),
                    Some("youtube")
                );
                assert_eq!(
                    workflow.steps[0]
                        .selection
                        .get("app")
                        .and_then(Value::as_str),
                    Some("chrome")
                );
                assert_eq!(
                    workflow.steps[0]
                        .selection
                        .get("result_kind")
                        .and_then(Value::as_str),
                    Some("video")
                );
            }
            _ => panic!("expected workflow continuation"),
        }
    }

    #[test]
    fn model_assisted_followup_roles_are_normalized_before_merge() {
        let params = json!({
            "operation": "screen_guided_followup_action",
            "browser_app": "chrome",
            "entities": {
                "rank": 1,
                "result_kind": "video"
            },
            "requires_screen_context": true
        });
        let resolution = resolve_workflow_continuation_with_model_params(
            Some(youtube_context()),
            &manifest_with_screen(true),
            "Aprimi il primo video nella scheda corrispettiva di google chrome",
            &params,
            0.84,
            2_000,
        );

        match resolution {
            Some(WorkflowContinuationResolution::Workflow(workflow)) => {
                let continuation = workflow.continuation.as_ref().expect("continuation");
                assert_eq!(
                    continuation.followup.source,
                    FollowupResolutionSource::ModelAssistedClassifier
                );
                assert_eq!(
                    continuation.followup.browser_hint.as_deref(),
                    Some("chrome")
                );
                assert_eq!(
                    continuation
                        .followup
                        .merge_diagnostic
                        .as_ref()
                        .and_then(|merge| merge.effective_provider.as_deref()),
                    Some("youtube")
                );
                assert_eq!(
                    continuation
                        .followup
                        .merge_diagnostic
                        .as_ref()
                        .and_then(|merge| merge.effective_query.as_deref()),
                    Some("shiva")
                );
            }
            _ => panic!("expected model-assisted workflow continuation"),
        }
    }

    #[test]
    fn edge_browser_detection_is_contextual_not_substring_based() {
        assert_eq!(browser_app_hint("edge of the screen"), None);
        assert_eq!(browser_app_hint("cutting edge tools"), None);
        assert_eq!(browser_app_hint("all'edge della schermata"), None);
        assert_eq!(browser_app_hint("apri microsoft edge"), Some("edge".into()));
        assert_eq!(browser_app_hint("open edge browser"), Some("edge".into()));
    }

    #[test]
    fn page_validation_refuses_provider_mismatch_after_manual_page_change() {
        let descriptor = first_video_descriptor();
        let capture = test_capture(1_900);
        let candidates = vec![ranked_candidate("google_result", "google", 1)];
        let validation =
            validate_continuation_page(&descriptor, Some(&capture), &[], &candidates, 2_000);

        assert_eq!(validation.status, SemanticPageValidationStatus::Mismatched);
        assert_eq!(validation.expected_provider.as_deref(), Some("youtube"));
        assert_eq!(validation.observed_provider.as_deref(), Some("google"));
        assert!(validation
            .mismatch_reason
            .as_deref()
            .unwrap_or("")
            .contains("provider"));
    }

    #[test]
    fn page_validation_ignores_capture_backend_as_content_provider() {
        let descriptor = first_video_descriptor();
        let mut capture = test_capture(1_900);
        capture.provider = "powershell_gdi".into();
        let page_evidence = vec![youtube_results_page_evidence("powershell_gdi")];
        let mut candidate = ranked_candidate("first_result", "powershell_gdi", 1);
        candidate.label = Some("Shiva - official video - YouTube".into());
        candidate.content_provider_hint = None;
        let validation = validate_continuation_page(
            &descriptor,
            Some(&capture),
            &page_evidence,
            &[candidate],
            2_000,
        );

        assert_eq!(validation.status, SemanticPageValidationStatus::Matched);
        assert_eq!(validation.expected_provider.as_deref(), Some("youtube"));
        assert_eq!(validation.observed_provider.as_deref(), Some("youtube"));
        assert_eq!(
            validation.capture_backend.as_deref(),
            Some("powershell_gdi")
        );
        assert!(validation
            .evidence
            .iter()
            .any(|entry| entry == "technical_backend_not_used_as_content_provider"));
    }

    #[test]
    fn stale_capture_requires_fresh_page_validation() {
        let descriptor = first_video_descriptor();
        let capture = test_capture(1_000);
        let validation = validate_continuation_page(
            &descriptor,
            Some(&capture),
            &[],
            &[ranked_candidate("youtube_result", "youtube", 1)],
            1_000 + MAX_RECENT_SCREEN_AGE_MS + 1,
        );

        assert_eq!(
            validation.status,
            SemanticPageValidationStatus::NeedsFreshCapture
        );
        assert!(validation.needs_fresh_capture);
    }

    #[test]
    fn scroll_policy_refuses_truthfully_when_scroll_primitive_is_unsupported() {
        let descriptor = first_video_descriptor();
        let validation = SemanticPageValidationResult {
            status: SemanticPageValidationStatus::LikelyMatched,
            expected_page_kind: BrowserPageKind::SearchResults,
            observed_page_kind: BrowserPageKind::Unknown,
            expected_provider: Some("youtube".into()),
            observed_provider: Some("youtube".into()),
            observed_browser_app: Some("chrome".into()),
            expected_query: Some("shiva".into()),
            observed_query: None,
            capture_backend: Some("powershell_gdi".into()),
            observation_source: Some("structured_vision".into()),
            confidence: 0.66,
            needs_fresh_capture: false,
            mismatch_reason: None,
            evidence: Vec::new(),
            captured_at_ms: Some(2_000),
            capture_age_ms: Some(100),
        };
        let decision = scroll_policy_for_regrounding(
            &descriptor,
            &capabilities_without_scroll(),
            0,
            2,
            &validation,
            0,
            false,
        );

        assert_eq!(decision.status, ScrollContinuationStatus::Unsupported);
        assert!(!decision.scroll_supported);
        assert!(decision.reason.contains("no safe scroll primitive"));
    }

    #[test]
    fn resolves_second_result_followup_with_rank_two() {
        let resolution = resolve_workflow_continuation(
            Some(youtube_context()),
            &manifest_with_screen(true),
            "apri il secondo",
            2_000,
        );

        match resolution {
            Some(WorkflowContinuationResolution::Workflow(workflow)) => {
                let continuation = workflow.continuation.expect("continuation");
                assert_eq!(
                    continuation
                        .followup
                        .result_reference
                        .and_then(|reference| reference.rank),
                    Some(2)
                );
                assert_eq!(
                    workflow.steps[0]
                        .selection
                        .get("rank")
                        .and_then(Value::as_u64),
                    Some(2)
                );
            }
            _ => panic!("expected second-result workflow continuation"),
        }
    }

    #[test]
    fn successive_followups_resolve_without_mutating_recent_context() {
        let context = youtube_context();
        let first = resolve_workflow_continuation(
            Some(context.clone()),
            &manifest_with_screen(true),
            "apri il primo",
            2_000,
        );
        let second = resolve_workflow_continuation(
            Some(context.clone()),
            &manifest_with_screen(true),
            "apri il secondo",
            2_001,
        );

        assert_eq!(context.provider.as_deref(), Some("youtube"));
        assert_eq!(context.query.as_deref(), Some("shiva"));
        assert_eq!(rank_from_resolution(first), Some(1));
        assert_eq!(rank_from_resolution(second), Some(2));
    }

    #[test]
    fn typing_followup_without_focus_builds_reground_then_type_workflow() {
        let resolution = resolve_workflow_continuation(
            Some(youtube_context()),
            &manifest_with_screen(true),
            "ora scrivi Shiva",
            2_000,
        );

        match resolution {
            Some(WorkflowContinuationResolution::Workflow(workflow)) => {
                assert_eq!(workflow.steps.len(), 2);
                assert_eq!(
                    workflow.steps[0].step_kind,
                    WorkflowStepKind::FocusSearchInput
                );
                assert_eq!(workflow.steps[1].step_kind, WorkflowStepKind::EnterText);
                assert_eq!(workflow.steps[1].value.as_deref(), Some("Shiva"));
            }
            _ => panic!("expected typing continuation workflow"),
        }
    }

    #[test]
    fn continue_without_resumable_workflow_refuses() {
        let resolution = resolve_workflow_continuation(
            Some(youtube_context()),
            &manifest_with_screen(true),
            "continua",
            2_000,
        );

        match resolution {
            Some(WorkflowContinuationResolution::Refusal(refusal)) => {
                assert_eq!(
                    refusal.policy.status,
                    ContinuationPolicyStatus::NoResumableWorkflow
                );
            }
            _ => panic!("expected no-resumable-workflow refusal"),
        }
    }

    #[test]
    fn refuses_stale_context_for_result_followup() {
        let mut context = RecentWorkflowContext {
            context_id: "ctx".into(),
            request_id: "request".into(),
            action_id: Some("action".into()),
            run_id: None,
            kind: RecentWorkflowContextKind::BrowserSearchResults,
            page_kind: BrowserPageKind::SearchResults,
            provider: Some("youtube".into()),
            app: Some("browser".into()),
            url: None,
            query: Some("shiva".into()),
            result_list: None,
            recent_focused_target: None,
            recent_selected_target: None,
            last_run_status: None,
            last_followup: None,
            resumable: false,
            continuation_allowed: true,
            created_at_ms: 1_000,
            updated_at_ms: 1_000,
            expires_at_ms: 2_000,
        };
        context.result_list = Some(ResultListContext {
            provider: Some("youtube".into()),
            query: Some("shiva".into()),
            item_kind: ResultListItemKind::Video,
            availability: ResultListAvailability::ExpectedOnScreen,
            observed_candidates: Vec::new(),
            expected_visible: true,
            created_at_ms: 1_000,
            updated_at_ms: 1_000,
            expires_at_ms: 2_000,
        });

        let resolution = resolve_workflow_continuation(
            Some(context),
            &manifest_with_screen(true),
            "clicca il primo risultato",
            5_000,
        );

        match resolution {
            Some(WorkflowContinuationResolution::Refusal(refusal)) => {
                assert_eq!(
                    refusal.policy.status,
                    ContinuationPolicyStatus::StaleWorkflowContext
                );
            }
            _ => panic!("expected stale-context refusal"),
        }
    }

    fn youtube_context() -> RecentWorkflowContext {
        RecentWorkflowContext {
            context_id: "ctx".into(),
            request_id: "request".into(),
            action_id: Some("action".into()),
            run_id: None,
            kind: RecentWorkflowContextKind::BrowserSearchResults,
            page_kind: BrowserPageKind::SearchResults,
            provider: Some("youtube".into()),
            app: Some("browser".into()),
            url: Some("https://www.youtube.com/results?search_query=shiva".into()),
            query: Some("shiva".into()),
            result_list: Some(ResultListContext {
                provider: Some("youtube".into()),
                query: Some("shiva".into()),
                item_kind: ResultListItemKind::Video,
                availability: ResultListAvailability::ExpectedOnScreen,
                observed_candidates: Vec::new(),
                expected_visible: true,
                created_at_ms: 1_000,
                updated_at_ms: 1_000,
                expires_at_ms: 181_000,
            }),
            recent_focused_target: None,
            recent_selected_target: None,
            last_run_status: None,
            last_followup: None,
            resumable: false,
            continuation_allowed: true,
            created_at_ms: 1_000,
            updated_at_ms: 1_000,
            expires_at_ms: 301_000,
        }
    }

    fn rank_from_resolution(resolution: Option<WorkflowContinuationResolution>) -> Option<u32> {
        match resolution? {
            WorkflowContinuationResolution::Workflow(workflow) => workflow
                .continuation
                .and_then(|continuation| continuation.followup.result_reference)
                .and_then(|reference| reference.rank),
            WorkflowContinuationResolution::Refusal(_) => None,
        }
    }

    fn first_video_descriptor() -> WorkflowContinuationDescriptor {
        match resolve_workflow_continuation(
            Some(youtube_context()),
            &manifest_with_screen(true),
            "aprimi il primo video",
            2_000,
        ) {
            Some(WorkflowContinuationResolution::Workflow(workflow)) => {
                workflow.continuation.expect("continuation")
            }
            _ => panic!("expected workflow descriptor"),
        }
    }

    fn test_capture(captured_at: u64) -> ScreenCaptureResult {
        ScreenCaptureResult {
            capture_id: "capture".into(),
            captured_at,
            image_path: "screen.png".into(),
            width: None,
            height: None,
            bytes: 10,
            provider: "test".into(),
        }
    }

    fn youtube_results_page_evidence(capture_backend: &str) -> PageSemanticEvidence {
        PageSemanticEvidence {
            browser_app_hint: Some("chrome".into()),
            content_provider_hint: Some("youtube".into()),
            page_kind_hint: Some("search_results".into()),
            query_hint: Some("shiva".into()),
            result_list_visible: Some(true),
            raw_confidence: Some(0.88),
            confidence: 0.88,
            evidence_sources: vec![
                PageEvidenceSource::StructuredVision,
                PageEvidenceSource::CaptureMetadata,
            ],
            capture_backend: Some(capture_backend.into()),
            observation_source: Some("structured_vision".into()),
            uncertainty: Vec::new(),
        }
    }

    fn ranked_candidate(id: &str, provider: &str, rank: u32) -> UITargetCandidate {
        UITargetCandidate {
            candidate_id: id.into(),
            role: UITargetRole::RankedResult,
            region: None,
            center_x: Some(640.0),
            center_y: Some(360.0),
            app_hint: Some("chrome".into()),
            browser_app_hint: Some("chrome".into()),
            provider_hint: Some(provider.into()),
            content_provider_hint: normalize_content_provider_hint(provider),
            page_kind_hint: Some("search_results".into()),
            capture_backend: None,
            observation_source: Some("test".into()),
            result_kind: Some("video".into()),
            confidence: 0.92,
            source: crate::ui_target_grounding::TargetGroundingSource::ScreenAnalysis,
            label: Some(format!("{provider} result")),
            rank: Some(rank),
            observed_at_ms: Some(2_000),
            reuse_eligible: true,
            supports_focus: false,
            supports_click: true,
            rationale: "test ranked result".into(),
        }
    }

    fn capabilities_without_scroll() -> UIPrimitiveCapabilitySet {
        UIPrimitiveCapabilitySet {
            platform: "test".into(),
            desktop_control_enabled: true,
            primitives: vec![UIPrimitiveCapability {
                primitive: UIPrimitiveKind::ScrollViewport,
                available: false,
                enabled: false,
                requires_screen_context: true,
                requires_high_confidence_target: false,
                requires_approval: false,
                platform_note: "scroll unsupported".into(),
            }],
        }
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
                allowed_permissions: Vec::new(),
                browser_enabled: true,
                desktop_control_enabled: true,
                allowed_roots: Vec::new(),
                terminal_allowed_commands: Vec::new(),
            },
        }
    }
}
