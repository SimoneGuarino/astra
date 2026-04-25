use crate::ui_target_grounding::{TargetRegion, UITargetCandidate};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Permission {
    FilesystemRead,
    FilesystemWrite,
    FilesystemSearch,
    TerminalSafe,
    TerminalDangerous,
    BrowserRead,
    BrowserAction,
    DesktopObserve,
    DesktopControl,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDescriptor {
    pub tool_name: String,
    pub category: String,
    pub description: String,
    pub required_permissions: Vec<Permission>,
    pub default_risk: RiskLevel,
    pub requires_confirmation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopActionRequest {
    pub tool_name: String,
    #[serde(default)]
    pub params: Value,
    #[serde(default)]
    pub preview_only: bool,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DesktopActionStatus {
    Executed,
    ApprovalRequired,
    Rejected,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopActionResponse {
    pub action_id: String,
    pub request_id: String,
    pub tool_name: String,
    pub status: DesktopActionStatus,
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub result: Option<Value>,
    #[serde(default)]
    pub risk_level: Option<RiskLevel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingApproval {
    pub action_id: String,
    pub request_id: String,
    pub tool_name: String,
    pub params: Value,
    pub risk_level: RiskLevel,
    pub reason: String,
    pub requested_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalDecisionRequest {
    pub action_id: String,
    #[serde(default)]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenObservationStatus {
    pub enabled: bool,
    pub provider: String,
    pub last_frame_at: Option<u64>,
    pub last_error: Option<String>,
    pub last_capture_path: Option<String>,
    pub capture_count: u64,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenCaptureResult {
    pub capture_id: String,
    pub captured_at: u64,
    pub image_path: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub bytes: u64,
    pub provider: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PageEvidenceSource {
    StructuredVision,
    TargetCandidate,
    CandidateLabel,
    RecentWorkflowContext,
    CaptureMetadata,
    Heuristic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageSemanticEvidence {
    #[serde(default)]
    pub browser_app_hint: Option<String>,
    #[serde(default)]
    pub content_provider_hint: Option<String>,
    #[serde(default)]
    pub page_kind_hint: Option<String>,
    #[serde(default)]
    pub query_hint: Option<String>,
    #[serde(default)]
    pub result_list_visible: Option<bool>,
    #[serde(default)]
    pub raw_confidence: Option<f32>,
    #[serde(default)]
    pub confidence: f32,
    #[serde(default)]
    pub evidence_sources: Vec<PageEvidenceSource>,
    #[serde(default)]
    pub capture_backend: Option<String>,
    #[serde(default)]
    pub observation_source: Option<String>,
    #[serde(default)]
    pub uncertainty: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VisibleEntityKind {
    ChannelHeader,
    ChannelResult,
    VideoResult,
    MixResult,
    PlaylistResult,
    HotelCard,
    PriceBlock,
    StarRating,
    SortControl,
    FilterChip,
    Avatar,
    TitleLink,
    Thumbnail,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VisibleResultKind {
    Video,
    Channel,
    Playlist,
    Mix,
    Hotel,
    Product,
    Repository,
    Generic,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClickRegion {
    pub region: TargetRegion,
    #[serde(default)]
    pub raw_confidence: Option<f32>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisibleEntity {
    pub entity_id: String,
    pub kind: VisibleEntityKind,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub raw_confidence: Option<f32>,
    pub confidence: f32,
    #[serde(default)]
    pub region: Option<TargetRegion>,
    #[serde(default)]
    pub attributes: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisibleResultItem {
    pub item_id: String,
    pub kind: VisibleResultKind,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub channel_name: Option<String>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub rank_overall: Option<u32>,
    #[serde(default)]
    pub rank_within_kind: Option<u32>,
    #[serde(default)]
    pub click_regions: HashMap<String, ClickRegion>,
    #[serde(default)]
    pub raw_confidence: Option<f32>,
    pub confidence: f32,
    #[serde(default)]
    pub rationale: Option<String>,
    #[serde(default)]
    pub attributes: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimaryListItem {
    pub item_id: String,
    pub rank: u32,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub item_kind: Option<String>,
    #[serde(default)]
    pub is_sponsored: Option<bool>,
    #[serde(default)]
    pub raw_confidence: Option<f32>,
    pub confidence: f32,
    #[serde(default)]
    pub click_regions: HashMap<String, ClickRegion>,
    #[serde(default)]
    pub attributes: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimaryList {
    pub cluster_id: String,
    pub container_kind: String,
    pub item_count: u32,
    #[serde(default)]
    pub items: Vec<PrimaryListItem>,
    #[serde(default)]
    pub raw_confidence: Option<f32>,
    #[serde(default)]
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageState {
    pub kind: String,
    pub dominant_content: String,
    #[serde(default)]
    pub list_visible: Option<bool>,
    #[serde(default)]
    pub detail_visible: Option<bool>,
    #[serde(default)]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionableControl {
    pub control_id: String,
    pub kind: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub region: Option<TargetRegion>,
    #[serde(default)]
    pub raw_confidence: Option<f32>,
    pub confidence: f32,
    #[serde(default)]
    pub attributes: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameUncertainty {
    pub code: String,
    pub message: String,
    pub severity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticScreenFrame {
    pub frame_id: String,
    pub captured_at: u64,
    #[serde(default)]
    pub image_path: Option<String>,
    pub page_evidence: PageSemanticEvidence,
    pub scene_summary: String,
    #[serde(default)]
    pub visible_entities: Vec<VisibleEntity>,
    #[serde(default)]
    pub visible_result_items: Vec<VisibleResultItem>,
    #[serde(default)]
    pub primary_list: Option<PrimaryList>,
    #[serde(default)]
    pub page_state: Option<PageState>,
    #[serde(default)]
    pub actionable_controls: Vec<ActionableControl>,
    #[serde(default)]
    pub legacy_target_candidates: Vec<UITargetCandidate>,
    #[serde(default)]
    pub uncertainty: Vec<FrameUncertainty>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GoalType {
    OpenListItem,
    OpenMediaResult,
    OpenChannel,
    FindBestOffer,
    InspectScreen,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalConstraints {
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub item_kind: Option<String>,
    #[serde(default)]
    pub result_kind: Option<VisibleResultKind>,
    #[serde(default)]
    pub rank_within_kind: Option<u32>,
    #[serde(default)]
    pub rank_overall: Option<u32>,
    #[serde(default)]
    pub entity_name: Option<String>,
    #[serde(default)]
    pub attributes: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalSpec {
    pub goal_id: String,
    pub goal_type: GoalType,
    pub constraints: GoalConstraints,
    pub success_condition: String,
    pub utterance: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlannerStepKind {
    ClickResultRegion,
    ClickEntityRegion,
    VerifyGoal,
    ReplanAfterPerception,
    RequestClarification,
    Refuse,
    NoOp,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlannerStepExecutionStatus {
    Executed,
    Skipped,
    Unsupported,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutableCoordinateInterpretation {
    ScreenValidated,
    WindowRelativeTranslated,
    RejectedSuspiciousGeometry,
    RejectedOutsideBrowserSurface,
    RejectedOutsideScreenBounds,
    RejectedUnsupportedCoordinateSpace,
    RejectedUntrustedGeometry,
    Unknown,
}

impl Default for ExecutableCoordinateInterpretation {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BrowserRecoveryStatus {
    NotNeeded,
    Attempted,
    Reacquired,
    Failed,
}

impl Default for BrowserRecoveryStatus {
    fn default() -> Self {
        Self::NotNeeded
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutableGeometryDiagnostic {
    #[serde(default)]
    pub raw_region: Option<TargetRegion>,
    #[serde(default)]
    pub interpreted_region: Option<TargetRegion>,
    #[serde(default)]
    pub raw_coordinate_space: Option<String>,
    #[serde(default)]
    pub interpretation: ExecutableCoordinateInterpretation,
    #[serde(default)]
    pub validation_passed: bool,
    #[serde(default)]
    pub translation_applied: bool,
    #[serde(default)]
    pub screen_bounds: Option<TargetRegion>,
    #[serde(default)]
    pub browser_window_bounds: Option<TargetRegion>,
    #[serde(default)]
    pub final_x: Option<i32>,
    #[serde(default)]
    pub final_y: Option<i32>,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerStepExecutionRecord {
    pub step_id: String,
    pub status: PlannerStepExecutionStatus,
    pub primitive: String,
    pub message: String,
    #[serde(default)]
    pub selected_target_candidate: Option<UITargetCandidate>,
    #[serde(default)]
    pub geometry: Option<ExecutableGeometryDiagnostic>,
    #[serde(default)]
    pub fresh_capture_required: bool,
    #[serde(default)]
    pub fresh_capture_used: bool,
    #[serde(default)]
    pub target_signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GoalVerificationStatus {
    GoalAchieved,
    GoalNotAchieved,
    PageChangedWrongOutcome,
    PageUnchanged,
    Ambiguous,
    ReplanRequired,
    Unsupported,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalVerificationRecord {
    pub iteration: usize,
    pub status: GoalVerificationStatus,
    pub confidence: f32,
    pub reason: String,
    #[serde(default)]
    pub frame_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VisibleActionabilityStatus {
    VisibleExecutable,
    VisibleTargetNeedsClickRegion,
    VisibleUnderGrounded,
    NoRelevantVisibleContent,
    LikelyOffscreen,
    Unknown,
}

impl Default for VisibleActionabilityStatus {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VisibleGroundingGap {
    WeakItemTyping,
    MissingClickRegion,
    MissingTitle,
    MissingRanking,
    RepeatedVisibleContent,
    PartialSemanticSignals,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutableFallbackSource {
    ActionableControl,
    LegacyTargetCandidate,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceSignalState {
    Missing,
    ExplicitZero,
    Low,
    Supported,
}

impl Default for ConfidenceSignalState {
    fn default() -> Self {
        Self::Missing
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExecutableTargetConfidenceDiagnostic {
    #[serde(default)]
    pub raw_item_confidence: Option<f32>,
    #[serde(default)]
    pub raw_region_confidence: Option<f32>,
    #[serde(default)]
    pub raw_page_confidence: Option<f32>,
    #[serde(default)]
    pub raw_planner_confidence: Option<f32>,
    #[serde(default)]
    pub item_confidence_state: ConfidenceSignalState,
    #[serde(default)]
    pub region_confidence_state: ConfidenceSignalState,
    #[serde(default)]
    pub page_confidence_state: ConfidenceSignalState,
    #[serde(default)]
    pub planner_confidence_state: ConfidenceSignalState,
    #[serde(default)]
    pub structural_semantic_confidence: f32,
    #[serde(default)]
    pub semantic_confidence: f32,
    #[serde(default)]
    pub region_confidence: f32,
    #[serde(default)]
    pub page_confidence: f32,
    #[serde(default)]
    pub derived_confidence: f32,
    #[serde(default)]
    pub required_threshold: f32,
    #[serde(default)]
    pub confidence_was_derived: bool,
    #[serde(default)]
    pub fallback_source: Option<ExecutableFallbackSource>,
    #[serde(default)]
    pub accepted: bool,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VisibleRefinementStrategy {
    None,
    TargetRegion,
    VisibleCluster,
    FullFrame,
}

impl Default for VisibleRefinementStrategy {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OffscreenInferenceStage {
    NotApplicable,
    DeferredPendingRefinement,
    EligibleAfterRefinement,
    ConfirmedAfterRefinement,
}

impl Default for OffscreenInferenceStage {
    fn default() -> Self {
        Self::NotApplicable
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisibleActionabilityDiagnostic {
    #[serde(default)]
    pub status: VisibleActionabilityStatus,
    #[serde(default)]
    pub relevant_visible_content: bool,
    #[serde(default)]
    pub target_visible_evidence: bool,
    #[serde(default)]
    pub refinement_eligible: bool,
    #[serde(default)]
    pub refinement_strategy: VisibleRefinementStrategy,
    #[serde(default)]
    pub offscreen_inference_stage: OffscreenInferenceStage,
    #[serde(default)]
    pub gaps: Vec<VisibleGroundingGap>,
    #[serde(default)]
    pub result_item_count: usize,
    #[serde(default)]
    pub weak_result_count: usize,
    #[serde(default)]
    pub missing_click_region_count: usize,
    #[serde(default)]
    pub missing_title_count: usize,
    #[serde(default)]
    pub missing_ranking_count: usize,
    #[serde(default)]
    pub entity_signal_count: usize,
    #[serde(default)]
    pub actionable_control_signal_count: usize,
    #[serde(default)]
    pub legacy_candidate_signal_count: usize,
    #[serde(default)]
    pub visible_refinement_attempts: usize,
    #[serde(default)]
    pub safe_fallback_available: bool,
    #[serde(default)]
    pub fallback_source_used: Option<ExecutableFallbackSource>,
}

impl Default for VisibleActionabilityDiagnostic {
    fn default() -> Self {
        Self {
            status: VisibleActionabilityStatus::Unknown,
            relevant_visible_content: false,
            target_visible_evidence: false,
            refinement_eligible: false,
            refinement_strategy: VisibleRefinementStrategy::None,
            offscreen_inference_stage: OffscreenInferenceStage::NotApplicable,
            gaps: Vec::new(),
            result_item_count: 0,
            weak_result_count: 0,
            missing_click_region_count: 0,
            missing_title_count: 0,
            missing_ranking_count: 0,
            entity_signal_count: 0,
            actionable_control_signal_count: 0,
            legacy_candidate_signal_count: 0,
            visible_refinement_attempts: 0,
            safe_fallback_available: false,
            fallback_source_used: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PerceptionRequestMode {
    TargetFocus,
    VisiblePageRefinement,
}

impl Default for PerceptionRequestMode {
    fn default() -> Self {
        Self::TargetFocus
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PerceptionRoutingDecision {
    TargetRegionAnchor,
    RegionlessTargetVisible,
    VisiblePageUnderGrounded,
}

impl Default for PerceptionRoutingDecision {
    fn default() -> Self {
        Self::VisiblePageUnderGrounded
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InteractionSurfaceKind {
    Browser,
    DesktopAgent,
    Terminal,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedInteractionSurface {
    pub kind: InteractionSurfaceKind,
    #[serde(default)]
    pub provider_hint: Option<String>,
    #[serde(default)]
    pub app_hint: Option<String>,
    #[serde(default)]
    pub page_kind_hint: Option<String>,
    #[serde(default)]
    pub bounds: Option<TargetRegion>,
    pub verified_at_ms: u64,
    pub source_frame_id: String,
    #[serde(default)]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FocusedPerceptionFailureReason {
    NoVerifiedSurface,
    SurfaceBoundsUnavailable,
    RequestedRegionOutsideSurface,
    SurfaceOwnershipLost,
    StructuredPerceptionEmpty,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceOwnershipStatus {
    NotRequired,
    Verified,
    Reacquired,
    Lost,
    Refused,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfaceOwnershipDiagnostic {
    pub iteration: usize,
    pub status: SurfaceOwnershipStatus,
    #[serde(default)]
    pub failure_reason: Option<FocusedPerceptionFailureReason>,
    #[serde(default)]
    pub surface: Option<VerifiedInteractionSurface>,
    #[serde(default)]
    pub observed_frame_id: Option<String>,
    #[serde(default)]
    pub provider_matches: Option<bool>,
    #[serde(default)]
    pub browser_evidence_present: bool,
    #[serde(default)]
    pub requested_region: Option<TargetRegion>,
    #[serde(default)]
    pub surface_bounds: Option<TargetRegion>,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusedPerceptionRequest {
    pub request_id: String,
    pub iteration: usize,
    pub reason: String,
    #[serde(default)]
    pub mode: PerceptionRequestMode,
    #[serde(default)]
    pub routing_decision: PerceptionRoutingDecision,
    #[serde(default)]
    pub refinement_strategy: VisibleRefinementStrategy,
    #[serde(default)]
    pub target_item_id: Option<String>,
    #[serde(default)]
    pub target_entity_id: Option<String>,
    #[serde(default)]
    pub region: Option<TargetRegion>,
    #[serde(default)]
    pub target_region_anchor_present: bool,
    #[serde(default)]
    pub verified_surface: Option<VerifiedInteractionSurface>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BrowserHandoffStatus {
    NotRequired,
    VisuallyVerified,
    ActivationUnsupported,
    ActivationFailed,
    PageNotVerified,
    SemanticFrameUnavailable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BrowserHandoffActivationStatus {
    NotAttempted,
    Executed,
    Unsupported,
    Failed,
}

impl Default for BrowserHandoffActivationStatus {
    fn default() -> Self {
        Self::NotAttempted
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BrowserHandoffFailureReason {
    BrowserHandoffRequired,
    BrowserActivationUnsupported,
    BrowserActivationFailed,
    BrowserPageNotVerified,
    SemanticFrameUnavailable,
    PermissionDenied,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BrowserPageSemanticKind {
    SearchResults,
    WatchPage,
    WebPage,
    Unknown,
}

impl Default for BrowserPageSemanticKind {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BrowserHandoffVerificationDecision {
    GoalSatisfied,
    NormalizedPageKind,
    SupportingEvidence,
    Rejected,
    NotRequired,
}

impl Default for BrowserHandoffVerificationDecision {
    fn default() -> Self {
        Self::Rejected
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BrowserHandoffVerificationDiagnostic {
    #[serde(default)]
    pub raw_page_kind_hint: Option<String>,
    #[serde(default)]
    pub normalized_page_kind: BrowserPageSemanticKind,
    #[serde(default)]
    pub decision: BrowserHandoffVerificationDecision,
    #[serde(default)]
    pub accepted: bool,
    #[serde(default)]
    pub provider_matches: bool,
    #[serde(default)]
    pub goal_expects_results_context: bool,
    #[serde(default)]
    pub generic_provider_page_kind_hint: bool,
    #[serde(default)]
    pub query_hint_present: bool,
    #[serde(default)]
    pub result_list_visible: bool,
    #[serde(default)]
    pub visible_result_item_count: usize,
    #[serde(default)]
    pub primary_list_item_count: usize,
    #[serde(default)]
    pub structural_list_surface_visible: bool,
    #[serde(default)]
    pub page_state_kind: Option<String>,
    #[serde(default)]
    pub page_state_dominant_content: Option<String>,
    #[serde(default)]
    pub visible_entity_signal_count: usize,
    #[serde(default)]
    pub legacy_candidate_signal_count: usize,
    #[serde(default)]
    pub scene_summary_result_hint: bool,
    #[serde(default)]
    pub supporting_signal_count: usize,
    #[serde(default)]
    pub supporting_evidence: Vec<String>,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserVisualHandoffRecord {
    pub iteration: usize,
    pub status: BrowserHandoffStatus,
    #[serde(default)]
    pub activation_status: BrowserHandoffActivationStatus,
    #[serde(default)]
    pub failure_reason: Option<BrowserHandoffFailureReason>,
    #[serde(default)]
    pub app_hint: Option<String>,
    #[serde(default)]
    pub provider_hint: Option<String>,
    #[serde(default)]
    pub page_kind_hint: Option<String>,
    #[serde(default)]
    pub verification: Option<BrowserHandoffVerificationDiagnostic>,
    pub activation_attempted: bool,
    pub page_verified: bool,
    #[serde(default)]
    pub frame_id: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
    pub attempts: usize,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct BrowserVisualHandoffResult {
    pub record: BrowserVisualHandoffRecord,
    pub verified_frame: Option<SemanticScreenFrame>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerStep {
    pub step_id: String,
    pub kind: PlannerStepKind,
    pub confidence: f32,
    pub rationale: String,
    #[serde(default)]
    pub target_item_id: Option<String>,
    #[serde(default)]
    pub target_entity_id: Option<String>,
    #[serde(default)]
    pub click_region_key: Option<String>,
    #[serde(default)]
    pub executable_candidate: Option<UITargetCandidate>,
    #[serde(default)]
    pub expected_state: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlannerContractSource {
    RustDeterministic,
    ModelAssisted,
    ModelAssistedFallback,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerContractInput {
    pub goal: GoalSpec,
    pub current_frame: SemanticScreenFrame,
    #[serde(default)]
    pub executed_steps: Vec<PlannerStepExecutionRecord>,
    #[serde(default)]
    pub verification_history: Vec<GoalVerificationRecord>,
    #[serde(default)]
    pub perception_requests: Vec<FocusedPerceptionRequest>,
    pub retry_budget: usize,
    pub retries_used: usize,
    #[serde(default)]
    pub visible_refinement_attempts: usize,
    #[serde(default)]
    pub max_visible_refinement_passes: usize,
    #[serde(default)]
    pub provider_hint: Option<String>,
    #[serde(default)]
    pub browser_app_hint: Option<String>,
    #[serde(default)]
    pub page_kind_hint: Option<String>,
    #[serde(default)]
    pub visible_actionability: VisibleActionabilityDiagnostic,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlannerDecisionStatus {
    Accepted,
    Normalized,
    Downgraded,
    Rejected,
    FallbackUsed,
}

impl Default for PlannerDecisionStatus {
    fn default() -> Self {
        Self::Accepted
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlannerRejectionReason {
    ModelUnavailable,
    LowConfidence,
    MalformedOutput,
    UnsupportedPrimitive,
    MissingTarget,
    FabricatedTarget,
    MissingClickRegion,
    ProviderMismatch,
    AmbiguousTarget,
    FocusedPerceptionRequested,
    VisiblePageRefinementRequested,
    LikelyOffscreenTarget,
    ScrollRequiredButUnsupported,
    DeterministicFallbackAdvised,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlannerVisibilityAssessment {
    VisibleGrounded,
    VisibleAmbiguous,
    VisibleUnderGrounded,
    VisibleTargetNeedsClickRegion,
    FocusedPerceptionNeeded,
    LikelyOffscreen,
    NotVisible,
    Unknown,
}

impl Default for PlannerVisibilityAssessment {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlannerScrollIntent {
    NotNeeded,
    LikelyNeeded,
    RequiredButUnsupported,
    FuturePrimitiveRequired,
}

impl Default for PlannerScrollIntent {
    fn default() -> Self {
        Self::NotNeeded
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerContractDecision {
    pub source: PlannerContractSource,
    pub proposed_step: PlannerStep,
    pub strategy_rationale: String,
    pub focused_perception_needed: bool,
    pub replan_needed: bool,
    #[serde(default)]
    pub expected_verification_target: Option<String>,
    pub planner_confidence: f32,
    pub accepted: bool,
    pub fallback_used: bool,
    #[serde(default)]
    pub rejection_reason: Option<String>,
    #[serde(default)]
    pub decision_status: PlannerDecisionStatus,
    #[serde(default)]
    pub rejection_code: Option<PlannerRejectionReason>,
    #[serde(default)]
    pub visibility_assessment: PlannerVisibilityAssessment,
    #[serde(default)]
    pub scroll_intent: PlannerScrollIntent,
    #[serde(default)]
    pub visible_actionability: VisibleActionabilityDiagnostic,
    #[serde(default)]
    pub target_confidence: Option<ExecutableTargetConfidenceDiagnostic>,
    #[serde(default)]
    pub normalized: bool,
    #[serde(default)]
    pub downgraded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerDecisionDiagnostic {
    pub iteration: usize,
    pub source: PlannerContractSource,
    pub accepted: bool,
    pub fallback_used: bool,
    pub planner_confidence: f32,
    #[serde(default)]
    pub strategy: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub decision_status: PlannerDecisionStatus,
    #[serde(default)]
    pub rejection_code: Option<PlannerRejectionReason>,
    #[serde(default)]
    pub visibility_assessment: PlannerVisibilityAssessment,
    #[serde(default)]
    pub scroll_intent: PlannerScrollIntent,
    #[serde(default)]
    pub visible_actionability: VisibleActionabilityDiagnostic,
    #[serde(default)]
    pub target_confidence: Option<ExecutableTargetConfidenceDiagnostic>,
    #[serde(default)]
    pub normalized: bool,
    #[serde(default)]
    pub downgraded: bool,
    #[serde(default)]
    pub focused_perception_needed: bool,
    #[serde(default)]
    pub replan_needed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GoalLoopStatus {
    Running,
    GoalAchieved,
    NeedsExecution,
    NeedsPerception,
    NeedsClarification,
    Refused,
    BrowserHandoffFailed,
    ScrollRequiredButUnsupported,
    BudgetExhausted,
    VerificationFailed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalLoopRun {
    pub run_id: String,
    pub goal: GoalSpec,
    pub status: GoalLoopStatus,
    pub iteration_count: usize,
    pub retry_budget: usize,
    pub retries_used: usize,
    #[serde(default)]
    pub current_strategy: Option<String>,
    #[serde(default)]
    pub fallback_strategy_state: Option<String>,
    #[serde(default)]
    pub frames: Vec<SemanticScreenFrame>,
    #[serde(default)]
    pub planner_steps: Vec<PlannerStep>,
    #[serde(default)]
    pub planner_diagnostics: Vec<PlannerDecisionDiagnostic>,
    #[serde(default)]
    pub executed_steps: Vec<PlannerStepExecutionRecord>,
    #[serde(default)]
    pub verification_history: Vec<GoalVerificationRecord>,
    #[serde(default)]
    pub focused_perception_requests: Vec<FocusedPerceptionRequest>,
    #[serde(default)]
    pub browser_handoff_history: Vec<BrowserVisualHandoffRecord>,
    #[serde(default)]
    pub browser_handoff: Option<BrowserVisualHandoffRecord>,
    #[serde(default)]
    pub verified_surface: Option<VerifiedInteractionSurface>,
    #[serde(default)]
    pub surface_diagnostics: Vec<SurfaceOwnershipDiagnostic>,
    pub focused_perception_used: bool,
    pub visible_refinement_used: bool,
    #[serde(default)]
    pub stale_capture_reuse_prevented: bool,
    #[serde(default)]
    pub browser_recovery_used: bool,
    #[serde(default)]
    pub browser_recovery_status: BrowserRecoveryStatus,
    #[serde(default)]
    pub post_action_progress_observed: bool,
    #[serde(default)]
    pub surface_ownership_lost: bool,
    #[serde(default)]
    pub focused_perception_failure_reason: Option<FocusedPerceptionFailureReason>,
    #[serde(default)]
    pub repeated_click_protection_triggered: bool,
    #[serde(default)]
    pub selected_target_candidate: Option<UITargetCandidate>,
    #[serde(default)]
    pub verifier_status: Option<String>,
    #[serde(default)]
    pub failure_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenAnalysisRequest {
    #[serde(default)]
    pub question: Option<String>,
    #[serde(default)]
    pub capture_fresh: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenAnalysisResult {
    pub analysis_id: String,
    pub request_id: String,
    pub captured_at: u64,
    pub image_path: String,
    pub model: String,
    pub provider: String,
    pub question: String,
    pub answer: String,
    #[serde(default)]
    pub ui_candidates: Vec<UITargetCandidate>,
    #[serde(default)]
    pub structured_candidates_error: Option<String>,
    #[serde(default)]
    pub semantic_frame: Option<SemanticScreenFrame>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopAuditEvent {
    pub audit_id: String,
    pub action_id: String,
    pub request_id: String,
    pub tool_name: String,
    pub stage: String,
    pub status: String,
    pub timestamp: u64,
    pub risk_level: RiskLevel,
    #[serde(default)]
    pub details: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopPolicySnapshot {
    pub allowed_roots: Vec<String>,
    pub terminal_allowed_commands: Vec<String>,
    pub allowed_permissions: Vec<Permission>,
    pub approval_required_for_high_risk: bool,
    pub browser_enabled: bool,
    pub desktop_control_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityToolAvailability {
    pub available: bool,
    pub enabled: bool,
    pub requires_approval: bool,
    pub state: CapabilityRuntimeState,
    #[serde(default)]
    pub disabled_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityRuntimeState {
    Unavailable,
    Disabled,
    ApprovalGated,
    Ready,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityToolState {
    pub tool_name: String,
    pub category: String,
    pub description: String,
    pub required_permissions: Vec<Permission>,
    pub default_risk: RiskLevel,
    pub requires_confirmation: bool,
    pub available: bool,
    pub enabled: bool,
    pub requires_approval: bool,
    pub state: CapabilityRuntimeState,
    #[serde(default)]
    pub disabled_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityScreenState {
    pub observation_supported: bool,
    pub observation_enabled: bool,
    pub capture_available: bool,
    pub analysis_available: bool,
    pub vision_model_available: bool,
    pub vision_model_name: Option<String>,
    pub recent_capture_available: bool,
    pub recent_capture_age_ms: Option<u64>,
    pub fresh_capture_available: bool,
    pub fresh_capture_requires_observation_enabled: bool,
    pub last_capture_path: Option<String>,
    pub last_frame_at: Option<u64>,
    pub provider: String,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityApprovalState {
    pub pending_count: usize,
    pub approval_required_for_high_risk: bool,
    #[serde(default)]
    pub pending_actions: Vec<CapabilityPendingApprovalSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityPendingApprovalSummary {
    pub action_id: String,
    pub tool_name: String,
    pub risk_level: RiskLevel,
    pub reason: String,
    pub requested_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityPermissionState {
    pub allowed_permissions: Vec<Permission>,
    pub browser_enabled: bool,
    pub desktop_control_enabled: bool,
    pub allowed_roots: Vec<String>,
    pub terminal_allowed_commands: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityManifest {
    pub version: String,
    pub generated_at: u64,
    pub tool_names: Vec<String>,
    pub enabled_tool_names: Vec<String>,
    pub disabled_tool_names: Vec<String>,
    pub tools: Vec<CapabilityToolState>,
    pub filesystem_read: CapabilityToolAvailability,
    pub filesystem_write: CapabilityToolAvailability,
    pub filesystem_search: CapabilityToolAvailability,
    pub terminal: CapabilityToolAvailability,
    pub browser_open: CapabilityToolAvailability,
    pub browser_search: CapabilityToolAvailability,
    pub desktop_launch: CapabilityToolAvailability,
    pub screen: CapabilityScreenState,
    pub approvals: CapabilityApprovalState,
    pub permissions: CapabilityPermissionState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionAvailability {
    pub available: bool,
    pub selected_model: Option<String>,
    pub candidates: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationRouteDiagnostic {
    pub message_excerpt: String,
    pub classifier_source: String,
    pub intent: String,
    pub target: Option<String>,
    pub action: Option<String>,
    #[serde(default)]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub extracted_params: Option<Value>,
    pub confidence: Option<f32>,
    pub routed_to: String,
    pub grounded: bool,
    pub fallback_used: bool,
    pub submit_action_called: bool,
    #[serde(default)]
    pub action_id: Option<String>,
    #[serde(default)]
    pub action_status: Option<String>,
    pub approval_created: bool,
    pub audit_expected: bool,
    pub rationale: Option<String>,
    pub error: Option<String>,
}
