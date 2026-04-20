use crate::ui_target_grounding::UITargetCandidate;
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
