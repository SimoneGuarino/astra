use crate::{
    assistant_tool_router::AssistantRouteDecision,
    context_broker,
    conversation_history::ConversationMessage,
    model_routing::{ollama_endpoint, resolve_active_ollama_model},
    OllamaChatResponse,
};
use chrono::Utc;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, time::Instant};

pub const MIN_ANSWER_FROM_CONTEXT_CONFIDENCE: f32 = 0.70;
pub const MIN_CONTEXT_BOUNDARY_CONFIDENCE: f32 = 0.60;
pub const NORMAL_CHAT_DIRECT_CONFIDENCE_THRESHOLD: f32 = 0.85;
#[allow(dead_code)]
pub const MIN_NORMAL_CHAT_CONFIDENCE: f32 = NORMAL_CHAT_DIRECT_CONFIDENCE_THRESHOLD;
pub const MIN_CLARIFY_CONFIDENCE: f32 = 0.60;
pub const MIN_REFUSE_CONFIDENCE: f32 = 0.80;
pub const MIN_NEEDS_TOOL_CONFIDENCE: f32 = 0.70;
pub const MIN_CONTEXT_SALIENCE_SCORE: f32 = 0.25;
pub const CONTEXT_SALIENCE_STALE_NORMAL_TURNS: u32 = 4;
pub const CONTEXT_SALIENCE_STALE_TURNS: u32 = 10;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AstraUserLanguage {
    Italian,
    English,
    Mixed,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkingContextFrame {
    pub current_topic: Option<String>,
    pub active_entities: Vec<String>,
    pub last_user_goal_summary: Option<String>,
    pub last_assistant_answer_summary: Option<String>,
    pub last_assistant_action: Option<String>,
    pub last_referenced_session: Option<SessionReference>,
    pub last_referenced_artifacts: Vec<ArtifactReference>,
    pub last_tool_result: Option<ToolResultFrame>,
    pub available_evidence_refs: Vec<EvidenceReference>,
    pub unresolved_followups: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending_governed_action: Option<PendingGovernedActionFrame>,
    pub salience: ContextSalience,
    pub confidence: f32,
    pub updated_at_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContextSalience {
    pub turn_age: u32,
    pub normal_chat_turns_since_update: u32,
    pub last_reinforced_at_ms: Option<i64>,
    pub salience_score: f32,
    pub stale: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionReference {
    pub session_id: String,
    pub source_kind: String,
    pub source_label: String,
    pub title: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactReference {
    pub artifact_id: String,
    pub artifact_kind: String,
    pub source_kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvidenceReference {
    pub evidence_id: String,
    pub evidence_kind: String,
    pub source_kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PendingGovernedActionFrame {
    pub present: bool,
    pub tool_name: String,
    pub intent: String,
    pub prerequisite: Option<String>,
    pub status: String,
    pub expires_at_present: bool,
    pub expired: bool,
    pub attempt_count: u8,
    pub metadata_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolResultFrame {
    pub tool_name: String,
    pub answer_kind: String,
    pub source_kind: String,
    pub source_label: String,
    pub session_id: Option<String>,
    pub used_evidence_ids: Vec<String>,
    pub evidence_count: usize,
    pub answer_summary: String,
    pub key_topics: Vec<String>,
    pub active_entities: Vec<String>,
    pub warnings: Vec<String>,
    pub confidence: Option<f32>,
    pub created_at_ms: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConversationOrchestratorDecision {
    AnswerFromContext(ContextAnswerPlan),
    AnswerFromContextBoundary(ContextAnswerPlan),
    NormalChatWithContext(ContextualChatPlan),
    ToolCall(AssistantRouteDecision),
    DeferToToolRouter(DeferToToolRouterPlan),
    NormalChat(NormalChatPlan),
    Clarify(ClarificationPlan),
    Refuse(RefusalPlan),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContextAnswerPlan {
    pub strategy: String,
    pub focus: Option<String>,
    pub context_ref: String,
    pub reason_code: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NormalChatPlan {
    pub intent_kind: Option<PlannerIntentKind>,
    pub capability_family: Option<PlannerCapabilityFamily>,
    pub requires_tool_arbitration: Option<bool>,
    pub requires_memory_lookup: Option<bool>,
    pub requires_governed_action: Option<bool>,
    pub requires_context_boundary: Option<bool>,
    pub safe_to_bypass_tools: Option<bool>,
    pub context_ref: Option<String>,
    pub reason_code: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContextualChatPlan {
    pub context_ref: String,
    pub reason_code: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeferToToolRouterPlan {
    pub reason: String,
    pub planner_failure_reason: Option<String>,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClarificationPlan {
    pub reason_code: String,
    pub message: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RefusalPlan {
    pub reason_code: String,
    pub message: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrchestratorPolicyAction {
    AcceptDecision,
    UseFullToolRouter {
        reason: OrchestratorFallbackReason,
    },
    DeferToFullToolRouter {
        reason: OrchestratorFallbackReason,
    },
    DowngradeToNormalChatWithContext {
        reason: NeedsToolPolicyReason,
    },
    DowngradeToContextBoundary {
        reason: NeedsToolPolicyReason,
    },
    #[allow(dead_code)]
    SafeClarify {
        reason: OrchestratorFallbackReason,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PlannerIntentKind {
    OrdinaryQuestion,
    CasualChat,
    GeneralKnowledge,
    SessionMemoryQuery,
    GovernedAction,
    ContextFollowup,
    ContextBoundary,
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PlannerCapabilityFamily {
    None,
    WorkSession,
    Meeting,
    ScreenContext,
    SessionMemory,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PlannerSafetyMetadata {
    pub intent_kind: Option<PlannerIntentKind>,
    pub capability_family: Option<PlannerCapabilityFamily>,
    pub requires_tool_arbitration: Option<bool>,
    pub requires_memory_lookup: Option<bool>,
    pub requires_governed_action: Option<bool>,
    pub requires_context_boundary: Option<bool>,
    pub safe_to_bypass_tools: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalChatArbitrationAction {
    AcceptDirectNormalChat,
    VerifyWithToolRouter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalChatArbitrationReason {
    SafeToBypassTools,
    LowConfidence,
    MissingSafetyFields,
    UnsafeToBypassTools,
    UnknownIntentKind,
    UnknownCapabilityFamily,
    CapabilityRequiresTools,
    RequiresToolArbitration,
    RequiresMemoryLookup,
    RequiresGovernedAction,
    RequiresContextBoundary,
    ToolAffinityRisk,
    PendingGovernedToolFlow,
    ExplicitUiAction,
    SlashCommand,
    ContextReference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormalChatArbitrationDecision {
    pub action: NormalChatArbitrationAction,
    pub reason: NormalChatArbitrationReason,
}

pub struct NormalChatArbitrationInput<'a> {
    pub plan: &'a NormalChatPlan,
    pub working_context: &'a WorkingContextFrame,
    pub tool_affinity_risk: Option<bool>,
    pub pending_governed_tool_flow: bool,
    pub explicit_ui_action: bool,
    pub slash_command: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ToolAffinitySignal {
    None,
    PlannerHighConfidence,
    ExplicitSlashCommand,
    UiAction,
    PendingToolContinuation,
    FullRouterResult,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NeedsToolPolicyInput {
    pub planner_confidence: Option<f32>,
    pub context_ref: Option<String>,
    pub last_tool_result_present: bool,
    pub pending_tool_action: bool,
    pub explicit_user_action: bool,
    pub slash_command: bool,
    pub ui_action: bool,
    pub planner_reason_code: Option<String>,
    pub tool_affinity: ToolAffinitySignal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum NeedsToolPolicyDecision {
    Accept { reason: NeedsToolPolicyReason },
    DowngradeToContextBoundary { reason: NeedsToolPolicyReason },
    DowngradeToNormalChatWithContext { reason: NeedsToolPolicyReason },
    Clarify { reason: NeedsToolPolicyReason },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NeedsToolPolicyReason {
    PlannerHighConfidence,
    ExplicitSlashCommand,
    ExplicitUserAction,
    UiAction,
    PendingToolContinuation,
    FullRouterResult,
    LowConfidenceWithLastToolResult,
    LowConfidenceWithoutToolAffinity,
    MissingPlannerConfidence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrchestratorFallbackReason {
    NormalChatLowConfidence,
    NormalChatUnsafeToBypassTools,
    AnswerFromContextLowConfidence,
    ClarifyLowConfidence,
    RefuseLowConfidence,
    MissingContextForContextAnswer,
    PlannerMalformed,
    PlannerEmpty,
    PlannerTimeout,
    PlannerEmptyNoGroundedContext,
    PlannerFailureNoGroundedContext,
    PlannerUnavailable,
    ToolAffinityUnresolved,
    NoLastToolResult,
}

#[derive(Debug, Clone, Serialize)]
pub struct AssistantOrchestratorDiagnostic {
    pub request_id: Option<String>,
    pub stage: String,
    pub planner_stage: String,
    pub working_context_present: bool,
    pub last_tool_result_present: bool,
    pub selected_route: Option<String>,
    pub context_ref: Option<String>,
    pub planner_model: Option<String>,
    pub planner_duration_ms: Option<u64>,
    pub planner_failure_reason: Option<String>,
    pub planner_confidence: Option<f32>,
    pub policy_action: Option<String>,
    pub fallback_policy: Option<String>,
    pub fallback_reason: Option<String>,
    pub planner_empty: bool,
    pub used_context_boundary_fallback: bool,
    pub normal_chat_context_injected: bool,
    pub normal_chat_bypassed_tool_router: Option<bool>,
    pub normal_chat_policy_action: Option<String>,
    pub normal_chat_policy_reason: Option<String>,
    pub normal_chat_direct_confidence_threshold: Option<f32>,
    pub normal_chat_accepted_directly: Option<bool>,
    pub planner_intent_kind: Option<String>,
    pub planner_capability_family: Option<String>,
    pub planner_requires_tool_arbitration: Option<bool>,
    pub planner_requires_memory_lookup: Option<bool>,
    pub planner_requires_governed_action: Option<bool>,
    pub planner_requires_context_boundary: Option<bool>,
    pub planner_safe_to_bypass_tools: Option<bool>,
    pub tool_router_invoked_reason: Option<String>,
    pub needs_tool_policy_action: Option<String>,
    pub needs_tool_policy_reason: Option<String>,
    pub needs_tool_confidence_threshold: Option<f32>,
    pub needs_tool_accepted: Option<bool>,
    pub tool_affinity_source: Option<String>,
    pub accepted_decision: bool,
    pub prompt_char_count: usize,
    pub prompt_budget_exceeded: bool,
    pub used_full_router: bool,
    pub tool_affinity_risk: Option<bool>,
    pub context_salience_score: Option<f32>,
    pub context_turn_age: Option<u32>,
    pub context_stale: Option<bool>,
    pub context_decay_action: Option<String>,
    pub context_continuation_policy_action: Option<String>,
    pub context_continuation_policy_reason: Option<String>,
    pub context_answer_first_attempted: Option<bool>,
    pub context_answer_fallback_used: Option<bool>,
    pub context_answer_empty_model_content: Option<bool>,
    pub expected_language: Option<String>,
    pub output_language: Option<String>,
    pub language_mismatch: Option<bool>,
    pub language_retry_attempted: Option<bool>,
    pub language_retry_succeeded: Option<bool>,
    pub budget_compaction_applied: Option<bool>,
    pub user_facing_context_label: Option<String>,
    pub sanitized_internal_context_refs: bool,
    pub tool_manifest_count: usize,
    pub metadata_only: bool,
}

#[derive(Debug, Clone)]
pub struct OrchestratorPlanAttempt {
    pub decision: ConversationOrchestratorDecision,
    pub diagnostic: AssistantOrchestratorDiagnostic,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlannerParseOutcome {
    pub decision: Option<ConversationOrchestratorDecision>,
    pub failure_reason: Option<String>,
    pub planner_confidence: Option<f32>,
    pub tool_affinity_risk: Option<bool>,
    pub planner_safety: PlannerSafetyMetadata,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContextAnswerOutput {
    pub answer: String,
    pub language: AstraUserLanguage,
    pub status: String,
    pub support: String,
    pub used_context_refs: Vec<String>,
    pub confidence: f32,
    pub warnings: Vec<String>,
    pub sanitized_internal_context_refs: bool,
}

#[derive(Debug, Clone)]
pub struct ContextAnswerAttempt {
    pub output: Option<ContextAnswerOutput>,
    pub diagnostic: AssistantOrchestratorDiagnostic,
}

impl AstraUserLanguage {
    pub fn code(self) -> &'static str {
        match self {
            Self::Italian => "it",
            Self::English => "en",
            Self::Mixed => "mixed",
            Self::Unknown => "unknown",
        }
    }

    pub fn instruction(self) -> &'static str {
        match self {
            Self::Italian => {
                "The user message language is Italian. All user-facing strings in the JSON answer MUST be in Italian. Do not answer in English unless the user explicitly asks for English."
            }
            Self::English => {
                "The user message language is English. All user-facing strings in the JSON answer MUST be in English."
            }
            Self::Mixed | Self::Unknown => {
                "Use the dominant language of the user message. If unclear, use the established conversation language; otherwise use concise Italian as the local default."
            }
        }
    }

    fn from_code(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "it" | "italian" | "italiano" => Some(Self::Italian),
            "en" | "english" | "inglese" => Some(Self::English),
            "mixed" | "misto" => Some(Self::Mixed),
            "unknown" | "unk" | "none" => Some(Self::Unknown),
            _ => None,
        }
    }
}

impl OrchestratorPolicyAction {
    pub fn label(&self) -> &'static str {
        match self {
            Self::AcceptDecision => "accept_decision",
            Self::UseFullToolRouter { .. } => "use_full_tool_router",
            Self::DeferToFullToolRouter { .. } => "defer_to_full_tool_router",
            Self::DowngradeToNormalChatWithContext { .. } => {
                "downgrade_to_normal_chat_with_context"
            }
            Self::DowngradeToContextBoundary { .. } => "downgrade_to_context_boundary",
            Self::SafeClarify { .. } => "safe_clarify",
        }
    }

    pub fn fallback_reason(&self) -> Option<&OrchestratorFallbackReason> {
        match self {
            Self::AcceptDecision
            | Self::DowngradeToNormalChatWithContext { .. }
            | Self::DowngradeToContextBoundary { .. } => None,
            Self::UseFullToolRouter { reason }
            | Self::DeferToFullToolRouter { reason }
            | Self::SafeClarify { reason } => Some(reason),
        }
    }

    pub fn needs_tool_reason(&self) -> Option<&NeedsToolPolicyReason> {
        match self {
            Self::DowngradeToNormalChatWithContext { reason }
            | Self::DowngradeToContextBoundary { reason } => Some(reason),
            _ => None,
        }
    }
}

impl OrchestratorFallbackReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NormalChatLowConfidence => "NormalChatLowConfidence",
            Self::NormalChatUnsafeToBypassTools => "NormalChatUnsafeToBypassTools",
            Self::AnswerFromContextLowConfidence => "AnswerFromContextLowConfidence",
            Self::ClarifyLowConfidence => "ClarifyLowConfidence",
            Self::RefuseLowConfidence => "RefuseLowConfidence",
            Self::MissingContextForContextAnswer => "MissingContextForContextAnswer",
            Self::PlannerMalformed => "PlannerMalformed",
            Self::PlannerEmpty => "PlannerEmpty",
            Self::PlannerTimeout => "PlannerTimeout",
            Self::PlannerEmptyNoGroundedContext => "planner_empty_no_grounded_context",
            Self::PlannerFailureNoGroundedContext => "planner_failure_no_grounded_context",
            Self::PlannerUnavailable => "PlannerUnavailable",
            Self::ToolAffinityUnresolved => "ToolAffinityUnresolved",
            Self::NoLastToolResult => "NoLastToolResult",
        }
    }
}

impl ToolAffinitySignal {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::PlannerHighConfidence => "planner_high_confidence",
            Self::ExplicitSlashCommand => "explicit_slash_command",
            Self::UiAction => "ui_action",
            Self::PendingToolContinuation => "pending_tool_continuation",
            Self::FullRouterResult => "full_router_result",
        }
    }
}

impl NeedsToolPolicyReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PlannerHighConfidence => "planner_high_confidence",
            Self::ExplicitSlashCommand => "explicit_slash_command",
            Self::ExplicitUserAction => "explicit_user_action",
            Self::UiAction => "ui_action",
            Self::PendingToolContinuation => "pending_tool_continuation",
            Self::FullRouterResult => "full_router_result",
            Self::LowConfidenceWithLastToolResult => "low_confidence_with_last_tool_result",
            Self::LowConfidenceWithoutToolAffinity => "low_confidence_without_tool_affinity",
            Self::MissingPlannerConfidence => "missing_planner_confidence",
        }
    }
}

impl PlannerIntentKind {
    fn from_raw(value: Option<&str>) -> Option<Self> {
        value.map(|value| match normalize_route(value).as_str() {
            "ordinaryquestion" => Self::OrdinaryQuestion,
            "casualchat" => Self::CasualChat,
            "generalknowledge" => Self::GeneralKnowledge,
            "sessionmemoryquery" => Self::SessionMemoryQuery,
            "governedaction" => Self::GovernedAction,
            "contextfollowup" => Self::ContextFollowup,
            "contextboundary" => Self::ContextBoundary,
            "unknown" => Self::Unknown,
            _ => Self::Unknown,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::OrdinaryQuestion => "ordinary_question",
            Self::CasualChat => "casual_chat",
            Self::GeneralKnowledge => "general_knowledge",
            Self::SessionMemoryQuery => "session_memory_query",
            Self::GovernedAction => "governed_action",
            Self::ContextFollowup => "context_followup",
            Self::ContextBoundary => "context_boundary",
            Self::Unknown => "unknown",
        }
    }

    fn direct_normal_chat_allowed(self) -> bool {
        matches!(
            self,
            Self::OrdinaryQuestion | Self::CasualChat | Self::GeneralKnowledge
        )
    }
}

impl PlannerCapabilityFamily {
    fn from_raw(value: Option<&str>) -> Option<Self> {
        value.map(|value| match normalize_route(value).as_str() {
            "none" => Self::None,
            "worksession" => Self::WorkSession,
            "meeting" => Self::Meeting,
            "screencontext" => Self::ScreenContext,
            "sessionmemory" => Self::SessionMemory,
            "unknown" => Self::Unknown,
            _ => Self::Unknown,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::WorkSession => "work_session",
            Self::Meeting => "meeting",
            Self::ScreenContext => "screen_context",
            Self::SessionMemory => "session_memory",
            Self::Unknown => "unknown",
        }
    }
}

impl PlannerSafetyMetadata {
    fn from_raw(raw: &RawPlannerOutput) -> Self {
        Self {
            intent_kind: PlannerIntentKind::from_raw(raw.intent_kind.as_deref()),
            capability_family: PlannerCapabilityFamily::from_raw(raw.capability_family.as_deref()),
            requires_tool_arbitration: raw.requires_tool_arbitration,
            requires_memory_lookup: raw.requires_memory_lookup,
            requires_governed_action: raw.requires_governed_action,
            requires_context_boundary: raw.requires_context_boundary,
            safe_to_bypass_tools: raw.safe_to_bypass_tools,
        }
    }
}

impl NormalChatArbitrationAction {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AcceptDirectNormalChat => "accept_direct_normal_chat",
            Self::VerifyWithToolRouter => "verify_with_tool_router",
        }
    }
}

impl NormalChatArbitrationReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SafeToBypassTools => "safe_to_bypass_tools",
            Self::LowConfidence => "low_confidence",
            Self::MissingSafetyFields => "missing_safety_fields",
            Self::UnsafeToBypassTools => "unsafe_to_bypass_tools",
            Self::UnknownIntentKind => "unknown_intent_kind",
            Self::UnknownCapabilityFamily => "unknown_capability_family",
            Self::CapabilityRequiresTools => "capability_requires_tools",
            Self::RequiresToolArbitration => "requires_tool_arbitration",
            Self::RequiresMemoryLookup => "requires_memory_lookup",
            Self::RequiresGovernedAction => "requires_governed_action",
            Self::RequiresContextBoundary => "requires_context_boundary",
            Self::ToolAffinityRisk => "tool_affinity_risk",
            Self::PendingGovernedToolFlow => "pending_governed_tool_flow",
            Self::ExplicitUiAction => "explicit_ui_action",
            Self::SlashCommand => "slash_command",
            Self::ContextReference => "context_reference",
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawPlannerOutput {
    route: Option<String>,
    context_ref: Option<String>,
    confidence: Option<f32>,
    tool_affinity_risk: Option<bool>,
    safe_to_bypass_tools: Option<bool>,
    intent_kind: Option<String>,
    capability_family: Option<String>,
    requires_governed_action: Option<bool>,
    requires_memory_lookup: Option<bool>,
    requires_context_boundary: Option<bool>,
    requires_tool_arbitration: Option<bool>,
    reason_code: Option<String>,
    answer_plan: Option<RawAnswerPlan>,
}

#[derive(Debug, Deserialize)]
struct RawAnswerPlan {
    strategy: Option<String>,
    focus: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawContextAnswerOutput {
    answer: Option<String>,
    language: Option<String>,
    status: Option<String>,
    support: Option<String>,
    #[serde(default)]
    used_context_refs: Vec<String>,
    confidence: Option<f32>,
    #[serde(default)]
    warnings: Vec<String>,
}

impl Default for WorkingContextFrame {
    fn default() -> Self {
        Self {
            current_topic: None,
            active_entities: Vec::new(),
            last_user_goal_summary: None,
            last_assistant_answer_summary: None,
            last_assistant_action: None,
            last_referenced_session: None,
            last_referenced_artifacts: Vec::new(),
            last_tool_result: None,
            available_evidence_refs: Vec::new(),
            unresolved_followups: Vec::new(),
            pending_governed_action: None,
            salience: ContextSalience::default(),
            confidence: 0.0,
            updated_at_ms: now_ms(),
        }
    }
}

impl Default for ContextSalience {
    fn default() -> Self {
        Self {
            turn_age: 0,
            normal_chat_turns_since_update: 0,
            last_reinforced_at_ms: None,
            salience_score: 0.0,
            stale: false,
        }
    }
}

impl ContextSalience {
    fn reset_for_tool_result(&mut self) {
        self.turn_age = 0;
        self.normal_chat_turns_since_update = 0;
        self.last_reinforced_at_ms = Some(now_ms());
        self.salience_score = 1.0;
        self.stale = false;
    }

    fn reinforce_from_context_answer(&mut self) {
        self.turn_age = self.turn_age.saturating_add(1);
        self.normal_chat_turns_since_update = 0;
        self.last_reinforced_at_ms = Some(now_ms());
        self.salience_score = self.salience_score.clamp(0.85, 1.0);
        self.stale = self.turn_age >= CONTEXT_SALIENCE_STALE_TURNS;
    }

    fn decay_from_context_boundary(&mut self) {
        self.turn_age = self.turn_age.saturating_add(1);
        self.normal_chat_turns_since_update = self.normal_chat_turns_since_update.saturating_add(1);
        self.salience_score = (self.salience_score * 0.82).max(0.0);
        self.stale = self.normal_chat_turns_since_update >= CONTEXT_SALIENCE_STALE_NORMAL_TURNS
            || self.turn_age >= CONTEXT_SALIENCE_STALE_TURNS
            || self.salience_score < MIN_CONTEXT_SALIENCE_SCORE;
    }

    fn decay_from_normal_chat(&mut self) {
        self.turn_age = self.turn_age.saturating_add(1);
        self.normal_chat_turns_since_update = self.normal_chat_turns_since_update.saturating_add(1);
        self.salience_score = (self.salience_score * 0.72).max(0.0);
        self.stale = self.normal_chat_turns_since_update >= CONTEXT_SALIENCE_STALE_NORMAL_TURNS
            || self.turn_age >= CONTEXT_SALIENCE_STALE_TURNS
            || self.salience_score < MIN_CONTEXT_SALIENCE_SCORE;
    }

    pub fn is_usable(&self) -> bool {
        !self.stale && self.salience_score >= MIN_CONTEXT_SALIENCE_SCORE
    }
}

impl WorkingContextFrame {
    pub fn has_working_context(&self) -> bool {
        (self.last_tool_result.is_some() && self.salience.is_usable())
            || self.current_topic.is_some()
            || self.last_assistant_answer_summary.is_some()
            || self.pending_governed_action.is_some()
    }

    pub fn last_tool_result_usable(&self) -> bool {
        self.last_tool_result.is_some() && self.salience.is_usable()
    }

    pub fn update_from_tool_result(&mut self, tool_result: ToolResultFrame) {
        self.current_topic = tool_result
            .key_topics
            .first()
            .cloned()
            .and_then(non_empty)
            .or_else(|| non_empty(tool_result.answer_summary.clone()));
        self.active_entities = tool_result.active_entities.clone();
        self.last_assistant_action = Some(tool_result.tool_name.clone());
        self.last_assistant_answer_summary = Some(tool_result.answer_summary.clone());
        self.last_referenced_session =
            tool_result
                .session_id
                .as_ref()
                .map(|session_id| SessionReference {
                    session_id: session_id.clone(),
                    source_kind: tool_result.source_kind.clone(),
                    source_label: tool_result.source_label.clone(),
                    title: None,
                });
        self.available_evidence_refs = tool_result
            .used_evidence_ids
            .iter()
            .map(|evidence_id| EvidenceReference {
                evidence_id: evidence_id.clone(),
                evidence_kind: "transcript".to_string(),
                source_kind: tool_result.source_kind.clone(),
            })
            .collect();
        self.confidence = tool_result.confidence.unwrap_or(0.72).clamp(0.0, 1.0);
        self.updated_at_ms = now_ms();
        self.salience.reset_for_tool_result();
        self.last_tool_result = Some(tool_result);
    }

    pub fn update_from_normal_chat(&mut self, user_message: &str, assistant_answer: &str) {
        self.last_user_goal_summary = non_empty(context_broker::bounded_text(user_message, 220));
        self.last_assistant_answer_summary =
            non_empty(context_broker::bounded_text(assistant_answer, 360));
        self.last_assistant_action = Some("normal_chat".to_string());
        self.updated_at_ms = now_ms();
        if self.confidence > 0.15 {
            self.confidence = (self.confidence * 0.9).max(0.15);
        }
        if self.last_tool_result.is_some() {
            self.salience.decay_from_normal_chat();
        }
    }

    pub fn update_from_context_answer(&mut self, user_message: &str, answer: &str) {
        self.last_user_goal_summary = non_empty(context_broker::bounded_text(user_message, 220));
        self.last_assistant_answer_summary = non_empty(context_broker::bounded_text(answer, 360));
        self.last_assistant_action = Some("answer_from_context".to_string());
        self.updated_at_ms = now_ms();
        if self.last_tool_result.is_some() {
            match classify_context_attachment(user_message, self) {
                ContextAttachmentKind::ExplicitEvidenceReference
                | ContextAttachmentKind::TopicPresenceCheck
                | ContextAttachmentKind::ContextDetailExpansion => {
                    self.salience.reinforce_from_context_answer();
                }
                ContextAttachmentKind::BoundaryGeneralKnowledge
                | ContextAttachmentKind::TopicOverlapOnly
                | ContextAttachmentKind::ToolBoundRequest
                | ContextAttachmentKind::Unrelated
                | ContextAttachmentKind::Ambiguous => {
                    self.salience.decay_from_context_boundary();
                }
            }
        }
    }
}

impl ToolResultFrame {
    #[allow(clippy::too_many_arguments)]
    pub fn compact(
        tool_name: impl Into<String>,
        answer_kind: impl Into<String>,
        source_kind: impl Into<String>,
        source_label: impl Into<String>,
        session_id: Option<String>,
        used_evidence_ids: Vec<String>,
        evidence_count: usize,
        answer_summary: impl Into<String>,
        warnings: Vec<String>,
        confidence: Option<f32>,
    ) -> Self {
        let raw_summary = answer_summary.into();
        let summary =
            context_broker::bounded_text(&sanitize_tool_result_answer_summary(&raw_summary), 700);
        let key_topics = extract_compact_topics(&summary, 8);
        let active_entities = extract_compact_entities(&summary, 8);
        Self {
            tool_name: tool_name.into(),
            answer_kind: answer_kind.into(),
            source_kind: source_kind.into(),
            source_label: source_label.into(),
            session_id,
            used_evidence_ids: dedupe_limited(used_evidence_ids, 24),
            evidence_count,
            answer_summary: summary,
            key_topics,
            active_entities,
            warnings: dedupe_limited(warnings, 8),
            confidence: confidence.map(|value| value.clamp(0.0, 1.0)),
            created_at_ms: now_ms(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextEvidenceSupport {
    Supported { matched_terms: Vec<String> },
    NotSupported { checked_terms: Vec<String> },
    Ambiguous { reason: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextAttachmentKind {
    ExplicitEvidenceReference,
    TopicPresenceCheck,
    ContextDetailExpansion,
    BoundaryGeneralKnowledge,
    TopicOverlapOnly,
    ToolBoundRequest,
    Unrelated,
    Ambiguous,
}

impl ContextAttachmentKind {
    fn as_reason_code(self) -> &'static str {
        match self {
            Self::ExplicitEvidenceReference => "explicit_evidence_reference",
            Self::TopicPresenceCheck => "topic_presence_check",
            Self::ContextDetailExpansion => "context_detail_expansion",
            Self::BoundaryGeneralKnowledge => "boundary_general_knowledge",
            Self::TopicOverlapOnly => "topic_overlap_only",
            Self::ToolBoundRequest => "tool_bound_request",
            Self::Unrelated => "unrelated",
            Self::Ambiguous => "ambiguous",
        }
    }

    fn should_answer_from_context(self) -> bool {
        matches!(
            self,
            Self::ExplicitEvidenceReference
                | Self::TopicPresenceCheck
                | Self::ContextDetailExpansion
        )
    }
}

pub(crate) fn sanitize_tool_result_answer_summary(input: &str) -> String {
    input
        .lines()
        .filter_map(sanitize_tool_result_summary_line)
        .collect::<Vec<_>>()
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

fn sanitize_tool_result_summary_line(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    let lower = trimmed.to_ascii_lowercase();
    let operational_prefix = [
        "fonte:",
        "source:",
        "nota:",
        "note:",
        "evidenze usate:",
        "evidenze disponibili:",
        "evidence used:",
        "available evidence:",
        "stt completeness:",
        "contesto usato:",
        "context used:",
    ]
    .iter()
    .any(|prefix| lower.starts_with(prefix));
    let internal_label = [
        "context_answer_synthesizer_fallback",
        "stt completeness:",
        "toolresultframe",
        "evidencereference",
        "metadata_only",
        "raw_model_output",
        "audit",
        "diagnostic",
    ]
    .iter()
    .any(|marker| lower.contains(marker));
    if operational_prefix || internal_label {
        return None;
    }
    let cleaned = trimmed
        .split_whitespace()
        .filter(|token| !is_segment_reference_token(token) && !is_uuidish_token(token))
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .trim_matches(|ch: char| matches!(ch, ',' | ';' | ':'))
        .to_string();
    (!cleaned.is_empty()).then_some(cleaned)
}

fn is_segment_reference_token(token: &str) -> bool {
    token
        .trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != ':' && ch != '-')
        .to_ascii_lowercase()
        .starts_with("segment:")
}

fn is_uuidish_token(token: &str) -> bool {
    let token = token.trim_matches(|ch: char| !ch.is_ascii_hexdigit() && ch != '-');
    let hex_count = token.chars().filter(|ch| ch.is_ascii_hexdigit()).count();
    let hyphen_count = token.chars().filter(|ch| *ch == '-').count();
    hex_count >= 32 && hyphen_count >= 4
}

pub fn extract_context_query_terms(user_message: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    normalized_context_tokens(user_message)
        .into_iter()
        .filter(|token| is_strong_context_query_term(token))
        .filter(|token| seen.insert(token.clone()))
        .take(8)
        .collect()
}

pub fn score_context_evidence_support(
    user_message: &str,
    tool_result: &ToolResultFrame,
) -> ContextEvidenceSupport {
    if looks_like_open_context_boundary_query(user_message) {
        return ContextEvidenceSupport::Ambiguous {
            reason: "open_boundary_question".to_string(),
        };
    }
    let terms = extract_context_query_terms(user_message);
    if terms.is_empty() {
        return ContextEvidenceSupport::Ambiguous {
            reason: "no_strong_query_terms".to_string(),
        };
    }
    let context_text = normalized_tool_result_context_text(tool_result);
    let evidence_tokens = normalized_context_tokens(&context_text)
        .into_iter()
        .collect::<HashSet<_>>();
    let matched_terms = terms
        .iter()
        .filter(|term| context_term_matches_evidence(term, &evidence_tokens))
        .cloned()
        .collect::<Vec<_>>();
    if matched_terms.is_empty() {
        ContextEvidenceSupport::NotSupported {
            checked_terms: terms,
        }
    } else {
        ContextEvidenceSupport::Supported { matched_terms }
    }
}

pub fn normalized_tool_result_context_text(tool_result: &ToolResultFrame) -> String {
    let mut parts = vec![sanitize_tool_result_answer_summary(
        &tool_result.answer_summary,
    )];
    parts.extend(
        tool_result
            .key_topics
            .iter()
            .take(8)
            .map(|value| context_broker::bounded_text(value, 90)),
    );
    parts.extend(
        tool_result
            .active_entities
            .iter()
            .take(8)
            .map(|value| context_broker::bounded_text(value, 90)),
    );
    context_broker::bounded_text(&parts.join(" "), 1_200)
}

fn normalized_context_tokens(input: &str) -> Vec<String> {
    normalize_context_text(input)
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

fn normalize_context_text(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    for ch in input.chars() {
        let normalized = match ch {
            'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å' => {
                'a'
            }
            'è' | 'é' | 'ê' | 'ë' | 'È' | 'É' | 'Ê' | 'Ë' => 'e',
            'ì' | 'í' | 'î' | 'ï' | 'Ì' | 'Í' | 'Î' | 'Ï' => 'i',
            'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' => 'o',
            'ù' | 'ú' | 'û' | 'ü' | 'Ù' | 'Ú' | 'Û' | 'Ü' => 'u',
            'ç' | 'Ç' => 'c',
            other if other.is_alphanumeric() => {
                for lower in other.to_lowercase() {
                    output.push(lower);
                }
                continue;
            }
            _ => ' ',
        };
        output.push(normalized);
    }
    output
}

fn is_strong_context_query_term(token: &str) -> bool {
    if token.chars().count() < 3 {
        return false;
    }
    !CONTEXT_QUERY_STOPWORDS.contains(&token)
}

fn looks_like_open_context_boundary_query(user_message: &str) -> bool {
    let tokens = normalized_context_tokens(user_message);
    let has_open_marker = tokens.iter().any(|token| {
        CONTEXT_OPEN_QUESTION_MARKERS
            .iter()
            .any(|marker| token == marker)
    });
    let has_presence_marker = tokens.iter().any(|token| {
        CONTEXT_PRESENCE_MARKERS
            .iter()
            .any(|marker| token == marker)
    });
    has_open_marker && !has_presence_marker
}

fn context_term_matches_evidence(term: &str, evidence_tokens: &HashSet<String>) -> bool {
    evidence_tokens.contains(term)
        || context_term_variants(term)
            .iter()
            .any(|variant| evidence_tokens.contains(variant))
}

fn context_term_variants(term: &str) -> Vec<String> {
    let mut variants = Vec::new();
    if let Some(stem) = term.strip_suffix('s') {
        variants.push(stem.to_string());
    }
    if let Some(stem) = term.strip_suffix('i') {
        variants.push(format!("{stem}o"));
        variants.push(format!("{stem}e"));
    }
    if let Some(stem) = term.strip_suffix('e') {
        variants.push(format!("{stem}a"));
        variants.push(format!("{stem}o"));
    }
    variants.push(format!("{term}s"));
    variants.push(format!("{term}i"));
    variants.sort();
    variants.dedup();
    variants
}


fn context_request_requires_tool_arbitration(user_message: &str) -> bool {
    let tokens = normalized_context_tokens(user_message);
    if tokens.is_empty() {
        return false;
    }
    let has_governed_lifecycle = tokens_contain_any(&tokens, CONTEXT_GOVERNED_ACTION_MARKERS);
    let has_screen_context_target = tokens_contain_any(&tokens, CONTEXT_SCREEN_CONTEXT_TARGET_MARKERS);
    let has_screen_context_action = tokens_contain_any(&tokens, CONTEXT_SCREEN_CONTEXT_ACTION_MARKERS);
    let has_session_target = tokens_contain_any(&tokens, CONTEXT_TOOL_SESSION_TARGET_MARKERS);
    let has_generation_action = tokens_contain_any(&tokens, CONTEXT_TOOL_GENERATION_ACTION_MARKERS);
    let has_intelligence_artifact = tokens_contain_any(&tokens, CONTEXT_TOOL_ARTIFACT_MARKERS);

    has_governed_lifecycle
        || (has_screen_context_target && has_screen_context_action)
        || (has_session_target && (has_generation_action || has_intelligence_artifact))
}

fn classify_context_attachment(
    user_message: &str,
    working_context: &WorkingContextFrame,
) -> ContextAttachmentKind {
    if !working_context.last_tool_result_usable() {
        return ContextAttachmentKind::Unrelated;
    }
    let Some(tool_result) = working_context.last_tool_result.as_ref() else {
        return ContextAttachmentKind::Unrelated;
    };
    let tokens = normalized_context_tokens(user_message);
    if tokens.is_empty() {
        return ContextAttachmentKind::Ambiguous;
    }
    if context_request_requires_tool_arbitration(user_message) {
        return ContextAttachmentKind::ToolBoundRequest;
    }

    let has_presence_marker = tokens_contain_any(&tokens, CONTEXT_PRESENCE_MARKERS);
    let has_explicit_evidence_reference =
        tokens_contain_any(&tokens, CONTEXT_EXPLICIT_EVIDENCE_MARKERS);
    let has_detail_marker = tokens_contain_any(&tokens, CONTEXT_DETAIL_MARKERS);
    let has_continuation_marker = tokens_contain_any(&tokens, CONTEXT_CONTINUATION_MARKERS);
    let has_open_boundary_marker = looks_like_open_context_boundary_query(user_message);
    let query_terms = extract_context_query_terms(user_message);
    let support = score_context_evidence_support(user_message, tool_result);
    let support_is_supported = matches!(support, ContextEvidenceSupport::Supported { .. });
    let support_is_not_supported = matches!(support, ContextEvidenceSupport::NotSupported { .. });

    if has_presence_marker
        || (has_continuation_marker
            && !has_open_boundary_marker
            && !has_detail_marker
            && !query_terms.is_empty())
    {
        return ContextAttachmentKind::TopicPresenceCheck;
    }

    if has_explicit_evidence_reference && !has_open_boundary_marker {
        return ContextAttachmentKind::ExplicitEvidenceReference;
    }

    if has_detail_marker && (support_is_supported || has_explicit_evidence_reference) {
        return ContextAttachmentKind::ContextDetailExpansion;
    }

    if has_open_boundary_marker {
        return ContextAttachmentKind::BoundaryGeneralKnowledge;
    }

    if support_is_supported {
        return ContextAttachmentKind::TopicOverlapOnly;
    }

    if support_is_not_supported && has_explicit_evidence_reference {
        return ContextAttachmentKind::TopicPresenceCheck;
    }

    if query_terms.is_empty() {
        ContextAttachmentKind::Ambiguous
    } else {
        ContextAttachmentKind::Unrelated
    }
}

fn tokens_contain_any(tokens: &[String], markers: &[&str]) -> bool {
    tokens
        .iter()
        .any(|token| markers.iter().any(|marker| token == marker))
}

fn contextual_chat_plan(reason_code: &str, _user_message: Option<&str>) -> ContextualChatPlan {
    ContextualChatPlan {
        context_ref: "last_tool_result".to_string(),
        reason_code: reason_code.to_string(),
        confidence: 0.72,
    }
}

fn context_attachment_reason(base_reason: &str, attachment: ContextAttachmentKind) -> String {
    format!("{base_reason}_{}", attachment.as_reason_code())
}

const CONTEXT_EXPLICIT_EVIDENCE_MARKERS: &[&str] = &[
    "archive",
    "archiviata",
    "archivio",
    "evidence",
    "evidenze",
    "recap",
    "transcript",
    "trascrizione",
];

const CONTEXT_GOVERNED_ACTION_MARKERS: &[&str] = &[
    "avvia",
    "avviamo",
    "avviare",
    "inizia",
    "iniziamo",
    "start",
    "stop",
    "ferma",
    "fermiamo",
    "termina",
    "terminiamo",
    "registra",
    "registrare",
];


const CONTEXT_SCREEN_CONTEXT_TARGET_MARKERS: &[&str] = &[
    "screen",
    "schermo",
    "screenshot",
    "capture",
    "cattura",
    "immagine",
];

const CONTEXT_SCREEN_CONTEXT_ACTION_MARKERS: &[&str] = &[
    "allega",
    "allegare",
    "attach",
    "aggiungi",
    "aggiungere",
    "collega",
    "collegare",
    "analizza",
    "analizzare",
];

const CONTEXT_TOOL_SESSION_TARGET_MARKERS: &[&str] = &[
    "attuale",
    "active",
    "corrente",
    "current",
    "intelligence",
    "intelligent",
    "meeting",
    "registrata",
    "registrazione",
    "session",
    "sessione",
    "transcript",
    "trascrizione",
    "work",
];

const CONTEXT_TOOL_GENERATION_ACTION_MARKERS: &[&str] = &[
    "crea",
    "creami",
    "elabora",
    "elaborami",
    "fammi",
    "fai",
    "genera",
    "generami",
    "generare",
    "generi",
    "produce",
    "produci",
    "produrre",
    "sintetizza",
    "sintetizzami",
    "riepiloga",
    "riepilogami",
];

const CONTEXT_TOOL_ARTIFACT_MARKERS: &[&str] = &[
    "analysis",
    "analisi",
    "completo",
    "completa",
    "dettagli",
    "dettagliato",
    "intelligence",
    "intelligent",
    "recap",
    "report",
    "riassunto",
    "riepilogo",
    "sintesi",
    "summary",
];

const CONTEXT_DETAIL_MARKERS: &[&str] = &[
    "approfondisci",
    "approfondire",
    "detail",
    "details",
    "dettagli",
    "dettaglio",
    "meglio",
    "parte",
    "spiega",
    "spiegami",
    "spiegare",
    "spieghi",
];

const CONTEXT_CONTINUATION_MARKERS: &[&str] = &[
    "allora",
    "quindi",
    "so",
    "then",
    "therefore",
    "invece",
    "dunque",
];

const CONTEXT_QUERY_STOPWORDS: &[&str] = &[
    "about",
    "abbiamo",
    "after",
    "anche",
    "argomento",
    "argomenti",
    "before",
    "che",
    "cosa",
    "dalla",
    "dalle",
    "degli",
    "dei",
    "del",
    "della",
    "delle",
    "did",
    "discusso",
    "discuss",
    "discussed",
    "era",
    "erano",
    "evidenze",
    "from",
    "have",
    "last",
    "meeting",
    "nella",
    "nelle",
    "parlava",
    "parlavamo",
    "parlato",
    "previous",
    "prima",
    "quindi",
    "recap",
    "recording",
    "riferimento",
    "session",
    "sessione",
    "sul",
    "sulla",
    "talk",
    "talked",
    "the",
    "topic",
    "transcript",
    "was",
    "were",
    "what",
];

const CONTEXT_OPEN_QUESTION_MARKERS: &[&str] = &[
    "before",
    "com",
    "come",
    "cosa",
    "formata",
    "formato",
    "formed",
    "how",
    "nata",
    "nato",
    "origin",
    "origine",
    "perche",
    "prima",
    "processo",
    "what",
    "why",
];

const CONTEXT_PRESENCE_MARKERS: &[&str] = &[
    "about",
    "discusso",
    "discussed",
    "parlava",
    "parlavamo",
    "parlato",
    "riferimento",
    "talked",
    "topic",
];

fn planner_failure_fallback(
    working_context: &WorkingContextFrame,
    reason_code: &str,
    user_message: Option<&str>,
) -> ConversationOrchestratorDecision {
    if working_context.last_tool_result_usable() {
        let attachment = classify_context_attachment(user_message.unwrap_or_default(), working_context);
        let attachment_reason = context_attachment_reason(reason_code, attachment);
        if matches!(attachment, ContextAttachmentKind::ToolBoundRequest) {
            let reason = if reason_code == "empty_model_content" {
                "planner_empty_no_grounded_context"
            } else {
                "planner_failure_no_grounded_context"
            };
            return ConversationOrchestratorDecision::DeferToToolRouter(DeferToToolRouterPlan {
                reason: reason.to_string(),
                planner_failure_reason: Some(reason_code.to_string()),
                confidence: None,
            });
        }
        if attachment.should_answer_from_context() {
            return ConversationOrchestratorDecision::AnswerFromContext(context_continuation_plan(
                &attachment_reason,
                user_message,
            ));
        }
        if matches!(
            attachment,
            ContextAttachmentKind::BoundaryGeneralKnowledge
                | ContextAttachmentKind::TopicOverlapOnly
                | ContextAttachmentKind::Ambiguous
        ) {
            return ConversationOrchestratorDecision::NormalChatWithContext(contextual_chat_plan(
                &attachment_reason,
                user_message,
            ));
        }
    }
    let reason = if reason_code == "empty_model_content" {
        "planner_empty_no_grounded_context"
    } else {
        "planner_failure_no_grounded_context"
    };
    ConversationOrchestratorDecision::DeferToToolRouter(DeferToToolRouterPlan {
        reason: reason.to_string(),
        planner_failure_reason: Some(reason_code.to_string()),
        confidence: None,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ContextContinuationPolicyAction {
    UseContextAnswerFirst,
    UseNormalChatWithContext,
    DeferToToolRouter,
    AskClarification,
    DecayContextAndNormalChat,
}

impl ContextContinuationPolicyAction {
    fn as_str(self) -> &'static str {
        match self {
            Self::UseContextAnswerFirst => "use_context_answer_first",
            Self::UseNormalChatWithContext => "use_normal_chat_with_context",
            Self::DeferToToolRouter => "defer_to_tool_router",
            Self::AskClarification => "ask_clarification",
            Self::DecayContextAndNormalChat => "decay_context_and_normal_chat",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextContinuationPolicyReason {
    PlannerEmptyWithSalientToolResult,
    PlannerFailureWithSalientToolResult,
    LowConfidenceContextualChat,
    PlannerSelectedContextualChat,
    PlannerFailureBoundaryGeneralKnowledge,
    PlannerFailureTopicOverlapOnly,
    PlannerFailureAmbiguousAttachment,
    ToolBoundContinuationRequest,
    PlannerSelectedToolBoundRequest,
    NoSalientToolResult,
    PendingGovernedActionPriority,
}

impl ContextContinuationPolicyReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::PlannerEmptyWithSalientToolResult => "planner_empty_with_salient_tool_result",
            Self::PlannerFailureWithSalientToolResult => "planner_failure_with_salient_tool_result",
            Self::LowConfidenceContextualChat => "low_confidence_contextual_chat",
            Self::PlannerSelectedContextualChat => "planner_selected_contextual_chat",
            Self::PlannerFailureBoundaryGeneralKnowledge => {
                "planner_failure_boundary_general_knowledge"
            }
            Self::PlannerFailureTopicOverlapOnly => "planner_failure_topic_overlap_only",
            Self::PlannerFailureAmbiguousAttachment => "planner_failure_ambiguous_attachment",
            Self::ToolBoundContinuationRequest => "tool_bound_continuation_request",
            Self::PlannerSelectedToolBoundRequest => "planner_selected_tool_bound_request",
            Self::NoSalientToolResult => "no_salient_tool_result",
            Self::PendingGovernedActionPriority => "pending_governed_action_priority",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextContinuationPolicyDecision {
    pub action: ContextContinuationPolicyAction,
    pub reason: ContextContinuationPolicyReason,
}

fn context_continuation_plan(reason_code: &str, user_message: Option<&str>) -> ContextAnswerPlan {
    ContextAnswerPlan {
        strategy: "context_continuation".to_string(),
        focus: user_message
            .and_then(|message| non_empty(context_broker::bounded_text(message, 160))),
        context_ref: "last_tool_result".to_string(),
        reason_code: reason_code.to_string(),
        confidence: MIN_ANSWER_FROM_CONTEXT_CONFIDENCE,
    }
}

fn apply_context_continuation_policy(
    decision: &ConversationOrchestratorDecision,
    working_context: &WorkingContextFrame,
    planner_failure_reason: Option<&str>,
    planner_confidence: Option<f32>,
    user_message: Option<&str>,
) -> Option<ContextContinuationPolicyDecision> {
    if working_context.pending_governed_action.is_some() {
        return Some(ContextContinuationPolicyDecision {
            action: ContextContinuationPolicyAction::UseNormalChatWithContext,
            reason: ContextContinuationPolicyReason::PendingGovernedActionPriority,
        });
    }

    if !working_context.last_tool_result_usable() {
        return match decision {
            ConversationOrchestratorDecision::NormalChatWithContext(_) => {
                Some(ContextContinuationPolicyDecision {
                    action: ContextContinuationPolicyAction::DecayContextAndNormalChat,
                    reason: ContextContinuationPolicyReason::NoSalientToolResult,
                })
            }
            _ => None,
        };
    }

    if let Some(reason) = planner_failure_reason {
        let attachment = classify_context_attachment(user_message.unwrap_or_default(), working_context);
        return Some(match attachment {
            ContextAttachmentKind::ExplicitEvidenceReference
            | ContextAttachmentKind::TopicPresenceCheck
            | ContextAttachmentKind::ContextDetailExpansion => ContextContinuationPolicyDecision {
                action: ContextContinuationPolicyAction::UseContextAnswerFirst,
                reason: if reason == "empty_model_content" {
                    ContextContinuationPolicyReason::PlannerEmptyWithSalientToolResult
                } else {
                    ContextContinuationPolicyReason::PlannerFailureWithSalientToolResult
                },
            },
            ContextAttachmentKind::BoundaryGeneralKnowledge => ContextContinuationPolicyDecision {
                action: ContextContinuationPolicyAction::UseNormalChatWithContext,
                reason: ContextContinuationPolicyReason::PlannerFailureBoundaryGeneralKnowledge,
            },
            ContextAttachmentKind::ToolBoundRequest => ContextContinuationPolicyDecision {
                action: ContextContinuationPolicyAction::DeferToToolRouter,
                reason: ContextContinuationPolicyReason::ToolBoundContinuationRequest,
            },
            ContextAttachmentKind::TopicOverlapOnly => ContextContinuationPolicyDecision {
                action: ContextContinuationPolicyAction::UseNormalChatWithContext,
                reason: ContextContinuationPolicyReason::PlannerFailureTopicOverlapOnly,
            },
            ContextAttachmentKind::Ambiguous => ContextContinuationPolicyDecision {
                action: ContextContinuationPolicyAction::UseNormalChatWithContext,
                reason: ContextContinuationPolicyReason::PlannerFailureAmbiguousAttachment,
            },
            ContextAttachmentKind::Unrelated => return None,
        });
    }

    if let ConversationOrchestratorDecision::AnswerFromContext(plan) = decision {
        if let Some(focus) = plan.focus.as_deref() {
            let attachment = classify_context_attachment(user_message.unwrap_or(focus), working_context);
            if matches!(attachment, ContextAttachmentKind::ToolBoundRequest) {
                return Some(ContextContinuationPolicyDecision {
                    action: ContextContinuationPolicyAction::DeferToToolRouter,
                    reason: ContextContinuationPolicyReason::PlannerSelectedToolBoundRequest,
                });
            }
            if matches!(
                attachment,
                ContextAttachmentKind::BoundaryGeneralKnowledge
                    | ContextAttachmentKind::TopicOverlapOnly
            ) {
                return Some(ContextContinuationPolicyDecision {
                    action: ContextContinuationPolicyAction::UseNormalChatWithContext,
                    reason: match attachment {
                        ContextAttachmentKind::BoundaryGeneralKnowledge => {
                            ContextContinuationPolicyReason::PlannerFailureBoundaryGeneralKnowledge
                        }
                        _ => ContextContinuationPolicyReason::PlannerFailureTopicOverlapOnly,
                    },
                });
            }
        }
    }

    if let ConversationOrchestratorDecision::NormalChatWithContext(plan) = decision {
        if planner_confidence.unwrap_or(plan.confidence) <= 0.0 {
            let attachment = classify_context_attachment(user_message.unwrap_or_default(), working_context);
            if matches!(attachment, ContextAttachmentKind::ToolBoundRequest) {
                return Some(ContextContinuationPolicyDecision {
                    action: ContextContinuationPolicyAction::DeferToToolRouter,
                    reason: ContextContinuationPolicyReason::ToolBoundContinuationRequest,
                });
            }
            if attachment.should_answer_from_context() {
                return Some(ContextContinuationPolicyDecision {
                    action: ContextContinuationPolicyAction::UseContextAnswerFirst,
                    reason: ContextContinuationPolicyReason::LowConfidenceContextualChat,
                });
            }
            return Some(ContextContinuationPolicyDecision {
                action: ContextContinuationPolicyAction::UseNormalChatWithContext,
                reason: match attachment {
                    ContextAttachmentKind::BoundaryGeneralKnowledge => {
                        ContextContinuationPolicyReason::PlannerFailureBoundaryGeneralKnowledge
                    }
                    ContextAttachmentKind::TopicOverlapOnly => {
                        ContextContinuationPolicyReason::PlannerFailureTopicOverlapOnly
                    }
                    ContextAttachmentKind::ToolBoundRequest => {
                        ContextContinuationPolicyReason::ToolBoundContinuationRequest
                    }
                    _ => ContextContinuationPolicyReason::PlannerFailureAmbiguousAttachment,
                },
            });
        }
        let attachment = classify_context_attachment(user_message.unwrap_or_default(), working_context);
        if matches!(attachment, ContextAttachmentKind::ToolBoundRequest) {
            return Some(ContextContinuationPolicyDecision {
                action: ContextContinuationPolicyAction::DeferToToolRouter,
                reason: ContextContinuationPolicyReason::PlannerSelectedToolBoundRequest,
            });
        }
        return Some(ContextContinuationPolicyDecision {
            action: ContextContinuationPolicyAction::UseNormalChatWithContext,
            reason: ContextContinuationPolicyReason::PlannerSelectedContextualChat,
        });
    }

    None
}

pub async fn plan_with_active_model(
    source: &str,
    user_message: &str,
    history: &[ConversationMessage],
    working_context: &WorkingContextFrame,
) -> OrchestratorPlanAttempt {
    let prompt =
        context_broker::build_context_planner_messages(user_message, history, working_context);
    let started = Instant::now();
    let model = resolve_active_ollama_model(user_message, source).await;
    let mut diagnostic = AssistantOrchestratorDiagnostic {
        request_id: None,
        stage: "context_planner".to_string(),
        planner_stage: "discourse_planner".to_string(),
        working_context_present: working_context.has_working_context(),
        last_tool_result_present: working_context.last_tool_result.is_some(),
        selected_route: None,
        context_ref: None,
        planner_model: Some(model.clone()),
        planner_duration_ms: None,
        planner_failure_reason: None,
        planner_confidence: None,
        policy_action: None,
        fallback_policy: None,
        fallback_reason: None,
        planner_empty: false,
        used_context_boundary_fallback: false,
        normal_chat_context_injected: false,
        normal_chat_bypassed_tool_router: None,
        normal_chat_policy_action: None,
        normal_chat_policy_reason: None,
        normal_chat_direct_confidence_threshold: None,
        normal_chat_accepted_directly: None,
        planner_intent_kind: None,
        planner_capability_family: None,
        planner_requires_tool_arbitration: None,
        planner_requires_memory_lookup: None,
        planner_requires_governed_action: None,
        planner_requires_context_boundary: None,
        planner_safe_to_bypass_tools: None,
        tool_router_invoked_reason: None,
        needs_tool_policy_action: None,
        needs_tool_policy_reason: None,
        needs_tool_confidence_threshold: None,
        needs_tool_accepted: None,
        tool_affinity_source: None,
        accepted_decision: false,
        prompt_char_count: prompt.prompt_char_count,
        prompt_budget_exceeded: prompt.prompt_budget_exceeded,
        used_full_router: false,
        tool_affinity_risk: None,
        context_salience_score: salience_score_for_diagnostic(working_context),
        context_turn_age: salience_turn_age_for_diagnostic(working_context),
        context_stale: salience_stale_for_diagnostic(working_context),
        context_decay_action: None,
        context_continuation_policy_action: None,
        context_continuation_policy_reason: None,
        context_answer_first_attempted: None,
        context_answer_fallback_used: None,
        context_answer_empty_model_content: None,
        expected_language: None,
        output_language: None,
        language_mismatch: None,
        language_retry_attempted: None,
        language_retry_succeeded: None,
        budget_compaction_applied: Some(prompt.budget_compaction_applied),
        user_facing_context_label: None,
        sanitized_internal_context_refs: false,
        tool_manifest_count: prompt.tool_manifest_count,
        metadata_only: true,
    };

    let fallback = planner_failure_fallback(working_context, "planner_failure", Some(user_message));
    let client = match Client::builder()
        .timeout(std::time::Duration::from_millis(8_000))
        .build()
    {
        Ok(client) => client,
        Err(_) => {
            diagnostic.planner_failure_reason = Some("endpoint_config".to_string());
            diagnostic.selected_route = Some(decision_route_label(&fallback).to_string());
            diagnostic.context_ref = decision_context_ref(&fallback).map(str::to_string);
            diagnostic.used_full_router = matches!(
                fallback,
                ConversationOrchestratorDecision::ToolCall(_)
                    | ConversationOrchestratorDecision::DeferToToolRouter(_)
            );
            diagnostic.normal_chat_context_injected = matches!(
                fallback,
                ConversationOrchestratorDecision::NormalChatWithContext(_)
            );
            diagnostic.planner_duration_ms = Some(started.elapsed().as_millis() as u64);
            return OrchestratorPlanAttempt {
                decision: fallback,
                diagnostic,
            };
        }
    };

    let response = match client
        .post(ollama_endpoint("/api/chat"))
        .json(&serde_json::json!({
            "model": model,
            "stream": false,
            "format": "json",
            "messages": prompt.messages,
            "options": {
                "temperature": 0.0,
                "top_p": 0.4,
                "repeat_penalty": 1.04,
                "num_predict": 220
            },
            "keep_alive": "30m"
        }))
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) if error.is_timeout() => {
            diagnostic.planner_failure_reason = Some("timeout".to_string());
            diagnostic.selected_route = Some(decision_route_label(&fallback).to_string());
            diagnostic.context_ref = decision_context_ref(&fallback).map(str::to_string);
            diagnostic.used_full_router = matches!(
                fallback,
                ConversationOrchestratorDecision::ToolCall(_)
                    | ConversationOrchestratorDecision::DeferToToolRouter(_)
            );
            diagnostic.normal_chat_context_injected = matches!(
                fallback,
                ConversationOrchestratorDecision::NormalChatWithContext(_)
            );
            diagnostic.planner_duration_ms = Some(started.elapsed().as_millis() as u64);
            return OrchestratorPlanAttempt {
                decision: fallback,
                diagnostic,
            };
        }
        Err(_) => {
            diagnostic.planner_failure_reason = Some("ollama_unavailable".to_string());
            diagnostic.selected_route = Some(decision_route_label(&fallback).to_string());
            diagnostic.context_ref = decision_context_ref(&fallback).map(str::to_string);
            diagnostic.used_full_router = matches!(
                fallback,
                ConversationOrchestratorDecision::ToolCall(_)
                    | ConversationOrchestratorDecision::DeferToToolRouter(_)
            );
            diagnostic.normal_chat_context_injected = matches!(
                fallback,
                ConversationOrchestratorDecision::NormalChatWithContext(_)
            );
            diagnostic.planner_duration_ms = Some(started.elapsed().as_millis() as u64);
            return OrchestratorPlanAttempt {
                decision: fallback,
                diagnostic,
            };
        }
    };
    if !response.status().is_success() {
        diagnostic.planner_failure_reason = Some("ollama_http_error".to_string());
        diagnostic.selected_route = Some(decision_route_label(&fallback).to_string());
        diagnostic.context_ref = decision_context_ref(&fallback).map(str::to_string);
        diagnostic.used_full_router = matches!(
            fallback,
            ConversationOrchestratorDecision::ToolCall(_)
                | ConversationOrchestratorDecision::DeferToToolRouter(_)
        );
        diagnostic.normal_chat_context_injected = matches!(
            fallback,
            ConversationOrchestratorDecision::NormalChatWithContext(_)
        );
        diagnostic.planner_duration_ms = Some(started.elapsed().as_millis() as u64);
        return OrchestratorPlanAttempt {
            decision: fallback,
            diagnostic,
        };
    }

    let body: OllamaChatResponse = match response.json().await {
        Ok(body) => body,
        Err(_) => {
            diagnostic.planner_failure_reason = Some("invalid_response_schema".to_string());
            diagnostic.selected_route = Some(decision_route_label(&fallback).to_string());
            diagnostic.context_ref = decision_context_ref(&fallback).map(str::to_string);
            diagnostic.used_full_router = matches!(
                fallback,
                ConversationOrchestratorDecision::ToolCall(_)
                    | ConversationOrchestratorDecision::DeferToToolRouter(_)
            );
            diagnostic.normal_chat_context_injected = matches!(
                fallback,
                ConversationOrchestratorDecision::NormalChatWithContext(_)
            );
            diagnostic.planner_duration_ms = Some(started.elapsed().as_millis() as u64);
            return OrchestratorPlanAttempt {
                decision: fallback,
                diagnostic,
            };
        }
    };
    let content = body
        .message
        .map(|message| message.content)
        .unwrap_or_default();
    let parsed = parse_context_planner_output(&content, working_context);
    let mut decision = parsed.decision.unwrap_or_else(|| {
        planner_failure_fallback(
            working_context,
            parsed
                .failure_reason
                .as_deref()
                .unwrap_or("planner_failure"),
            Some(user_message),
        )
    });
    diagnostic.planner_failure_reason = parsed.failure_reason;
    diagnostic.planner_confidence = parsed.planner_confidence;
    diagnostic.tool_affinity_risk = parsed.tool_affinity_risk;
    apply_planner_safety_to_diagnostic(&mut diagnostic, &parsed.planner_safety);
    if let Some(context_policy) = apply_context_continuation_policy(
        &decision,
        working_context,
        diagnostic.planner_failure_reason.as_deref(),
        diagnostic.planner_confidence,
        Some(user_message),
    ) {
        diagnostic.context_continuation_policy_action =
            Some(context_policy.action.as_str().to_string());
        diagnostic.context_continuation_policy_reason =
            Some(context_policy.reason.as_str().to_string());
        if context_policy.action == ContextContinuationPolicyAction::UseContextAnswerFirst {
            decision = ConversationOrchestratorDecision::AnswerFromContext(
                context_continuation_plan(context_policy.reason.as_str(), Some(user_message)),
            );
            diagnostic.context_answer_first_attempted = Some(true);
        } else if context_policy.action == ContextContinuationPolicyAction::DeferToToolRouter {
            decision = ConversationOrchestratorDecision::DeferToToolRouter(DeferToToolRouterPlan {
                reason: context_policy.reason.as_str().to_string(),
                planner_failure_reason: diagnostic.planner_failure_reason.clone(),
                confidence: diagnostic.planner_confidence,
            });
        } else if context_policy.action == ContextContinuationPolicyAction::UseNormalChatWithContext
            && matches!(decision, ConversationOrchestratorDecision::AnswerFromContext(_))
        {
            decision = ConversationOrchestratorDecision::NormalChatWithContext(contextual_chat_plan(
                context_policy.reason.as_str(),
                Some(user_message),
            ));
        }
    }
    diagnostic.selected_route = Some(decision_route_label(&decision).to_string());
    diagnostic.context_ref = decision_context_ref(&decision).map(str::to_string);
    diagnostic.used_full_router = matches!(
        decision,
        ConversationOrchestratorDecision::ToolCall(_)
            | ConversationOrchestratorDecision::DeferToToolRouter(_)
    );
    diagnostic.planner_empty =
        diagnostic.planner_failure_reason.as_deref() == Some("empty_model_content");
    diagnostic.used_context_boundary_fallback =
        diagnostic.planner_failure_reason.is_some() && working_context.last_tool_result_usable();
    diagnostic.normal_chat_context_injected = matches!(
        decision,
        ConversationOrchestratorDecision::NormalChatWithContext(_)
    );
    if diagnostic.used_context_boundary_fallback {
        diagnostic.fallback_policy = Some(
            match &decision {
                ConversationOrchestratorDecision::AnswerFromContext(_) => {
                    "context_answer_on_planner_failure"
                }
                ConversationOrchestratorDecision::NormalChatWithContext(_) => {
                    "normal_chat_with_context_on_planner_failure"
                }
                _ => "context_attachment_on_planner_failure",
            }
            .to_string(),
        );
    }
    if diagnostic.used_full_router {
        diagnostic.tool_router_invoked_reason = match &decision {
            ConversationOrchestratorDecision::DeferToToolRouter(plan) => Some(plan.reason.clone()),
            _ => Some("planner_needs_tool".to_string()),
        };
    }
    diagnostic.planner_duration_ms = Some(started.elapsed().as_millis() as u64);

    OrchestratorPlanAttempt {
        decision,
        diagnostic,
    }
}

pub async fn synthesize_context_answer_with_active_model(
    source: &str,
    user_message: &str,
    working_context: &WorkingContextFrame,
    plan: &ContextAnswerPlan,
) -> ContextAnswerAttempt {
    let expected_language = detect_user_language(user_message);
    let Some(tool_result) = working_context.last_tool_result.as_ref() else {
        return ContextAnswerAttempt {
            output: None,
            diagnostic: context_answer_diagnostic(
                None,
                None,
                0,
                false,
                false,
                "missing_tool_result",
                working_context,
                expected_language,
            ),
        };
    };
    let prompt = context_broker::build_context_answer_messages(
        user_message,
        working_context,
        tool_result,
        expected_language,
    );
    let started = Instant::now();
    let model = resolve_active_ollama_model(user_message, source).await;
    let mut diagnostic = AssistantOrchestratorDiagnostic {
        request_id: None,
        stage: "context_answer".to_string(),
        planner_stage: "context_answer".to_string(),
        working_context_present: true,
        last_tool_result_present: true,
        selected_route: Some(
            if plan.strategy == "context_boundary" {
                "answer_from_context_boundary"
            } else {
                "answer_from_context"
            }
            .to_string(),
        ),
        context_ref: Some(plan.context_ref.clone()),
        planner_model: Some(model.clone()),
        planner_duration_ms: None,
        planner_failure_reason: None,
        planner_confidence: Some(plan.confidence),
        policy_action: Some("accept_decision".to_string()),
        fallback_policy: None,
        fallback_reason: None,
        planner_empty: false,
        used_context_boundary_fallback: plan.strategy == "context_boundary",
        normal_chat_context_injected: false,
        normal_chat_bypassed_tool_router: None,
        normal_chat_policy_action: None,
        normal_chat_policy_reason: None,
        normal_chat_direct_confidence_threshold: None,
        normal_chat_accepted_directly: None,
        planner_intent_kind: None,
        planner_capability_family: None,
        planner_requires_tool_arbitration: None,
        planner_requires_memory_lookup: None,
        planner_requires_governed_action: None,
        planner_requires_context_boundary: None,
        planner_safe_to_bypass_tools: None,
        tool_router_invoked_reason: None,
        needs_tool_policy_action: None,
        needs_tool_policy_reason: None,
        needs_tool_confidence_threshold: None,
        needs_tool_accepted: None,
        tool_affinity_source: None,
        accepted_decision: true,
        prompt_char_count: prompt.prompt_char_count,
        prompt_budget_exceeded: prompt.prompt_budget_exceeded,
        used_full_router: false,
        tool_affinity_risk: None,
        context_salience_score: salience_score_for_diagnostic(working_context),
        context_turn_age: salience_turn_age_for_diagnostic(working_context),
        context_stale: salience_stale_for_diagnostic(working_context),
        context_decay_action: Some("reinforce_on_context_answer".to_string()),
        context_continuation_policy_action: None,
        context_continuation_policy_reason: None,
        context_answer_first_attempted: Some(true),
        context_answer_fallback_used: Some(false),
        context_answer_empty_model_content: Some(false),
        expected_language: Some(expected_language.code().to_string()),
        output_language: None,
        language_mismatch: None,
        language_retry_attempted: Some(false),
        language_retry_succeeded: Some(false),
        budget_compaction_applied: Some(prompt.budget_compaction_applied),
        user_facing_context_label: Some(user_facing_context_label(tool_result)),
        sanitized_internal_context_refs: false,
        tool_manifest_count: 0,
        metadata_only: true,
    };
    let client = match Client::builder()
        .timeout(std::time::Duration::from_millis(10_000))
        .build()
    {
        Ok(client) => client,
        Err(_) => {
            diagnostic.planner_failure_reason = Some("endpoint_config".to_string());
            diagnostic.planner_duration_ms = Some(started.elapsed().as_millis() as u64);
            diagnostic.context_answer_fallback_used = Some(true);
            diagnostic.context_answer_empty_model_content = Some(false);
            return ContextAnswerAttempt {
                output: Some(fallback_context_answer(
                    tool_result,
                    plan,
                    expected_language,
                )),
                diagnostic,
            };
        }
    };
    let response = match client
        .post(ollama_endpoint("/api/chat"))
        .json(&serde_json::json!({
            "model": model,
            "stream": false,
            "format": "json",
            "messages": prompt.messages,
            "options": {
                "temperature": 0.0,
                "top_p": 0.4,
                "repeat_penalty": 1.04,
                "num_predict": 320
            },
            "keep_alive": "30m"
        }))
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) if error.is_timeout() => {
            diagnostic.planner_failure_reason = Some("timeout".to_string());
            diagnostic.planner_duration_ms = Some(started.elapsed().as_millis() as u64);
            diagnostic.context_answer_fallback_used = Some(true);
            diagnostic.context_answer_empty_model_content = Some(false);
            return ContextAnswerAttempt {
                output: Some(fallback_context_answer(
                    tool_result,
                    plan,
                    expected_language,
                )),
                diagnostic,
            };
        }
        Err(_) => {
            diagnostic.planner_failure_reason = Some("ollama_unavailable".to_string());
            diagnostic.planner_duration_ms = Some(started.elapsed().as_millis() as u64);
            diagnostic.context_answer_fallback_used = Some(true);
            diagnostic.context_answer_empty_model_content = Some(false);
            return ContextAnswerAttempt {
                output: Some(fallback_context_answer(
                    tool_result,
                    plan,
                    expected_language,
                )),
                diagnostic,
            };
        }
    };
    if !response.status().is_success() {
        diagnostic.planner_failure_reason = Some("ollama_http_error".to_string());
        diagnostic.planner_duration_ms = Some(started.elapsed().as_millis() as u64);
        diagnostic.context_answer_fallback_used = Some(true);
        diagnostic.context_answer_empty_model_content = Some(false);
        return ContextAnswerAttempt {
            output: Some(fallback_context_answer(
                tool_result,
                plan,
                expected_language,
            )),
            diagnostic,
        };
    }
    let body: OllamaChatResponse = match response.json().await {
        Ok(body) => body,
        Err(_) => {
            diagnostic.planner_failure_reason = Some("invalid_response_schema".to_string());
            diagnostic.planner_duration_ms = Some(started.elapsed().as_millis() as u64);
            diagnostic.context_answer_fallback_used = Some(true);
            diagnostic.context_answer_empty_model_content = Some(false);
            return ContextAnswerAttempt {
                output: Some(fallback_context_answer(
                    tool_result,
                    plan,
                    expected_language,
                )),
                diagnostic,
            };
        }
    };
    let content = body
        .message
        .map(|message| message.content)
        .unwrap_or_default();
    let mut output = parse_context_answer_output(&content).or_else(|| {
        diagnostic.planner_failure_reason = Some(
            if content.trim().is_empty() {
                "empty_model_content"
            } else {
                "invalid_context_answer_json"
            }
            .to_string(),
        );
        diagnostic.context_answer_fallback_used = Some(true);
        diagnostic.context_answer_empty_model_content = Some(content.trim().is_empty());
        Some(fallback_context_answer(
            tool_result,
            plan,
            expected_language,
        ))
    });
    if let Some(output) = output.as_mut() {
        let actual_language = effective_output_language(output);
        let mismatch = context_answer_language_mismatch(expected_language, actual_language);
        diagnostic.output_language = Some(actual_language.code().to_string());
        diagnostic.language_mismatch = Some(mismatch);
        if mismatch {
            diagnostic.language_retry_attempted = Some(true);
            if let Some(mut corrected) = retry_context_answer_language_with_active_model(
                &client,
                &model,
                output,
                expected_language,
                tool_result,
            )
            .await
            {
                let corrected_language = effective_output_language(&corrected);
                if !context_answer_language_mismatch(expected_language, corrected_language) {
                    corrected.language = corrected_language;
                    *output = corrected;
                    diagnostic.language_retry_succeeded = Some(true);
                    diagnostic.output_language = Some(corrected_language.code().to_string());
                } else {
                    *output = fallback_context_answer(tool_result, plan, expected_language);
                    diagnostic.language_retry_succeeded = Some(false);
                    diagnostic.context_answer_fallback_used = Some(true);
                    diagnostic.output_language = Some(output.language.code().to_string());
                }
            } else {
                *output = fallback_context_answer(tool_result, plan, expected_language);
                diagnostic.language_retry_succeeded = Some(false);
                diagnostic.context_answer_fallback_used = Some(true);
                diagnostic.output_language = Some(output.language.code().to_string());
            }
        } else {
            output.language = actual_language;
        }
        diagnostic.sanitized_internal_context_refs = sanitize_context_answer_output(output);
        diagnostic.output_language = Some(output.language.code().to_string());
    }
    diagnostic.planner_duration_ms = Some(started.elapsed().as_millis() as u64);
    ContextAnswerAttempt { output, diagnostic }
}

async fn retry_context_answer_language_with_active_model(
    client: &Client,
    model: &str,
    output: &ContextAnswerOutput,
    expected_language: AstraUserLanguage,
    tool_result: &ToolResultFrame,
) -> Option<ContextAnswerOutput> {
    let prompt = context_broker::build_context_answer_language_retry_messages(
        output,
        expected_language,
        tool_result,
    );
    let response = client
        .post(ollama_endpoint("/api/chat"))
        .json(&serde_json::json!({
            "model": model,
            "stream": false,
            "format": "json",
            "messages": prompt.messages,
            "options": {
                "temperature": 0.0,
                "top_p": 0.3,
                "repeat_penalty": 1.02,
                "num_predict": 240
            },
            "keep_alive": "30m"
        }))
        .send()
        .await
        .ok()?;
    if !response.status().is_success() {
        return None;
    }
    let body: OllamaChatResponse = response.json().await.ok()?;
    let content = body
        .message
        .map(|message| message.content)
        .unwrap_or_default();
    parse_context_answer_output(&content)
}

pub fn parse_context_planner_output(
    content: &str,
    _working_context: &WorkingContextFrame,
) -> PlannerParseOutcome {
    let Some(candidate) = extract_json_object(content) else {
        return PlannerParseOutcome {
            decision: None,
            failure_reason: Some(if content.trim().is_empty() {
                "empty_model_content".to_string()
            } else {
                "invalid_json".to_string()
            }),
            planner_confidence: None,
            tool_affinity_risk: None,
            planner_safety: PlannerSafetyMetadata::default(),
        };
    };
    let raw: RawPlannerOutput = match serde_json::from_str(candidate) {
        Ok(value) => value,
        Err(_) => {
            return PlannerParseOutcome {
                decision: None,
                failure_reason: Some("invalid_schema".to_string()),
                planner_confidence: None,
                tool_affinity_risk: None,
                planner_safety: PlannerSafetyMetadata::default(),
            }
        }
    };
    let planner_safety = PlannerSafetyMetadata::from_raw(&raw);
    let route = normalize_route(raw.route.as_deref().unwrap_or("normal_chat"));
    let confidence = raw.confidence.unwrap_or(0.0).clamp(0.0, 1.0);
    let reason_code = raw
        .reason_code
        .clone()
        .unwrap_or_else(|| "unspecified".to_string());
    let context_ref = raw
        .context_ref
        .clone()
        .unwrap_or_else(|| "none".to_string());
    let decision = match route.as_str() {
        "answerfromcontext" => {
            let answer_plan = raw.answer_plan.unwrap_or(RawAnswerPlan {
                strategy: Some("none".to_string()),
                focus: None,
            });
            ConversationOrchestratorDecision::AnswerFromContext(ContextAnswerPlan {
                strategy: answer_plan
                    .strategy
                    .unwrap_or_else(|| "none".to_string()),
                focus: answer_plan
                    .focus
                    .and_then(|value| non_empty(context_broker::bounded_text(&value, 120))),
                context_ref,
                reason_code,
                confidence,
            })
        }
        "answerfromcontextboundary" | "generalanswerwithcontextboundary" => {
            let answer_plan = raw.answer_plan.unwrap_or(RawAnswerPlan {
                strategy: Some("context_boundary".to_string()),
                focus: None,
            });
            ConversationOrchestratorDecision::AnswerFromContextBoundary(ContextAnswerPlan {
                strategy: answer_plan.strategy.unwrap_or_else(|| "context_boundary".to_string()),
                focus: answer_plan
                    .focus
                    .and_then(|value| non_empty(context_broker::bounded_text(&value, 160))),
                context_ref,
                reason_code,
                confidence,
            })
        }
        "needstool" => ConversationOrchestratorDecision::ToolCall(AssistantRouteDecision::NormalChat),
        "normalchatwithcontext" => {
            ConversationOrchestratorDecision::NormalChatWithContext(ContextualChatPlan {
                context_ref,
                reason_code,
                confidence,
            })
        }
        "normalchat" => ConversationOrchestratorDecision::NormalChat(NormalChatPlan {
            intent_kind: planner_safety.intent_kind,
            capability_family: planner_safety.capability_family,
            requires_tool_arbitration: planner_safety.requires_tool_arbitration,
            requires_memory_lookup: planner_safety.requires_memory_lookup,
            requires_governed_action: planner_safety.requires_governed_action,
            requires_context_boundary: planner_safety.requires_context_boundary,
            safe_to_bypass_tools: planner_safety.safe_to_bypass_tools,
            context_ref: non_empty(context_ref.clone()),
            reason_code,
            confidence,
        }),
        "clarify" => ConversationOrchestratorDecision::Clarify(ClarificationPlan {
            reason_code,
            message:
                "Mi serve un riferimento in piu per capire se devo usare il contesto precedente o recuperare nuove evidenze."
                    .to_string(),
            confidence,
        }),
        "refuse" => ConversationOrchestratorDecision::Refuse(RefusalPlan {
            reason_code,
            message:
                "Non posso gestire questa richiesta in modo sicuro con il contesto disponibile."
                    .to_string(),
            confidence,
        }),
        _ => {
            return PlannerParseOutcome {
                decision: None,
                failure_reason: Some("invalid_route".to_string()),
                planner_confidence: Some(confidence),
                tool_affinity_risk: raw.tool_affinity_risk,
                planner_safety,
            }
        }
    };
    PlannerParseOutcome {
        decision: Some(decision),
        failure_reason: None,
        planner_confidence: Some(confidence),
        tool_affinity_risk: raw.tool_affinity_risk,
        planner_safety,
    }
}

pub fn parse_context_answer_output(content: &str) -> Option<ContextAnswerOutput> {
    let candidate = extract_json_object(content)?;
    let raw: RawContextAnswerOutput = serde_json::from_str(candidate).ok()?;
    let language = raw
        .language
        .as_deref()
        .map(AstraUserLanguage::from_code)
        .unwrap_or(Some(AstraUserLanguage::Unknown))?;
    let mut status = raw
        .status
        .unwrap_or_else(|| "partial".to_string())
        .trim()
        .to_ascii_lowercase();
    if !matches!(
        status.as_str(),
        "answered" | "partial" | "insufficient_context" | "boundary_answer"
    ) {
        return None;
    }
    let support = raw
        .support
        .unwrap_or_else(|| "supported_by_context".to_string())
        .trim()
        .to_ascii_lowercase();
    if !matches!(
        support.as_str(),
        "supported_by_context" | "not_in_context" | "general_knowledge_with_context"
    ) {
        return None;
    }
    let mut answer = raw.answer.unwrap_or_default().trim().to_string();
    if answer.is_empty() && status == "insufficient_context" {
        answer = "Non ho contesto sufficiente per rispondere senza recuperare nuove evidenze."
            .to_string();
    }
    if answer.is_empty() {
        return None;
    }
    let allowed = ["last_tool_result", "working_topic"];
    if raw
        .used_context_refs
        .iter()
        .any(|item| !allowed.contains(&item.as_str()))
    {
        return None;
    }
    if raw.used_context_refs.is_empty() && status == "answered" {
        status = "partial".to_string();
    }
    Some(ContextAnswerOutput {
        answer,
        language,
        status,
        support,
        used_context_refs: raw.used_context_refs,
        confidence: raw.confidence.unwrap_or(0.0).clamp(0.0, 1.0),
        warnings: raw
            .warnings
            .into_iter()
            .map(|warning| context_broker::bounded_text(&warning, 160))
            .filter(|warning| {
                !warning.trim().is_empty() && !is_internal_context_answer_warning(warning)
            })
            .take(6)
            .collect(),
        sanitized_internal_context_refs: false,
    })
}

pub fn render_context_answer(
    tool_result: &ToolResultFrame,
    output: &ContextAnswerOutput,
) -> String {
    let mut output = output.clone();
    sanitize_context_answer_output(&mut output);
    let answer = output.answer.trim();
    if answer.to_lowercase().starts_with("fonte:") {
        return answer.to_string();
    }
    let mut lines = vec![
        format!(
            "Fonte: {}.",
            tool_result.source_label.trim().trim_end_matches('.')
        ),
        answer.to_string(),
    ];
    for warning in output
        .warnings
        .iter()
        .filter(|warning| !is_internal_context_answer_warning(warning))
        .take(3)
    {
        lines.push(format!("Nota: {warning}"));
    }
    lines.join("\n")
}

pub fn sanitize_context_answer_output(output: &mut ContextAnswerOutput) -> bool {
    let original = output.answer.clone();
    let sanitized_lines = original
        .lines()
        .filter_map(|line| {
            let lower = line.to_ascii_lowercase();
            let leaks_internal_ref = [
                "last_tool_result",
                "working_context",
                "toolresultframe",
                "evidencereference",
                "contextframe",
                "context used:",
                "contesto usato:",
                "context_answer_synthesizer_fallback",
                "stt completeness:",
                "evidenze usate:",
                "evidenze disponibili:",
                "segment:",
            ]
            .iter()
            .any(|marker| lower.contains(marker));
            (!leaks_internal_ref).then(|| line.trim_end())
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string();
    let changed = sanitized_lines != original.trim();
    if changed && !sanitized_lines.is_empty() {
        output.answer = sanitized_lines;
        output.sanitized_internal_context_refs = true;
    }
    output.sanitized_internal_context_refs
}

fn is_internal_context_answer_warning(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    [
        "context_answer_synthesizer_fallback",
        "toolresultframe",
        "evidencereference",
        "metadata_only",
        "raw_model_output",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

pub fn user_facing_context_label(tool_result: &ToolResultFrame) -> String {
    format!(
        "Fonte: {}.",
        tool_result.source_label.trim().trim_end_matches('.')
    )
}

pub fn build_normal_chat_with_context_preamble(
    working_context: &WorkingContextFrame,
    plan: &ContextualChatPlan,
) -> Option<String> {
    let tool_result = working_context.last_tool_result.as_ref()?;
    let topics = tool_result
        .key_topics
        .iter()
        .take(5)
        .map(|topic| context_broker::bounded_text(topic, 80))
        .collect::<Vec<_>>()
        .join(", ");
    let boundary_mode = plan.reason_code.contains("boundary_general_knowledge")
        || plan.reason_code.contains("topic_overlap_only")
        || plan.reason_code.contains("normal_chat_with_context");
    let instruction = if boundary_mode {
        "Questo recap e solo contesto secondario, non la fonte primaria della risposta. Rispondi normalmente alla domanda dell'utente usando conoscenza generale quando necessario; cita il recap solo per separare cosa risulta dalla registrazione da cosa e spiegazione generale. Non aprire la risposta con 'Fonte' e non attribuire al transcript dettagli che non contiene."
    } else {
        "Usa il recap come contesto conversazionale, ma se rispondi con conoscenza generale distinguila chiaramente dalle evidenze della registrazione e non attribuire al transcript dettagli non presenti."
    };
    Some(format!(
        "Contesto conversazionale compatto da '{}'. Recap sintetico: {}. Temi: {}. {instruction}",
        context_broker::bounded_text(&tool_result.source_label, 120),
        context_broker::bounded_text(&tool_result.answer_summary, 360),
        if topics.trim().is_empty() { "non specificati".to_string() } else { topics },
    ))
}

pub fn detect_user_language(message: &str) -> AstraUserLanguage {
    let (italian, english) = score_user_language_markers(message);
    if italian < 1.5 && english < 1.5 {
        return AstraUserLanguage::Unknown;
    }
    if italian > 0.0 && english > 0.0 {
        let larger = italian.max(english);
        let smaller = italian.min(english);
        if larger / smaller.max(0.1) < 1.35 {
            return AstraUserLanguage::Mixed;
        }
    }
    if italian > english {
        AstraUserLanguage::Italian
    } else {
        AstraUserLanguage::English
    }
}

fn detect_answer_language(answer: &str) -> AstraUserLanguage {
    detect_user_language(answer)
}

fn score_user_language_markers(text: &str) -> (f32, f32) {
    let italian_markers = [
        "di",
        "cosa",
        "abbiamo",
        "parlato",
        "parlava",
        "nel",
        "nella",
        "nell'ultima",
        "mi",
        "fai",
        "un",
        "una",
        "si",
        "no",
        "non",
        "in",
        "base",
        "alla",
        "della",
        "dei",
        "delle",
        "puoi",
        "puo",
        "era",
        "stato",
        "dove",
        "come",
        "perche",
    ];
    let english_markers = [
        "what",
        "did",
        "we",
        "discuss",
        "talk",
        "talked",
        "last",
        "recording",
        "so",
        "was",
        "it",
        "about",
        "the",
        "not",
        "based",
        "previous",
        "answer",
        "can",
        "you",
        "give",
        "me",
    ];
    let mut italian = 0.0;
    let mut english = 0.0;
    for token in text
        .to_lowercase()
        .split(|character: char| !character.is_alphabetic() && character != '\'')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        if italian_markers.contains(&token) {
            italian += 1.0;
        }
        if english_markers.contains(&token) {
            english += 1.0;
        }
    }
    (italian, english)
}

fn effective_output_language(output: &ContextAnswerOutput) -> AstraUserLanguage {
    let detected = detect_answer_language(&output.answer);
    if matches!(detected, AstraUserLanguage::Unknown) {
        output.language
    } else {
        detected
    }
}

fn context_answer_language_mismatch(
    expected: AstraUserLanguage,
    actual: AstraUserLanguage,
) -> bool {
    matches!(
        (expected, actual),
        (AstraUserLanguage::Italian, AstraUserLanguage::English)
            | (AstraUserLanguage::English, AstraUserLanguage::Italian)
    )
}

pub fn decision_route_label(decision: &ConversationOrchestratorDecision) -> &'static str {
    match decision {
        ConversationOrchestratorDecision::AnswerFromContext(_) => "answer_from_context",
        ConversationOrchestratorDecision::AnswerFromContextBoundary(_) => {
            "answer_from_context_boundary"
        }
        ConversationOrchestratorDecision::NormalChatWithContext(_) => "normal_chat_with_context",
        ConversationOrchestratorDecision::ToolCall(_) => "needs_tool",
        ConversationOrchestratorDecision::DeferToToolRouter(_) => "defer_to_tool_router",
        ConversationOrchestratorDecision::NormalChat(_) => "normal_chat",
        ConversationOrchestratorDecision::Clarify(_) => "clarify",
        ConversationOrchestratorDecision::Refuse(_) => "refuse",
    }
}

pub fn decision_context_ref(decision: &ConversationOrchestratorDecision) -> Option<&str> {
    match decision {
        ConversationOrchestratorDecision::AnswerFromContext(plan)
        | ConversationOrchestratorDecision::AnswerFromContextBoundary(plan) => {
            Some(&plan.context_ref)
        }
        ConversationOrchestratorDecision::NormalChatWithContext(plan) => Some(&plan.context_ref),
        ConversationOrchestratorDecision::NormalChat(plan) => plan.context_ref.as_deref(),
        _ => None,
    }
}

pub fn apply_normal_chat_arbitration_policy(
    input: &NormalChatArbitrationInput<'_>,
) -> NormalChatArbitrationDecision {
    let verify = |reason| NormalChatArbitrationDecision {
        action: NormalChatArbitrationAction::VerifyWithToolRouter,
        reason,
    };

    if input.plan.confidence < NORMAL_CHAT_DIRECT_CONFIDENCE_THRESHOLD {
        return verify(NormalChatArbitrationReason::LowConfidence);
    }
    if input.slash_command {
        return verify(NormalChatArbitrationReason::SlashCommand);
    }
    if input.explicit_ui_action {
        return verify(NormalChatArbitrationReason::ExplicitUiAction);
    }
    if input.pending_governed_tool_flow {
        return verify(NormalChatArbitrationReason::PendingGovernedToolFlow);
    }
    if input.tool_affinity_risk == Some(true) {
        return verify(NormalChatArbitrationReason::ToolAffinityRisk);
    }

    let Some(intent_kind) = input.plan.intent_kind else {
        return verify(NormalChatArbitrationReason::MissingSafetyFields);
    };
    if matches!(intent_kind, PlannerIntentKind::Unknown) {
        return verify(NormalChatArbitrationReason::UnknownIntentKind);
    }
    if !intent_kind.direct_normal_chat_allowed() {
        return verify(NormalChatArbitrationReason::UnsafeToBypassTools);
    }

    match input.plan.capability_family {
        Some(PlannerCapabilityFamily::None) => {}
        Some(PlannerCapabilityFamily::Unknown) => {
            return verify(NormalChatArbitrationReason::UnknownCapabilityFamily);
        }
        Some(_) => return verify(NormalChatArbitrationReason::CapabilityRequiresTools),
        None => return verify(NormalChatArbitrationReason::MissingSafetyFields),
    }

    if input.plan.requires_tool_arbitration.is_none()
        || input.plan.requires_memory_lookup.is_none()
        || input.plan.requires_governed_action.is_none()
        || input.plan.requires_context_boundary.is_none()
        || input.plan.safe_to_bypass_tools.is_none()
    {
        return verify(NormalChatArbitrationReason::MissingSafetyFields);
    }
    if input.plan.safe_to_bypass_tools != Some(true) {
        return verify(NormalChatArbitrationReason::UnsafeToBypassTools);
    }
    if input.plan.requires_tool_arbitration == Some(true) {
        return verify(NormalChatArbitrationReason::RequiresToolArbitration);
    }
    if input.plan.requires_memory_lookup == Some(true) {
        return verify(NormalChatArbitrationReason::RequiresMemoryLookup);
    }
    if input.plan.requires_governed_action == Some(true) {
        return verify(NormalChatArbitrationReason::RequiresGovernedAction);
    }
    if input.plan.requires_context_boundary == Some(true) {
        return verify(NormalChatArbitrationReason::RequiresContextBoundary);
    }
    if input
        .plan
        .context_ref
        .as_deref()
        .is_some_and(|value| normalize_route(value) != "none")
        && input.working_context.last_tool_result_usable()
    {
        return verify(NormalChatArbitrationReason::ContextReference);
    }

    NormalChatArbitrationDecision {
        action: NormalChatArbitrationAction::AcceptDirectNormalChat,
        reason: NormalChatArbitrationReason::SafeToBypassTools,
    }
}

fn normal_chat_fallback_reason(
    arbitration: NormalChatArbitrationDecision,
) -> OrchestratorFallbackReason {
    match arbitration.reason {
        NormalChatArbitrationReason::LowConfidence => {
            OrchestratorFallbackReason::NormalChatLowConfidence
        }
        _ => OrchestratorFallbackReason::NormalChatUnsafeToBypassTools,
    }
}


fn planner_safety_requires_router(diagnostic: &AssistantOrchestratorDiagnostic) -> bool {
    diagnostic.planner_requires_governed_action == Some(true)
        || diagnostic.planner_requires_memory_lookup == Some(true)
        || diagnostic.planner_requires_tool_arbitration == Some(true)
        || diagnostic.planner_safe_to_bypass_tools == Some(false)
        || matches!(
            diagnostic.planner_intent_kind.as_deref(),
            Some("governed_action") | Some("session_memory_query")
        )
        || matches!(
            diagnostic.planner_capability_family.as_deref(),
            Some("work_session") | Some("meeting") | Some("screen_context") | Some("session_memory")
        )
}

pub fn apply_orchestrator_policy(
    attempt: &OrchestratorPlanAttempt,
    working_context: &WorkingContextFrame,
) -> OrchestratorPolicyAction {
    match &attempt.decision {
        ConversationOrchestratorDecision::AnswerFromContext(plan) => {
            if plan.confidence < MIN_ANSWER_FROM_CONTEXT_CONFIDENCE {
                return OrchestratorPolicyAction::UseFullToolRouter {
                    reason: OrchestratorFallbackReason::AnswerFromContextLowConfidence,
                };
            }
            if plan.context_ref != "last_tool_result" {
                return OrchestratorPolicyAction::UseFullToolRouter {
                    reason: OrchestratorFallbackReason::MissingContextForContextAnswer,
                };
            }
            if !working_context.last_tool_result_usable() {
                return OrchestratorPolicyAction::UseFullToolRouter {
                    reason: OrchestratorFallbackReason::NoLastToolResult,
                };
            }
            OrchestratorPolicyAction::AcceptDecision
        }
        ConversationOrchestratorDecision::AnswerFromContextBoundary(plan) => {
            if plan.confidence < MIN_CONTEXT_BOUNDARY_CONFIDENCE {
                return OrchestratorPolicyAction::SafeClarify {
                    reason: OrchestratorFallbackReason::AnswerFromContextLowConfidence,
                };
            }
            if plan.context_ref != "last_tool_result" || working_context.last_tool_result.is_none()
            {
                return OrchestratorPolicyAction::UseFullToolRouter {
                    reason: OrchestratorFallbackReason::NoLastToolResult,
                };
            }
            if !working_context.last_tool_result_usable() {
                return OrchestratorPolicyAction::SafeClarify {
                    reason: OrchestratorFallbackReason::NoLastToolResult,
                };
            }
            OrchestratorPolicyAction::AcceptDecision
        }
        ConversationOrchestratorDecision::NormalChatWithContext(plan) => {
            if plan.confidence < MIN_CLARIFY_CONFIDENCE {
                return OrchestratorPolicyAction::SafeClarify {
                    reason: OrchestratorFallbackReason::ToolAffinityUnresolved,
                };
            }
            if plan.context_ref != "last_tool_result" || working_context.last_tool_result.is_none()
            {
                return OrchestratorPolicyAction::UseFullToolRouter {
                    reason: OrchestratorFallbackReason::NoLastToolResult,
                };
            }
            if !working_context.last_tool_result_usable() {
                return OrchestratorPolicyAction::SafeClarify {
                    reason: OrchestratorFallbackReason::NoLastToolResult,
                };
            }
            OrchestratorPolicyAction::AcceptDecision
        }
        ConversationOrchestratorDecision::NormalChat(plan) => {
            let arbitration = apply_normal_chat_arbitration_policy(&NormalChatArbitrationInput {
                plan,
                working_context,
                tool_affinity_risk: attempt.diagnostic.tool_affinity_risk,
                pending_governed_tool_flow: working_context.pending_governed_action.is_some(),
                explicit_ui_action: false,
                slash_command: false,
            });
            match arbitration.action {
                NormalChatArbitrationAction::AcceptDirectNormalChat => {
                    OrchestratorPolicyAction::AcceptDecision
                }
                NormalChatArbitrationAction::VerifyWithToolRouter => {
                    OrchestratorPolicyAction::UseFullToolRouter {
                        reason: normal_chat_fallback_reason(arbitration),
                    }
                }
            }
        }
        ConversationOrchestratorDecision::Clarify(plan) => {
            if planner_safety_requires_router(&attempt.diagnostic) {
                return OrchestratorPolicyAction::UseFullToolRouter {
                    reason: OrchestratorFallbackReason::ToolAffinityUnresolved,
                };
            }
            if plan.confidence < MIN_CLARIFY_CONFIDENCE {
                return OrchestratorPolicyAction::UseFullToolRouter {
                    reason: OrchestratorFallbackReason::ClarifyLowConfidence,
                };
            }
            OrchestratorPolicyAction::AcceptDecision
        }
        ConversationOrchestratorDecision::Refuse(plan) => {
            if planner_safety_requires_router(&attempt.diagnostic) {
                return OrchestratorPolicyAction::UseFullToolRouter {
                    reason: OrchestratorFallbackReason::ToolAffinityUnresolved,
                };
            }
            if plan.confidence < MIN_REFUSE_CONFIDENCE {
                return OrchestratorPolicyAction::UseFullToolRouter {
                    reason: OrchestratorFallbackReason::RefuseLowConfidence,
                };
            }
            OrchestratorPolicyAction::AcceptDecision
        }
        ConversationOrchestratorDecision::DeferToToolRouter(plan) => {
            let reason = if plan.reason == "planner_empty_no_grounded_context" {
                OrchestratorFallbackReason::PlannerEmptyNoGroundedContext
            } else {
                OrchestratorFallbackReason::PlannerFailureNoGroundedContext
            };
            OrchestratorPolicyAction::DeferToFullToolRouter { reason }
        }
        ConversationOrchestratorDecision::ToolCall(_) => {
            if let Some(reason) = planner_failure_to_fallback_reason(
                attempt.diagnostic.planner_failure_reason.as_deref(),
            ) {
                if working_context.last_tool_result_usable()
                    && matches!(
                        reason,
                        OrchestratorFallbackReason::PlannerEmpty
                            | OrchestratorFallbackReason::PlannerMalformed
                            | OrchestratorFallbackReason::PlannerTimeout
                            | OrchestratorFallbackReason::PlannerUnavailable
                    )
                {
                    return OrchestratorPolicyAction::SafeClarify { reason };
                }
                OrchestratorPolicyAction::UseFullToolRouter { reason }
            } else {
                match apply_needs_tool_policy(&NeedsToolPolicyInput {
                    planner_confidence: attempt.diagnostic.planner_confidence,
                    context_ref: attempt.diagnostic.context_ref.clone(),
                    last_tool_result_present: working_context.last_tool_result_usable(),
                    pending_tool_action: working_context.pending_governed_action.is_some(),
                    explicit_user_action: false,
                    slash_command: false,
                    ui_action: false,
                    planner_reason_code: attempt
                        .diagnostic
                        .planner_failure_reason
                        .clone()
                        .or_else(|| Some("discourse_planner_needs_tool".to_string())),
                    tool_affinity: if working_context.pending_governed_action.is_some() {
                        ToolAffinitySignal::PendingToolContinuation
                    } else {
                        needs_tool_affinity_from_confidence(attempt.diagnostic.planner_confidence)
                    },
                }) {
                    NeedsToolPolicyDecision::Accept { .. } => {
                        OrchestratorPolicyAction::AcceptDecision
                    }
                    NeedsToolPolicyDecision::DowngradeToContextBoundary { reason } => {
                        OrchestratorPolicyAction::DowngradeToContextBoundary { reason }
                    }
                    NeedsToolPolicyDecision::DowngradeToNormalChatWithContext { reason } => {
                        OrchestratorPolicyAction::DowngradeToNormalChatWithContext { reason }
                    }
                    NeedsToolPolicyDecision::Clarify { .. } => {
                        OrchestratorPolicyAction::SafeClarify {
                            reason: OrchestratorFallbackReason::ToolAffinityUnresolved,
                        }
                    }
                }
            }
        }
    }
}

pub fn needs_tool_affinity_from_confidence(confidence: Option<f32>) -> ToolAffinitySignal {
    if confidence.is_some_and(|value| value >= MIN_NEEDS_TOOL_CONFIDENCE) {
        ToolAffinitySignal::PlannerHighConfidence
    } else {
        ToolAffinitySignal::None
    }
}

pub fn apply_needs_tool_policy(input: &NeedsToolPolicyInput) -> NeedsToolPolicyDecision {
    if input.slash_command {
        return NeedsToolPolicyDecision::Accept {
            reason: NeedsToolPolicyReason::ExplicitSlashCommand,
        };
    }
    if input.ui_action {
        return NeedsToolPolicyDecision::Accept {
            reason: NeedsToolPolicyReason::UiAction,
        };
    }
    if input.explicit_user_action {
        return NeedsToolPolicyDecision::Accept {
            reason: NeedsToolPolicyReason::ExplicitUserAction,
        };
    }
    if input.pending_tool_action {
        return NeedsToolPolicyDecision::Accept {
            reason: NeedsToolPolicyReason::PendingToolContinuation,
        };
    }
    if matches!(input.tool_affinity, ToolAffinitySignal::FullRouterResult) {
        return NeedsToolPolicyDecision::Accept {
            reason: NeedsToolPolicyReason::FullRouterResult,
        };
    }
    if input
        .planner_confidence
        .is_some_and(|value| value >= MIN_NEEDS_TOOL_CONFIDENCE)
    {
        return NeedsToolPolicyDecision::Accept {
            reason: NeedsToolPolicyReason::PlannerHighConfidence,
        };
    }
    if input.last_tool_result_present {
        return NeedsToolPolicyDecision::DowngradeToNormalChatWithContext {
            reason: NeedsToolPolicyReason::LowConfidenceWithLastToolResult,
        };
    }
    if input.planner_confidence.is_none() {
        return NeedsToolPolicyDecision::Clarify {
            reason: NeedsToolPolicyReason::MissingPlannerConfidence,
        };
    }
    NeedsToolPolicyDecision::Clarify {
        reason: NeedsToolPolicyReason::LowConfidenceWithoutToolAffinity,
    }
}

fn apply_planner_safety_to_diagnostic(
    diagnostic: &mut AssistantOrchestratorDiagnostic,
    safety: &PlannerSafetyMetadata,
) {
    diagnostic.planner_intent_kind = safety.intent_kind.map(|value| value.as_str().to_string());
    diagnostic.planner_capability_family = safety
        .capability_family
        .map(|value| value.as_str().to_string());
    diagnostic.planner_requires_tool_arbitration = safety.requires_tool_arbitration;
    diagnostic.planner_requires_memory_lookup = safety.requires_memory_lookup;
    diagnostic.planner_requires_governed_action = safety.requires_governed_action;
    diagnostic.planner_requires_context_boundary = safety.requires_context_boundary;
    diagnostic.planner_safe_to_bypass_tools = safety.safe_to_bypass_tools;
}

fn normal_chat_diagnostic_reason(
    diagnostic: &AssistantOrchestratorDiagnostic,
    policy: &OrchestratorPolicyAction,
) -> &'static str {
    if matches!(policy, OrchestratorPolicyAction::AcceptDecision) {
        return NormalChatArbitrationReason::SafeToBypassTools.as_str();
    }
    if policy
        .fallback_reason()
        .is_some_and(|reason| reason == &OrchestratorFallbackReason::NormalChatLowConfidence)
    {
        return NormalChatArbitrationReason::LowConfidence.as_str();
    }
    if diagnostic.planner_intent_kind.is_none()
        || diagnostic.planner_capability_family.is_none()
        || diagnostic.planner_requires_tool_arbitration.is_none()
        || diagnostic.planner_requires_memory_lookup.is_none()
        || diagnostic.planner_requires_governed_action.is_none()
        || diagnostic.planner_requires_context_boundary.is_none()
        || diagnostic.planner_safe_to_bypass_tools.is_none()
    {
        return NormalChatArbitrationReason::MissingSafetyFields.as_str();
    }
    if diagnostic.planner_intent_kind.as_deref() == Some("unknown") {
        return NormalChatArbitrationReason::UnknownIntentKind.as_str();
    }
    if diagnostic.planner_capability_family.as_deref() == Some("unknown") {
        return NormalChatArbitrationReason::UnknownCapabilityFamily.as_str();
    }
    if diagnostic.planner_capability_family.as_deref() != Some("none") {
        return NormalChatArbitrationReason::CapabilityRequiresTools.as_str();
    }
    if diagnostic.planner_requires_tool_arbitration == Some(true) {
        return NormalChatArbitrationReason::RequiresToolArbitration.as_str();
    }
    if diagnostic.planner_requires_memory_lookup == Some(true) {
        return NormalChatArbitrationReason::RequiresMemoryLookup.as_str();
    }
    if diagnostic.planner_requires_governed_action == Some(true) {
        return NormalChatArbitrationReason::RequiresGovernedAction.as_str();
    }
    if diagnostic.planner_requires_context_boundary == Some(true) {
        return NormalChatArbitrationReason::RequiresContextBoundary.as_str();
    }
    if diagnostic.planner_safe_to_bypass_tools != Some(true) {
        return NormalChatArbitrationReason::UnsafeToBypassTools.as_str();
    }
    if diagnostic
        .context_ref
        .as_deref()
        .is_some_and(|value| normalize_route(value) != "none")
    {
        return NormalChatArbitrationReason::ContextReference.as_str();
    }
    NormalChatArbitrationReason::UnsafeToBypassTools.as_str()
}

pub fn apply_policy_to_diagnostic(
    diagnostic: &mut AssistantOrchestratorDiagnostic,
    policy: &OrchestratorPolicyAction,
) {
    diagnostic.policy_action = Some(policy.label().to_string());
    diagnostic.fallback_policy = match policy {
        OrchestratorPolicyAction::AcceptDecision => None,
        OrchestratorPolicyAction::UseFullToolRouter { .. } => {
            Some("use_full_tool_router".to_string())
        }
        OrchestratorPolicyAction::DeferToFullToolRouter { reason } => {
            Some(reason.as_str().to_string())
        }
        OrchestratorPolicyAction::DowngradeToNormalChatWithContext { .. } => {
            Some("downgrade_to_normal_chat_with_context".to_string())
        }
        OrchestratorPolicyAction::DowngradeToContextBoundary { .. } => {
            Some("downgrade_to_context_boundary".to_string())
        }
        OrchestratorPolicyAction::SafeClarify { .. } => Some("safe_clarify".to_string()),
    };
    diagnostic.fallback_reason = policy
        .fallback_reason()
        .map(|reason| reason.as_str().to_string());
    if diagnostic.selected_route.as_deref() == Some("needs_tool") {
        diagnostic.needs_tool_confidence_threshold = Some(MIN_NEEDS_TOOL_CONFIDENCE);
        diagnostic.needs_tool_accepted =
            Some(matches!(policy, OrchestratorPolicyAction::AcceptDecision));
        diagnostic.needs_tool_policy_action = Some(
            match policy {
                OrchestratorPolicyAction::AcceptDecision => "accept",
                OrchestratorPolicyAction::UseFullToolRouter { .. } => "use_full_tool_router",
                OrchestratorPolicyAction::DeferToFullToolRouter { .. } => {
                    "defer_to_full_tool_router"
                }
                OrchestratorPolicyAction::DowngradeToNormalChatWithContext { .. } => {
                    "downgrade_to_normal_chat_with_context"
                }
                OrchestratorPolicyAction::DowngradeToContextBoundary { .. } => {
                    "downgrade_to_context_boundary"
                }
                OrchestratorPolicyAction::SafeClarify { .. } => "clarify",
            }
            .to_string(),
        );
        diagnostic.needs_tool_policy_reason = policy
            .needs_tool_reason()
            .map(|reason| reason.as_str().to_string())
            .or_else(|| {
                if matches!(policy, OrchestratorPolicyAction::AcceptDecision) {
                    Some("planner_high_confidence".to_string())
                } else {
                    diagnostic.fallback_reason.clone()
                }
            });
        diagnostic.tool_affinity_source = Some(
            needs_tool_affinity_from_confidence(diagnostic.planner_confidence)
                .as_str()
                .to_string(),
        );
    }
    if diagnostic.selected_route.as_deref() == Some("normal_chat") {
        let accepted_directly = matches!(policy, OrchestratorPolicyAction::AcceptDecision);
        diagnostic.normal_chat_policy_action = Some(
            if accepted_directly {
                NormalChatArbitrationAction::AcceptDirectNormalChat
            } else {
                NormalChatArbitrationAction::VerifyWithToolRouter
            }
            .as_str()
            .to_string(),
        );
        diagnostic.normal_chat_policy_reason =
            Some(normal_chat_diagnostic_reason(diagnostic, policy).to_string());
        diagnostic.normal_chat_direct_confidence_threshold =
            Some(NORMAL_CHAT_DIRECT_CONFIDENCE_THRESHOLD);
        diagnostic.normal_chat_accepted_directly = Some(accepted_directly);
        diagnostic.normal_chat_bypassed_tool_router = Some(accepted_directly);
    }
    diagnostic.accepted_decision = matches!(policy, OrchestratorPolicyAction::AcceptDecision);
    diagnostic.planner_empty =
        diagnostic.planner_failure_reason.as_deref() == Some("empty_model_content");
    if matches!(
        policy,
        OrchestratorPolicyAction::UseFullToolRouter { .. }
            | OrchestratorPolicyAction::DeferToFullToolRouter { .. }
            | OrchestratorPolicyAction::SafeClarify { .. }
            | OrchestratorPolicyAction::DowngradeToNormalChatWithContext { .. }
            | OrchestratorPolicyAction::DowngradeToContextBoundary { .. }
    ) {
        diagnostic.used_full_router = matches!(
            policy,
            OrchestratorPolicyAction::UseFullToolRouter { .. }
                | OrchestratorPolicyAction::DeferToFullToolRouter { .. }
        );
    }
    if matches!(
        policy,
        OrchestratorPolicyAction::DowngradeToNormalChatWithContext { .. }
    ) {
        diagnostic.normal_chat_context_injected = true;
    }
    if diagnostic.normal_chat_context_injected {
        diagnostic.normal_chat_bypassed_tool_router = Some(true);
    }
    if diagnostic.used_full_router {
        diagnostic.normal_chat_bypassed_tool_router = Some(false);
    }
    if matches!(
        policy,
        OrchestratorPolicyAction::DowngradeToContextBoundary { .. }
    ) {
        diagnostic.used_context_boundary_fallback = true;
    }
    if diagnostic.used_full_router {
        let reason = if diagnostic.selected_route.as_deref() == Some("needs_tool")
            && diagnostic.needs_tool_accepted == Some(true)
        {
            "accepted_needs_tool_high_confidence".to_string()
        } else if diagnostic.selected_route.as_deref() == Some("normal_chat")
            && diagnostic.normal_chat_policy_action.as_deref() == Some("verify_with_tool_router")
        {
            "normal_chat_arbitration".to_string()
        } else {
            diagnostic
                .fallback_reason
                .clone()
                .unwrap_or_else(|| "needs_tool".to_string())
        };
        diagnostic.tool_router_invoked_reason = Some(reason);
    }
}

fn planner_failure_to_fallback_reason(
    failure_reason: Option<&str>,
) -> Option<OrchestratorFallbackReason> {
    match failure_reason {
        Some("empty_model_content") => Some(OrchestratorFallbackReason::PlannerEmpty),
        Some("timeout") => Some(OrchestratorFallbackReason::PlannerTimeout),
        Some("ollama_unavailable" | "endpoint_config" | "ollama_http_error") => {
            Some(OrchestratorFallbackReason::PlannerUnavailable)
        }
        Some("invalid_json" | "invalid_schema" | "invalid_route" | "invalid_response_schema") => {
            Some(OrchestratorFallbackReason::PlannerMalformed)
        }
        Some(_) => Some(OrchestratorFallbackReason::PlannerMalformed),
        None => None,
    }
}

pub fn now_ms() -> i64 {
    Utc::now().timestamp_millis()
}

fn salience_score_for_diagnostic(working_context: &WorkingContextFrame) -> Option<f32> {
    working_context
        .last_tool_result
        .as_ref()
        .map(|_| working_context.salience.salience_score)
}

fn salience_turn_age_for_diagnostic(working_context: &WorkingContextFrame) -> Option<u32> {
    working_context
        .last_tool_result
        .as_ref()
        .map(|_| working_context.salience.turn_age)
}

fn salience_stale_for_diagnostic(working_context: &WorkingContextFrame) -> Option<bool> {
    working_context
        .last_tool_result
        .as_ref()
        .map(|_| working_context.salience.stale)
}

fn fallback_context_answer(
    tool_result: &ToolResultFrame,
    plan: &ContextAnswerPlan,
    expected_language: AstraUserLanguage,
) -> ContextAnswerOutput {
    let language = match expected_language {
        AstraUserLanguage::English => AstraUserLanguage::English,
        _ => AstraUserLanguage::Italian,
    };
    let boundary_mode = matches!(
        plan.strategy.as_str(),
        "context_boundary" | "normal_chat_with_context" | "general_context_boundary"
    );
    let context_continuation_mode = plan.strategy == "context_continuation";
    let summary = fallback_context_summary(tool_result, language);
    let mut status = if boundary_mode {
        "boundary_answer".to_string()
    } else if context_continuation_mode {
        "insufficient_context".to_string()
    } else {
        "partial".to_string()
    };
    let mut support = if boundary_mode || context_continuation_mode {
        "not_in_context".to_string()
    } else {
        "supported_by_context".to_string()
    };
    let answer = if context_continuation_mode {
        let user_message = plan.focus.as_deref().unwrap_or_default();
        match score_context_evidence_support(user_message, tool_result) {
            ContextEvidenceSupport::Supported { matched_terms } => {
                status = "answered".to_string();
                support = "supported_by_context".to_string();
                let topic = format_context_terms_for_answer(&matched_terms);
                if matches!(language, AstraUserLanguage::English) {
                    format!(
                        "Yes. The evidence from the last session supports a reference to {topic}. In particular, the session covered: {summary}"
                    )
                } else {
                    format!(
                        "Si, nelle evidenze dell'ultima sessione risulta un riferimento a {topic}. In particolare, la sessione trattava: {summary}"
                    )
                }
            }
            ContextEvidenceSupport::NotSupported { checked_terms } => {
                status = "answered".to_string();
                support = "not_in_context".to_string();
                let topic = format_context_terms_for_answer(&checked_terms);
                if matches!(language, AstraUserLanguage::English) {
                    format!(
                        "No. The evidence from the last session does not show a reference to {topic}. The session was instead about: {summary}"
                    )
                } else {
                    format!(
                        "No, nelle evidenze dell'ultima sessione non risulta un riferimento a {topic}. La sessione parlava invece di: {summary}"
                    )
                }
            }
            ContextEvidenceSupport::Ambiguous { .. } => {
                if matches!(language, AstraUserLanguage::English) {
                    format!(
                        "The available transcript evidence does not directly establish that part. From the session evidence, the discussion was about: {summary}. I can separate transcript evidence from general knowledge if you want to continue from there."
                    )
                } else {
                    format!(
                        "Nel transcript disponibile non emerge direttamente questa parte. Dalle evidenze risulta che la sessione parlava di: {summary}. Posso separare le evidenze dalla conoscenza generale se vuoi proseguire da li."
                    )
                }
            }
        }
    } else if boundary_mode {
        let focus = plan
            .focus
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or("questa domanda");
        if matches!(language, AstraUserLanguage::English) {
            format!(
                "The available transcript recap does not directly establish an answer to {focus}. From the prior context, the session was about: {}. I can answer from general knowledge, but that would be separate from the recording evidence.",
                summary
            )
        } else {
            format!(
                "Nel recap disponibile non ho evidenze dirette per rispondere con certezza a {focus}. Dal contesto precedente, la sessione parlava di: {}. Posso darti una spiegazione generale, distinguendola dalle evidenze della registrazione.",
                summary
            )
        }
    } else if matches!(
        plan.strategy.as_str(),
        "verify_entity_against_context" | "compare"
    ) {
        let focus = plan
            .focus
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or("questa affermazione");
        match score_context_evidence_support(focus, tool_result) {
            ContextEvidenceSupport::Supported { .. } => {
                status = "answered".to_string();
                support = "supported_by_context".to_string();
                if matches!(language, AstraUserLanguage::English) {
                    format!(
                        "Yes. Based on the previous recap from {}, the reference to {focus} is supported: {summary}",
                        tool_result.source_label
                    )
                } else {
                    format!(
                        "Si. In base al recap di {}, il riferimento a {focus} risulta supportato: {summary}",
                        tool_result.source_label
                    )
                }
            }
            ContextEvidenceSupport::NotSupported { .. } => {
                status = "answered".to_string();
                support = "not_in_context".to_string();
                if matches!(language, AstraUserLanguage::English) {
                    format!(
                        "No. Based on the previous recap from {}, it was not about {focus}. It was about: {summary}",
                        tool_result.source_label
                    )
                } else {
                    format!(
                        "No, in base al recap di {}, non si parlava di {focus}. Si parlava di: {summary}",
                        tool_result.source_label
                    )
                }
            }
            ContextEvidenceSupport::Ambiguous { .. } => {
                status = "insufficient_context".to_string();
                support = "not_in_context".to_string();
                if matches!(language, AstraUserLanguage::English) {
                    format!(
                        "The previous recap is not specific enough to verify {focus} deterministically. The session was about: {summary}"
                    )
                } else {
                    format!(
                        "Il recap precedente non e abbastanza specifico per verificare {focus} in modo deterministico. La sessione parlava di: {summary}"
                    )
                }
            }
        }
    } else if matches!(language, AstraUserLanguage::English) {
        format!(
            "Based on the previous recap from {}: {}",
            tool_result.source_label, summary
        )
    } else {
        format!(
            "In base al recap di {}: {}",
            tool_result.source_label, summary
        )
    };
    ContextAnswerOutput {
        answer,
        language,
        status,
        support,
        used_context_refs: vec!["last_tool_result".to_string()],
        confidence: 0.62,
        warnings: Vec::new(),
        sanitized_internal_context_refs: false,
    }
}

fn fallback_context_summary(tool_result: &ToolResultFrame, language: AstraUserLanguage) -> String {
    let summary = sanitize_tool_result_answer_summary(&tool_result.answer_summary);
    if !summary.trim().is_empty() {
        return context_broker::bounded_text(&summary, 420);
    }
    let topics = tool_result
        .key_topics
        .iter()
        .take(5)
        .map(|topic| context_broker::bounded_text(topic, 80))
        .collect::<Vec<_>>()
        .join(", ");
    if !topics.trim().is_empty() {
        return topics;
    }
    if matches!(language, AstraUserLanguage::English) {
        "the bounded summary from the previous session".to_string()
    } else {
        "il riepilogo sintetico disponibile della sessione".to_string()
    }
}

fn format_context_terms_for_answer(terms: &[String]) -> String {
    let terms = terms
        .iter()
        .take(3)
        .map(|term| capitalize_context_term(term))
        .collect::<Vec<_>>();
    match terms.as_slice() {
        [] => "questo tema".to_string(),
        [one] => one.clone(),
        [first, second] => format!("{first} e {second}"),
        [first, middle @ .., last] => format!(
            "{}, e {last}",
            [first.as_str()]
                .into_iter()
                .chain(middle.iter().map(String::as_str))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn capitalize_context_term(term: &str) -> String {
    let mut chars = term.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

#[allow(clippy::too_many_arguments)]
fn context_answer_diagnostic(
    selected_route: Option<String>,
    context_ref: Option<String>,
    prompt_char_count: usize,
    prompt_budget_exceeded: bool,
    budget_compaction_applied: bool,
    failure_reason: &str,
    working_context: &WorkingContextFrame,
    expected_language: AstraUserLanguage,
) -> AssistantOrchestratorDiagnostic {
    AssistantOrchestratorDiagnostic {
        request_id: None,
        stage: "context_answer".to_string(),
        planner_stage: "context_answer".to_string(),
        working_context_present: working_context.has_working_context(),
        last_tool_result_present: working_context.last_tool_result.is_some(),
        selected_route,
        context_ref,
        planner_model: None,
        planner_duration_ms: None,
        planner_failure_reason: Some(failure_reason.to_string()),
        planner_confidence: None,
        policy_action: Some("safe_clarify".to_string()),
        fallback_policy: Some("safe_clarify".to_string()),
        fallback_reason: Some("MissingContextForContextAnswer".to_string()),
        planner_empty: false,
        used_context_boundary_fallback: false,
        normal_chat_context_injected: false,
        normal_chat_bypassed_tool_router: None,
        normal_chat_policy_action: None,
        normal_chat_policy_reason: None,
        normal_chat_direct_confidence_threshold: None,
        normal_chat_accepted_directly: None,
        planner_intent_kind: None,
        planner_capability_family: None,
        planner_requires_tool_arbitration: None,
        planner_requires_memory_lookup: None,
        planner_requires_governed_action: None,
        planner_requires_context_boundary: None,
        planner_safe_to_bypass_tools: None,
        tool_router_invoked_reason: None,
        needs_tool_policy_action: None,
        needs_tool_policy_reason: None,
        needs_tool_confidence_threshold: None,
        needs_tool_accepted: None,
        tool_affinity_source: None,
        accepted_decision: false,
        prompt_char_count,
        prompt_budget_exceeded,
        used_full_router: false,
        tool_affinity_risk: None,
        context_salience_score: salience_score_for_diagnostic(working_context),
        context_turn_age: salience_turn_age_for_diagnostic(working_context),
        context_stale: salience_stale_for_diagnostic(working_context),
        context_decay_action: None,
        context_continuation_policy_action: None,
        context_continuation_policy_reason: None,
        context_answer_first_attempted: Some(false),
        context_answer_fallback_used: Some(false),
        context_answer_empty_model_content: Some(false),
        expected_language: Some(expected_language.code().to_string()),
        output_language: None,
        language_mismatch: None,
        language_retry_attempted: Some(false),
        language_retry_succeeded: Some(false),
        budget_compaction_applied: Some(budget_compaction_applied),
        user_facing_context_label: None,
        sanitized_internal_context_refs: false,
        tool_manifest_count: 0,
        metadata_only: true,
    }
}

fn normalize_route(value: &str) -> String {
    value
        .chars()
        .flat_map(char::to_lowercase)
        .filter(|ch| ch.is_alphanumeric())
        .collect()
}

fn extract_json_object(content: &str) -> Option<&str> {
    let start = content.find('{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, ch) in content[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let end = start + offset;
                    return Some(&content[start..=end]);
                }
            }
            _ => {}
        }
    }
    None
}

fn extract_compact_topics(summary: &str, limit: usize) -> Vec<String> {
    let stopwords = [
        "della", "delle", "degli", "dallo", "dalla", "dalle", "about", "with", "that", "this",
        "sessione", "parlava", "abbiamo", "summary", "fonte", "contesto",
    ];
    let mut seen = HashSet::new();
    summary
        .split(|ch: char| !ch.is_alphanumeric())
        .map(str::trim)
        .filter(|token| token.chars().count() >= 5)
        .filter(|token| !stopwords.contains(&token.to_ascii_lowercase().as_str()))
        .filter_map(|token| {
            let normalized = token.to_ascii_lowercase();
            if seen.insert(normalized) {
                Some(token.to_string())
            } else {
                None
            }
        })
        .take(limit)
        .collect()
}

fn extract_compact_entities(summary: &str, limit: usize) -> Vec<String> {
    let mut seen = HashSet::new();
    summary
        .split_whitespace()
        .map(|token| token.trim_matches(|ch: char| !ch.is_alphanumeric()))
        .filter(|token| token.chars().next().is_some_and(char::is_uppercase))
        .filter(|token| token.chars().count() >= 3)
        .filter_map(|token| {
            let normalized = token.to_ascii_lowercase();
            if seen.insert(normalized) {
                Some(token.to_string())
            } else {
                None
            }
        })
        .take(limit)
        .collect()
}

fn dedupe_limited(values: Vec<String>, limit: usize) -> Vec<String> {
    let mut seen = HashSet::new();
    values
        .into_iter()
        .filter_map(|value| non_empty(context_broker::bounded_text(&value, 180)))
        .filter(|value| seen.insert(value.clone()))
        .take(limit)
        .collect()
}

fn non_empty(value: String) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn earth_frame() -> WorkingContextFrame {
        let mut frame = WorkingContextFrame::default();
        frame.update_from_tool_result(ToolResultFrame::compact(
            "work_session.recap",
            "work_session_recap",
            "session_archive_transcript",
            "ultima sessione archiviata",
            Some("session-earth".to_string()),
            vec!["segment:1".to_string(), "segment:2".to_string()],
            2,
            "La sessione parlava della formazione della Terra primordiale, dell'oceano di magma, dell'atmosfera primitiva e dei primi mari.",
            vec!["STT incomplete_drain_timeout".to_string()],
            Some(0.9),
        ));
        frame
    }

    fn attempt_for(
        decision: ConversationOrchestratorDecision,
        confidence: Option<f32>,
        tool_affinity_risk: Option<bool>,
        failure_reason: Option<&str>,
    ) -> OrchestratorPlanAttempt {
        let planner_safety = match &decision {
            ConversationOrchestratorDecision::NormalChat(plan) => PlannerSafetyMetadata {
                intent_kind: plan.intent_kind,
                capability_family: plan.capability_family,
                requires_tool_arbitration: plan.requires_tool_arbitration,
                requires_memory_lookup: plan.requires_memory_lookup,
                requires_governed_action: plan.requires_governed_action,
                requires_context_boundary: plan.requires_context_boundary,
                safe_to_bypass_tools: plan.safe_to_bypass_tools,
            },
            _ => PlannerSafetyMetadata::default(),
        };
        let mut diagnostic = AssistantOrchestratorDiagnostic {
            request_id: None,
            stage: "context_planner".to_string(),
            planner_stage: "discourse_planner".to_string(),
            working_context_present: true,
            last_tool_result_present: true,
            selected_route: Some(decision_route_label(&decision).to_string()),
            context_ref: decision_context_ref(&decision).map(str::to_string),
            planner_model: Some("test-model".to_string()),
            planner_duration_ms: Some(1),
            planner_failure_reason: failure_reason.map(str::to_string),
            planner_confidence: confidence,
            policy_action: None,
            fallback_policy: None,
            fallback_reason: None,
            planner_empty: failure_reason == Some("empty_model_content"),
            used_context_boundary_fallback: false,
            normal_chat_context_injected: matches!(
                decision,
                ConversationOrchestratorDecision::NormalChatWithContext(_)
            ),
            normal_chat_bypassed_tool_router: None,
            normal_chat_policy_action: None,
            normal_chat_policy_reason: None,
            normal_chat_direct_confidence_threshold: None,
            normal_chat_accepted_directly: None,
            planner_intent_kind: None,
            planner_capability_family: None,
            planner_requires_tool_arbitration: None,
            planner_requires_memory_lookup: None,
            planner_requires_governed_action: None,
            planner_requires_context_boundary: None,
            planner_safe_to_bypass_tools: None,
            tool_router_invoked_reason: None,
            needs_tool_policy_action: None,
            needs_tool_policy_reason: None,
            needs_tool_confidence_threshold: None,
            needs_tool_accepted: None,
            tool_affinity_source: None,
            accepted_decision: false,
            prompt_char_count: 1200,
            prompt_budget_exceeded: false,
            used_full_router: matches!(
                decision,
                ConversationOrchestratorDecision::ToolCall(_)
                    | ConversationOrchestratorDecision::DeferToToolRouter(_)
            ),
            tool_affinity_risk,
            context_salience_score: Some(1.0),
            context_turn_age: Some(0),
            context_stale: Some(false),
            context_decay_action: None,
            context_continuation_policy_action: None,
            context_continuation_policy_reason: None,
            context_answer_first_attempted: None,
            context_answer_fallback_used: None,
            context_answer_empty_model_content: None,
            expected_language: None,
            output_language: None,
            language_mismatch: None,
            language_retry_attempted: None,
            language_retry_succeeded: None,
            budget_compaction_applied: Some(false),
            user_facing_context_label: None,
            sanitized_internal_context_refs: false,
            tool_manifest_count: 0,
            metadata_only: true,
        };
        apply_planner_safety_to_diagnostic(&mut diagnostic, &planner_safety);
        OrchestratorPlanAttempt {
            diagnostic,
            decision,
        }
    }

    fn context_plan(confidence: f32) -> ContextAnswerPlan {
        ContextAnswerPlan {
            strategy: "verify_entity_against_context".to_string(),
            focus: Some("Marte".to_string()),
            context_ref: "last_tool_result".to_string(),
            reason_code: "claim_check_against_last_answer".to_string(),
            confidence,
        }
    }

    fn safe_normal_chat_plan(confidence: f32) -> NormalChatPlan {
        NormalChatPlan {
            intent_kind: Some(PlannerIntentKind::OrdinaryQuestion),
            capability_family: Some(PlannerCapabilityFamily::None),
            requires_tool_arbitration: Some(false),
            requires_memory_lookup: Some(false),
            requires_governed_action: Some(false),
            requires_context_boundary: Some(false),
            safe_to_bypass_tools: Some(true),
            context_ref: Some("none".to_string()),
            reason_code: "ordinary_chat".to_string(),
            confidence,
        }
    }

    fn unsafe_normal_chat_plan(confidence: f32) -> NormalChatPlan {
        NormalChatPlan {
            intent_kind: None,
            capability_family: None,
            requires_tool_arbitration: None,
            requires_memory_lookup: None,
            requires_governed_action: None,
            requires_context_boundary: None,
            safe_to_bypass_tools: None,
            context_ref: Some("none".to_string()),
            reason_code: "unsafe_or_legacy_normal_chat".to_string(),
            confidence,
        }
    }

    fn boundary_plan(confidence: f32) -> ContextAnswerPlan {
        ContextAnswerPlan {
            strategy: "context_boundary".to_string(),
            focus: Some("prima dell'impatto iniziale cosa c'era?".to_string()),
            context_ref: "last_tool_result".to_string(),
            reason_code: "contextual_question_beyond_evidence".to_string(),
            confidence,
        }
    }

    #[test]
    fn evidence_grounded_tool_answer_creates_tool_result_frame() {
        let frame = earth_frame();
        let tool = frame.last_tool_result.expect("tool result");

        assert_eq!(tool.source_kind, "session_archive_transcript");
        assert_eq!(tool.session_id.as_deref(), Some("session-earth"));
        assert_eq!(tool.used_evidence_ids, vec!["segment:1", "segment:2"]);
        assert!(tool.answer_summary.contains("Terra primordiale"));
        assert!(tool.key_topics.iter().any(|topic| topic.contains("Terra")));
    }

    #[test]
    fn tool_result_frame_does_not_store_raw_transcript_text_field() {
        let tool = earth_frame().last_tool_result.expect("tool result");
        let serialized = serde_json::to_value(&tool).expect("tool frame json");

        assert!(serialized.get("answer_summary").is_some());
        assert!(serialized.get("transcript").is_none());
        assert!(serialized.get("raw_transcript").is_none());
        assert!(serialized.get("raw_model_output").is_none());
    }

    #[test]
    fn normal_chat_update_preserves_last_grounded_context() {
        let mut frame = earth_frame();
        let previous_tool = frame.last_tool_result.clone();

        frame.update_from_normal_chat("chi sei?", "Sono Astra.");

        assert_eq!(frame.last_tool_result, previous_tool);
        assert_eq!(frame.last_assistant_action.as_deref(), Some("normal_chat"));
    }

    #[test]
    fn planner_answer_from_context_does_not_require_full_router() {
        let frame = earth_frame();
        let parsed = parse_context_planner_output(
            r#"{
              "route": "answer_from_context",
              "context_ref": "last_tool_result",
              "confidence": 0.91,
              "reason_code": "claim_check_against_last_answer",
              "answer_plan": {"strategy": "verify_entity_against_context", "focus": "Marte"}
            }"#,
            &frame,
        );

        assert!(matches!(
            parsed.decision,
            Some(ConversationOrchestratorDecision::AnswerFromContext(_))
        ));
    }

    #[test]
    fn planner_answer_from_context_boundary_parses_without_tool_router() {
        let frame = earth_frame();
        let parsed = parse_context_planner_output(
            r#"{
              "route": "answer_from_context_boundary",
              "context_ref": "last_tool_result",
              "confidence": 0.82,
              "reason_code": "contextual_question_beyond_evidence",
              "answer_plan": {"strategy": "context_boundary", "focus": "prima dell'impatto iniziale"}
            }"#,
            &frame,
        );

        assert!(matches!(
            parsed.decision,
            Some(ConversationOrchestratorDecision::AnswerFromContextBoundary(
                _
            ))
        ));
    }

    #[test]
    fn planner_normal_chat_with_context_parses_without_tool_router() {
        let frame = earth_frame();
        let parsed = parse_context_planner_output(
            r#"{
              "route": "normal_chat_with_context",
              "context_ref": "last_tool_result",
              "confidence": 0.86,
              "reason_code": "general_question_about_current_topic",
              "answer_plan": {"strategy": "none", "focus": null}
            }"#,
            &frame,
        );

        assert!(matches!(
            parsed.decision,
            Some(ConversationOrchestratorDecision::NormalChatWithContext(_))
        ));
    }

    #[test]
    fn policy_normal_chat_below_threshold_falls_back_to_full_router() {
        let frame = WorkingContextFrame::default();
        let attempt = attempt_for(
            ConversationOrchestratorDecision::NormalChat(safe_normal_chat_plan(
                MIN_NORMAL_CHAT_CONFIDENCE - 0.01,
            )),
            Some(MIN_NORMAL_CHAT_CONFIDENCE - 0.01),
            None,
            None,
        );

        assert!(matches!(
            apply_orchestrator_policy(&attempt, &frame),
            OrchestratorPolicyAction::UseFullToolRouter {
                reason: OrchestratorFallbackReason::NormalChatLowConfidence
            }
        ));
    }

    #[test]
    fn policy_normal_chat_above_threshold_is_accepted() {
        let frame = WorkingContextFrame::default();
        let attempt = attempt_for(
            ConversationOrchestratorDecision::NormalChat(safe_normal_chat_plan(
                MIN_NORMAL_CHAT_CONFIDENCE,
            )),
            Some(MIN_NORMAL_CHAT_CONFIDENCE),
            Some(false),
            None,
        );

        assert_eq!(
            apply_orchestrator_policy(&attempt, &frame),
            OrchestratorPolicyAction::AcceptDecision
        );
    }

    #[test]
    fn safe_normal_chat_diagnostic_marks_direct_acceptance() {
        let frame = WorkingContextFrame::default();
        let mut attempt = attempt_for(
            ConversationOrchestratorDecision::NormalChat(safe_normal_chat_plan(0.94)),
            Some(0.94),
            Some(false),
            None,
        );
        attempt.diagnostic.request_id = Some("request-safe-normal-chat".to_string());

        let policy = apply_orchestrator_policy(&attempt, &frame);
        apply_policy_to_diagnostic(&mut attempt.diagnostic, &policy);

        assert_eq!(policy, OrchestratorPolicyAction::AcceptDecision);
        assert_eq!(
            attempt.diagnostic.normal_chat_policy_action.as_deref(),
            Some("accept_direct_normal_chat")
        );
        assert_eq!(
            attempt.diagnostic.normal_chat_policy_reason.as_deref(),
            Some("safe_to_bypass_tools")
        );
        assert_eq!(attempt.diagnostic.normal_chat_accepted_directly, Some(true));
        assert!(!attempt.diagnostic.used_full_router);
        assert_eq!(
            attempt.diagnostic.planner_intent_kind.as_deref(),
            Some("ordinary_question")
        );
        assert_eq!(
            attempt.diagnostic.planner_capability_family.as_deref(),
            Some("none")
        );
        assert_eq!(attempt.diagnostic.planner_safe_to_bypass_tools, Some(true));
        assert!(attempt.diagnostic.request_id.is_some());
        assert!(attempt.diagnostic.metadata_only);
    }

    #[test]
    fn normal_chat_missing_safety_fields_verifies_with_tool_router() {
        let frame = WorkingContextFrame::default();
        let mut attempt = attempt_for(
            ConversationOrchestratorDecision::NormalChat(unsafe_normal_chat_plan(0.95)),
            Some(0.95),
            Some(false),
            None,
        );

        let policy = apply_orchestrator_policy(&attempt, &frame);
        apply_policy_to_diagnostic(&mut attempt.diagnostic, &policy);

        assert!(matches!(
            policy,
            OrchestratorPolicyAction::UseFullToolRouter {
                reason: OrchestratorFallbackReason::NormalChatUnsafeToBypassTools
            }
        ));
        assert_eq!(
            attempt.diagnostic.normal_chat_policy_action.as_deref(),
            Some("verify_with_tool_router")
        );
        assert_eq!(
            attempt.diagnostic.normal_chat_policy_reason.as_deref(),
            Some("missing_safety_fields")
        );
        assert_eq!(
            attempt.diagnostic.tool_router_invoked_reason.as_deref(),
            Some("normal_chat_arbitration")
        );
        assert!(attempt.diagnostic.used_full_router);
    }

    #[test]
    fn normal_chat_safe_to_bypass_none_verifies_with_tool_router() {
        let frame = WorkingContextFrame::default();
        let mut plan = safe_normal_chat_plan(0.95);
        plan.safe_to_bypass_tools = None;
        let attempt = attempt_for(
            ConversationOrchestratorDecision::NormalChat(plan),
            Some(0.95),
            Some(false),
            None,
        );

        assert!(matches!(
            apply_orchestrator_policy(&attempt, &frame),
            OrchestratorPolicyAction::UseFullToolRouter {
                reason: OrchestratorFallbackReason::NormalChatUnsafeToBypassTools
            }
        ));
    }

    #[test]
    fn normal_chat_unknown_capability_verifies_with_tool_router() {
        let frame = WorkingContextFrame::default();
        let mut plan = safe_normal_chat_plan(0.95);
        plan.capability_family = Some(PlannerCapabilityFamily::Unknown);
        let attempt = attempt_for(
            ConversationOrchestratorDecision::NormalChat(plan),
            Some(0.95),
            Some(false),
            None,
        );

        assert!(matches!(
            apply_orchestrator_policy(&attempt, &frame),
            OrchestratorPolicyAction::UseFullToolRouter {
                reason: OrchestratorFallbackReason::NormalChatUnsafeToBypassTools
            }
        ));
    }

    #[test]
    fn normal_chat_requires_governed_action_verifies_with_tool_router() {
        let frame = WorkingContextFrame::default();
        let mut plan = safe_normal_chat_plan(0.95);
        plan.intent_kind = Some(PlannerIntentKind::GovernedAction);
        plan.requires_governed_action = Some(true);
        plan.safe_to_bypass_tools = Some(false);
        let mut attempt = attempt_for(
            ConversationOrchestratorDecision::NormalChat(plan),
            Some(0.95),
            Some(false),
            None,
        );

        let policy = apply_orchestrator_policy(&attempt, &frame);
        apply_policy_to_diagnostic(&mut attempt.diagnostic, &policy);

        assert!(matches!(
            policy,
            OrchestratorPolicyAction::UseFullToolRouter {
                reason: OrchestratorFallbackReason::NormalChatUnsafeToBypassTools
            }
        ));
        assert_eq!(
            attempt.diagnostic.normal_chat_policy_action.as_deref(),
            Some("verify_with_tool_router")
        );
        assert_eq!(
            attempt.diagnostic.planner_requires_governed_action,
            Some(true)
        );
    }

    #[test]
    fn normal_chat_requires_memory_lookup_verifies_with_tool_router() {
        let frame = WorkingContextFrame::default();
        let mut plan = safe_normal_chat_plan(0.95);
        plan.intent_kind = Some(PlannerIntentKind::SessionMemoryQuery);
        plan.capability_family = Some(PlannerCapabilityFamily::SessionMemory);
        plan.requires_memory_lookup = Some(true);
        plan.safe_to_bypass_tools = Some(false);
        let attempt = attempt_for(
            ConversationOrchestratorDecision::NormalChat(plan),
            Some(0.95),
            Some(false),
            None,
        );

        assert!(matches!(
            apply_orchestrator_policy(&attempt, &frame),
            OrchestratorPolicyAction::UseFullToolRouter {
                reason: OrchestratorFallbackReason::NormalChatUnsafeToBypassTools
            }
        ));
    }

    #[test]
    fn normal_chat_work_session_capability_verifies_with_tool_router() {
        let frame = WorkingContextFrame::default();
        let mut plan = safe_normal_chat_plan(0.95);
        plan.capability_family = Some(PlannerCapabilityFamily::WorkSession);
        plan.safe_to_bypass_tools = Some(false);
        let attempt = attempt_for(
            ConversationOrchestratorDecision::NormalChat(plan),
            Some(0.95),
            Some(false),
            None,
        );

        assert!(matches!(
            apply_orchestrator_policy(&attempt, &frame),
            OrchestratorPolicyAction::UseFullToolRouter {
                reason: OrchestratorFallbackReason::NormalChatUnsafeToBypassTools
            }
        ));
    }

    #[test]
    fn policy_answer_from_context_below_threshold_uses_full_router() {
        let frame = earth_frame();
        let attempt = attempt_for(
            ConversationOrchestratorDecision::AnswerFromContext(context_plan(
                MIN_ANSWER_FROM_CONTEXT_CONFIDENCE - 0.01,
            )),
            Some(MIN_ANSWER_FROM_CONTEXT_CONFIDENCE - 0.01),
            None,
            None,
        );

        assert!(matches!(
            apply_orchestrator_policy(&attempt, &frame),
            OrchestratorPolicyAction::UseFullToolRouter {
                reason: OrchestratorFallbackReason::AnswerFromContextLowConfidence
            }
        ));
    }

    #[test]
    fn policy_answer_from_context_without_last_tool_result_uses_full_router() {
        let frame = WorkingContextFrame::default();
        let attempt = attempt_for(
            ConversationOrchestratorDecision::AnswerFromContext(context_plan(0.93)),
            Some(0.93),
            None,
            None,
        );

        assert!(matches!(
            apply_orchestrator_policy(&attempt, &frame),
            OrchestratorPolicyAction::UseFullToolRouter {
                reason: OrchestratorFallbackReason::NoLastToolResult
            }
        ));
    }

    #[test]
    fn planner_empty_with_last_tool_result_presence_check_uses_context_answer_first() {
        let frame = earth_frame();
        let decision = planner_failure_fallback(
            &frame,
            "empty_model_content",
            Some("quindi si parlava della luna?"),
        );

        assert!(matches!(
            decision,
            ConversationOrchestratorDecision::AnswerFromContext(_)
        ));
        assert!(!matches!(
            decision,
            ConversationOrchestratorDecision::ToolCall(_)
        ));
    }

    #[test]
    fn planner_empty_with_last_tool_result_general_question_uses_normal_chat_with_context() {
        let frame = earth_frame();
        let decision = planner_failure_fallback(
            &frame,
            "empty_model_content",
            Some("ma la terra come si e formata?"),
        );

        assert!(matches!(
            decision,
            ConversationOrchestratorDecision::NormalChatWithContext(_)
        ));
    }

    #[test]
    fn planner_empty_with_last_tool_result_boundary_question_uses_normal_chat_with_context() {
        let frame = earth_frame();
        let decision = planner_failure_fallback(
            &frame,
            "empty_model_content",
            Some("prima degli impatti cosa c'era?"),
        );

        assert!(matches!(
            decision,
            ConversationOrchestratorDecision::NormalChatWithContext(_)
        ));
    }

    #[test]
    fn context_attachment_distinguishes_presence_from_general_topic_overlap() {
        let frame = earth_frame();
        assert_eq!(
            classify_context_attachment("parlava della terra la registrazione?", &frame),
            ContextAttachmentKind::TopicPresenceCheck
        );
        assert_eq!(
            classify_context_attachment("ma la terra come si e formata?", &frame),
            ContextAttachmentKind::BoundaryGeneralKnowledge
        );
    }

    #[test]
    fn planner_empty_with_governed_action_defers_to_tool_router() {
        let frame = earth_frame();
        let decision = planner_failure_fallback(
            &frame,
            "empty_model_content",
            Some("iniziamo una nuova sessione di registrazione"),
        );

        assert!(matches!(
            decision,
            ConversationOrchestratorDecision::DeferToToolRouter(_)
        ));
    }

    #[test]
    fn context_policy_downgrades_answer_from_context_general_question() {
        let frame = earth_frame();
        let decision = ConversationOrchestratorDecision::AnswerFromContext(ContextAnswerPlan {
            strategy: "context_continuation".to_string(),
            focus: Some("ma la terra come si e formata?".to_string()),
            context_ref: "last_tool_result".to_string(),
            reason_code: "model_selected_context".to_string(),
            confidence: 0.91,
        });

        let policy = apply_context_continuation_policy(
            &decision,
            &frame,
            None,
            Some(0.91),
            Some("ma la terra come si e formata?"),
        )
        .expect("context policy");

        assert_eq!(
            policy.action,
            ContextContinuationPolicyAction::UseNormalChatWithContext
        );
        assert_eq!(
            policy.reason,
            ContextContinuationPolicyReason::PlannerFailureBoundaryGeneralKnowledge
        );
    }

    #[test]
    fn context_continuation_policy_promotes_zero_confidence_contextual_chat() {
        let frame = earth_frame();
        let decision =
            ConversationOrchestratorDecision::NormalChatWithContext(ContextualChatPlan {
                context_ref: "last_tool_result".to_string(),
                reason_code: "planner_empty_contextual_followup".to_string(),
                confidence: 0.0,
            });

        let policy = apply_context_continuation_policy(&decision, &frame, None, Some(0.0), Some("quindi si parlava della terra?"))
            .expect("context policy");

        assert_eq!(
            policy.action,
            ContextContinuationPolicyAction::UseContextAnswerFirst
        );
        assert_eq!(
            policy.reason,
            ContextContinuationPolicyReason::LowConfidenceContextualChat
        );
    }

    #[test]
    fn planner_empty_without_last_tool_result_defers_to_full_router() {
        let frame = WorkingContextFrame::default();
        let decision = planner_failure_fallback(&frame, "empty_model_content", None);
        let mut attempt = attempt_for(decision, None, None, Some("empty_model_content"));
        attempt.diagnostic.working_context_present = frame.has_working_context();
        attempt.diagnostic.last_tool_result_present = frame.last_tool_result.is_some();
        attempt.diagnostic.context_salience_score = None;
        attempt.diagnostic.context_turn_age = None;
        attempt.diagnostic.context_stale = None;

        let policy = apply_orchestrator_policy(&attempt, &frame);
        apply_policy_to_diagnostic(&mut attempt.diagnostic, &policy);

        assert!(matches!(
            attempt.decision,
            ConversationOrchestratorDecision::DeferToToolRouter(_)
        ));
        assert!(matches!(
            policy,
            OrchestratorPolicyAction::DeferToFullToolRouter {
                reason: OrchestratorFallbackReason::PlannerEmptyNoGroundedContext
            }
        ));
        assert_eq!(
            attempt.diagnostic.selected_route.as_deref(),
            Some("defer_to_tool_router")
        );
        assert_eq!(
            attempt.diagnostic.policy_action.as_deref(),
            Some("defer_to_full_tool_router")
        );
        assert_eq!(
            attempt.diagnostic.fallback_policy.as_deref(),
            Some("planner_empty_no_grounded_context")
        );
        assert_eq!(
            attempt.diagnostic.tool_router_invoked_reason.as_deref(),
            Some("planner_empty_no_grounded_context")
        );
        assert!(attempt.diagnostic.used_full_router);
        assert!(!attempt.diagnostic.accepted_decision);
    }

    #[test]
    fn planner_empty_toolcall_policy_does_not_use_full_router_with_last_tool_result() {
        let frame = earth_frame();
        let attempt = attempt_for(
            ConversationOrchestratorDecision::ToolCall(AssistantRouteDecision::NormalChat),
            None,
            None,
            Some("empty_model_content"),
        );

        assert!(matches!(
            apply_orchestrator_policy(&attempt, &frame),
            OrchestratorPolicyAction::SafeClarify {
                reason: OrchestratorFallbackReason::PlannerEmpty
            }
        ));
    }

    #[test]
    fn policy_clarify_below_threshold_uses_full_router() {
        let frame = WorkingContextFrame::default();
        let attempt = attempt_for(
            ConversationOrchestratorDecision::Clarify(ClarificationPlan {
                reason_code: "unclear".to_string(),
                message: "Mi serve un riferimento in piu.".to_string(),
                confidence: MIN_CLARIFY_CONFIDENCE - 0.01,
            }),
            Some(MIN_CLARIFY_CONFIDENCE - 0.01),
            None,
            None,
        );

        assert!(matches!(
            apply_orchestrator_policy(&attempt, &frame),
            OrchestratorPolicyAction::UseFullToolRouter {
                reason: OrchestratorFallbackReason::ClarifyLowConfidence
            }
        ));
    }

    #[test]
    fn policy_refuse_below_threshold_uses_full_router() {
        let frame = WorkingContextFrame::default();
        let attempt = attempt_for(
            ConversationOrchestratorDecision::Refuse(RefusalPlan {
                reason_code: "unsafe".to_string(),
                message: "Non posso gestire questa richiesta.".to_string(),
                confidence: MIN_REFUSE_CONFIDENCE - 0.01,
            }),
            Some(MIN_REFUSE_CONFIDENCE - 0.01),
            None,
            None,
        );

        assert!(matches!(
            apply_orchestrator_policy(&attempt, &frame),
            OrchestratorPolicyAction::UseFullToolRouter {
                reason: OrchestratorFallbackReason::RefuseLowConfidence
            }
        ));
    }

    #[test]
    fn latest_session_low_confidence_normal_chat_cannot_skip_full_router() {
        let frame = WorkingContextFrame::default();
        let mut attempt = attempt_for(
            ConversationOrchestratorDecision::NormalChat(safe_normal_chat_plan(0.64)),
            Some(0.64),
            None,
            None,
        );

        let policy = apply_orchestrator_policy(&attempt, &frame);
        apply_policy_to_diagnostic(&mut attempt.diagnostic, &policy);

        assert!(matches!(
            policy,
            OrchestratorPolicyAction::UseFullToolRouter {
                reason: OrchestratorFallbackReason::NormalChatLowConfidence
            }
        ));
        assert_eq!(
            attempt.diagnostic.selected_route.as_deref(),
            Some("normal_chat")
        );
        assert_eq!(attempt.diagnostic.planner_confidence, Some(0.64));
        assert_eq!(
            attempt.diagnostic.policy_action.as_deref(),
            Some("use_full_tool_router")
        );
        assert_eq!(
            attempt.diagnostic.fallback_reason.as_deref(),
            Some("NormalChatLowConfidence")
        );
        assert!(attempt.diagnostic.used_full_router);
        assert!(!attempt.diagnostic.accepted_decision);
    }

    #[test]
    fn context_followup_high_confidence_answer_from_context_is_accepted() {
        let frame = earth_frame();
        let attempt = attempt_for(
            ConversationOrchestratorDecision::AnswerFromContext(context_plan(0.91)),
            Some(0.91),
            None,
            None,
        );

        assert_eq!(
            apply_orchestrator_policy(&attempt, &frame),
            OrchestratorPolicyAction::AcceptDecision
        );
    }

    #[test]
    fn orchestrator_policy_diagnostic_fields_are_metadata_only() {
        let frame = WorkingContextFrame::default();
        let mut attempt = attempt_for(
            ConversationOrchestratorDecision::NormalChat(safe_normal_chat_plan(0.42)),
            Some(0.42),
            Some(true),
            None,
        );
        let policy = apply_orchestrator_policy(&attempt, &frame);
        apply_policy_to_diagnostic(&mut attempt.diagnostic, &policy);
        let serialized = serde_json::to_value(&attempt.diagnostic).expect("diagnostic json");

        assert_eq!(serialized["metadata_only"], true);
        let confidence = serialized["planner_confidence"]
            .as_f64()
            .expect("planner confidence");
        assert!((confidence - 0.42).abs() < 0.001);
        assert_eq!(serialized["policy_action"], "use_full_tool_router");
        assert_eq!(serialized["fallback_reason"], "NormalChatLowConfidence");
        assert_eq!(serialized["accepted_decision"], false);
        assert_eq!(serialized["used_full_router"], true);
        assert_eq!(serialized["tool_affinity_risk"], true);
        assert!(serialized.get("context_salience_score").is_some());
        assert!(serialized.get("context_turn_age").is_some());
        assert!(serialized.get("context_stale").is_some());
        assert!(serialized.get("needs_tool_policy_action").is_some());
        assert!(serialized
            .get("context_continuation_policy_action")
            .is_some());
        assert!(serialized.get("context_answer_fallback_used").is_some());
        assert!(serialized.get("user_message").is_none());
        assert!(serialized.get("raw_prompt").is_none());
        assert!(serialized.get("raw_model_output").is_none());
        assert!(serialized.get("transcript_text").is_none());
        assert!(serialized.get("answer").is_none());
    }

    #[test]
    fn planner_needs_tool_invokes_full_router_path() {
        let frame = earth_frame();
        let parsed = parse_context_planner_output(
            r#"{
              "route": "needs_tool",
              "context_ref": "none",
              "confidence": 0.88,
              "reason_code": "show_evidence_requested",
              "answer_plan": {"strategy": "none", "focus": null}
            }"#,
            &frame,
        );

        assert!(matches!(
            parsed.decision,
            Some(ConversationOrchestratorDecision::ToolCall(_))
        ));
    }

    #[test]
    fn valid_needs_tool_policy_invokes_full_router_path() {
        let frame = earth_frame();
        let mut attempt = attempt_for(
            ConversationOrchestratorDecision::ToolCall(AssistantRouteDecision::NormalChat),
            Some(0.91),
            None,
            None,
        );
        let policy = apply_orchestrator_policy(&attempt, &frame);
        apply_policy_to_diagnostic(&mut attempt.diagnostic, &policy);

        assert_eq!(policy, OrchestratorPolicyAction::AcceptDecision);
        assert_eq!(attempt.diagnostic.needs_tool_accepted, Some(true));
        assert_eq!(
            attempt.diagnostic.needs_tool_policy_reason.as_deref(),
            Some("planner_high_confidence")
        );
        assert_eq!(
            attempt.diagnostic.tool_router_invoked_reason.as_deref(),
            Some("accepted_needs_tool_high_confidence")
        );
    }

    #[test]
    fn low_confidence_needs_tool_with_context_downgrades_to_contextual_chat() {
        let frame = earth_frame();
        let mut attempt = attempt_for(
            ConversationOrchestratorDecision::ToolCall(AssistantRouteDecision::NormalChat),
            Some(0.31),
            None,
            None,
        );
        let policy = apply_orchestrator_policy(&attempt, &frame);
        apply_policy_to_diagnostic(&mut attempt.diagnostic, &policy);

        assert!(matches!(
            policy,
            OrchestratorPolicyAction::DowngradeToNormalChatWithContext {
                reason: NeedsToolPolicyReason::LowConfidenceWithLastToolResult
            }
        ));
        assert_eq!(attempt.diagnostic.needs_tool_accepted, Some(false));
        assert_eq!(
            attempt.diagnostic.needs_tool_policy_action.as_deref(),
            Some("downgrade_to_normal_chat_with_context")
        );
        assert_eq!(
            attempt.diagnostic.needs_tool_policy_reason.as_deref(),
            Some("low_confidence_with_last_tool_result")
        );
        assert!(!attempt.diagnostic.used_full_router);
        assert!(attempt.diagnostic.normal_chat_context_injected);
    }

    #[test]
    fn normal_chat_with_context_diagnostic_marks_tool_router_bypass() {
        let frame = earth_frame();
        let mut attempt = attempt_for(
            ConversationOrchestratorDecision::NormalChatWithContext(ContextualChatPlan {
                context_ref: "last_tool_result".to_string(),
                reason_code: "contextual_general_answer".to_string(),
                confidence: 0.9,
            }),
            Some(0.9),
            None,
            None,
        );
        let policy = apply_orchestrator_policy(&attempt, &frame);
        apply_policy_to_diagnostic(&mut attempt.diagnostic, &policy);

        assert_eq!(policy, OrchestratorPolicyAction::AcceptDecision);
        assert!(attempt.diagnostic.normal_chat_context_injected);
        assert_eq!(
            attempt.diagnostic.normal_chat_bypassed_tool_router,
            Some(true)
        );
        assert!(!attempt.diagnostic.used_full_router);
    }

    #[test]
    fn explicit_slash_command_accepts_needs_tool_policy_without_confidence() {
        let decision = apply_needs_tool_policy(&NeedsToolPolicyInput {
            planner_confidence: None,
            context_ref: None,
            last_tool_result_present: false,
            pending_tool_action: false,
            explicit_user_action: false,
            slash_command: true,
            ui_action: false,
            planner_reason_code: Some("explicit_shortcut".to_string()),
            tool_affinity: ToolAffinitySignal::ExplicitSlashCommand,
        });

        assert!(matches!(
            decision,
            NeedsToolPolicyDecision::Accept {
                reason: NeedsToolPolicyReason::ExplicitSlashCommand
            }
        ));
    }

    #[test]
    fn stale_context_is_not_accepted_for_answer_from_context() {
        let mut frame = earth_frame();
        for index in 0..CONTEXT_SALIENCE_STALE_NORMAL_TURNS {
            frame.update_from_normal_chat(
                &format!("unrelated {index}"),
                "Una risposta ordinaria senza contesto Work Session.",
            );
        }
        let attempt = attempt_for(
            ConversationOrchestratorDecision::AnswerFromContext(context_plan(0.91)),
            Some(0.91),
            None,
            None,
        );

        assert!(frame.last_tool_result.is_some());
        assert!(frame.salience.stale);
        assert!(matches!(
            apply_orchestrator_policy(&attempt, &frame),
            OrchestratorPolicyAction::UseFullToolRouter {
                reason: OrchestratorFallbackReason::NoLastToolResult
            }
        ));
    }

    #[test]
    fn salience_starts_high_reinforces_and_decays() {
        let mut frame = earth_frame();
        assert_eq!(frame.salience.salience_score, 1.0);
        assert!(!frame.salience.stale);

        frame.update_from_context_answer(
            "quindi si parlava della terra?",
            "Risposta dal contesto precedente.",
        );
        assert!(frame.salience.salience_score >= 0.85);
        assert_eq!(frame.salience.normal_chat_turns_since_update, 0);

        frame.update_from_context_answer(
            "ma la terra come si e formata?",
            "Risposta generale con contesto precedente.",
        );
        assert!(frame.salience.salience_score < 1.0);
        assert_eq!(frame.salience.normal_chat_turns_since_update, 1);

        frame.update_from_normal_chat("chi sei?", "Sono Astra.");
        assert!(frame.salience.salience_score < 0.85);
        assert_eq!(frame.salience.normal_chat_turns_since_update, 2);
    }

    #[test]
    fn new_tool_result_replaces_stale_context_salience() {
        let mut frame = earth_frame();
        for index in 0..CONTEXT_SALIENCE_STALE_NORMAL_TURNS {
            frame.update_from_normal_chat(&format!("topic {index}"), "normal chat");
        }
        assert!(frame.salience.stale);

        frame.update_from_tool_result(ToolResultFrame::compact(
            "work_session.recall",
            "work_session_recall",
            "session_archive_transcript",
            "sessione archiviata",
            Some("session-new".to_string()),
            vec!["segment:new".to_string()],
            1,
            "Nuovo risultato grounded dalla memoria sessione.",
            Vec::new(),
            Some(0.88),
        ));

        assert!(!frame.salience.stale);
        assert_eq!(frame.salience.turn_age, 0);
        assert_eq!(frame.salience.salience_score, 1.0);
        assert_eq!(
            frame
                .last_tool_result
                .as_ref()
                .and_then(|tool| tool.session_id.as_deref()),
            Some("session-new")
        );
    }

    #[test]
    fn planner_normal_chat_routes_normal_chat() {
        let frame = earth_frame();
        let parsed = parse_context_planner_output(
            r#"{
              "route": "normal_chat",
              "context_ref": "none",
              "confidence": 0.95,
              "reason_code": "new_general_topic",
              "answer_plan": {"strategy": "none", "focus": null}
            }"#,
            &frame,
        );

        assert!(matches!(
            parsed.decision,
            Some(ConversationOrchestratorDecision::NormalChat(_))
        ));
    }

    #[test]
    fn planner_safety_fields_parse_into_typed_metadata() {
        let frame = WorkingContextFrame::default();
        let parsed = parse_context_planner_output(
            r#"{
              "route": "normal_chat",
              "intent_kind": "ordinary_question",
              "capability_family": "none",
              "requires_tool_arbitration": false,
              "requires_memory_lookup": false,
              "requires_governed_action": false,
              "requires_context_boundary": false,
              "safe_to_bypass_tools": true,
              "context_ref": "none",
              "confidence": 0.94,
              "reason_code": "ordinary_question",
              "answer_plan": {"strategy": "none", "focus": null}
            }"#,
            &frame,
        );

        assert_eq!(
            parsed.planner_safety.intent_kind,
            Some(PlannerIntentKind::OrdinaryQuestion)
        );
        assert_eq!(
            parsed.planner_safety.capability_family,
            Some(PlannerCapabilityFamily::None)
        );
        assert_eq!(parsed.planner_safety.safe_to_bypass_tools, Some(true));
        match parsed.decision.expect("normal chat decision") {
            ConversationOrchestratorDecision::NormalChat(plan) => {
                assert_eq!(plan.intent_kind, Some(PlannerIntentKind::OrdinaryQuestion));
                assert_eq!(plan.capability_family, Some(PlannerCapabilityFamily::None));
                assert_eq!(plan.safe_to_bypass_tools, Some(true));
            }
            other => panic!("expected normal chat, got {other:?}"),
        }
    }

    #[test]
    fn legacy_normal_chat_schema_is_not_directly_safe() {
        let frame = WorkingContextFrame::default();
        let parsed = parse_context_planner_output(
            r#"{
              "route": "normal_chat",
              "confidence": 0.95,
              "reason_code": "legacy_schema"
            }"#,
            &frame,
        );
        let decision = parsed.decision.expect("normal chat decision");
        let mut attempt = attempt_for(decision, parsed.planner_confidence, Some(false), None);
        apply_planner_safety_to_diagnostic(&mut attempt.diagnostic, &parsed.planner_safety);

        let policy = apply_orchestrator_policy(&attempt, &frame);
        apply_policy_to_diagnostic(&mut attempt.diagnostic, &policy);

        assert!(matches!(
            policy,
            OrchestratorPolicyAction::UseFullToolRouter {
                reason: OrchestratorFallbackReason::NormalChatUnsafeToBypassTools
            }
        ));
        assert_eq!(
            attempt.diagnostic.normal_chat_policy_action.as_deref(),
            Some("verify_with_tool_router")
        );
    }

    #[test]
    fn ordinary_normal_chat_is_not_hijacked_by_context() {
        let frame = earth_frame();
        let parsed = parse_context_planner_output(
            r#"{
              "route": "normal_chat",
              "context_ref": "none",
              "confidence": 0.95,
              "reason_code": "ordinary_identity_question",
              "answer_plan": {"strategy": "none", "focus": null}
            }"#,
            &frame,
        );
        let decision = parsed.decision.expect("normal chat decision");

        assert_eq!(decision_route_label(&decision), "normal_chat");
    }

    #[test]
    fn malformed_planner_output_falls_back_without_phrase_routing() {
        let frame = earth_frame();
        let parsed = parse_context_planner_output("not json", &frame);

        assert!(parsed.decision.is_none());
        assert_eq!(parsed.failure_reason.as_deref(), Some("invalid_json"));
    }

    #[test]
    fn production_routing_has_no_banned_phrase_checks() {
        let files = [
            (
                "conversation_orchestrator.rs",
                include_str!("conversation_orchestrator.rs"),
            ),
            ("context_broker.rs", include_str!("context_broker.rs")),
            (
                "assistant_tool_router.rs",
                include_str!("assistant_tool_router.rs"),
            ),
            ("work_session_chat.rs", include_str!("work_session_chat.rs")),
            ("lib.rs", include_str!("lib.rs")),
        ];
        let banned_patterns = [
            ".contains(\"registrazione\")",
            ".contains(\"sessione\")",
            ".contains(\"recap\")",
            ".contains(\"trascrizione\")",
            ".contains(\"iniziamo\")",
            ".contains(\"luna\")",
            ".contains(\"terra\")",
            ".contains(\"sole\")",
            ".contains(\"marte\")",
            "contains_any(",
        ];

        for (file_name, contents) in files {
            let production = contents.split("#[cfg(test)]").next().unwrap_or(contents);
            for pattern in banned_patterns {
                assert!(
                    !production.contains(pattern),
                    "{file_name} contains banned routing pattern {pattern}"
                );
            }
        }
    }

    #[test]
    fn context_answer_mars_followup_returns_grounded_no() {
        let frame = earth_frame();
        let tool = frame.last_tool_result.as_ref().expect("tool result");
        let plan = ContextAnswerPlan {
            strategy: "verify_entity_against_context".to_string(),
            focus: Some("Marte".to_string()),
            context_ref: "last_tool_result".to_string(),
            reason_code: "claim_check_against_last_answer".to_string(),
            confidence: 0.91,
        };
        let output = fallback_context_answer(tool, &plan, AstraUserLanguage::Italian);
        let rendered = render_context_answer(tool, &output);

        assert!(rendered.contains("Fonte: ultima sessione archiviata"));
        assert!(rendered.contains("No,"));
        assert!(rendered.contains("Terra primordiale"));
        assert!(rendered.contains("Marte"));
        assert!(!rendered.contains("last_tool_result"));
    }

    #[test]
    fn extract_context_query_terms_keeps_meaningful_topic_only() {
        assert_eq!(
            extract_context_query_terms("quindi si parlava della luna?"),
            vec!["luna"]
        );
        assert_eq!(
            extract_context_query_terms("parlava della terra?"),
            vec!["terra"]
        );
    }

    #[test]
    fn fallback_context_answer_topic_supported_returns_yes() {
        let frame = earth_frame();
        let tool = frame.last_tool_result.as_ref().expect("tool result");
        let plan = context_continuation_plan(
            "planner_empty_with_salient_tool_result",
            Some("parlava della terra?"),
        );

        let output = fallback_context_answer(tool, &plan, AstraUserLanguage::Italian);

        assert_eq!(output.status, "answered");
        assert_eq!(output.support, "supported_by_context");
        assert!(output.answer.starts_with("Si,"));
        assert!(output.answer.contains("Terra"));
        assert!(output.answer.contains("oceano di magma"));
    }

    #[test]
    fn fallback_context_answer_topic_not_supported_luna_returns_no() {
        let frame = earth_frame();
        let tool = frame.last_tool_result.as_ref().expect("tool result");
        let plan = context_continuation_plan(
            "planner_empty_with_salient_tool_result",
            Some("quindi si parlava della luna?"),
        );

        let output = fallback_context_answer(tool, &plan, AstraUserLanguage::Italian);

        assert_eq!(output.status, "answered");
        assert_eq!(output.support, "not_in_context");
        assert!(output.answer.starts_with("No,"));
        assert!(output.answer.contains("Luna"));
        assert!(output.answer.contains("Terra primordiale"));
    }

    #[test]
    fn fallback_context_answer_topic_not_supported_sole_returns_no() {
        let frame = earth_frame();
        let tool = frame.last_tool_result.as_ref().expect("tool result");
        let plan = context_continuation_plan(
            "planner_empty_with_salient_tool_result",
            Some("quindi si parlava del sole?"),
        );

        let output = fallback_context_answer(tool, &plan, AstraUserLanguage::Italian);

        assert!(output.answer.starts_with("No,"));
        assert!(output.answer.contains("Sole"));
        assert!(output.answer.contains("Terra primordiale"));
    }

    #[test]
    fn fallback_context_answer_ambiguous_open_boundary_question_does_not_false_no() {
        let frame = earth_frame();
        let tool = frame.last_tool_result.as_ref().expect("tool result");
        let plan = context_continuation_plan(
            "planner_empty_with_salient_tool_result",
            Some("prima degli impatti cosa c'era?"),
        );

        let output = fallback_context_answer(tool, &plan, AstraUserLanguage::Italian);

        assert_eq!(output.status, "insufficient_context");
        assert_eq!(output.support, "not_in_context");
        assert!(!output.answer.starts_with("No,"));
        assert!(output.answer.contains("Nel transcript disponibile"));
        assert!(output.answer.contains("Terra primordiale"));
    }

    #[test]
    fn fallback_context_answer_does_not_emit_internal_warning() {
        let frame = earth_frame();
        let tool = frame.last_tool_result.as_ref().expect("tool result");
        let plan = context_continuation_plan(
            "planner_empty_with_salient_tool_result",
            Some("quindi si parlava della luna?"),
        );

        let output = fallback_context_answer(tool, &plan, AstraUserLanguage::Italian);
        let rendered = render_context_answer(tool, &output);

        assert!(output.warnings.is_empty());
        assert!(!rendered.contains("context_answer_synthesizer_fallback"));
    }

    #[test]
    fn tool_result_summary_sanitizer_removes_stt_completeness() {
        let cleaned = sanitize_tool_result_answer_summary(
            "Fonte: ultima sessione archiviata\nSTT completeness: incomplete_drain_timeout\nLa sessione parlava della Terra primordiale.",
        );

        assert!(!cleaned.contains("STT completeness"));
        assert!(!cleaned.contains("Fonte:"));
        assert!(cleaned.contains("Terra primordiale"));
    }

    #[test]
    fn tool_result_summary_sanitizer_removes_evidence_ids() {
        let cleaned = sanitize_tool_result_answer_summary(
            "La sessione parlava della Terra. Evidenze: segment:dc9e21f2-0000-0000-0000-000000000000 9d403d4c-0000-0000-0000-000000000000",
        );

        assert!(!cleaned.contains("segment:"));
        assert!(!cleaned.contains("9d403d4c-0000-0000-0000-000000000000"));
        assert!(cleaned.contains("Terra"));
    }

    #[test]
    fn context_boundary_answer_marks_evidence_boundary() {
        let frame = earth_frame();
        let tool = frame.last_tool_result.as_ref().expect("tool result");
        let output =
            fallback_context_answer(tool, &boundary_plan(0.82), AstraUserLanguage::Italian);
        let rendered = render_context_answer(tool, &output);

        assert_eq!(output.status, "boundary_answer");
        assert_eq!(output.support, "not_in_context");
        assert!(rendered.contains("Fonte: ultima sessione archiviata"));
        assert!(rendered.contains("non ho evidenze dirette"));
        assert!(rendered.contains("registrazione"));
        assert!(!rendered.contains("routing tool-aware"));
    }

    #[test]
    fn normal_chat_with_context_preamble_is_compact_and_boundary_aware() {
        let frame = earth_frame();
        let plan = ContextualChatPlan {
            context_ref: "last_tool_result".to_string(),
            reason_code: "general_question_about_current_topic".to_string(),
            confidence: 0.91,
        };
        let preamble =
            build_normal_chat_with_context_preamble(&frame, &plan).expect("context preamble");

        assert!(preamble.contains("recap Work Session"));
        assert!(preamble.contains("evidenze della registrazione"));
        assert!(!preamble.contains("segment:1"));
        assert!(preamble.chars().count() < 900);
    }

    #[test]
    fn context_answer_rejects_unknown_context_ref() {
        assert!(parse_context_answer_output(
            r#"{"answer":"No","status":"answered","used_context_refs":["raw_transcript"],"confidence":0.9,"warnings":[]}"#
        )
        .is_none());
    }

    #[test]
    fn context_answer_parser_requires_non_empty_answer() {
        assert!(parse_context_answer_output(
            r#"{"answer":"","status":"answered","used_context_refs":["last_tool_result"],"confidence":0.9,"warnings":[]}"#
        )
        .is_none());
    }

    #[test]
    fn language_detector_identifies_italian_followups() {
        assert_eq!(
            detect_user_language("quindi si parlava di marte?"),
            AstraUserLanguage::Italian
        );
        assert_eq!(
            detect_user_language("di cosa abbiamo parlato nell'ultima sessione?"),
            AstraUserLanguage::Italian
        );
        assert_eq!(
            detect_user_language("mi fai un recap?"),
            AstraUserLanguage::Italian
        );
    }

    #[test]
    fn language_detector_identifies_english_followups() {
        assert_eq!(
            detect_user_language("so was it about Mars?"),
            AstraUserLanguage::English
        );
        assert_eq!(
            detect_user_language("what did we discuss in the last recording?"),
            AstraUserLanguage::English
        );
    }

    #[test]
    fn italian_expected_language_detects_english_context_answer_mismatch() {
        let output = parse_context_answer_output(
            r#"{
              "answer": "No, the discussion was about the earliest phase of Earth, not Mars.",
              "language": "en",
              "status": "answered",
              "used_context_refs": ["last_tool_result"],
              "confidence": 0.95,
              "warnings": []
            }"#,
        )
        .expect("context answer output");

        assert_eq!(output.language, AstraUserLanguage::English);
        assert!(context_answer_language_mismatch(
            AstraUserLanguage::Italian,
            effective_output_language(&output)
        ));
    }

    #[test]
    fn corrected_italian_context_answer_is_accepted() {
        let output = parse_context_answer_output(
            r#"{
              "answer": "No, in base al recap della sessione archiviata, si parlava della fase primordiale della Terra, non di Marte.",
              "language": "it",
              "status": "answered",
              "used_context_refs": ["last_tool_result"],
              "confidence": 0.91,
              "warnings": []
            }"#,
        )
        .expect("corrected context answer output");

        assert_eq!(
            effective_output_language(&output),
            AstraUserLanguage::Italian
        );
        assert!(!context_answer_language_mismatch(
            AstraUserLanguage::Italian,
            effective_output_language(&output)
        ));
    }

    #[test]
    fn english_expected_language_accepts_english_answer() {
        let output = parse_context_answer_output(
            r#"{
              "answer": "No. Based on the previous recap, it was about early Earth, not Mars.",
              "language": "en",
              "status": "answered",
              "used_context_refs": ["last_tool_result"],
              "confidence": 0.91,
              "warnings": []
            }"#,
        )
        .expect("english context answer output");

        assert!(!context_answer_language_mismatch(
            AstraUserLanguage::English,
            effective_output_language(&output)
        ));
    }

    #[test]
    fn mixed_or_unknown_expected_language_does_not_false_fail() {
        let output = parse_context_answer_output(
            r#"{
              "answer": "No. It was about Terra primordiale, not Mars.",
              "language": "mixed",
              "status": "answered",
              "used_context_refs": ["last_tool_result"],
              "confidence": 0.83,
              "warnings": []
            }"#,
        )
        .expect("mixed context answer output");

        assert!(!context_answer_language_mismatch(
            AstraUserLanguage::Mixed,
            effective_output_language(&output)
        ));
        assert!(!context_answer_language_mismatch(
            AstraUserLanguage::Unknown,
            effective_output_language(&output)
        ));
    }

    #[test]
    fn correction_failure_uses_deterministic_italian_fallback() {
        let frame = earth_frame();
        let tool = frame.last_tool_result.as_ref().expect("tool result");
        let output = fallback_context_answer(tool, &context_plan(0.91), AstraUserLanguage::Italian);

        assert_eq!(output.language, AstraUserLanguage::Italian);
        assert!(output.answer.contains("in base al recap"));
        assert!(output.answer.contains("Terra primordiale"));
    }

    #[test]
    fn context_answer_empty_model_content_uses_deterministic_tool_result_fallback() {
        let frame = earth_frame();
        let tool = frame.last_tool_result.as_ref().expect("tool result");
        let plan = context_continuation_plan(
            "planner_empty_with_salient_tool_result",
            Some("quindi si parlava della luna?"),
        );

        let output = parse_context_answer_output("")
            .unwrap_or_else(|| fallback_context_answer(tool, &plan, AstraUserLanguage::Italian));
        let rendered = render_context_answer(tool, &output);

        assert_eq!(output.status, "answered");
        assert!(rendered.contains("Fonte: ultima sessione archiviata"));
        assert!(rendered.contains("non risulta un riferimento a Luna"));
        assert!(rendered.contains("Terra primordiale"));
        assert!(!rendered.contains("chat normale"));
        assert!(!rendered.contains("context_answer_synthesizer_fallback"));
    }

    #[test]
    fn sanitizer_removes_internal_context_ref_lines() {
        let mut output = ContextAnswerOutput {
            answer: "No, si parlava della Terra.\nContesto usato: last_tool_result.".to_string(),
            language: AstraUserLanguage::Italian,
            status: "answered".to_string(),
            support: "supported_by_context".to_string(),
            used_context_refs: vec!["last_tool_result".to_string()],
            confidence: 0.92,
            warnings: Vec::new(),
            sanitized_internal_context_refs: false,
        };

        assert!(sanitize_context_answer_output(&mut output));
        assert_eq!(output.answer, "No, si parlava della Terra.");
        assert!(!output.answer.contains("last_tool_result"));
    }

    #[test]
    fn sanitizer_removes_english_internal_context_ref_lines() {
        let mut output = ContextAnswerOutput {
            answer: "No, it was about Earth.\nContext used: last_tool_result.".to_string(),
            language: AstraUserLanguage::English,
            status: "answered".to_string(),
            support: "supported_by_context".to_string(),
            used_context_refs: vec!["last_tool_result".to_string()],
            confidence: 0.92,
            warnings: Vec::new(),
            sanitized_internal_context_refs: false,
        };

        assert!(sanitize_context_answer_output(&mut output));
        assert_eq!(output.answer, "No, it was about Earth.");
        assert!(!output.answer.contains("last_tool_result"));
    }

    #[test]
    fn render_context_answer_uses_source_label_without_internal_refs() {
        let frame = earth_frame();
        let tool = frame.last_tool_result.as_ref().expect("tool result");
        let output = ContextAnswerOutput {
            answer: "No, si parlava della Terra.\nContesto usato: last_tool_result.".to_string(),
            language: AstraUserLanguage::Italian,
            status: "answered".to_string(),
            support: "supported_by_context".to_string(),
            used_context_refs: vec!["last_tool_result".to_string()],
            confidence: 0.92,
            warnings: Vec::new(),
            sanitized_internal_context_refs: false,
        };
        let rendered = render_context_answer(tool, &output);

        assert!(rendered.starts_with("Fonte: ultima sessione archiviata."));
        assert!(rendered.contains("No, si parlava della Terra."));
        assert!(!rendered.contains("last_tool_result"));
        assert!(!rendered.contains("Contesto usato"));
    }

    #[test]
    fn context_answer_diagnostic_tracks_language_and_sanitization_metadata() {
        let frame = earth_frame();
        let diagnostic = context_answer_diagnostic(
            Some("answer_from_context".to_string()),
            Some("last_tool_result".to_string()),
            1800,
            false,
            true,
            "test",
            &frame,
            AstraUserLanguage::Italian,
        );
        let serialized = serde_json::to_value(&diagnostic).expect("diagnostic json");

        assert_eq!(serialized["expected_language"], "it");
        assert_eq!(serialized["budget_compaction_applied"], true);
        assert_eq!(serialized["sanitized_internal_context_refs"], false);
        assert!(serialized.get("raw_prompt").is_none());
        assert!(serialized.get("raw_model_output").is_none());
        assert!(serialized.get("answer").is_none());
    }
    #[test]
    fn tool_bound_intelligent_transcript_recap_does_not_use_last_context() {
        let frame = earth_frame();
        assert_eq!(
            classify_context_attachment(
                "sull'attuale ultima sessione registrata mi generi un intelligent transcript recap",
                &frame,
            ),
            ContextAttachmentKind::ToolBoundRequest
        );
        let decision = planner_failure_fallback(
            &frame,
            "empty_model_content",
            Some("sull'attuale ultima sessione registrata mi generi un intelligent transcript recap"),
        );
        assert!(matches!(
            decision,
            ConversationOrchestratorDecision::DeferToToolRouter(_)
        ));
    }

    #[test]
    fn context_policy_defers_tool_bound_contextual_chat_to_router() {
        let frame = earth_frame();
        let decision = ConversationOrchestratorDecision::NormalChatWithContext(ContextualChatPlan {
            context_ref: "last_tool_result".to_string(),
            reason_code: "model_selected_contextual_chat".to_string(),
            confidence: 0.91,
        });
        let policy = apply_context_continuation_policy(
            &decision,
            &frame,
            None,
            Some(0.91),
            Some("mi generi un intelligent transcript recap dell'attuale sessione?"),
        )
        .expect("context policy");
        assert_eq!(policy.action, ContextContinuationPolicyAction::DeferToToolRouter);
        assert_eq!(
            policy.reason,
            ContextContinuationPolicyReason::PlannerSelectedToolBoundRequest
        );
    }

    #[test]
    fn planner_refuse_with_governed_action_safety_verifies_with_tool_router() {
        let frame = earth_frame();
        let mut attempt = attempt_for(
            ConversationOrchestratorDecision::Refuse(RefusalPlan {
                reason_code: "governed_action".to_string(),
                message: "refuse".to_string(),
                confidence: 1.0,
            }),
            Some(1.0),
            None,
            None,
        );
        attempt.diagnostic.planner_intent_kind = Some("governed_action".to_string());
        attempt.diagnostic.planner_capability_family = Some("work_session".to_string());
        attempt.diagnostic.planner_requires_governed_action = Some(true);
        attempt.diagnostic.planner_safe_to_bypass_tools = Some(false);

        assert!(matches!(
            apply_orchestrator_policy(&attempt, &frame),
            OrchestratorPolicyAction::UseFullToolRouter { .. }
        ));
    }


}
