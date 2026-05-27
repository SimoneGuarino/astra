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
pub const MIN_NORMAL_CHAT_CONFIDENCE: f32 = 0.85;
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

#[derive(Debug, Deserialize)]
struct RawPlannerOutput {
    route: Option<String>,
    context_ref: Option<String>,
    confidence: Option<f32>,
    tool_affinity_risk: Option<bool>,
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
            self.salience.reinforce_from_context_answer();
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
        let summary = context_broker::bounded_text(&answer_summary.into(), 700);
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

fn planner_failure_fallback(
    working_context: &WorkingContextFrame,
    reason_code: &str,
) -> ConversationOrchestratorDecision {
    if working_context.last_tool_result_usable() {
        return ConversationOrchestratorDecision::NormalChatWithContext(ContextualChatPlan {
            context_ref: "last_tool_result".to_string(),
            reason_code: reason_code.to_string(),
            confidence: MIN_NORMAL_CHAT_CONFIDENCE,
        });
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

    let fallback = planner_failure_fallback(working_context, "planner_failure");
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
    let decision = parsed.decision.unwrap_or_else(|| {
        planner_failure_fallback(
            working_context,
            parsed
                .failure_reason
                .as_deref()
                .unwrap_or("planner_failure"),
        )
    });
    diagnostic.planner_failure_reason = parsed.failure_reason;
    diagnostic.planner_confidence = parsed.planner_confidence;
    diagnostic.tool_affinity_risk = parsed.tool_affinity_risk;
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
        diagnostic.fallback_policy = Some("context_boundary_on_planner_failure".to_string());
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
        diagnostic.planner_failure_reason = Some("invalid_context_answer_json".to_string());
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
                    diagnostic.output_language = Some(output.language.code().to_string());
                }
            } else {
                *output = fallback_context_answer(tool_result, plan, expected_language);
                diagnostic.language_retry_succeeded = Some(false);
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
            }
        }
    };
    let route = normalize_route(raw.route.as_deref().unwrap_or("normal_chat"));
    let confidence = raw.confidence.unwrap_or(0.0).clamp(0.0, 1.0);
    let reason_code = raw.reason_code.unwrap_or_else(|| "unspecified".to_string());
    let context_ref = raw.context_ref.unwrap_or_else(|| "none".to_string());
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
            }
        }
    };
    PlannerParseOutcome {
        decision: Some(decision),
        failure_reason: None,
        planner_confidence: Some(confidence),
        tool_affinity_risk: raw.tool_affinity_risk,
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
            .filter(|warning| !warning.trim().is_empty())
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
    for warning in output.warnings.iter().take(3) {
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

pub fn user_facing_context_label(tool_result: &ToolResultFrame) -> String {
    format!(
        "Fonte: {}.",
        tool_result.source_label.trim().trim_end_matches('.')
    )
}

pub fn build_normal_chat_with_context_preamble(
    working_context: &WorkingContextFrame,
    _plan: &ContextualChatPlan,
) -> Option<String> {
    let tool_result = working_context.last_tool_result.as_ref()?;
    let topics = tool_result
        .key_topics
        .iter()
        .take(5)
        .map(|topic| context_broker::bounded_text(topic, 80))
        .collect::<Vec<_>>()
        .join(", ");
    Some(format!(
        "Contesto conversazionale compatto: l'utente sta proseguendo una conversazione basata su un recap Work Session da '{}'. Recap sintetico: {}. Temi: {}. Se rispondi con conoscenza generale, distinguila chiaramente dalle evidenze della registrazione e non attribuire al transcript dettagli non presenti.",
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
        _ => None,
    }
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
            if plan.confidence < MIN_NORMAL_CHAT_CONFIDENCE {
                return OrchestratorPolicyAction::UseFullToolRouter {
                    reason: OrchestratorFallbackReason::NormalChatLowConfidence,
                };
            }
            if attempt.diagnostic.tool_affinity_risk == Some(true) {
                return OrchestratorPolicyAction::UseFullToolRouter {
                    reason: OrchestratorFallbackReason::ToolAffinityUnresolved,
                };
            }
            OrchestratorPolicyAction::AcceptDecision
        }
        ConversationOrchestratorDecision::Clarify(plan) => {
            if plan.confidence < MIN_CLARIFY_CONFIDENCE {
                return OrchestratorPolicyAction::UseFullToolRouter {
                    reason: OrchestratorFallbackReason::ClarifyLowConfidence,
                };
            }
            OrchestratorPolicyAction::AcceptDecision
        }
        ConversationOrchestratorDecision::Refuse(plan) => {
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
                    pending_tool_action: false,
                    explicit_user_action: false,
                    slash_command: false,
                    ui_action: false,
                    planner_reason_code: attempt
                        .diagnostic
                        .planner_failure_reason
                        .clone()
                        .or_else(|| Some("discourse_planner_needs_tool".to_string())),
                    tool_affinity: needs_tool_affinity_from_confidence(
                        attempt.diagnostic.planner_confidence,
                    ),
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
    let answer = if boundary_mode {
        let focus = plan
            .focus
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or("questa domanda");
        if matches!(language, AstraUserLanguage::English) {
            format!(
                "The available transcript recap does not directly establish an answer to {focus}. From the prior context, the session was about: {}. I can answer from general knowledge, but that would be separate from the recording evidence.",
                tool_result.answer_summary
            )
        } else {
            format!(
                "Nel recap disponibile non ho evidenze dirette per rispondere con certezza a {focus}. Dal contesto precedente, la sessione parlava di: {}. Posso darti una spiegazione generale, distinguendola dalle evidenze della registrazione.",
                tool_result.answer_summary
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
        let focus_supported = plan.focus.as_ref().is_some_and(|focus| {
            let needle = focus.to_lowercase();
            let haystack = format!(
                "{} {} {}",
                tool_result.answer_summary,
                tool_result.key_topics.join(" "),
                tool_result.active_entities.join(" ")
            )
            .to_lowercase();
            !needle.trim().is_empty() && haystack.contains(needle.trim())
        });
        match (language, focus_supported) {
            (AstraUserLanguage::English, true) => format!(
                "Yes. Based on the previous recap from {}, the reference to {focus} is supported: {}",
                tool_result.source_label, tool_result.answer_summary
            ),
            (AstraUserLanguage::English, false) => format!(
                "No. Based on the previous recap from {}, it was not about {focus}. It was about: {}",
                tool_result.source_label, tool_result.answer_summary
            ),
            (_, true) => format!(
                "Si. In base al recap di {}, il riferimento a {focus} risulta supportato: {}",
                tool_result.source_label, tool_result.answer_summary
            ),
            (_, false) => format!(
                "No, in base al recap di {}, non si parlava di {focus}. Si parlava di: {}",
                tool_result.source_label, tool_result.answer_summary
            ),
        }
    } else if matches!(language, AstraUserLanguage::English) {
        format!(
            "Based on the previous recap from {}: {}",
            tool_result.source_label, tool_result.answer_summary
        )
    } else {
        format!(
            "In base al recap di {}: {}",
            tool_result.source_label, tool_result.answer_summary
        )
    };
    ContextAnswerOutput {
        answer,
        language,
        status: if boundary_mode {
            "boundary_answer".to_string()
        } else {
            "partial".to_string()
        },
        support: if boundary_mode {
            "not_in_context".to_string()
        } else {
            "supported_by_context".to_string()
        },
        used_context_refs: vec!["last_tool_result".to_string()],
        confidence: 0.62,
        warnings: vec!["context_answer_synthesizer_fallback".to_string()],
        sanitized_internal_context_refs: false,
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
        OrchestratorPlanAttempt {
            diagnostic: AssistantOrchestratorDiagnostic {
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
            },
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
            ConversationOrchestratorDecision::NormalChat(NormalChatPlan {
                reason_code: "ordinary_chat".to_string(),
                confidence: MIN_NORMAL_CHAT_CONFIDENCE - 0.01,
            }),
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
            ConversationOrchestratorDecision::NormalChat(NormalChatPlan {
                reason_code: "ordinary_chat".to_string(),
                confidence: MIN_NORMAL_CHAT_CONFIDENCE,
            }),
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
    fn planner_empty_with_last_tool_result_uses_contextual_chat_fallback() {
        let frame = earth_frame();
        let decision = planner_failure_fallback(&frame, "empty_model_content");

        assert!(matches!(
            decision,
            ConversationOrchestratorDecision::NormalChatWithContext(_)
        ));
        assert!(!matches!(
            decision,
            ConversationOrchestratorDecision::ToolCall(_)
        ));
    }

    #[test]
    fn planner_empty_without_last_tool_result_defers_to_full_router() {
        let frame = WorkingContextFrame::default();
        let decision = planner_failure_fallback(&frame, "empty_model_content");
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
            ConversationOrchestratorDecision::NormalChat(NormalChatPlan {
                reason_code: "low_confidence_data_access".to_string(),
                confidence: 0.64,
            }),
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
            ConversationOrchestratorDecision::NormalChat(NormalChatPlan {
                reason_code: "ordinary_chat".to_string(),
                confidence: 0.42,
            }),
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

        frame.update_from_context_answer("quindi?", "Risposta dal contesto precedente.");
        assert!(frame.salience.salience_score >= 0.85);
        assert_eq!(frame.salience.normal_chat_turns_since_update, 0);

        frame.update_from_normal_chat("chi sei?", "Sono Astra.");
        assert!(frame.salience.salience_score < 1.0);
        assert_eq!(frame.salience.normal_chat_turns_since_update, 1);
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
}
