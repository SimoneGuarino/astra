use crate::{
    desktop_agent_types::{
        PlannerContractDecision, PlannerContractInput, PlannerContractSource,
        PlannerDecisionStatus, PlannerRejectionReason, PlannerScrollIntent, PlannerStep,
        PlannerStepKind, PlannerVisibilityAssessment,
    },
    semantic_frame::GoalLoopDriverFuture,
};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use std::env;
use uuid::Uuid;

const OLLAMA_BASE_URL: &str = "http://127.0.0.1:11434";

#[derive(Clone)]
pub struct ModelAssistedPlanner {
    client: Client,
}

impl ModelAssistedPlanner {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }

    pub fn plan<'a>(
        &'a self,
        input: &'a PlannerContractInput,
    ) -> GoalLoopDriverFuture<'a, Result<Option<PlannerContractDecision>, String>> {
        Box::pin(async move {
            if !planner_enabled() {
                return Ok(None);
            }
            let model = self.select_model().await?;
            let response = self.call_planner_model(&model, input).await?;
            if response.deterministic_fallback_advised.unwrap_or(false) {
                return Err("model planner advised deterministic fallback".into());
            }
            Ok(Some(response.into_contract_decision(input)))
        })
    }

    async fn select_model(&self) -> Result<String, String> {
        let installed = self.fetch_installed_models().await.unwrap_or_default();
        let candidates = env::var("ASTRA_PLANNER_MODEL_CANDIDATES")
            .unwrap_or_else(|_| "gpt-oss:20b,qwen3:14b,qwen3:8b,llama3.1:8b".into())
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();

        select_first_available(&candidates, &installed)
            .or_else(|| candidates.first().cloned())
            .ok_or_else(|| "no planner model candidates configured".into())
    }

    async fn fetch_installed_models(&self) -> Result<Vec<String>, String> {
        let response = self
            .client
            .get(format!("{OLLAMA_BASE_URL}/api/tags"))
            .send()
            .await
            .map_err(|error| format!("Ollama planner tags request failed: {error}"))?;
        if !response.status().is_success() {
            return Err(format!(
                "Ollama planner tags HTTP error: {}",
                response.status()
            ));
        }
        let body: Value = response
            .json()
            .await
            .map_err(|error| format!("Ollama planner tags parse failed: {error}"))?;
        Ok(body
            .get("models")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.get("name").and_then(Value::as_str))
            .map(ToOwned::to_owned)
            .collect())
    }

    async fn call_planner_model(
        &self,
        model: &str,
        input: &PlannerContractInput,
    ) -> Result<PlannerModelResponse, String> {
        let payload = json!({
            "model": model,
            "stream": false,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "num_ctx": 8192
            },
            "messages": [
                {"role": "system", "content": planner_system_prompt()},
                {"role": "user", "content": planner_user_prompt(input)}
            ]
        });

        let response = self
            .client
            .post(format!("{OLLAMA_BASE_URL}/api/chat"))
            .json(&payload)
            .send()
            .await
            .map_err(|error| format!("Ollama planner request failed: {error}"))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!("Ollama planner HTTP error {status}: {body}"));
        }
        let body: Value = response
            .json()
            .await
            .map_err(|error| format!("Ollama planner response parse failed: {error}"))?;
        let content = body
            .get("message")
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "Ollama planner returned an empty response".to_string())?;
        let json_text = extract_json_object(content)
            .ok_or_else(|| "planner response did not contain a JSON object".to_string())?;
        serde_json::from_str::<PlannerModelResponse>(json_text)
            .map_err(|error| format!("planner JSON schema parse failed: {error}"))
    }
}

#[derive(Debug, Clone, Deserialize)]
struct PlannerModelResponse {
    strategy: String,
    action: String,
    rationale: String,
    confidence: f32,
    #[serde(default)]
    target_item_id: Option<String>,
    #[serde(default)]
    target_entity_id: Option<String>,
    #[serde(default)]
    click_region_key: Option<String>,
    #[serde(default)]
    expected_state: Option<String>,
    #[serde(default)]
    focused_perception_needed: Option<bool>,
    #[serde(default)]
    replan_needed: Option<bool>,
    #[serde(default)]
    deterministic_fallback_advised: Option<bool>,
    #[serde(default)]
    visibility_assessment: Option<String>,
    #[serde(default)]
    scroll_needed: Option<bool>,
    #[serde(default)]
    scroll_reason: Option<String>,
}

impl PlannerModelResponse {
    fn into_contract_decision(self, input: &PlannerContractInput) -> PlannerContractDecision {
        let kind = planner_step_kind(&self.action);
        let expected_state = self
            .expected_state
            .clone()
            .or_else(|| Some(input.goal.success_condition.clone()));
        let confidence = self.confidence.clamp(0.0, 1.0);
        let step = PlannerStep {
            step_id: Uuid::new_v4().to_string(),
            kind: kind.clone(),
            confidence,
            rationale: self.rationale.clone(),
            target_item_id: self.target_item_id.clone(),
            target_entity_id: self.target_entity_id.clone(),
            click_region_key: self.click_region_key.clone(),
            executable_candidate: None,
            expected_state: expected_state.clone(),
        };
        PlannerContractDecision {
            source: PlannerContractSource::ModelAssisted,
            proposed_step: step,
            strategy_rationale: self.strategy,
            focused_perception_needed: self
                .focused_perception_needed
                .unwrap_or(kind == PlannerStepKind::ReplanAfterPerception),
            replan_needed: self
                .replan_needed
                .unwrap_or(kind == PlannerStepKind::ReplanAfterPerception),
            expected_verification_target: expected_state,
            planner_confidence: confidence,
            accepted: false,
            fallback_used: false,
            rejection_reason: self.scroll_reason.clone(),
            decision_status: PlannerDecisionStatus::Accepted,
            rejection_code: planner_rejection_code_for_model_response(
                self.scroll_needed,
                &self.scroll_reason,
            ),
            visibility_assessment: planner_visibility_assessment(
                self.visibility_assessment.as_deref(),
                kind == PlannerStepKind::ReplanAfterPerception,
                self.scroll_needed.unwrap_or(false),
            ),
            scroll_intent: planner_scroll_intent(self.scroll_needed.unwrap_or(false)),
            visible_actionability: input.visible_actionability.clone(),
            target_confidence: None,
            normalized: false,
            downgraded: false,
        }
    }
}

fn planner_enabled() -> bool {
    env::var("ASTRA_PLANNER_LLM_ENABLED")
        .map(|value| !matches!(value.trim(), "0" | "false" | "FALSE" | "off" | "OFF"))
        .unwrap_or(true)
}

fn planner_system_prompt() -> &'static str {
    concat!(
        "You are Astra Planner, a planning-only model behind a Rust-governed desktop agent. ",
        "Return only strict JSON. Do not execute actions. Do not invent screen targets. ",
        "You may only choose among visible item/entity ids in the provided semantic frame. ",
        "Allowed action values: click_result_region, click_entity_region, focused_perception, ",
        "request_clarification, refuse, verify_goal, no_op, use_deterministic. ",
        "If visible_actionability says refinement is still eligible, prefer focused_perception over off-screen reasoning. ",
        "If the page context appears correct but the target is not visible and scrolling would likely be required, ",
        "set action=focused_perception, visibility_assessment=likely_offscreen, scroll_needed=true, and do not invent a target id. ",
        "If uncertain, choose focused_perception or use_deterministic. ",
        "Never output pixel coordinates. Rust will validate and bind all targets before execution."
    )
}

fn planner_user_prompt(input: &PlannerContractInput) -> String {
    let compact = json!({
        "goal": input.goal,
        "page_evidence": input.current_frame.page_evidence,
        "scene_summary": input.current_frame.scene_summary,
        "visible_entities": input.current_frame.visible_entities,
        "visible_result_items": input.current_frame.visible_result_items,
        "actionable_controls": input.current_frame.actionable_controls,
        "legacy_target_candidates": input.current_frame.legacy_target_candidates,
        "uncertainty": input.current_frame.uncertainty,
        "visible_actionability": input.visible_actionability,
        "executed_steps": input.executed_steps,
        "verification_history": input.verification_history,
        "perception_requests": input.perception_requests,
        "retry_budget": input.retry_budget,
        "retries_used": input.retries_used,
        "visible_refinement_attempts": input.visible_refinement_attempts,
        "max_visible_refinement_passes": input.max_visible_refinement_passes,
        "provider_hint": input.provider_hint,
        "browser_app_hint": input.browser_app_hint,
        "page_kind_hint": input.page_kind_hint,
    });
    format!(
        "Planner contract input:\n{}\n\nReturn JSON with keys: strategy, action, rationale, confidence, target_item_id, target_entity_id, click_region_key, expected_state, focused_perception_needed, replan_needed, deterministic_fallback_advised, visibility_assessment, scroll_needed, scroll_reason. Valid visibility_assessment values: visible_grounded, visible_ambiguous, visible_under_grounded, visible_target_needs_click_region, focused_perception_needed, likely_offscreen, not_visible, unknown.",
        compact
    )
}

fn planner_visibility_assessment(
    value: Option<&str>,
    focused_replan: bool,
    scroll_needed: bool,
) -> PlannerVisibilityAssessment {
    if scroll_needed {
        return PlannerVisibilityAssessment::LikelyOffscreen;
    }
    match value.map(normalize_action).as_deref() {
        Some("visible_grounded") | Some("grounded") => PlannerVisibilityAssessment::VisibleGrounded,
        Some("visible_ambiguous") | Some("ambiguous") => {
            PlannerVisibilityAssessment::VisibleAmbiguous
        }
        Some("visible_under_grounded") | Some("visible_undergrounded") => {
            PlannerVisibilityAssessment::VisibleUnderGrounded
        }
        Some("visible_target_needs_click_region")
        | Some("visible_missing_click_region")
        | Some("missing_click_region") => {
            PlannerVisibilityAssessment::VisibleTargetNeedsClickRegion
        }
        Some("focused_perception_needed") | Some("needs_focused_perception") => {
            PlannerVisibilityAssessment::FocusedPerceptionNeeded
        }
        Some("likely_offscreen") | Some("offscreen") => {
            PlannerVisibilityAssessment::LikelyOffscreen
        }
        Some("not_visible") => PlannerVisibilityAssessment::NotVisible,
        _ if focused_replan => PlannerVisibilityAssessment::FocusedPerceptionNeeded,
        _ => PlannerVisibilityAssessment::Unknown,
    }
}

fn planner_scroll_intent(scroll_needed: bool) -> PlannerScrollIntent {
    if scroll_needed {
        PlannerScrollIntent::RequiredButUnsupported
    } else {
        PlannerScrollIntent::NotNeeded
    }
}

fn planner_rejection_code_for_model_response(
    scroll_needed: Option<bool>,
    scroll_reason: &Option<String>,
) -> Option<PlannerRejectionReason> {
    if scroll_needed.unwrap_or(false) {
        Some(PlannerRejectionReason::ScrollRequiredButUnsupported)
    } else if scroll_reason.is_some() {
        Some(PlannerRejectionReason::LikelyOffscreenTarget)
    } else {
        None
    }
}

fn planner_step_kind(value: &str) -> PlannerStepKind {
    match normalize_action(value).as_str() {
        "click_result_region" => PlannerStepKind::ClickResultRegion,
        "click_entity_region" => PlannerStepKind::ClickEntityRegion,
        "focused_perception" | "replan_after_perception" => PlannerStepKind::ReplanAfterPerception,
        "request_clarification" => PlannerStepKind::RequestClarification,
        "refuse" => PlannerStepKind::Refuse,
        "verify_goal" => PlannerStepKind::VerifyGoal,
        "no_op" => PlannerStepKind::NoOp,
        _ => PlannerStepKind::ReplanAfterPerception,
    }
}

fn normalize_action(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('-', "_")
}

fn select_first_available(candidates: &[String], installed_models: &[String]) -> Option<String> {
    let installed_lower = installed_models
        .iter()
        .map(|value| value.to_ascii_lowercase())
        .collect::<Vec<_>>();
    candidates.iter().find_map(|candidate| {
        let exact = candidate.to_ascii_lowercase();
        if installed_lower.iter().any(|installed| installed == &exact) {
            return Some(candidate.clone());
        }
        let base = exact.split(':').next().unwrap_or(&exact).to_string();
        installed_models.iter().find_map(|installed| {
            let installed_lower = installed.to_ascii_lowercase();
            (installed_lower == base || installed_lower.starts_with(&(base.clone() + ":")))
                .then(|| installed.clone())
        })
    })
}

fn extract_json_object(content: &str) -> Option<&str> {
    let trimmed = content.trim();
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        return Some(trimmed);
    }
    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    (end > start).then_some(trimmed[start..=end].trim())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desktop_agent_types::{
        GoalConstraints, GoalSpec, GoalType, PageEvidenceSource, PageSemanticEvidence,
        SemanticScreenFrame, VisibleResultKind,
    };

    #[test]
    fn parses_model_response_into_non_executable_contract_decision() {
        let response = PlannerModelResponse {
            strategy: "open first video".into(),
            action: "click_result_region".into(),
            rationale: "visible video result matches the goal".into(),
            confidence: 0.91,
            target_item_id: Some("video_1".into()),
            target_entity_id: None,
            click_region_key: Some("title".into()),
            expected_state: Some("media_watch_page_visible".into()),
            focused_perception_needed: Some(false),
            replan_needed: Some(false),
            deterministic_fallback_advised: Some(false),
            visibility_assessment: Some("visible_grounded".into()),
            scroll_needed: Some(false),
            scroll_reason: None,
        };
        let input = PlannerContractInput {
            goal: GoalSpec {
                goal_id: "goal".into(),
                goal_type: GoalType::OpenMediaResult,
                constraints: GoalConstraints {
                    provider: Some("youtube".into()),
                    result_kind: Some(VisibleResultKind::Video),
                    rank_within_kind: Some(1),
                    rank_overall: None,
                    entity_name: None,
                    attributes: Value::Null,
                },
                success_condition: "video_watch_page_open".into(),
                utterance: "aprimi il primo video".into(),
                confidence: 0.9,
            },
            current_frame: SemanticScreenFrame {
                frame_id: "frame".into(),
                captured_at: 1,
                image_path: None,
                page_evidence: PageSemanticEvidence {
                    browser_app_hint: Some("chrome".into()),
                    content_provider_hint: Some("youtube".into()),
                    page_kind_hint: Some("search_results".into()),
                    query_hint: Some("shiva".into()),
                    result_list_visible: Some(true),
                    raw_confidence: Some(0.9),
                    confidence: 0.9,
                    evidence_sources: vec![PageEvidenceSource::StructuredVision],
                    capture_backend: Some("powershell_gdi".into()),
                    observation_source: Some("test".into()),
                    uncertainty: Vec::new(),
                },
                scene_summary: "YouTube results".into(),
                visible_entities: Vec::new(),
                visible_result_items: Vec::new(),
                actionable_controls: Vec::new(),
                legacy_target_candidates: Vec::new(),
                uncertainty: Vec::new(),
            },
            executed_steps: Vec::new(),
            verification_history: Vec::new(),
            perception_requests: Vec::new(),
            retry_budget: 3,
            retries_used: 0,
            visible_refinement_attempts: 0,
            max_visible_refinement_passes: 1,
            provider_hint: Some("youtube".into()),
            browser_app_hint: Some("chrome".into()),
            page_kind_hint: Some("search_results".into()),
            visible_actionability: Default::default(),
        };

        let decision = response.into_contract_decision(&input);

        assert_eq!(decision.source, PlannerContractSource::ModelAssisted);
        assert!(!decision.accepted);
        assert!(decision.proposed_step.executable_candidate.is_none());
        assert_eq!(
            decision.proposed_step.target_item_id.as_deref(),
            Some("video_1")
        );
    }
}
