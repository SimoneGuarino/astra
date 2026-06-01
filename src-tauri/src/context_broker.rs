use crate::{
    assistant_tool_router::available_tool_manifest,
    conversation_history::ConversationMessage,
    conversation_orchestrator::{
        AstraUserLanguage, ContextAnswerOutput, ToolResultFrame, WorkingContextFrame,
    },
};
use serde_json::{json, Value};

pub const CONTEXT_PLANNER_TARGET_CHARS: usize = 2_500;
pub const CONTEXT_PLANNER_HARD_CAP_CHARS: usize = 3_500;
pub const CONTEXT_ANSWER_TARGET_CHARS: usize = 2_500;
pub const CONTEXT_ANSWER_HARD_CAP_CHARS: usize = 3_500;
pub const FULL_ROUTER_TARGET_CHARS: usize = 7_000;
pub const FULL_ROUTER_HARD_CAP_CHARS: usize = 9_000;

#[derive(Debug, Clone)]
pub struct PromptBuild {
    pub messages: Vec<Value>,
    pub prompt_char_count: usize,
    pub prompt_budget_exceeded: bool,
    pub budget_compaction_applied: bool,
    pub tool_manifest_count: usize,
}

pub fn prompt_char_count(messages: &[Value]) -> usize {
    messages
        .iter()
        .filter_map(|message| message.get("content").and_then(Value::as_str))
        .map(str::len)
        .sum()
}

pub fn build_context_planner_messages(
    user_message: &str,
    history: &[ConversationMessage],
    working_context: &WorkingContextFrame,
) -> PromptBuild {
    let mut budget_compaction_applied = false;
    let mut messages =
        context_planner_messages_with_limits(user_message, history, working_context, 3, 140, 520);
    let mut prompt_chars = prompt_char_count(&messages);
    if prompt_chars > CONTEXT_PLANNER_TARGET_CHARS {
        budget_compaction_applied = true;
        messages = context_planner_messages_with_limits(
            user_message,
            history,
            working_context,
            2,
            100,
            320,
        );
        prompt_chars = prompt_char_count(&messages);
    }
    if prompt_chars > CONTEXT_PLANNER_TARGET_CHARS {
        budget_compaction_applied = true;
        messages = context_planner_messages_with_limits(
            user_message,
            history,
            working_context,
            1,
            80,
            220,
        );
        prompt_chars = prompt_char_count(&messages);
    }
    if prompt_chars > CONTEXT_PLANNER_TARGET_CHARS {
        budget_compaction_applied = true;
        messages = context_planner_messages_minimal(user_message, working_context);
        prompt_chars = prompt_char_count(&messages);
    }

    PromptBuild {
        messages,
        prompt_char_count: prompt_chars,
        prompt_budget_exceeded: prompt_chars > CONTEXT_PLANNER_HARD_CAP_CHARS,
        budget_compaction_applied,
        tool_manifest_count: 0,
    }
}

pub fn build_context_answer_messages(
    user_message: &str,
    _working_context: &WorkingContextFrame,
    tool_result: &ToolResultFrame,
    expected_language: AstraUserLanguage,
) -> PromptBuild {
    let mut budget_compaction_applied = tool_result.answer_summary.chars().count() > 520
        || tool_result.key_topics.len() > 6
        || tool_result.active_entities.len() > 6
        || tool_result.warnings.len() > 3;
    let mut messages = context_answer_messages_with_summary_limit(
        user_message,
        tool_result,
        expected_language,
        520,
    );
    let mut prompt_chars = prompt_char_count(&messages);
    if prompt_chars > CONTEXT_ANSWER_TARGET_CHARS {
        budget_compaction_applied = true;
        messages = context_answer_messages_with_summary_limit(
            user_message,
            tool_result,
            expected_language,
            300,
        );
        prompt_chars = prompt_char_count(&messages);
    }
    if prompt_chars > CONTEXT_ANSWER_HARD_CAP_CHARS {
        budget_compaction_applied = true;
        messages =
            context_answer_messages_minimal(user_message, tool_result, expected_language, 180);
        prompt_chars = prompt_char_count(&messages);
    }
    if prompt_chars > CONTEXT_ANSWER_HARD_CAP_CHARS {
        budget_compaction_applied = true;
        messages =
            context_answer_messages_minimal(user_message, tool_result, expected_language, 100);
        prompt_chars = prompt_char_count(&messages);
    }

    PromptBuild {
        messages,
        prompt_char_count: prompt_chars,
        prompt_budget_exceeded: prompt_chars > CONTEXT_ANSWER_HARD_CAP_CHARS,
        budget_compaction_applied,
        tool_manifest_count: 0,
    }
}

pub fn filtered_tool_manifest_json(
    working_context: Option<&WorkingContextFrame>,
    include_when_to_use: bool,
) -> Value {
    let context_is_work_session = working_context
        .and_then(|frame| frame.last_tool_result.as_ref())
        .is_some_and(|tool| tool.tool_name.starts_with("work_session."));
    let tools = available_tool_manifest()
        .into_iter()
        .filter(|tool| {
            if context_is_work_session {
                tool.tool.starts_with("work_session.")
            } else {
                true
            }
        })
        .map(|item| {
            if include_when_to_use {
                json!({
                    "tool": item.tool,
                    "allowed_targets": item.allowed_targets,
                    "mode": item.mode,
                    "requires_evidence_synthesis": item.requires_evidence_synthesis,
                    "when_to_use": item.when_to_use,
                })
            } else {
                json!({
                    "tool": item.tool,
                    "allowed_targets": item.allowed_targets,
                    "mode": item.mode,
                    "requires_evidence_synthesis": item.requires_evidence_synthesis,
                })
            }
        })
        .collect::<Vec<_>>();
    json!(tools)
}

#[allow(dead_code)]
pub fn tool_manifest_count(value: &Value) -> usize {
    value.as_array().map_or(0, Vec::len)
}

pub fn compact_recent_turns(
    history: &[ConversationMessage],
    max_turns: usize,
    max_chars_per_turn: usize,
) -> Vec<Value> {
    history
        .iter()
        .rev()
        .take(max_turns)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|turn| {
            json!({
                "role": turn.role,
                "text": bounded_text(&turn.content, max_chars_per_turn),
            })
        })
        .collect()
}

#[allow(dead_code)]
pub fn compact_working_context_value(
    working_context: &WorkingContextFrame,
    answer_summary_limit: usize,
) -> Value {
    json!({
        "current_topic": working_context
            .current_topic
            .as_ref()
            .map(|value| bounded_text(value, 160)),
        "active_entities": working_context.active_entities.iter().take(8).collect::<Vec<_>>(),
        "last_user_goal_summary": working_context
            .last_user_goal_summary
            .as_ref()
            .map(|value| bounded_text(value, 220)),
        "last_assistant_answer_summary": working_context
            .last_assistant_answer_summary
            .as_ref()
            .map(|value| bounded_text(value, answer_summary_limit)),
        "last_assistant_action": working_context.last_assistant_action,
        "last_referenced_session": working_context.last_referenced_session,
        "last_referenced_artifacts": working_context.last_referenced_artifacts.iter().take(8).collect::<Vec<_>>(),
        "last_tool_result": working_context.last_tool_result.as_ref().map(|tool| compact_tool_result_value(tool, answer_summary_limit)),
        "available_evidence_refs": working_context.available_evidence_refs.iter().take(12).collect::<Vec<_>>(),
        "unresolved_followups": working_context.unresolved_followups.iter().take(4).collect::<Vec<_>>(),
        "pending_governed_action": working_context.pending_governed_action.as_ref().map(|action| json!({
            "present": action.present,
            "tool_name": action.tool_name,
            "intent": action.intent,
            "prerequisite": action.prerequisite,
            "status": action.status,
            "expires_at_present": action.expires_at_present,
            "expired": action.expired,
            "attempt_count": action.attempt_count,
            "metadata_only": true,
        })),
        "confidence": working_context.confidence,
        "updated_at_ms": working_context.updated_at_ms,
    })
}

pub fn compact_discourse_context_value(
    working_context: &WorkingContextFrame,
    answer_summary_limit: usize,
    item_limit: usize,
) -> Value {
    json!({
        "current_topic": working_context
            .current_topic
            .as_ref()
            .map(|value| bounded_text(value, 120)),
        "active_entities": working_context
            .active_entities
            .iter()
            .take(item_limit)
            .map(|value| bounded_text(value, 80))
            .collect::<Vec<_>>(),
        "last_user_goal_summary": working_context
            .last_user_goal_summary
            .as_ref()
            .map(|value| bounded_text(value, 140)),
        "last_assistant_answer_summary": working_context
            .last_assistant_answer_summary
            .as_ref()
            .map(|value| bounded_text(value, answer_summary_limit)),
        "last_assistant_action": working_context.last_assistant_action,
        "last_tool_result": working_context
            .last_tool_result
            .as_ref()
            .filter(|_| working_context.salience.is_usable())
            .map(|tool| compact_context_answer_tool_result_value(tool, answer_summary_limit, item_limit)),
        "context_salience": working_context.last_tool_result.as_ref().map(|_| json!({
            "turn_age": working_context.salience.turn_age,
            "normal_chat_turns_since_update": working_context.salience.normal_chat_turns_since_update,
            "salience_score": working_context.salience.salience_score,
            "stale": working_context.salience.stale,
        })),
        "pending_governed_action": working_context.pending_governed_action.as_ref().map(|action| json!({
            "present": action.present,
            "tool_name": action.tool_name,
            "intent": action.intent,
            "prerequisite": action.prerequisite,
            "status": action.status,
            "expires_at_present": action.expires_at_present,
            "expired": action.expired,
            "metadata_only": true,
        })),
        "confidence": working_context.confidence,
    })
}

#[allow(dead_code)]
pub fn compact_tool_result_value(
    tool_result: &ToolResultFrame,
    answer_summary_limit: usize,
) -> Value {
    json!({
        "tool_name": tool_result.tool_name,
        "answer_kind": tool_result.answer_kind,
        "source_kind": tool_result.source_kind,
        "source_label": bounded_text(&tool_result.source_label, 120),
        "session_id": tool_result.session_id,
        "used_evidence_ids": tool_result.used_evidence_ids.iter().take(12).collect::<Vec<_>>(),
        "evidence_count": tool_result.evidence_count,
        "answer_summary": bounded_text(&tool_result.answer_summary, answer_summary_limit),
        "key_topics": tool_result.key_topics.iter().take(10).collect::<Vec<_>>(),
        "active_entities": tool_result.active_entities.iter().take(10).collect::<Vec<_>>(),
        "warnings": tool_result.warnings.iter().take(4).collect::<Vec<_>>(),
        "confidence": tool_result.confidence,
        "created_at_ms": tool_result.created_at_ms,
    })
}

pub fn bounded_text(value: &str, max_chars: usize) -> String {
    let trimmed = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if trimmed.chars().count() <= max_chars {
        return trimmed;
    }
    let mut text = trimmed.chars().take(max_chars).collect::<String>();
    text.push_str("...");
    text
}

fn context_planner_messages_with_limits(
    user_message: &str,
    history: &[ConversationMessage],
    working_context: &WorkingContextFrame,
    max_turns: usize,
    max_chars_per_turn: usize,
    answer_summary_limit: usize,
) -> Vec<Value> {
    let system = concat!(
        "You are Astra's discourse planner, not user-facing. ",
        "Return JSON only; do not answer or call tools. ",
        "Classify the next assistant action from compact conversation context; Rust validates the final route. ",
        "Set safe_to_bypass_tools=true only for ordinary chat that needs no tools, session memory, transcript evidence, screen context, or governed action. ",
        "If unsure, or if the user asks to start/stop recording, recap/search/read a session, generate transcript summaries/intelligence, recall meeting memory, attach screen context, or perform any governed action, set safe_to_bypass_tools=false. ",
        "Requests like intelligent transcript recap, complete recap, meeting intelligence, attach screenshot/screen, or current/active/latest session analysis are needs_tool; do not answer them from last_tool_result. ",
        "If pending_governed_action is present and the user is ready to continue it, use needs_tool and keep safe_to_bypass_tools=false. ",
        "Use answer_from_context only for topic-presence checks and direct evidence follow-ups such as whether a specific topic was mentioned. ",
        "Use normal_chat_with_context when the user asks a general or boundary knowledge question inspired by prior context; the prior result is context, not the source of the full answer."
    );
    let payload = json!({
        "task": "plan_next_assistant_action",
        "user_message": bounded_text(user_message, 520),
        "recent_turns": compact_recent_turns(history, max_turns, max_chars_per_turn),
        "working_context": compact_discourse_context_value(working_context, answer_summary_limit, 5),
        "allowed_routes": [
            "answer_from_context",
            "answer_from_context_boundary",
            "normal_chat_with_context",
            "needs_tool",
            "normal_chat",
            "clarify",
            "refuse"
        ],
        "output_schema": {
            "route": "answer_from_context|answer_from_context_boundary|normal_chat_with_context|needs_tool|normal_chat|clarify|refuse",
            "intent_kind": "ordinary_question|casual_chat|general_knowledge|session_memory_query|governed_action|context_followup|context_boundary|unknown",
            "capability_family": "none|work_session|meeting|screen_context|session_memory|unknown",
            "requires_tool_arbitration": false,
            "requires_memory_lookup": false,
            "requires_governed_action": false,
            "requires_context_boundary": false,
            "safe_to_bypass_tools": false,
            "context_ref": "last_tool_result|working_topic|none",
            "confidence": 0.0,
            "tool_affinity_risk": false,
            "reason_code": "short_machine_reason",
            "answer_plan": {
                "strategy": "verify_entity_against_context|expand_summary|explain_previous_answer|compare|context_boundary|none",
                "focus": "string|null"
            }
        },
        "rules": [
            "No chain-of-thought.",
            "No answer_from_context without supporting context_ref.",
            "Use answer_from_context for questions that ask whether something was present, mentioned, discussed, or supported by the previous transcript/session evidence.",
            "Use answer_from_context_boundary when the user explicitly asks for transcript-grounded evidence but the evidence is insufficient.",
            "Use normal_chat_with_context when the user asks a general explanation, causal/how/why/origin question, or boundary question inspired by the transcript. In that case, answer with general knowledge and separate it from transcript evidence.",
            "For ordinary questions unrelated to tools/context, use normal_chat with capability_family=none and safe_to_bypass_tools=true.",
            "For a pending governed action continuation, use needs_tool with capability_family=work_session or meeting.",
            "Never mark safe_to_bypass_tools=true for session memory, transcript, meeting/work-session state, screen context, or governed actions.",
            "No browser, terminal, filesystem, email, cloud, or autonomous actions."
        ]
    });
    vec![
        json!({"role": "system", "content": system}),
        json!({"role": "user", "content": payload.to_string()}),
    ]
}

fn context_planner_messages_minimal(
    user_message: &str,
    working_context: &WorkingContextFrame,
) -> Vec<Value> {
    let system = concat!(
        "You are Astra's discourse planner, not user-facing. JSON only. ",
        "Rust arbitrates final execution. ",
        "safe_to_bypass_tools=true only for ordinary chat needing no tools, session memory, transcript evidence, screen context, or governed action. ",
        "If pending_governed_action exists or unsure, safe_to_bypass_tools=false. ",
        "Use context boundary routes for contextual questions beyond supplied evidence."
    );
    let payload = json!({
        "task": "plan_next_assistant_action",
        "user_message": bounded_text(user_message, 180),
        "working_context": compact_discourse_context_value(working_context, 120, 3),
        "allowed_routes": ["answer_from_context", "answer_from_context_boundary", "normal_chat_with_context", "needs_tool", "normal_chat", "clarify", "refuse"],
        "output_schema": {
            "route": "answer_from_context|answer_from_context_boundary|normal_chat_with_context|needs_tool|normal_chat|clarify|refuse",
            "intent_kind": "ordinary_question|casual_chat|general_knowledge|session_memory_query|governed_action|context_followup|context_boundary|unknown",
            "capability_family": "none|work_session|meeting|screen_context|session_memory|unknown",
            "requires_tool_arbitration": false,
            "requires_memory_lookup": false,
            "requires_governed_action": false,
            "requires_context_boundary": false,
            "safe_to_bypass_tools": false,
            "context_ref": "last_tool_result|working_topic|none",
            "confidence": 0.0,
            "tool_affinity_risk": false,
            "reason_code": "short_machine_reason",
            "answer_plan": {"strategy": "verify_entity_against_context|expand_summary|explain_previous_answer|compare|context_boundary|none", "focus": "string|null"}
        }
    });
    vec![
        json!({"role": "system", "content": system}),
        json!({"role": "user", "content": payload.to_string()}),
    ]
}

fn context_answer_messages_with_summary_limit(
    user_message: &str,
    tool_result: &ToolResultFrame,
    expected_language: AstraUserLanguage,
    answer_summary_limit: usize,
) -> Vec<Value> {
    let system = concat!(
        "You are Astra's context-grounded answer synthesizer. ",
        "Use the compact tool result frame as the evidence boundary. ",
        "If the user's claim is not supported, say so. ",
        "If the user asks whether the previous session was about a topic or entity, answer Yes or No first. ",
        "If supported, briefly explain from the provided context; if not supported, say no and summarize what the session was actually about. ",
        "If general knowledge is useful, clearly mark it as separate from transcript evidence. ",
        "Do not put STT completeness, source labels, evidence IDs, or operational diagnostics in answer; put needed warnings only in warnings. ",
        "Do not call tools. Return strict JSON only. Keep JSON keys in English."
    );
    let payload = json!({
        "user_message": bounded_text(user_message, 520),
        "expected_language": expected_language.code(),
        "language_instruction": expected_language.instruction(),
        "tool_result": compact_context_answer_tool_result_value(tool_result, answer_summary_limit, 6),
        "output_schema": {
            "answer": "string",
            "language": "it|en|mixed|unknown",
            "status": "answered|partial|insufficient_context|boundary_answer",
            "support": "supported_by_context|not_in_context|general_knowledge_with_context",
            "used_context_refs": ["last_tool_result"],
            "confidence": 0.0,
            "warnings": ["string"]
        },
        "rules": [
            "Use only the tool_result answer_summary, topics, entities, source label, warnings, and evidence count.",
            "For topic-presence checks, answer yes/no directly when the compact context is sufficient.",
            "If a claim or entity is not supported by the tool_result, state that limitation.",
            "For boundary questions, clearly distinguish transcript evidence from general knowledge.",
            "Do not include STT completeness, source labels, evidence IDs, or operational diagnostics inside answer.",
            "If warnings are needed, put them only in warnings.",
            "Do not present external facts as transcript evidence."
        ]
    });
    vec![
        json!({"role": "system", "content": system}),
        json!({"role": "user", "content": payload.to_string()}),
    ]
}

fn context_answer_messages_minimal(
    user_message: &str,
    tool_result: &ToolResultFrame,
    expected_language: AstraUserLanguage,
    answer_summary_limit: usize,
) -> Vec<Value> {
    let system = concat!(
        "Context-grounded answer. JSON only. No tools. ",
        "Use compact_context as evidence boundary. If unsupported, say so. ",
        "For topic-presence checks, answer Yes or No first. ",
        "Keep operational diagnostics out of answer; warnings only in warnings. ",
        "Separate any general knowledge from transcript evidence. Keep JSON keys in English."
    );
    let payload = json!({
        "user_message": bounded_text(user_message, 220),
        "expected_language": expected_language.code(),
        "language_instruction": expected_language.instruction(),
        "compact_context": compact_context_answer_tool_result_value(tool_result, answer_summary_limit, 4),
        "output_schema": {
            "answer": "string",
            "language": "it|en|mixed|unknown",
            "status": "answered|partial|insufficient_context|boundary_answer",
            "support": "supported_by_context|not_in_context|general_knowledge_with_context",
            "used_context_refs": ["last_tool_result"],
            "confidence": 0.0,
            "warnings": ["string"]
        }
    });
    vec![
        json!({"role": "system", "content": system}),
        json!({"role": "user", "content": payload.to_string()}),
    ]
}

pub fn build_context_answer_language_retry_messages(
    output: &ContextAnswerOutput,
    expected_language: AstraUserLanguage,
    tool_result: &ToolResultFrame,
) -> PromptBuild {
    let messages = context_answer_language_retry_messages(output, expected_language, tool_result);
    let prompt_char_count = prompt_char_count(&messages);
    PromptBuild {
        messages,
        prompt_char_count,
        prompt_budget_exceeded: prompt_char_count > CONTEXT_ANSWER_TARGET_CHARS,
        budget_compaction_applied: false,
        tool_manifest_count: 0,
    }
}

fn context_answer_language_retry_messages(
    output: &ContextAnswerOutput,
    expected_language: AstraUserLanguage,
    tool_result: &ToolResultFrame,
) -> Vec<Value> {
    let system = concat!(
        "Rewrite the answer into the expected language. Preserve meaning. Do not add facts. ",
        "Return strict JSON only with English keys."
    );
    let payload = json!({
        "expected_language": expected_language.code(),
        "language_instruction": expected_language.instruction(),
        "answer_to_rewrite": bounded_text(&output.answer, 700),
        "compact_context": compact_context_answer_tool_result_value(tool_result, 180, 4),
        "output_schema": {
            "answer": "string",
            "language": "it|en|mixed|unknown",
            "status": output.status,
            "support": output.support,
            "used_context_refs": ["last_tool_result"],
            "confidence": 0.0,
            "warnings": ["string"]
        }
    });
    vec![
        json!({"role": "system", "content": system}),
        json!({"role": "user", "content": payload.to_string()}),
    ]
}

pub fn compact_context_answer_tool_result_value(
    tool_result: &ToolResultFrame,
    answer_summary_limit: usize,
    item_limit: usize,
) -> Value {
    json!({
        "source_label": bounded_text(&tool_result.source_label, 120),
        "answer_summary": bounded_text(&tool_result.answer_summary, answer_summary_limit),
        "key_topics": tool_result.key_topics.iter().take(item_limit).map(|value| bounded_text(value, 90)).collect::<Vec<_>>(),
        "active_entities": tool_result.active_entities.iter().take(item_limit).map(|value| bounded_text(value, 90)).collect::<Vec<_>>(),
        "warnings": tool_result.warnings.iter().take(3).map(|value| bounded_text(value, 140)).collect::<Vec<_>>(),
        "evidence_count": tool_result.evidence_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation_orchestrator::{ContextSalience, WorkingContextFrame};

    #[test]
    fn planner_prompt_stays_below_budget_for_short_followup() {
        let frame = WorkingContextFrame {
            last_tool_result: Some(crate::conversation_orchestrator::ToolResultFrame {
                tool_name: "work_session.recap".to_string(),
                answer_kind: "work_session_recap".to_string(),
                source_kind: "session_archive_transcript".to_string(),
                source_label: "ultima sessione archiviata".to_string(),
                session_id: Some("session-1".to_string()),
                used_evidence_ids: vec!["segment:1".to_string()],
                evidence_count: 1,
                answer_summary: "La sessione parlava della formazione della Terra primordiale, oceano di magma, atmosfera primitiva e primi mari.".to_string(),
                key_topics: vec!["Terra primordiale".to_string()],
                active_entities: vec!["Terra".to_string()],
                warnings: Vec::new(),
                confidence: Some(0.9),
                created_at_ms: 1,
            }),
            salience: ContextSalience {
                turn_age: 0,
                normal_chat_turns_since_update: 0,
                last_reinforced_at_ms: Some(1),
                salience_score: 1.0,
                stale: false,
            },
            ..Default::default()
        };

        let prompt = build_context_planner_messages("quindi si parlava di marte?", &[], &frame);

        assert!(prompt.prompt_char_count <= CONTEXT_PLANNER_TARGET_CHARS);
        assert!(!prompt.prompt_budget_exceeded);
        assert_eq!(prompt.tool_manifest_count, 0);
    }

    #[test]
    fn planner_prompt_compaction_keeps_discourse_stage_under_budget() {
        let frame = WorkingContextFrame {
            last_assistant_answer_summary: Some("x".repeat(10_000)),
            last_tool_result: Some(crate::conversation_orchestrator::ToolResultFrame {
                tool_name: "work_session.recap".to_string(),
                answer_kind: "work_session_recap".to_string(),
                source_kind: "session_archive_transcript".to_string(),
                source_label: "ultima sessione archiviata".to_string(),
                session_id: Some("session-1".to_string()),
                used_evidence_ids: vec!["segment:1".to_string()],
                evidence_count: 1,
                answer_summary: "x".repeat(10_000),
                key_topics: Vec::new(),
                active_entities: Vec::new(),
                warnings: Vec::new(),
                confidence: Some(0.7),
                created_at_ms: 1,
            }),
            salience: ContextSalience {
                turn_age: 0,
                normal_chat_turns_since_update: 0,
                last_reinforced_at_ms: Some(1),
                salience_score: 1.0,
                stale: false,
            },
            ..Default::default()
        };

        let prompt =
            build_context_planner_messages("puoi verificare questa affermazione?", &[], &frame);

        assert!(prompt.budget_compaction_applied);
        assert!(!prompt.prompt_budget_exceeded);
        assert!(prompt.prompt_char_count <= CONTEXT_PLANNER_TARGET_CHARS);
        assert_eq!(prompt.tool_manifest_count, 0);
    }

    #[test]
    fn context_planner_prompt_does_not_include_full_tool_manifest() {
        let frame = frame_with_tool_summary("La sessione parlava della Terra primordiale.");
        let prompt = build_context_planner_messages("fammi una domanda di contesto", &[], &frame);
        let serialized = serde_json::to_string(&prompt.messages).expect("planner prompt json");

        assert_eq!(prompt.tool_manifest_count, 0);
        assert!(!serialized.contains("work_session.recap"));
        assert!(serialized.contains("allowed_routes"));
        assert!(serialized.contains("normal_chat_with_context"));
        assert!(serialized.contains("answer_from_context_boundary"));
    }

    #[test]
    fn filtered_manifest_removes_verbose_descriptions_when_compact() {
        let manifest = filtered_tool_manifest_json(None, false);
        let serialized = serde_json::to_string(&manifest).expect("manifest json");

        assert!(serialized.contains("work_session.recap"));
        assert!(!serialized.contains("when_to_use"));
        assert!(!serialized.contains("description"));
    }

    #[test]
    fn context_answer_prompt_for_short_followup_stays_under_hard_cap() {
        let frame = frame_with_tool_summary("La sessione parlava della formazione della Terra primordiale, oceano di magma, atmosfera primitiva e primi mari.");
        let tool = frame.last_tool_result.as_ref().expect("tool result");
        let prompt = build_context_answer_messages(
            "quindi si parlava di marte?",
            &frame,
            tool,
            AstraUserLanguage::Italian,
        );

        assert!(prompt.prompt_char_count <= CONTEXT_ANSWER_HARD_CAP_CHARS);
        assert!(!prompt.prompt_budget_exceeded);
    }

    #[test]
    fn context_answer_prompt_compacts_large_tool_result() {
        let frame = frame_with_tool_summary(&"Terra primordiale ".repeat(900));
        let tool = frame.last_tool_result.as_ref().expect("tool result");
        let prompt = build_context_answer_messages(
            "puoi verificare questa affermazione?",
            &frame,
            tool,
            AstraUserLanguage::Italian,
        );

        assert!(prompt.budget_compaction_applied);
        assert!(!prompt.prompt_budget_exceeded);
        assert!(prompt.prompt_char_count <= CONTEXT_ANSWER_HARD_CAP_CHARS);
    }

    #[test]
    fn context_answer_prompt_uses_compact_tool_result_without_evidence_ids() {
        let frame = frame_with_tool_summary("La sessione parlava della Terra primordiale.");
        let tool = frame.last_tool_result.as_ref().expect("tool result");
        let value = compact_context_answer_tool_result_value(tool, 120, 4);

        assert_eq!(value["source_label"], "ultima sessione archiviata");
        assert!(value.get("used_evidence_ids").is_none());
        assert_eq!(value["evidence_count"], 1);
    }

    #[test]
    fn context_answer_prompt_instructs_yes_no_topic_checks() {
        let frame = frame_with_tool_summary("La sessione parlava della Terra primordiale.");
        let tool = frame.last_tool_result.as_ref().expect("tool result");
        let prompt = build_context_answer_messages(
            "quindi si parlava di questo tema?",
            &frame,
            tool,
            AstraUserLanguage::Italian,
        );
        let serialized = serde_json::to_string(&prompt.messages).expect("prompt json");

        assert!(serialized.contains("answer Yes or No first"));
        assert!(serialized.contains("topic-presence checks"));
    }

    #[test]
    fn context_answer_prompt_keeps_warnings_out_of_answer() {
        let frame = frame_with_tool_summary("La sessione parlava della Terra primordiale.");
        let tool = frame.last_tool_result.as_ref().expect("tool result");
        let prompt = build_context_answer_messages(
            "puoi verificare questa affermazione?",
            &frame,
            tool,
            AstraUserLanguage::Italian,
        );
        let serialized = serde_json::to_string(&prompt.messages).expect("prompt json");

        assert!(serialized.contains("put needed warnings only in warnings"));
        assert!(serialized.contains("Do not include STT completeness"));
    }

    fn frame_with_tool_summary(summary: &str) -> WorkingContextFrame {
        WorkingContextFrame {
            last_tool_result: Some(crate::conversation_orchestrator::ToolResultFrame {
                tool_name: "work_session.recap".to_string(),
                answer_kind: "work_session_recap".to_string(),
                source_kind: "session_archive_transcript".to_string(),
                source_label: "ultima sessione archiviata".to_string(),
                session_id: Some("session-1".to_string()),
                used_evidence_ids: vec!["segment:1".to_string()],
                evidence_count: 1,
                answer_summary: summary.to_string(),
                key_topics: vec![
                    "Terra primordiale".to_string(),
                    "oceano di magma".to_string(),
                    "atmosfera primitiva".to_string(),
                    "primi mari".to_string(),
                ],
                active_entities: vec!["Terra".to_string()],
                warnings: Vec::new(),
                confidence: Some(0.9),
                created_at_ms: 1,
            }),
            salience: ContextSalience {
                turn_age: 0,
                normal_chat_turns_since_update: 0,
                last_reinforced_at_ms: Some(1),
                salience_score: 1.0,
                stale: false,
            },
            ..Default::default()
        }
    }
}
