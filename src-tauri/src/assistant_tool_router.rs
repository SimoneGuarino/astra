use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolFamily {
    WorkSession,
    Meeting,
    SessionMemory,
    ScreenContext,
    NormalAssistant,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolTarget {
    pub kind: String,
    pub session_id: Option<String>,
    #[serde(default)]
    pub object_type: Option<String>,
    #[serde(default)]
    pub object_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssistantToolIntent {
    pub tool_family: ToolFamily,
    pub tool_name: String,
    pub intent: String,
    pub target: ToolTarget,
    pub confidence: f32,
    pub language: Option<String>,
    pub query: Option<String>,
    pub reason_code: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClarificationRequest {
    pub message: String,
    pub confidence: f32,
    pub reason_code: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafeRefusal {
    pub message: String,
    pub reason_code: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AssistantRouteDecision {
    NormalChat,
    ToolCall(AssistantToolIntent),
    Clarify(ClarificationRequest),
    Refuse(SafeRefusal),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RouterFailureReason {
    OllamaUnavailable,
    EmptyResponse,
    MalformedJson,
    InvalidSchema,
    LowConfidence,
    InvalidTool,
    InvalidTarget,
    Timeout,
    EndpointConfig,
    ModelRoutingUnavailable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AssistantToolRouterRuntimeResult {
    Routed(AssistantRouteDecision),
    NormalChat {
        confidence: f32,
        reason_code: String,
    },
    Clarify {
        message: String,
        reason_code: String,
    },
    Refuse {
        message: String,
        reason_code: String,
    },
    Unavailable {
        reason: RouterFailureReason,
    },
    Malformed {
        reason: RouterFailureReason,
        raw_len: usize,
    },
    EmptyModelContent {
        model: String,
    },
    Timeout {
        timeout_ms: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouterParseOutcome {
    pub result: AssistantToolRouterRuntimeResult,
    pub repair_attempted: bool,
    pub repair_succeeded: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct AssistantToolManifestItem {
    pub tool: &'static str,
    pub governed_tool: &'static str,
    pub description: &'static str,
    pub when_to_use: &'static str,
    pub allowed_targets: &'static [&'static str],
    pub mode: &'static str,
    pub requires_evidence_synthesis: bool,
    pub safety: &'static str,
}

#[derive(Debug, Deserialize)]
struct RawRouterTarget {
    kind: Option<String>,
    session_id: Option<String>,
    #[serde(default)]
    object_type: Option<String>,
    #[serde(default)]
    object_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawRouterProposal {
    route: Option<String>,
    tool: Option<String>,
    intent: Option<String>,
    target: Option<Value>,
    #[serde(default, alias = "object_type")]
    object: Option<String>,
    #[serde(default)]
    object_ids: Vec<String>,
    confidence: Option<f32>,
    language: Option<String>,
    query: Option<String>,
    reason_code: Option<String>,
}

pub fn available_tool_manifest() -> Vec<AssistantToolManifestItem> {
    vec![
        AssistantToolManifestItem {
            tool: "work_session.start",
            governed_tool: "meeting.session.start",
            description: "Start a governed Work Session with audio capture and transcript.",
            when_to_use: "User wants to start recording, take notes, begin a meeting or work session.",
            allowed_targets: &["none"],
            mode: "write_governed",
            requires_evidence_synthesis: false,
            safety: "Requires meeting/session capture permissions and explicit consent.",
        },
        AssistantToolManifestItem {
            tool: "work_session.stop",
            governed_tool: "meeting.session.stop",
            description: "Stop the active Work Session and archive it.",
            when_to_use: "User wants to stop, finish, end, or save the current Work Session.",
            allowed_targets: &["active_session", "none"],
            mode: "write_governed",
            requires_evidence_synthesis: false,
            safety: "Uses bounded STT drain/finalization; no unbounded wait.",
        },
        AssistantToolManifestItem {
            tool: "work_session.stop_and_recap",
            governed_tool: "meeting.session.stop + meeting.intelligence.generate",
            description: "Stop the active Work Session and then generate/read a recap.",
            when_to_use: "User asks to stop and generate a recap or summary.",
            allowed_targets: &["active_session", "none"],
            mode: "write_governed_then_read",
            requires_evidence_synthesis: true,
            safety: "Preserves bounded drain and validates transcript availability.",
        },
        AssistantToolManifestItem {
            tool: "work_session.recap",
            governed_tool: "meeting.intelligence.generate/read",
            description: "Generate or read a recap of an active, completed, or archived Work Session.",
            when_to_use: "User asks for a recap, summary, what was collected, or what was discussed.",
            allowed_targets: &[
                "last_referenced_session",
                "latest_archived_session",
                "last_completed_session",
                "active_session",
                "archived_sessions",
                "none",
            ],
            mode: "read_only_governed",
            requires_evidence_synthesis: true,
            safety: "Uses local evidence only and does not mutate transcript.",
        },
        AssistantToolManifestItem {
            tool: "work_session.generate_intelligence",
            governed_tool: "meeting.intelligence.generate",
            description: "Generate structured Meeting Intelligence artifacts from transcript evidence.",
            when_to_use: "User explicitly asks for Meeting Intelligence, decisions, action items, risks, open questions, structured notes, technical recap, or follow-up artifacts.",
            allowed_targets: &[
                "last_referenced_session",
                "latest_archived_session",
                "last_completed_session",
                "active_session",
                "none",
            ],
            mode: "read_only_governed",
            requires_evidence_synthesis: true,
            safety: "Generates derived evidence-linked artifacts only; raw transcript remains the source of truth.",
        },
        AssistantToolManifestItem {
            tool: "work_session.transcript_summary",
            governed_tool: "meeting.session.read / meeting.session.archive.read",
            description: "Summarize or analyze transcript content from a governed Work Session archive.",
            when_to_use: "User asks what the transcript, recording, last registration, or saved session content was about.",
            allowed_targets: &[
                "last_referenced_session",
                "latest_archived_session",
                "last_completed_session",
                "active_session",
                "archived_sessions",
                "none",
            ],
            mode: "read_only_governed",
            requires_evidence_synthesis: true,
            safety: "Retrieves bounded local transcript evidence for answer synthesis; no audio paths or screenshots.",
        },
        AssistantToolManifestItem {
            tool: "work_session.details",
            governed_tool: "meeting.session.read / meeting.session.archive.read",
            description: "Show detailed information about a referenced, active, completed, or archived Work Session.",
            when_to_use: "User asks for more details, tell me more, or a contextual follow-up about a previous Work Session answer.",
            allowed_targets: &[
                "last_referenced_session",
                "latest_archived_session",
                "last_completed_session",
                "active_session",
                "archived_sessions",
                "none",
            ],
            mode: "read_only_governed",
            requires_evidence_synthesis: true,
            safety: "Reads local metadata/artifacts through governed paths.",
        },
        AssistantToolManifestItem {
            tool: "work_session.technical_recap",
            governed_tool: "meeting.intelligence.generate/read",
            description: "Return the technical recap artifact for a Work Session.",
            when_to_use: "User asks for a technical recap or engineering summary of a session.",
            allowed_targets: &[
                "last_referenced_session",
                "latest_archived_session",
                "last_completed_session",
                "active_session",
                "none",
            ],
            mode: "read_only_governed",
            requires_evidence_synthesis: true,
            safety: "Uses local Meeting Intelligence evidence.",
        },
        AssistantToolManifestItem {
            tool: "work_session.followup_draft",
            governed_tool: "meeting.followup.draft",
            description: "Draft a follow-up message from a Work Session.",
            when_to_use: "User asks for a follow-up draft, mail draft, or written next-step summary.",
            allowed_targets: &[
                "last_referenced_session",
                "latest_archived_session",
                "last_completed_session",
                "active_session",
                "none",
            ],
            mode: "read_only_governed",
            requires_evidence_synthesis: true,
            safety: "Draft only; never sends email.",
        },
        AssistantToolManifestItem {
            tool: "work_session.recall",
            governed_tool: "meeting.recall.answer",
            description: "Answer questions over archived sessions using evidence.",
            when_to_use: "User asks what was decided, discussed, seen, or remembered from previous sessions.",
            allowed_targets: &[
                "archived_sessions",
                "latest_archived_session",
                "last_referenced_session",
                "last_completed_session",
                "active_session",
                "none",
            ],
            mode: "read_only_governed",
            requires_evidence_synthesis: true,
            safety: "Audit uses query hash/length; answer must be evidence-linked.",
        },
        AssistantToolManifestItem {
            tool: "work_session.search",
            governed_tool: "meeting.session.search",
            description: "Search archived Session Memory lexically.",
            when_to_use: "User asks to search session memory or find sessions by topic.",
            allowed_targets: &["archived_sessions", "latest_archived_session", "none"],
            mode: "read_only_governed",
            requires_evidence_synthesis: false,
            safety: "Local lexical search only.",
        },
        AssistantToolManifestItem {
            tool: "work_session.attach_screen",
            governed_tool: "meeting.screen_context.attach_current",
            description: "Attach current screen context to the active Work Session.",
            when_to_use: "User asks to save or attach what is currently visible.",
            allowed_targets: &["current_screen", "active_session", "none"],
            mode: "observe_only_governed",
            requires_evidence_synthesis: false,
            safety: "Observe-only; no clicking, DesktopControl, or browser automation.",
        },
        AssistantToolManifestItem {
            tool: "work_session.show_evidence",
            governed_tool: "meeting.recall.answer / cached evidence rendering",
            description: "Show evidence snippets from a previous Work Session answer.",
            when_to_use: "User asks for evidence, proof, sources, or show me the evidence.",
            allowed_targets: &[
                "last_referenced_session",
                "latest_archived_session",
                "last_completed_session",
                "active_session",
                "archived_sessions",
                "none",
            ],
            mode: "read_only_governed",
            requires_evidence_synthesis: true,
            safety: "Bounded snippets only; no full transcript dump.",
        },
        AssistantToolManifestItem {
            tool: "work_session.status",
            governed_tool: "meeting.session.read",
            description: "Show compact active/completed/archive Work Session status.",
            when_to_use: "User asks if a session is active, how many segments remain, or current Work Session status.",
            allowed_targets: &[
                "last_referenced_session",
                "latest_archived_session",
                "last_completed_session",
                "active_session",
                "none",
            ],
            mode: "read_only_governed",
            requires_evidence_synthesis: false,
            safety: "Metadata/status only.",
        },
        AssistantToolManifestItem {
            tool: "work_session.open_details",
            governed_tool: "meeting.session.read",
            description: "Open or point to the Meeting inspector/details view.",
            when_to_use: "User explicitly asks to open the Meeting details panel.",
            allowed_targets: &[
                "last_referenced_session",
                "latest_archived_session",
                "last_completed_session",
                "active_session",
                "none",
            ],
            mode: "ui_affordance",
            requires_evidence_synthesis: false,
            safety: "UI affordance only.",
        },
    ]
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn parse_router_decision(content: &str) -> Option<AssistantRouteDecision> {
    match parse_router_runtime_result(content, "") {
        AssistantToolRouterRuntimeResult::Routed(decision) => Some(decision),
        AssistantToolRouterRuntimeResult::NormalChat { .. } => {
            Some(AssistantRouteDecision::NormalChat)
        }
        AssistantToolRouterRuntimeResult::Clarify {
            message,
            reason_code,
        } => Some(AssistantRouteDecision::Clarify(ClarificationRequest {
            message,
            confidence: 0.0,
            reason_code,
        })),
        AssistantToolRouterRuntimeResult::Refuse {
            message,
            reason_code,
        } => Some(AssistantRouteDecision::Refuse(SafeRefusal {
            message,
            reason_code,
        })),
        _ => None,
    }
}

pub fn parse_router_runtime_result(content: &str, model: &str) -> AssistantToolRouterRuntimeResult {
    parse_router_runtime_result_with_repair(content, model).result
}

pub fn parse_router_runtime_result_with_repair(content: &str, model: &str) -> RouterParseOutcome {
    if content.trim().is_empty() {
        return RouterParseOutcome {
            result: AssistantToolRouterRuntimeResult::EmptyModelContent {
                model: model.to_string(),
            },
            repair_attempted: false,
            repair_succeeded: false,
        };
    }

    let trimmed = content.trim();
    let mut candidates = vec![trimmed.to_string()];
    if let Some(json) = extract_json_object(trimmed) {
        let candidate = json.trim().to_string();
        if !candidates.iter().any(|value| value == &candidate) {
            candidates.push(candidate);
        }
    }

    for candidate in candidates {
        let extraction_repaired = candidate.trim() != trimmed;
        if let Some(result) = parse_router_json_candidate(&candidate) {
            return RouterParseOutcome {
                result,
                repair_attempted: extraction_repaired,
                repair_succeeded: extraction_repaired,
            };
        }

        let repaired = repair_common_router_json(&candidate);
        if repaired != candidate {
            if let Some(result) = parse_router_json_candidate(&repaired) {
                return RouterParseOutcome {
                    result,
                    repair_attempted: true,
                    repair_succeeded: true,
                };
            }
        }
    }

    RouterParseOutcome {
        result: AssistantToolRouterRuntimeResult::Malformed {
            reason: RouterFailureReason::MalformedJson,
            raw_len: content.len(),
        },
        repair_attempted: true,
        repair_succeeded: false,
    }
}

fn parse_router_json_candidate(content: &str) -> Option<AssistantToolRouterRuntimeResult> {
    let proposal: RawRouterProposal = match serde_json::from_str(content) {
        Ok(value) => value,
        Err(_) => return None,
    };
    let confidence = proposal.confidence.unwrap_or(0.0).clamp(0.0, 1.0);
    let reason_code = proposal
        .reason_code
        .clone()
        .unwrap_or_else(|| "unspecified".to_string());
    Some(match validate_router_proposal(proposal) {
        Ok(AssistantRouteDecision::NormalChat) => AssistantToolRouterRuntimeResult::NormalChat {
            confidence,
            reason_code,
        },
        Ok(AssistantRouteDecision::Clarify(clarify)) => AssistantToolRouterRuntimeResult::Clarify {
            message: clarify.message,
            reason_code: clarify.reason_code,
        },
        Ok(AssistantRouteDecision::Refuse(refusal)) => AssistantToolRouterRuntimeResult::Refuse {
            message: refusal.message,
            reason_code: refusal.reason_code,
        },
        Ok(decision) => AssistantToolRouterRuntimeResult::Routed(decision),
        Err(reason) => AssistantToolRouterRuntimeResult::Malformed {
            reason,
            raw_len: content.len(),
        },
    })
}

fn validate_router_proposal(
    proposal: RawRouterProposal,
) -> Result<AssistantRouteDecision, RouterFailureReason> {
    let route = normalize_token(proposal.route.as_deref().unwrap_or("normal_chat"));
    let confidence = proposal.confidence.unwrap_or(0.0).clamp(0.0, 1.0);
    let reason_code = proposal
        .reason_code
        .clone()
        .unwrap_or_else(|| "unspecified".to_string());
    match route.as_str() {
        "normalchat" => Ok(AssistantRouteDecision::NormalChat),
        "refuse" => Ok(AssistantRouteDecision::Refuse(SafeRefusal {
            message: "Non posso gestire questa richiesta con gli strumenti disponibili in modo sicuro.".to_string(),
            reason_code,
        })),
        "clarify" => Ok(AssistantRouteDecision::Clarify(ClarificationRequest {
            message: "Vuoi che usi gli strumenti Work Session? Dimmi se ti riferisci all'ultima sessione, alla sessione attiva o alla memoria archiviata.".to_string(),
            confidence,
            reason_code,
        })),
        "toolcall" => validate_tool_call(proposal, confidence, reason_code),
        _ => Err(RouterFailureReason::InvalidSchema),
    }
}

fn validate_tool_call(
    proposal: RawRouterProposal,
    confidence: f32,
    reason_code: String,
) -> Result<AssistantRouteDecision, RouterFailureReason> {
    let tool_name = proposal.tool.ok_or(RouterFailureReason::InvalidTool)?;
    let normalized_tool = normalize_tool_name(&tool_name);
    if !known_tool(&normalized_tool) {
        return Err(RouterFailureReason::InvalidTool);
    }
    if confidence < 0.6 {
        return Ok(AssistantRouteDecision::NormalChat);
    }
    if confidence < 0.8 {
        return Ok(AssistantRouteDecision::Clarify(ClarificationRequest {
            message: "La richiesta sembra richiedere uno strumento Astra, ma non e abbastanza chiara per eseguirlo. Vuoi usare l'ultima Work Session, la sessione attiva o la memoria archiviata?".to_string(),
            confidence,
            reason_code,
        }));
    }

    let target = raw_router_target_from(proposal.target, proposal.object, proposal.object_ids)?;
    let target_kind = target.kind.unwrap_or_else(|| "none".to_string());
    if !target_allowed(&normalized_tool, &target_kind) {
        return Err(RouterFailureReason::InvalidTarget);
    }

    Ok(AssistantRouteDecision::ToolCall(AssistantToolIntent {
        tool_family: tool_family_for(&normalized_tool),
        tool_name: normalized_tool,
        intent: proposal.intent.unwrap_or_else(|| "unknown".to_string()),
        target: ToolTarget {
            kind: target_kind,
            session_id: target.session_id,
            object_type: target.object_type,
            object_ids: target.object_ids,
        },
        confidence,
        language: proposal.language,
        query: proposal.query,
        reason_code,
    }))
}

fn raw_router_target_from(
    value: Option<Value>,
    object_type: Option<String>,
    object_ids: Vec<String>,
) -> Result<RawRouterTarget, RouterFailureReason> {
    match value {
        None | Some(Value::Null) => Ok(RawRouterTarget {
            kind: Some("none".to_string()),
            session_id: None,
            object_type,
            object_ids,
        }),
        Some(Value::String(kind)) => Ok(RawRouterTarget {
            kind: Some(kind),
            session_id: None,
            object_type,
            object_ids,
        }),
        Some(Value::Object(map)) => {
            let mut target: RawRouterTarget = serde_json::from_value(Value::Object(map))
                .map_err(|_| RouterFailureReason::InvalidSchema)?;
            if target.object_type.is_none() {
                target.object_type = object_type;
            }
            if target.object_ids.is_empty() {
                target.object_ids = object_ids;
            }
            Ok(target)
        }
        _ => Err(RouterFailureReason::InvalidTarget),
    }
}

fn known_tool(tool: &str) -> bool {
    matches!(
        tool,
        "work_session.start"
            | "work_session.stop"
            | "work_session.stop_and_recap"
            | "work_session.recap"
            | "work_session.generate_intelligence"
            | "work_session.transcript_summary"
            | "work_session.details"
            | "work_session.technical_recap"
            | "work_session.followup_draft"
            | "work_session.recall"
            | "work_session.search"
            | "work_session.attach_screen"
            | "work_session.show_evidence"
            | "work_session.status"
            | "work_session.open_details"
    )
}

fn tool_family_for(tool: &str) -> ToolFamily {
    match tool {
        "work_session.recall" | "work_session.search" => ToolFamily::SessionMemory,
        "work_session.attach_screen" => ToolFamily::ScreenContext,
        tool if tool.starts_with("work_session.") => ToolFamily::WorkSession,
        _ => ToolFamily::NormalAssistant,
    }
}

fn target_allowed(tool: &str, target_kind: &str) -> bool {
    let target = normalize_token(target_kind);
    match tool {
        "work_session.attach_screen" => {
            matches!(
                target.as_str(),
                "currentscreen" | "activesession" | "none" | "unknown" | ""
            )
        }
        "work_session.start" => matches!(target.as_str(), "none" | "unknown" | ""),
        "work_session.stop" | "work_session.stop_and_recap" => {
            matches!(target.as_str(), "activesession" | "none" | "unknown" | "")
        }
        "work_session.details"
        | "work_session.recap"
        | "work_session.generate_intelligence"
        | "work_session.transcript_summary"
        | "work_session.technical_recap"
        | "work_session.followup_draft"
        | "work_session.show_evidence"
        | "work_session.status"
        | "work_session.open_details" => matches!(
            target.as_str(),
            "lastreferencedsession"
                | "latestarchivedsession"
                | "lastcompletedsession"
                | "activesession"
                | "archivedsessions"
                | "none"
                | "unknown"
                | ""
        ),
        "work_session.recall" | "work_session.search" => matches!(
            target.as_str(),
            "archivedsessions"
                | "latestarchivedsession"
                | "lastreferencedsession"
                | "lastcompletedsession"
                | "activesession"
                | "none"
                | "unknown"
                | ""
        ),
        _ => false,
    }
}

fn normalize_tool_name(tool: &str) -> String {
    tool.trim()
        .chars()
        .flat_map(char::to_lowercase)
        .map(|ch| if ch == '-' { '_' } else { ch })
        .collect()
}

fn normalize_token(value: &str) -> String {
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

fn repair_common_router_json(content: &str) -> String {
    let mut repaired = content.trim().to_string();
    if repaired.starts_with("```") {
        repaired = repaired
            .trim_start_matches("```json")
            .trim_start_matches("```JSON")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim()
            .to_string();
    }
    remove_trailing_json_commas(&repaired)
}

fn remove_trailing_json_commas(content: &str) -> String {
    let chars = content.chars().collect::<Vec<_>>();
    let mut output = String::with_capacity(content.len());
    for (index, ch) in chars.iter().enumerate() {
        if *ch == ',' {
            let next_non_ws = chars
                .iter()
                .skip(index + 1)
                .find(|candidate| !candidate.is_whitespace());
            if matches!(next_non_ws, Some('}' | ']')) {
                continue;
            }
        }
        output.push(*ch);
    }
    output
}

#[allow(dead_code)]
pub fn tool_manifest_json() -> Value {
    serde_json::json!(available_tool_manifest())
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn compact_tool_manifest_json() -> Value {
    serde_json::json!(available_tool_manifest()
        .into_iter()
        .map(|item| serde_json::json!({
            "tool": item.tool,
            "allowed_targets": item.allowed_targets,
            "mode": item.mode,
            "requires_evidence_synthesis": item.requires_evidence_synthesis,
            "when_to_use": item.when_to_use,
        }))
        .collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_work_session_details_tool_call() {
        let decision = parse_router_decision(
            r#"{
                "route": "tool_call",
                "tool": "work_session.details",
                "intent": "generate_details",
                "target": {
                    "kind": "last_referenced_session",
                    "session_id": "session-1",
                    "object_type": "recap",
                    "object_ids": ["seg-1"]
                },
                "confidence": 0.86,
                "language": "it",
                "query": null,
                "reason_code": "contextual_followup"
            }"#,
        )
        .expect("router decision");

        match decision {
            AssistantRouteDecision::ToolCall(intent) => {
                assert_eq!(intent.tool_name, "work_session.details");
                assert_eq!(intent.target.kind, "last_referenced_session");
                assert_eq!(intent.target.object_type.as_deref(), Some("recap"));
                assert_eq!(intent.target.object_ids, vec!["seg-1".to_string()]);
            }
            _ => panic!("expected tool call"),
        }
    }

    #[test]
    fn manifest_exposes_transcript_summary_as_evidence_synthesis_tool() {
        let manifest = available_tool_manifest();
        let transcript_tool = manifest
            .iter()
            .find(|item| item.tool == "work_session.transcript_summary")
            .expect("transcript summary tool");

        assert!(transcript_tool.requires_evidence_synthesis);
        assert_eq!(transcript_tool.mode, "read_only_governed");
        assert!(transcript_tool
            .allowed_targets
            .contains(&"last_referenced_session"));
        assert!(transcript_tool.safety.contains("bounded"));
    }

    #[test]
    fn low_confidence_tool_call_clarifies() {
        let decision = parse_router_decision(
            r#"{
                "route": "tool_call",
                "tool": "work_session.recap",
                "intent": "generate_recap",
                "target": {"kind": "latest_archived_session", "session_id": null},
                "confidence": 0.72,
                "language": "it",
                "query": null,
                "reason_code": "weak_recap"
            }"#,
        )
        .expect("router decision");

        assert!(matches!(decision, AssistantRouteDecision::Clarify(_)));
    }

    #[test]
    fn runtime_result_distinguishes_empty_model_content() {
        let result = parse_router_runtime_result("   ", "gpt-oss:20b");

        assert_eq!(
            result,
            AssistantToolRouterRuntimeResult::EmptyModelContent {
                model: "gpt-oss:20b".to_string()
            }
        );
    }

    #[test]
    fn runtime_result_distinguishes_malformed_json() {
        let result = parse_router_runtime_result("not json", "gpt-oss:20b");

        assert_eq!(
            result,
            AssistantToolRouterRuntimeResult::Malformed {
                reason: RouterFailureReason::MalformedJson,
                raw_len: "not json".len(),
            }
        );
    }

    #[test]
    fn runtime_result_rejects_unknown_tool() {
        let result = parse_router_runtime_result(
            r#"{
                "route": "tool_call",
                "tool": "desktop.control",
                "intent": "click",
                "target": {"kind": "current_screen", "session_id": null},
                "confidence": 0.99,
                "reason_code": "unsafe"
            }"#,
            "gpt-oss:20b",
        );

        assert!(matches!(
            result,
            AssistantToolRouterRuntimeResult::Malformed {
                reason: RouterFailureReason::InvalidTool,
                ..
            }
        ));
    }

    #[test]
    fn runtime_result_rejects_invalid_target() {
        let result = parse_router_runtime_result(
            r#"{
                "route": "tool_call",
                "tool": "work_session.attach_screen",
                "intent": "attach_screen",
                "target": {"kind": "archived_sessions", "session_id": null},
                "confidence": 0.95,
                "reason_code": "bad_target"
            }"#,
            "gpt-oss:20b",
        );

        assert!(matches!(
            result,
            AssistantToolRouterRuntimeResult::Malformed {
                reason: RouterFailureReason::InvalidTarget,
                ..
            }
        ));
    }

    #[test]
    fn runtime_result_returns_normal_chat_metadata() {
        let result = parse_router_runtime_result(
            r#"{
                "route": "normal_chat",
                "tool": null,
                "intent": null,
                "target": {"kind": "none", "session_id": null},
                "confidence": 0.91,
                "reason_code": "ordinary_chat"
            }"#,
            "gpt-oss:20b",
        );

        assert_eq!(
            result,
            AssistantToolRouterRuntimeResult::NormalChat {
                confidence: 0.91,
                reason_code: "ordinary_chat".to_string()
            }
        );
    }

    #[test]
    fn compact_router_json_parses_to_tool_call() {
        let result = parse_router_runtime_result(
            r#"{
                "route": "tool_call",
                "tool": "work_session.transcript_summary",
                "target": "latest_archived_session",
                "object": "transcript",
                "confidence": 0.91,
                "query": "last recording transcript summary",
                "reason_code": "user_asks_last_recording_content"
            }"#,
            "gpt-oss:20b",
        );

        match result {
            AssistantToolRouterRuntimeResult::Routed(AssistantRouteDecision::ToolCall(intent)) => {
                assert_eq!(intent.tool_name, "work_session.transcript_summary");
                assert_eq!(intent.target.kind, "latest_archived_session");
                assert_eq!(intent.target.object_type.as_deref(), Some("transcript"));
                assert_eq!(intent.confidence, 0.91);
            }
            other => panic!("expected compact tool call, got {other:?}"),
        }
    }

    #[test]
    fn malformed_router_output_with_extractable_object_is_repaired_by_extraction() {
        let outcome = parse_router_runtime_result_with_repair(
            r#"Here is the route:
            {
                "route": "normal_chat",
                "tool": null,
                "target": "none",
                "confidence": 0.83,
                "reason_code": "ordinary_chat"
            }
            Done."#,
            "gpt-oss:20b",
        );

        assert!(outcome.repair_attempted);
        assert!(outcome.repair_succeeded);
        assert!(matches!(
            outcome.result,
            AssistantToolRouterRuntimeResult::NormalChat { .. }
        ));
    }

    #[test]
    fn malformed_router_output_with_trailing_comma_is_repaired() {
        let outcome = parse_router_runtime_result_with_repair(
            r#"{
                "route": "normal_chat",
                "tool": null,
                "target": "none",
                "confidence": 0.83,
                "reason_code": "ordinary_chat",
            }"#,
            "gpt-oss:20b",
        );

        assert!(outcome.repair_attempted);
        assert!(outcome.repair_succeeded);
        assert!(matches!(
            outcome.result,
            AssistantToolRouterRuntimeResult::NormalChat { .. }
        ));
    }

    #[test]
    fn rejects_unsafe_unknown_tool() {
        assert!(parse_router_decision(
            r#"{
                "route": "tool_call",
                "tool": "desktop.control",
                "intent": "click",
                "target": {"kind": "current_screen", "session_id": null},
                "confidence": 0.99,
                "reason_code": "unsafe"
            }"#,
        )
        .is_none());
    }
}
