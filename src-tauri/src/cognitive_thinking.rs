use crate::{
    conversation_history::ConversationMessage,
    desktop_agent_types::CapabilityManifest,
    memory::retrieval::MemoryContextPacket,
    model_routing::ollama_endpoint,
    speech_events::AssistantDeepSearchOptions,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    env,
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingPlan {
    pub request_id: String,
    pub intent_summary: String,
    #[serde(default)]
    pub self_questions: Vec<ThinkingQuestion>,
    pub memory_assessment: MemoryAssessment,
    pub evidence_assessment: EvidenceAssessment,
    pub route: ThinkingRoute,
    pub deep_search: DeepSearchDecision,
    pub tool_decision: ToolDecision,
    pub uncertainty: ThinkingUncertainty,
    #[serde(default)]
    pub user_visible_trace: Vec<UserVisibleThinkingStep>,
    pub confidence: f32,
    pub planner_source: String,
    pub duration_ms: u64,
    #[serde(default)]
    pub warnings: Vec<String>,
}

const MAX_USER_VISIBLE_THINKING_STEPS: usize = 8;
const MAX_USER_VISIBLE_TITLE_CHARS: usize = 96;
const MAX_USER_VISIBLE_DETAIL_CHARS: usize = 220;

impl ThinkingPlan {
    pub fn should_auto_run_deep_search(&self, options: &AssistantDeepSearchOptions) -> bool {
        options.auto_when_needed
            && !options.enabled
            && self.deep_search.is_needed()
            && matches!(
                self.route,
                ThinkingRoute::DeepSearchRequired | ThinkingRoute::MemoryGroundedAnswer | ThinkingRoute::DirectAnswer
            )
    }

    pub fn safe_user_trace(&self) -> Vec<UserVisibleThinkingStep> {
        if self.user_visible_trace.is_empty() {
            fallback_trace_for_plan(self)
        } else {
            self.user_visible_trace
                .iter()
                .take(MAX_USER_VISIBLE_THINKING_STEPS)
                .map(|step| UserVisibleThinkingStep {
                    phase: sanitize_phase(&step.phase),
                    title: sanitize_user_visible_text(&step.title, MAX_USER_VISIBLE_TITLE_CHARS),
                    detail: step
                        .detail
                        .as_deref()
                        .map(|detail| sanitize_user_visible_text(detail, MAX_USER_VISIBLE_DETAIL_CHARS)),
                    confidence: step.confidence.map(|value| value.clamp(0.0, 1.0)),
                })
                .collect()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingQuestion {
    pub question: String,
    pub purpose: ThinkingQuestionPurpose,
    #[serde(default)]
    pub answer_summary: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingQuestionPurpose {
    Intent,
    Memory,
    Evidence,
    Tool,
    DeepSearch,
    Safety,
    Synthesis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAssessment {
    pub relevant: bool,
    pub coverage: f32,
    pub node_count: usize,
    #[serde(default)]
    pub missing_information: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceAssessment {
    pub has_local_evidence: bool,
    pub has_current_session_evidence: bool,
    pub requires_current_information: bool,
    pub requires_external_sources: bool,
    pub evidence_summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingRoute {
    DirectAnswer,
    MemoryGroundedAnswer,
    ToolArbitrationRequired,
    DeepSearchRequired,
    ClarifyRequired,
    Refuse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchDecision {
    pub needed: bool,
    pub reason: DeepSearchReason,
    #[serde(default)]
    pub query_hint: Option<String>,
}

impl DeepSearchDecision {
    pub fn is_needed(&self) -> bool {
        self.needed && !matches!(self.reason, DeepSearchReason::NotNeeded | DeepSearchReason::BlockedByPolicy)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeepSearchReason {
    NotNeeded,
    UnknownTopic,
    CurrentInformation,
    LowMemoryCoverage,
    HighStakes,
    BlockedByPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDecision {
    pub tool_required: bool,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub candidate_tool: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingUncertainty {
    pub level: ThinkingUncertaintyLevel,
    #[serde(default)]
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingUncertaintyLevel {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserVisibleThinkingStep {
    pub phase: String,
    pub title: String,
    #[serde(default)]
    pub detail: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawThinkingPlan {
    #[serde(default)]
    intent_summary: Option<String>,
    #[serde(default)]
    self_questions: Vec<ThinkingQuestion>,
    #[serde(default)]
    memory_assessment: Option<MemoryAssessment>,
    #[serde(default)]
    evidence_assessment: Option<EvidenceAssessment>,
    #[serde(default)]
    route: Option<ThinkingRoute>,
    #[serde(default)]
    deep_search: Option<DeepSearchDecision>,
    #[serde(default)]
    tool_decision: Option<ToolDecision>,
    #[serde(default)]
    uncertainty: Option<ThinkingUncertainty>,
    #[serde(default)]
    user_visible_trace: Vec<UserVisibleThinkingStep>,
    #[serde(default)]
    confidence: Option<f32>,
    #[serde(default)]
    warnings: Vec<String>,
}

pub async fn build_thinking_plan(
    request_id: &str,
    message: &str,
    history: &[ConversationMessage],
    memory_context: Option<&MemoryContextPacket>,
    manifest: &CapabilityManifest,
    deep_search_options: &AssistantDeepSearchOptions,
) -> ThinkingPlan {
    let started = Instant::now();
    if !thinking_llm_enabled() {
        return heuristic_plan(
            request_id,
            message,
            history,
            memory_context,
            manifest,
            deep_search_options,
            started,
            Some("ASTRA_THINKING_LLM_ENABLED disabled".into()),
        );
    }

    match call_thinking_model(message, history, memory_context, manifest, deep_search_options).await {
        Ok(raw) => normalize_raw_plan(request_id, message, memory_context, raw, started),
        Err(error) => heuristic_plan(
            request_id,
            message,
            history,
            memory_context,
            manifest,
            deep_search_options,
            started,
            Some(format!("thinking planner fallback: {error}")),
        ),
    }
}

async fn call_thinking_model(
    message: &str,
    history: &[ConversationMessage],
    memory_context: Option<&MemoryContextPacket>,
    manifest: &CapabilityManifest,
    deep_search_options: &AssistantDeepSearchOptions,
) -> Result<RawThinkingPlan, String> {
    let model = env::var("ASTRA_THINKING_MODEL")
        .or_else(|_| env::var("ASTRA_PLANNER_MODEL"))
        .or_else(|_| env::var("ASTRA_ACTIVE_MODEL"))
        .unwrap_or_else(|_| "gpt-oss:20b".to_string());
    let payload = json!({
        "model": model,
        "stream": false,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "num_ctx": 8192
        },
        "messages": [
            {"role": "system", "content": thinking_system_prompt()},
            {"role": "user", "content": thinking_user_prompt(message, history, memory_context, manifest, deep_search_options)}
        ]
    });

    let client = Client::builder()
        .timeout(Duration::from_secs(thinking_timeout_secs()))
        .build()
        .map_err(|error| format!("Ollama thinking client build failed: {error}"))?;

    let response = client
        .post(ollama_endpoint("/api/chat"))
        .json(&payload)
        .send()
        .await
        .map_err(|error| format!("Ollama thinking request failed: {error}"))?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("Ollama thinking HTTP error {status}: {}", cap_text(&body, 360)));
    }
    let body: Value = response
        .json()
        .await
        .map_err(|error| format!("Ollama thinking response parse failed: {error}"))?;
    let content = body
        .get("message")
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "Ollama thinking returned an empty response".to_string())?;
    let json_text = extract_json_object(content)
        .ok_or_else(|| "thinking response did not contain a JSON object".to_string())?;
    serde_json::from_str::<RawThinkingPlan>(json_text)
        .map_err(|error| format!("thinking JSON schema parse failed: {error}"))
}

fn normalize_raw_plan(
    request_id: &str,
    message: &str,
    memory_context: Option<&MemoryContextPacket>,
    raw: RawThinkingPlan,
    started: Instant,
) -> ThinkingPlan {
    let heuristic = heuristic_classification(message, memory_context);
    let route = raw.route.unwrap_or_else(|| heuristic.route.clone());
    let deep_search = raw.deep_search.unwrap_or_else(|| heuristic.deep_search.clone());
    let memory_assessment = raw.memory_assessment.unwrap_or_else(|| memory_assessment(memory_context, &heuristic));
    let evidence_assessment = raw
        .evidence_assessment
        .unwrap_or_else(|| evidence_assessment(&heuristic, memory_context));

    ThinkingPlan {
        request_id: request_id.to_string(),
        intent_summary: sanitize_user_visible_text(
            raw.intent_summary
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| heuristic.intent_summary.as_str()),
            220,
        ),
        self_questions: sanitize_questions(raw.self_questions, &heuristic),
        memory_assessment,
        evidence_assessment,
        route,
        deep_search,
        tool_decision: raw.tool_decision.unwrap_or_else(|| heuristic.tool_decision.clone()),
        uncertainty: raw.uncertainty.unwrap_or_else(|| heuristic.uncertainty.clone()),
        user_visible_trace: sanitize_trace(raw.user_visible_trace),
        confidence: raw.confidence.unwrap_or(heuristic.confidence).clamp(0.0, 1.0),
        planner_source: "llm_json".into(),
        duration_ms: started.elapsed().as_millis() as u64,
        warnings: raw
            .warnings
            .into_iter()
            .take(6)
            .map(|warning| sanitize_user_visible_text(&warning, 220))
            .collect(),
    }
}

#[derive(Debug, Clone)]
struct HeuristicClassification {
    intent_summary: String,
    route: ThinkingRoute,
    deep_search: DeepSearchDecision,
    tool_decision: ToolDecision,
    uncertainty: ThinkingUncertainty,
    confidence: f32,
    requires_current_information: bool,
    local_or_memory_query: bool,
}

fn heuristic_plan(
    request_id: &str,
    message: &str,
    history: &[ConversationMessage],
    memory_context: Option<&MemoryContextPacket>,
    manifest: &CapabilityManifest,
    deep_search_options: &AssistantDeepSearchOptions,
    started: Instant,
    warning: Option<String>,
) -> ThinkingPlan {
    let heuristic = heuristic_classification(message, memory_context);
    let mut warnings = warning.into_iter().collect::<Vec<_>>();
    if history.is_empty() {
        warnings.push("no_prior_history_for_this_request".into());
    }
    if manifest.enabled_tool_names.is_empty() {
        warnings.push("no_enabled_tools_in_capability_manifest".into());
    }
    if heuristic.deep_search.is_needed() && !deep_search_options.auto_when_needed && !deep_search_options.enabled {
        warnings.push("deep_search_needed_but_auto_mode_disabled".into());
    }

    ThinkingPlan {
        request_id: request_id.to_string(),
        intent_summary: heuristic.intent_summary.clone(),
        self_questions: sanitize_questions(Vec::new(), &heuristic),
        memory_assessment: memory_assessment(memory_context, &heuristic),
        evidence_assessment: evidence_assessment(&heuristic, memory_context),
        route: heuristic.route.clone(),
        deep_search: heuristic.deep_search.clone(),
        tool_decision: heuristic.tool_decision.clone(),
        uncertainty: heuristic.uncertainty.clone(),
        user_visible_trace: fallback_trace_for_heuristic(&heuristic, memory_context),
        confidence: heuristic.confidence,
        planner_source: "deterministic_heuristic".into(),
        duration_ms: started.elapsed().as_millis() as u64,
        warnings,
    }
}

fn heuristic_classification(message: &str, memory_context: Option<&MemoryContextPacket>) -> HeuristicClassification {
    let text = message.to_lowercase();
    let memory_nodes = memory_context.map(|packet| packet.nodes.len()).unwrap_or(0);
    let has_memory = memory_nodes > 0;
    let explicit_tool = contains_any(&text, &["apri", "clicca", "scrivi nel file", "leggi il file", "terminale", "browser", "cerca su google", "youtube", "cartella", "screenshot", "schermo"]);
    let local_or_memory_query = contains_any(&text, &["come mi chiamo", "sai chi sono", "ricordi", "memoria", "ultima sessione", "registrazione", "transcript", "trascrizione", "cosa abbiamo parlato", "work session", "meeting"]);
    let requires_current_information = contains_any(&text, &["ultime", "attuale", "aggiornato", "oggi", "ora", "news", "prezzo", "versione corrente", "recenti", "2026", "2025", "cerca online", "web", "internet"]);
    let high_stakes = contains_any(&text, &["legale", "medico", "finanziario", "investimento", "contratto", "normativa", "sicurezza", "privacy"]);
    let asks_research = contains_any(&text, &["deep search", "approfondisci", "fonti", "ricerca", "paper", "documentati", "non conosci"]);

    let mut deep_search = DeepSearchDecision {
        needed: false,
        reason: DeepSearchReason::NotNeeded,
        query_hint: None,
    };
    let route = if explicit_tool && !asks_research {
        ThinkingRoute::ToolArbitrationRequired
    } else if local_or_memory_query {
        if has_memory {
            ThinkingRoute::MemoryGroundedAnswer
        } else {
            ThinkingRoute::ToolArbitrationRequired
        }
    } else if requires_current_information || asks_research || (high_stakes && !has_memory) {
        deep_search = DeepSearchDecision {
            needed: true,
            reason: if high_stakes {
                DeepSearchReason::HighStakes
            } else if requires_current_information {
                DeepSearchReason::CurrentInformation
            } else if !has_memory {
                DeepSearchReason::LowMemoryCoverage
            } else {
                DeepSearchReason::UnknownTopic
            },
            query_hint: Some(cap_text(message, 180)),
        };
        ThinkingRoute::DeepSearchRequired
    } else if has_memory {
        ThinkingRoute::MemoryGroundedAnswer
    } else {
        ThinkingRoute::DirectAnswer
    };

    let tool_decision = ToolDecision {
        tool_required: matches!(route, ThinkingRoute::ToolArbitrationRequired),
        reason: matches!(route, ThinkingRoute::ToolArbitrationRequired).then(|| {
            if local_or_memory_query {
                "La richiesta dipende da sessioni, transcript o stato locale governato.".into()
            } else {
                "La richiesta implica interazione con browser, desktop, file o superfici osservabili.".into()
            }
        }),
        candidate_tool: None,
    };

    let uncertainty = ThinkingUncertainty {
        level: if deep_search.is_needed() || (local_or_memory_query && !has_memory) {
            ThinkingUncertaintyLevel::Medium
        } else {
            ThinkingUncertaintyLevel::Low
        },
        reasons: if deep_search.is_needed() {
            vec!["Servono evidenze aggiornate o fonti esterne prima della sintesi finale.".into()]
        } else if local_or_memory_query && !has_memory {
            vec!["La memoria semantica non copre completamente la richiesta; serve routing locale/sessione.".into()]
        } else {
            Vec::new()
        },
    };

    let intent_summary = if local_or_memory_query {
        "Richiesta legata alla memoria locale, identità, sessioni o transcript di Astra.".into()
    } else if explicit_tool {
        "Richiesta operativa che potrebbe richiedere strumenti governati.".into()
    } else if deep_search.is_needed() {
        "Richiesta informativa che potrebbe richiedere ricerca esterna governata.".into()
    } else {
        "Richiesta conversazionale o di sintesi rispondibile con il contesto disponibile.".into()
    };

    HeuristicClassification {
        intent_summary,
        route,
        deep_search,
        tool_decision,
        uncertainty,
        confidence: if has_memory || explicit_tool || requires_current_information { 0.72 } else { 0.64 },
        requires_current_information,
        local_or_memory_query,
    }
}

fn memory_assessment(memory_context: Option<&MemoryContextPacket>, heuristic: &HeuristicClassification) -> MemoryAssessment {
    let node_count = memory_context.map(|packet| packet.nodes.len()).unwrap_or(0);
    MemoryAssessment {
        relevant: node_count > 0,
        coverage: if node_count >= 6 { 0.74 } else if node_count > 0 { 0.48 } else { 0.0 },
        node_count,
        missing_information: if node_count == 0 && heuristic.local_or_memory_query {
            vec!["Nessun nodo memoria sufficiente nel packet corrente; serve routing verso sessioni/transcript se disponibili.".into()]
        } else if heuristic.requires_current_information {
            vec!["La memoria locale potrebbe non essere aggiornata; servono fonti esterne governate.".into()]
        } else {
            Vec::new()
        },
    }
}

fn evidence_assessment(heuristic: &HeuristicClassification, memory_context: Option<&MemoryContextPacket>) -> EvidenceAssessment {
    let has_memory = memory_context.map(|packet| !packet.nodes.is_empty()).unwrap_or(false);
    EvidenceAssessment {
        has_local_evidence: has_memory || heuristic.local_or_memory_query,
        has_current_session_evidence: heuristic.local_or_memory_query,
        requires_current_information: heuristic.requires_current_information,
        requires_external_sources: heuristic.deep_search.is_needed(),
        evidence_summary: if heuristic.deep_search.is_needed() {
            "Servono fonti esterne o aggiornate; Deep Search può produrre evidenze bounded.".into()
        } else if has_memory {
            "La Memory Graph ha restituito contesto locale da integrare prudentemente.".into()
        } else {
            "Nessuna evidenza locale forte richiesta prima della risposta finale.".into()
        },
    }
}

fn sanitize_questions(
    questions: Vec<ThinkingQuestion>,
    heuristic: &HeuristicClassification,
) -> Vec<ThinkingQuestion> {
    let mut sanitized = questions
        .into_iter()
        .take(8)
        .filter(|question| !question.question.trim().is_empty())
        .map(|question| ThinkingQuestion {
            question: sanitize_user_visible_text(&question.question, 160),
            purpose: question.purpose,
            answer_summary: question
                .answer_summary
                .as_deref()
                .map(|value| sanitize_user_visible_text(value, 220)),
            confidence: question.confidence.map(|value| value.clamp(0.0, 1.0)),
        })
        .collect::<Vec<_>>();
    if sanitized.is_empty() {
        sanitized = vec![
            ThinkingQuestion {
                question: "Cosa sta chiedendo davvero l'utente?".into(),
                purpose: ThinkingQuestionPurpose::Intent,
                answer_summary: Some(heuristic.intent_summary.clone()),
                confidence: Some(heuristic.confidence),
            },
            ThinkingQuestion {
                question: "La memoria locale o la sessione corrente sono più affidabili della chat generica?".into(),
                purpose: ThinkingQuestionPurpose::Memory,
                answer_summary: Some(if heuristic.local_or_memory_query { "Sì, devo privilegiare memoria, transcript o routing locale governato." } else { "Solo se il Memory Graph restituisce evidenze rilevanti." }.into()),
                confidence: Some(0.7),
            },
            ThinkingQuestion {
                question: "Serve Deep Search prima di rispondere?".into(),
                purpose: ThinkingQuestionPurpose::DeepSearch,
                answer_summary: Some(if heuristic.deep_search.is_needed() { "Sì, la richiesta sembra richiedere fonti aggiornate o copertura esterna." } else { "No, non emergono segnali forti di ricerca esterna necessaria." }.into()),
                confidence: Some(0.68),
            },
        ];
    }
    sanitized
}

fn sanitize_trace(trace: Vec<UserVisibleThinkingStep>) -> Vec<UserVisibleThinkingStep> {
    trace
        .into_iter()
        .take(MAX_USER_VISIBLE_THINKING_STEPS)
        .filter(|step| !step.title.trim().is_empty())
        .map(|step| UserVisibleThinkingStep {
            phase: sanitize_phase(&step.phase),
            title: sanitize_user_visible_text(&step.title, MAX_USER_VISIBLE_TITLE_CHARS),
            detail: step
                .detail
                .as_deref()
                .map(|detail| sanitize_user_visible_text(detail, MAX_USER_VISIBLE_DETAIL_CHARS)),
            confidence: step.confidence.map(|value| value.clamp(0.0, 1.0)),
        })
        .collect()
}

fn fallback_trace_for_plan(plan: &ThinkingPlan) -> Vec<UserVisibleThinkingStep> {
    let heuristic = HeuristicClassification {
        intent_summary: plan.intent_summary.clone(),
        route: plan.route.clone(),
        deep_search: plan.deep_search.clone(),
        tool_decision: plan.tool_decision.clone(),
        uncertainty: plan.uncertainty.clone(),
        confidence: plan.confidence,
        requires_current_information: plan.evidence_assessment.requires_current_information,
        local_or_memory_query: plan.evidence_assessment.has_current_session_evidence,
    };
    fallback_trace_for_heuristic(&heuristic, None)
}

fn fallback_trace_for_heuristic(
    heuristic: &HeuristicClassification,
    memory_context: Option<&MemoryContextPacket>,
) -> Vec<UserVisibleThinkingStep> {
    let memory_nodes = memory_context.map(|packet| packet.nodes.len()).unwrap_or(0);
    let mut steps = vec![
        UserVisibleThinkingStep {
            phase: "intent".into(),
            title: "Capisco l'intento".into(),
            detail: Some(heuristic.intent_summary.clone()),
            confidence: Some(heuristic.confidence),
        },
        UserVisibleThinkingStep {
            phase: "memory".into(),
            title: "Controllo la memoria locale".into(),
            detail: Some(if memory_nodes > 0 {
                format!("Ho trovato {memory_nodes} nodo/i rilevanti nella Memory Graph.")
            } else {
                "Non ho abbastanza memoria semantica diretta per considerarla sufficiente.".into()
            }),
            confidence: Some(if memory_nodes > 0 { 0.72 } else { 0.44 }),
        },
        UserVisibleThinkingStep {
            phase: "routing".into(),
            title: "Valuto il percorso governato".into(),
            detail: Some(match heuristic.route {
                ThinkingRoute::ToolArbitrationRequired => "La richiesta deve passare dai tool governati, non da un'azione libera del modello.",
                ThinkingRoute::DeepSearchRequired => "La richiesta può richiedere ricerca bounded prima della sintesi.",
                ThinkingRoute::MemoryGroundedAnswer => "La risposta può essere ancorata al contesto memoria disponibile.",
                ThinkingRoute::ClarifyRequired => "Prima di procedere serve una precisazione dell'utente.",
                ThinkingRoute::Refuse => "La richiesta richiede una risposta sicura o un rifiuto governato.",
                ThinkingRoute::DirectAnswer => "La richiesta sembra rispondibile direttamente con il contesto disponibile.",
            }.into()),
            confidence: Some(heuristic.confidence),
        },
    ];
    if heuristic.deep_search.is_needed() {
        steps.push(UserVisibleThinkingStep {
            phase: "deep_search".into(),
            title: "Decido se attivare Deep Search".into(),
            detail: Some("La copertura locale non basta o servono informazioni aggiornate; userò ricerca governata se consentita.".into()),
            confidence: Some(0.7),
        });
    }
    steps.push(UserVisibleThinkingStep {
        phase: "synthesis".into(),
        title: "Preparo la sintesi finale".into(),
        detail: Some("Terrò separate ipotesi, memoria, evidenze e azioni validate dal runtime.".into()),
        confidence: Some(0.76),
    });
    steps
}

fn thinking_system_prompt() -> &'static str {
    "You are AstraOS governed cognitive planner. You do not answer the user. You produce a concise JSON-only thinking plan for Rust runtime validation. Do not reveal chain-of-thought. Use short self-question summaries only. Rust owns final decisions for tools, permissions, policy, audit, browser/desktop/file actions, Deep Search and memory writes. Never authorize blind clicking, fabricated coordinates, unsafe autonomous actions, unbounded loops or raw hidden reasoning. Return only a JSON object matching the requested schema."
}

fn thinking_user_prompt(
    message: &str,
    history: &[ConversationMessage],
    memory_context: Option<&MemoryContextPacket>,
    manifest: &CapabilityManifest,
    deep_search_options: &AssistantDeepSearchOptions,
) -> String {
    let history_excerpt = history
        .iter()
        .rev()
        .take(6)
        .map(|entry| format!("{}: {}", entry.role, cap_text(&entry.content, 360)))
        .collect::<Vec<_>>()
        .join("\n");
    let memory_nodes = memory_context.map(|packet| packet.nodes.len()).unwrap_or(0);
    let memory_excerpt = memory_context
        .map(|packet| {
            packet
                .nodes
                .iter()
                .take(6)
                .map(|node| format!("- {} | score {:.2} | {}", cap_text(&node.title, 80), node.score, cap_text(&node.summary, 260)))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();
    let enabled_tools = manifest
        .enabled_tool_names
        .iter()
        .take(24)
        .cloned()
        .collect::<Vec<_>>();

    format!(
        "User request:\n{}\n\nRecent conversation excerpt:\n{}\n\nMemory packet: {} node(s)\n{}\n\nEnabled governed tools/capabilities: {:?}\n\nDeep Search options: enabled={}, auto_when_needed={}\n\nReturn JSON only with this shape:\n{{\"intent_summary\": string, \"self_questions\": [{{\"question\": string, \"purpose\": \"intent|memory|evidence|tool|deep_search|safety|synthesis\", \"answer_summary\": string, \"confidence\": number}}], \"memory_assessment\": {{\"relevant\": boolean, \"coverage\": number, \"node_count\": number, \"missing_information\": [string]}}, \"evidence_assessment\": {{\"has_local_evidence\": boolean, \"has_current_session_evidence\": boolean, \"requires_current_information\": boolean, \"requires_external_sources\": boolean, \"evidence_summary\": string}}, \"route\": \"direct_answer|memory_grounded_answer|tool_arbitration_required|deep_search_required|clarify_required|refuse\", \"deep_search\": {{\"needed\": boolean, \"reason\": \"not_needed|unknown_topic|current_information|low_memory_coverage|high_stakes|blocked_by_policy\", \"query_hint\": string|null}}, \"tool_decision\": {{\"tool_required\": boolean, \"reason\": string|null, \"candidate_tool\": string|null}}, \"uncertainty\": {{\"level\": \"low|medium|high\", \"reasons\": [string]}}, \"user_visible_trace\": [{{\"phase\": string, \"title\": string, \"detail\": string, \"confidence\": number}}], \"confidence\": number, \"warnings\": [string]}}",
        cap_text(message, 900),
        if history_excerpt.is_empty() { "none" } else { &history_excerpt },
        memory_nodes,
        if memory_excerpt.is_empty() { "none" } else { &memory_excerpt },
        enabled_tools,
        deep_search_options.enabled,
        deep_search_options.auto_when_needed,
    )
}

fn thinking_llm_enabled() -> bool {
    env::var("ASTRA_THINKING_LLM_ENABLED")
        .map(|value| !matches!(value.trim(), "0" | "false" | "FALSE" | "off" | "OFF"))
        .unwrap_or(true)
}

fn thinking_timeout_secs() -> u64 {
    env::var("ASTRA_THINKING_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map(|value| value.clamp(2, 45))
        .unwrap_or(12)
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

fn sanitize_phase(value: &str) -> String {
    let normalized = value
        .trim()
        .to_lowercase()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch
            } else if matches!(ch, '_' | '-') {
                '_'
            } else {
                ' '
            }
        })
        .collect::<String>();

    let compact = normalized
        .split_whitespace()
        .collect::<Vec<_>>()
        .join("_");

    canonical_thinking_phase(&compact)
}

fn canonical_thinking_phase(value: &str) -> String {
    let phase = value.trim_matches('_');
    let canonical = match phase {
        "intent" | "intent_summary" | "understand_intent" => "intent",
        "memory" | "memory_retrieval" | "local_memory" | "rag" => "memory",
        "evidence" | "evidence_assessment" | "sources" => "evidence",
        "tool" | "tools" | "tool_arbitration" | "tool_decision" => "tool_arbitration",
        "routing" | "route" | "governed_route" => "routing",
        "deep_search" | "deep_search_decision" | "research" => "deep_search_decision",
        "safety" | "policy" | "policy_safety" => "safety",
        "synthesis" | "answer" | "response" => "synthesis",
        "clarify" | "clarification" | "clarify_required" => "clarify_required",
        "quality" | "quality_review" | "thinking_quality" => "quality",
        "refuse" | "safe_refusal" => "refuse",
        "thinking" | "" => "thinking",
        other if other.contains("intent") => "intent",
        other if other.contains("memory") || other.contains("rag") => "memory",
        other if other.contains("evidence") || other.contains("source") => "evidence",
        other if other.contains("tool") => "tool_arbitration",
        other if other.contains("routing") || other.contains("route") => "routing",
        other if other.contains("deep_search") || other.contains("research") => "deep_search_decision",
        other if other.contains("safety") || other.contains("policy") => "safety",
        other if other.contains("clarify") => "clarify_required",
        other if other.contains("quality") => "quality",
        other if other.contains("synthesis") || other.contains("answer") || other.contains("response") => "synthesis",
        _ => "thinking",
    };
    canonical.into()
}

fn sanitize_user_visible_text(value: &str, max_chars: usize) -> String {
    let capped = cap_text(value, max_chars);
    let lower = capped.to_lowercase();
    let forbidden = [
        "chain-of-thought",
        "chain of thought",
        "hidden reasoning",
        "private reasoning",
        "scratchpad",
        "internal monologue",
        "step-by-step reasoning",
        "raw reasoning",
        "ragionamento nascosto",
        "pensiero nascosto",
        "monologo interno",
        "traccia privata",
    ];
    if forbidden.iter().any(|marker| lower.contains(marker)) {
        return "Traccia sintetica governata disponibile; ragionamento interno non esposto.".into();
    }
    capped
}

fn cap_text(value: &str, max_chars: usize) -> String {
    let trimmed = value.trim().replace('\0', " ");
    if trimmed.chars().count() <= max_chars {
        return trimmed;
    }
    let mut capped = trimmed.chars().take(max_chars.saturating_sub(1)).collect::<String>();
    capped.push('…');
    capped
}

fn extract_json_object(content: &str) -> Option<&str> {
    let start = content.find('{')?;
    let end = content.rfind('}')?;
    (end >= start).then_some(&content[start..=end])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_plan() -> ThinkingPlan {
        ThinkingPlan {
            request_id: "req-test".into(),
            intent_summary: "Test intent".into(),
            self_questions: Vec::new(),
            memory_assessment: MemoryAssessment {
                relevant: false,
                coverage: 0.0,
                node_count: 0,
                missing_information: Vec::new(),
            },
            evidence_assessment: EvidenceAssessment {
                has_local_evidence: false,
                has_current_session_evidence: false,
                requires_current_information: false,
                requires_external_sources: false,
                evidence_summary: "No external evidence required.".into(),
            },
            route: ThinkingRoute::DirectAnswer,
            deep_search: DeepSearchDecision {
                needed: false,
                reason: DeepSearchReason::NotNeeded,
                query_hint: None,
            },
            tool_decision: ToolDecision {
                tool_required: false,
                reason: None,
                candidate_tool: None,
            },
            uncertainty: ThinkingUncertainty {
                level: ThinkingUncertaintyLevel::Low,
                reasons: Vec::new(),
            },
            user_visible_trace: Vec::new(),
            confidence: 0.82,
            planner_source: "unit_test".into(),
            duration_ms: 1,
            warnings: Vec::new(),
        }
    }

    #[test]
    fn safe_user_trace_never_exposes_raw_reasoning_markers() {
        let mut plan = base_plan();
        plan.user_visible_trace = vec![UserVisibleThinkingStep {
            phase: "intent:bad/value".into(),
            title: "Here is my chain-of-thought".into(),
            detail: Some("private reasoning scratchpad".into()),
            confidence: Some(1.4),
        }];

        let trace = plan.safe_user_trace();
        assert_eq!(trace[0].phase, "intent");
        assert!(trace[0].title.contains("Traccia sintetica governata"));
        assert!(trace[0].detail.as_deref().unwrap_or_default().contains("Traccia sintetica governata"));
        assert_eq!(trace[0].confidence, Some(1.0));
    }

    #[test]
    fn auto_deep_search_requires_auto_mode_needed_and_not_manual_enabled() {
        let mut plan = base_plan();
        plan.route = ThinkingRoute::DeepSearchRequired;
        plan.deep_search = DeepSearchDecision {
            needed: true,
            reason: DeepSearchReason::CurrentInformation,
            query_hint: Some("latest".into()),
        };

        let enabled = AssistantDeepSearchOptions {
            enabled: false,
            auto_when_needed: true,
            ..Default::default()
        };
        let manual = AssistantDeepSearchOptions {
            enabled: true,
            auto_when_needed: true,
            ..Default::default()
        };
        let disabled = AssistantDeepSearchOptions {
            enabled: false,
            auto_when_needed: false,
            ..Default::default()
        };

        assert!(plan.should_auto_run_deep_search(&enabled));
        assert!(!plan.should_auto_run_deep_search(&manual));
        assert!(!plan.should_auto_run_deep_search(&disabled));
    }

    #[test]
    fn auto_deep_search_never_bypasses_tool_arbitration() {
        let mut plan = base_plan();
        plan.route = ThinkingRoute::ToolArbitrationRequired;
        plan.deep_search = DeepSearchDecision {
            needed: true,
            reason: DeepSearchReason::CurrentInformation,
            query_hint: Some("latest browser workflow".into()),
        };
        plan.tool_decision = ToolDecision {
            tool_required: true,
            reason: Some("browser action requires governed tool arbitration".into()),
            candidate_tool: Some("browser.open".into()),
        };

        let options = AssistantDeepSearchOptions {
            enabled: false,
            auto_when_needed: true,
            ..Default::default()
        };

        assert!(!plan.should_auto_run_deep_search(&options));
    }

    #[test]
    fn regression_direct_answer_stays_direct_without_tools_or_deep_search() {
        let classification = heuristic_classification(
            "spiegami in modo semplice come funziona una closure in Rust",
            None,
        );

        assert_eq!(classification.route, ThinkingRoute::DirectAnswer);
        assert!(!classification.deep_search.is_needed());
        assert!(!classification.tool_decision.tool_required);
        assert!(!classification.local_or_memory_query);
    }

    #[test]
    fn regression_memory_bound_request_does_not_trigger_deep_search() {
        let classification = heuristic_classification(
            "ricordi cosa abbiamo detto nell'ultima sessione registrata?",
            None,
        );

        assert!(classification.local_or_memory_query);
        assert!(!classification.deep_search.is_needed());
        assert!(matches!(
            classification.route,
            ThinkingRoute::MemoryGroundedAnswer | ThinkingRoute::ToolArbitrationRequired
        ));
    }

    #[test]
    fn regression_current_info_request_requires_auto_deep_search_candidate() {
        let classification = heuristic_classification(
            "quali sono le ultime novità aggiornate su AstraOS oggi?",
            None,
        );

        assert_eq!(classification.route, ThinkingRoute::DeepSearchRequired);
        assert!(classification.requires_current_information);
        assert!(classification.deep_search.is_needed());
        assert_eq!(classification.deep_search.reason, DeepSearchReason::CurrentInformation);
    }

    #[test]
    fn regression_tool_bound_request_stays_inside_governed_tool_arbitration() {
        let classification = heuristic_classification(
            "apri il browser e clicca il primo risultato della pagina",
            None,
        );

        assert_eq!(classification.route, ThinkingRoute::ToolArbitrationRequired);
        assert!(classification.tool_decision.tool_required);
        assert!(!classification.deep_search.is_needed());
    }

    #[test]
    fn thinking_phase_names_are_canonicalized_for_stable_ui_contract() {
        assert_eq!(sanitize_phase("deep-search decision"), "deep_search_decision");
        assert_eq!(sanitize_phase("tool:decision"), "tool_arbitration");
        assert_eq!(sanitize_phase("policy/safety"), "safety");
        assert_eq!(sanitize_phase("governed route"), "routing");
        assert_eq!(sanitize_phase("unknown private label"), "thinking");
    }

    #[test]
    fn regression_low_confidence_plans_keep_user_visible_trace_metadata_only() {
        let mut plan = base_plan();
        plan.confidence = 0.31;
        plan.uncertainty = ThinkingUncertainty {
            level: ThinkingUncertaintyLevel::High,
            reasons: vec!["input ambiguo".into()],
        };
        plan.user_visible_trace = vec![UserVisibleThinkingStep {
            phase: "clarify_required".into(),
            title: "Valuto se serve una precisazione".into(),
            detail: Some("La richiesta ha bassa confidenza; mostro solo una sintesi governata.".into()),
            confidence: Some(0.31),
        }];

        let trace = plan.safe_user_trace();
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].phase, "clarify_required");
        assert_eq!(trace[0].confidence, Some(0.31));
        assert!(!trace[0].title.to_lowercase().contains("chain-of-thought"));
    }
}

