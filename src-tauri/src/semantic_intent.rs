use crate::{
    assistant_context::build_capability_context_v2, desktop_agent_types::CapabilityManifest,
};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{env, time::Duration};

const OLLAMA_BASE_URL: &str = "http://127.0.0.1:11434";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticIntentKind {
    CapabilityQuestion,
    ToolActionRequest,
    ScreenQuestion,
    ScreenAnalysisRequest,
    ApprovalStatusQuestion,
    NormalChat,
}

impl SemanticIntentKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CapabilityQuestion => "capability_question",
            Self::ToolActionRequest => "tool_action_request",
            Self::ScreenQuestion => "screen_question",
            Self::ScreenAnalysisRequest => "screen_analysis_request",
            Self::ApprovalStatusQuestion => "approval_status_question",
            Self::NormalChat => "normal_chat",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityTarget {
    Screen,
    Browser,
    Terminal,
    FilesystemRead,
    FilesystemWrite,
    FilesystemSearch,
    DesktopLaunch,
    Approval,
    General,
    Unknown,
}

impl CapabilityTarget {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Screen => "screen",
            Self::Browser => "browser",
            Self::Terminal => "terminal",
            Self::FilesystemRead => "filesystem_read",
            Self::FilesystemWrite => "filesystem_write",
            Self::FilesystemSearch => "filesystem_search",
            Self::DesktopLaunch => "desktop_launch",
            Self::Approval => "approval",
            Self::General => "general",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticAction {
    BrowserSearch,
    BrowserOpenUrl,
    FilesystemRead,
    FilesystemSearch,
    FilesystemWrite,
    TerminalRun,
    DesktopLaunchApp,
    None,
    Unknown,
}

impl SemanticAction {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::BrowserSearch => "browser_search",
            Self::BrowserOpenUrl => "browser_open_url",
            Self::FilesystemRead => "filesystem_read",
            Self::FilesystemSearch => "filesystem_search",
            Self::FilesystemWrite => "filesystem_write",
            Self::TerminalRun => "terminal_run",
            Self::DesktopLaunchApp => "desktop_launch_app",
            Self::None => "none",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SemanticScreenRequest {
    pub capture_fresh: Option<bool>,
    pub reuse_recent: Option<bool>,
    pub state_question: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct SemanticIntent {
    pub kind: SemanticIntentKind,
    pub target: CapabilityTarget,
    pub action: SemanticAction,
    pub params: Value,
    pub screen: SemanticScreenRequest,
    pub confidence: f32,
    pub language: Option<String>,
    pub rationale: Option<String>,
}

pub async fn classify_intent(
    message: &str,
    manifest: &CapabilityManifest,
) -> Result<SemanticIntent, String> {
    let timeout_ms = env::var("ASTRA_INTENT_CLASSIFIER_TIMEOUT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(4_000);
    let client = Client::builder()
        .timeout(Duration::from_millis(timeout_ms))
        .build()
        .map_err(|error| format!("intent classifier client setup failed: {error}"))?;

    let installed_models = fetch_installed_models(&client).await.unwrap_or_default();
    let model = select_intent_model(&installed_models);
    let facts = serde_json::to_string(&build_capability_context_v2(manifest).facts)
        .unwrap_or_else(|_| "{}".to_string());
    let user_message = message.chars().take(1_200).collect::<String>();
    let payload = json!({
        "model": model,
        "stream": false,
        "format": "json",
        "keep_alive": "30m",
        "options": {
            "temperature": 0.0,
            "top_p": 0.1,
            "num_predict": 220
        },
        "messages": [
            {
                "role": "system",
                "content": classifier_system_prompt()
            },
            {
                "role": "user",
                "content": format!("Runtime facts JSON:\n{facts}\n\nUser message:\n{user_message}")
            }
        ]
    });

    let response = client
        .post(format!("{OLLAMA_BASE_URL}/api/chat"))
        .json(&payload)
        .send()
        .await
        .map_err(|error| format!("intent classifier request failed: {error}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("intent classifier HTTP error {status}: {body}"));
    }

    let body: Value = response
        .json()
        .await
        .map_err(|error| format!("intent classifier response parse failed: {error}"))?;
    let content = body
        .get("message")
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .ok_or_else(|| "intent classifier response did not contain message.content".to_string())?;

    parse_intent_json(content)
}

fn classifier_system_prompt() -> &'static str {
    concat!(
        "You classify one user message for Astra, a governed local desktop assistant. ",
        "You interpret intent semantically across English, Italian, mixed language, paraphrases, and imperfect STT. ",
        "You do not decide whether execution is allowed; Rust validates every tool, permission, approval, and screen state. ",
        "Return exactly one JSON object and no prose. Schema: ",
        "{\"intent\":\"capability_question|tool_action_request|screen_question|screen_analysis_request|approval_status_question|normal_chat\",",
        "\"target\":\"screen|browser|terminal|filesystem_read|filesystem_write|filesystem_search|desktop_launch|approval|general|unknown\",",
        "\"action\":\"browser_search|browser_open_url|filesystem_read|filesystem_search|filesystem_write|terminal_run|desktop_launch_app|none|unknown\",",
        "\"params\":{\"query\":null,\"url\":null,\"path\":null,\"pattern\":null,\"content\":null,\"mode\":null,\"command\":null,\"args\":[],\"cwd\":null,\"app\":null},",
        "\"screen\":{\"capture_fresh\":null,\"reuse_recent\":null,\"state_question\":null},",
        "\"confidence\":0.0,\"language\":\"en|it|mixed|unknown\",\"rationale\":\"short reason\"}. ",
        "Use capability_question when the user asks what Astra can do. ",
        "Use screen_question for screen access/state questions such as whether Astra can see, observe, capture, or analyze the screen. ",
        "Use screen_analysis_request when the user wants visible screen content interpreted, asks what they are looking at, what is wrong here, what to click, or wants a fresh screen analysis. ",
        "Use approval_status_question for pending approval, confirmation, or whether something is waiting. ",
        "Use tool_action_request only when the user is asking Astra to perform a concrete desktop/browser/file/terminal action. ",
        "Use normal_chat for ordinary conversation, advice, explanation, coding help, or ambiguous requests that should not execute a tool."
    )
}

#[derive(Debug, Deserialize)]
struct RawIntent {
    intent: Option<String>,
    target: Option<String>,
    action: Option<String>,
    tool_name: Option<String>,
    params: Option<Value>,
    parameters: Option<Value>,
    screen: Option<RawScreenRequest>,
    confidence: Option<f32>,
    language: Option<String>,
    rationale: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawScreenRequest {
    capture_fresh: Option<bool>,
    reuse_recent: Option<bool>,
    state_question: Option<bool>,
}

fn parse_intent_json(content: &str) -> Result<SemanticIntent, String> {
    let json_text = extract_json_object(content)
        .ok_or_else(|| "intent classifier did not return a JSON object".to_string())?;
    let raw = serde_json::from_str::<RawIntent>(json_text)
        .map_err(|error| format!("intent classifier JSON validation failed: {error}"))?;

    let mut kind = parse_intent_kind(raw.intent.as_deref());
    let target = parse_target(raw.target.as_deref());
    let action = parse_action(raw.action.as_deref())
        .or_else(|| parse_action_from_tool_name(raw.tool_name.as_deref()))
        .unwrap_or(SemanticAction::Unknown);
    if matches!(kind, SemanticIntentKind::NormalChat) && raw.tool_name.is_some() {
        kind = SemanticIntentKind::ToolActionRequest;
    }
    let confidence = raw.confidence.unwrap_or(0.0).clamp(0.0, 1.0);
    let screen = raw.screen.unwrap_or(RawScreenRequest {
        capture_fresh: None,
        reuse_recent: None,
        state_question: None,
    });
    let params = raw.params.or(raw.parameters).unwrap_or_else(|| json!({}));

    Ok(SemanticIntent {
        kind,
        target,
        action,
        params,
        screen: SemanticScreenRequest {
            capture_fresh: screen.capture_fresh,
            reuse_recent: screen.reuse_recent,
            state_question: screen.state_question,
        },
        confidence,
        language: raw.language.map(|value| value.trim().to_ascii_lowercase()),
        rationale: raw
            .rationale
            .map(|value| value.trim().chars().take(180).collect::<String>())
            .filter(|value| !value.is_empty()),
    })
}

fn extract_json_object(content: &str) -> Option<&str> {
    let start = content.find('{')?;
    let end = content.rfind('}')?;
    (end >= start).then(|| &content[start..=end])
}

fn parse_intent_kind(value: Option<&str>) -> SemanticIntentKind {
    match normalize(value).as_deref() {
        Some("capability_question") => SemanticIntentKind::CapabilityQuestion,
        Some("tool_action_request") => SemanticIntentKind::ToolActionRequest,
        Some("screen_question") => SemanticIntentKind::ScreenQuestion,
        Some("screen_analysis_request") => SemanticIntentKind::ScreenAnalysisRequest,
        Some("approval_status_question") => SemanticIntentKind::ApprovalStatusQuestion,
        Some("normal_chat") => SemanticIntentKind::NormalChat,
        _ => SemanticIntentKind::NormalChat,
    }
}

fn parse_target(value: Option<&str>) -> CapabilityTarget {
    match normalize(value).as_deref() {
        Some("screen") => CapabilityTarget::Screen,
        Some("browser") => CapabilityTarget::Browser,
        Some("terminal") => CapabilityTarget::Terminal,
        Some("filesystem_read") => CapabilityTarget::FilesystemRead,
        Some("filesystem_write") => CapabilityTarget::FilesystemWrite,
        Some("filesystem_search") => CapabilityTarget::FilesystemSearch,
        Some("desktop_launch") => CapabilityTarget::DesktopLaunch,
        Some("approval") => CapabilityTarget::Approval,
        Some("general") => CapabilityTarget::General,
        Some("unknown") => CapabilityTarget::Unknown,
        _ => CapabilityTarget::Unknown,
    }
}

fn parse_action(value: Option<&str>) -> Option<SemanticAction> {
    match normalize(value).as_deref() {
        Some("browser_search") => Some(SemanticAction::BrowserSearch),
        Some("browser_open_url") => Some(SemanticAction::BrowserOpenUrl),
        Some("filesystem_read") => Some(SemanticAction::FilesystemRead),
        Some("filesystem_search") => Some(SemanticAction::FilesystemSearch),
        Some("filesystem_write") => Some(SemanticAction::FilesystemWrite),
        Some("terminal_run") => Some(SemanticAction::TerminalRun),
        Some("desktop_launch_app") => Some(SemanticAction::DesktopLaunchApp),
        Some("none") => Some(SemanticAction::None),
        Some("unknown") => Some(SemanticAction::Unknown),
        _ => None,
    }
}

fn parse_action_from_tool_name(value: Option<&str>) -> Option<SemanticAction> {
    match normalize(value).as_deref() {
        Some("browser.search") | Some("browser_search") => Some(SemanticAction::BrowserSearch),
        Some("browser.open") | Some("browser_open") => Some(SemanticAction::BrowserOpenUrl),
        Some("filesystem.read_text") | Some("filesystem_read_text") => {
            Some(SemanticAction::FilesystemRead)
        }
        Some("filesystem.search") | Some("filesystem_search") => {
            Some(SemanticAction::FilesystemSearch)
        }
        Some("filesystem.write_text") | Some("filesystem_write_text") => {
            Some(SemanticAction::FilesystemWrite)
        }
        Some("terminal.run") | Some("terminal_run") => Some(SemanticAction::TerminalRun),
        Some("desktop.launch_app") | Some("desktop_launch_app") => {
            Some(SemanticAction::DesktopLaunchApp)
        }
        _ => None,
    }
}

fn normalize(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.to_ascii_lowercase().replace('-', "_"))
}

async fn fetch_installed_models(client: &Client) -> Result<Vec<String>, String> {
    let response = client
        .get(format!("{OLLAMA_BASE_URL}/api/tags"))
        .send()
        .await
        .map_err(|error| format!("Ollama tags request failed: {error}"))?;

    if !response.status().is_success() {
        return Err(format!("Ollama tags HTTP error: {}", response.status()));
    }

    let body: Value = response
        .json()
        .await
        .map_err(|error| format!("Ollama tags parse failed: {error}"))?;

    Ok(body
        .get("models")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.get("name").and_then(Value::as_str))
        .map(ToOwned::to_owned)
        .collect())
}

fn select_intent_model(installed_models: &[String]) -> String {
    let candidates = env::var("ASTRA_MODEL_INTENT_CANDIDATES")
        .unwrap_or_else(|_| "qwen3:8b,llama3.1:8b,gpt-oss:20b,qwen3:14b".to_string())
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();

    select_first_available(&candidates, installed_models)
        .or_else(|| candidates.first().cloned())
        .unwrap_or_else(|| "qwen3:8b".to_string())
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
