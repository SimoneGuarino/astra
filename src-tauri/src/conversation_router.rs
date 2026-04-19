use serde_json::json;

use crate::{
    desktop_agent::DesktopAgentRuntime,
    desktop_agent_types::{CapabilityManifest, DesktopActionRequest, DesktopActionResponse, ScreenAnalysisRequest},
};

pub enum ConversationRoute {
    DirectResponse(String),
    ActionResponse(DesktopActionResponse),
    ScreenAnalysis(ScreenAnalysisResultEnvelope),
    Continue,
}

pub struct ScreenAnalysisResultEnvelope {
    pub answer: String,
    pub model: String,
    pub image_path: String,
}

pub async fn route_message(
    runtime: &DesktopAgentRuntime,
    manifest: &CapabilityManifest,
    message: &str,
) -> Result<ConversationRoute, String> {
    let normalized = message.trim();
    let lower = normalized.to_lowercase();

    if let Some(response) = capability_question_response(manifest, &lower) {
        return Ok(ConversationRoute::DirectResponse(response));
    }

    if is_pending_approval_question(&lower) {
        let approvals = runtime.pending_approvals();
        let response = if approvals.is_empty() {
            "There are no pending approvals right now.".to_string()
        } else {
            let names = approvals.iter().map(|approval| approval.tool_name.clone()).collect::<Vec<_>>().join(", ");
            format!("There are {} pending approval(s): {}.", approvals.len(), names)
        };
        return Ok(ConversationRoute::DirectResponse(response));
    }

    if is_screen_analysis_request(&lower) {
        if !manifest.screen.observation_supported {
            return Ok(ConversationRoute::DirectResponse("Screen observation is not available on this system yet.".into()));
        }
        if !manifest.screen.observation_enabled {
            return Ok(ConversationRoute::DirectResponse("I can analyze the screen, but screen observation is currently disabled. Enable observation first and I can capture the current screen for you.".into()));
        }
        if !manifest.screen.analysis_available {
            return Ok(ConversationRoute::DirectResponse("I can capture the screen, but no local vision model is currently available for screen analysis.".into()));
        }

        let capture_fresh = lower.contains("right now") || lower.contains("adesso") || lower.contains("current screen") || lower.contains("questa schermata") || !manifest.screen.recent_capture_available;
        let result = runtime.analyze_screen(ScreenAnalysisRequest { question: Some(normalized.to_string()), capture_fresh }).await?;
        return Ok(ConversationRoute::ScreenAnalysis(ScreenAnalysisResultEnvelope { answer: result.answer, model: result.model, image_path: result.image_path }));
    }

    if let Some(action) = infer_action_request(&lower, normalized) {
        let result = runtime.submit_action(uuid::Uuid::new_v4().to_string(), action)?;
        return Ok(ConversationRoute::ActionResponse(result));
    }

    Ok(ConversationRoute::Continue)
}

fn capability_question_response(manifest: &CapabilityManifest, lower: &str) -> Option<String> {
    if asks_about_screen(lower) {
        return Some(if manifest.screen.analysis_available && manifest.screen.observation_enabled {
            if manifest.screen.recent_capture_available {
                "Yes. I can capture and analyze the current screen, and I already have a recent screen capture available if you want me to use it.".into()
            } else {
                "Yes. I can capture and analyze the current screen when observation is enabled. There is no recent capture yet, so I would take a fresh screenshot first.".into()
            }
        } else if manifest.screen.observation_supported && !manifest.screen.observation_enabled {
            "I can analyze the screen, but screen observation is currently disabled. If you enable it, I can capture and analyze the current screen.".into()
        } else if manifest.screen.capture_available && !manifest.screen.analysis_available {
            "I can capture the screen, but screen analysis is not currently available because no compatible local vision model is ready.".into()
        } else {
            "Screen observation is not currently available in this runtime.".into()
        });
    }
    if asks_about_browser(lower) { return Some(tool_capability_response("browser actions", &manifest.browser_open, &manifest.browser_search)); }
    if asks_about_terminal(lower) { return Some(single_tool_response("terminal commands", &manifest.terminal)); }
    if asks_about_file_read(lower) { return Some(single_tool_response("file reading", &manifest.filesystem_read)); }
    if asks_about_file_write(lower) { return Some(single_tool_response("file writing", &manifest.filesystem_write)); }
    if asks_about_file_search(lower) { return Some(single_tool_response("file search", &manifest.filesystem_search)); }
    if asks_about_desktop_control(lower) { return Some(single_tool_response("desktop application launch", &manifest.desktop_launch)); }
    None
}

fn infer_action_request(lower: &str, original: &str) -> Option<DesktopActionRequest> {
    if lower.starts_with("search the web for ") || lower.starts_with("cerca sul web ") {
        let query = original.split_once(' ').map(|(_, rest)| rest).unwrap_or(original).trim();
        return Some(DesktopActionRequest { tool_name: "browser.search".into(), params: json!({"query": query}), preview_only: false, reason: Some("User requested a web search".into()) });
    }
    if lower.starts_with("open browser") || lower.starts_with("apri il browser") {
        return Some(DesktopActionRequest { tool_name: "desktop.launch_app".into(), params: json!({"path": browser_executable()}), preview_only: false, reason: Some("User requested opening the browser".into()) });
    }
    if lower.starts_with("open ") && (lower.contains("http://") || lower.contains("https://")) {
        let url = original.trim_start_matches("open ").trim();
        return Some(DesktopActionRequest { tool_name: "browser.open".into(), params: json!({"url": url}), preview_only: false, reason: Some("User requested opening a URL".into()) });
    }
    if lower.starts_with("read file ") || lower.starts_with("leggi il file ") {
        let path = original.split_once(' ').map(|(_, rest)| rest).unwrap_or(original).trim();
        return Some(DesktopActionRequest { tool_name: "filesystem.read_text".into(), params: json!({"path": path}), preview_only: false, reason: Some("User requested reading a file".into()) });
    }
    if lower.starts_with("search files for ") || lower.starts_with("cerca file ") {
        let pattern = original.split_once(' ').map(|(_, rest)| rest).unwrap_or(original).trim();
        return Some(DesktopActionRequest { tool_name: "filesystem.search".into(), params: json!({"pattern": pattern, "max_results": 25}), preview_only: false, reason: Some("User requested searching files".into()) });
    }
    if lower.starts_with("run terminal command ") || lower.starts_with("esegui comando ") {
        let command = original.split_once(' ').map(|(_, rest)| rest).unwrap_or(original).trim();
        let mut parts = command.split_whitespace();
        let cmd = parts.next()?;
        let args = parts.map(|value| value.to_string()).collect::<Vec<_>>();
        return Some(DesktopActionRequest { tool_name: "terminal.run".into(), params: json!({"command": cmd, "args": args}), preview_only: false, reason: Some("User requested running a terminal command".into()) });
    }
    None
}

fn tool_capability_response(name: &str, primary: &crate::desktop_agent_types::CapabilityToolAvailability, secondary: &crate::desktop_agent_types::CapabilityToolAvailability) -> String {
    if primary.available || secondary.available {
        if primary.enabled || secondary.enabled {
            if primary.requires_approval || secondary.requires_approval {
                format!("Yes. {name} are available, but some actions may require approval before execution.")
            } else {
                format!("Yes. {name} are available and ready to use.")
            }
        } else {
            format!("{name} exist in this runtime, but they are currently disabled by policy or permissions.")
        }
    } else { format!("{name} are not currently available in this runtime.") }
}
fn single_tool_response(name: &str, availability: &crate::desktop_agent_types::CapabilityToolAvailability) -> String {
    if !availability.available { format!("{name} are not currently available in this runtime.") }
    else if !availability.enabled { format!("{name} are available in Astra, but they are currently disabled by policy or permissions.") }
    else if availability.requires_approval { format!("{name} are available, but they may require your approval before execution.") }
    else { format!("Yes. {name} are available and ready to use.") }
}
fn asks_about_screen(lower: &str) -> bool { lower.contains("see the screen") || lower.contains("can you see the screen") || lower.contains("vedere lo schermo") || lower.contains("what am i looking at") || lower.contains("cosa sto vedendo") || lower.contains("help me on this screen") || lower.contains("questa schermata") || lower.contains("what's wrong here") || lower.contains("what is wrong here") }
fn asks_about_browser(lower: &str) -> bool { lower.contains("open the browser") || lower.contains("use the browser") || lower.contains("search the web") || lower.contains("browser") || lower.contains("navigare") }
fn asks_about_terminal(lower: &str) -> bool { lower.contains("terminal") || lower.contains("shell") || lower.contains("command line") || lower.contains("run commands") }
fn asks_about_file_read(lower: &str) -> bool { lower.contains("read file") || lower.contains("read files") || lower.contains("leggere file") }
fn asks_about_file_write(lower: &str) -> bool { lower.contains("write file") || lower.contains("write files") || lower.contains("modify file") || lower.contains("scrivere file") || lower.contains("modificare file") }
fn asks_about_file_search(lower: &str) -> bool { lower.contains("search files") || lower.contains("find files") || lower.contains("cercare file") }
fn asks_about_desktop_control(lower: &str) -> bool { lower.contains("open applications") || lower.contains("launch app") || lower.contains("aprire programmi") || lower.contains("desktop control") }
fn is_pending_approval_question(lower: &str) -> bool { lower.contains("pending approval") || lower.contains("pending approvals") || lower.contains("need my approval") || lower.contains("approvazioni") || lower.contains("approval for that") }
fn is_screen_analysis_request(lower: &str) -> bool { lower.contains("what am i looking at") || lower.contains("cosa sto vedendo") || lower.contains("what's wrong here") || lower.contains("what is wrong here") || lower.contains("what should i click") || lower.contains("what does this error mean") || lower.contains("help me on this screen") || lower.contains("analyze the screen") || lower.contains("analizza lo schermo") }
fn browser_executable() -> &'static str {
    if cfg!(target_os = "windows") {
        r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    } else {
        "xdg-open"
    }
}