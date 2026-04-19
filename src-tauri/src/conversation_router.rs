use serde_json::{json, Value};

use crate::{
    assistant_context::capability_label,
    desktop_agent::DesktopAgentRuntime,
    desktop_agent_types::{
        CapabilityManifest, CapabilityToolAvailability, ConversationRouteDiagnostic,
        DesktopActionRequest, DesktopActionResponse, DesktopActionStatus, ScreenAnalysisRequest,
    },
    semantic_intent::{
        classify_intent, CapabilityTarget, SemanticAction, SemanticIntent, SemanticIntentKind,
        SemanticScreenRequest,
    },
};

const MIN_CLASSIFIER_CONFIDENCE: f32 = 0.45;

pub enum ConversationRoute {
    DirectResponse(String),
    ActionResponse(DesktopActionResponse),
    ScreenAnalysis(ScreenAnalysisResultEnvelope),
    Continue,
}

pub struct ConversationRouteResult {
    pub route: ConversationRoute,
    pub diagnostic: ConversationRouteDiagnostic,
}

pub struct ScreenAnalysisResultEnvelope {
    pub response_text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResponseLanguage {
    English,
    Italian,
}

pub async fn route_message(
    runtime: &DesktopAgentRuntime,
    manifest: &CapabilityManifest,
    message: &str,
) -> Result<ConversationRouteResult, String> {
    let normalized = message.trim();
    let lower = normalized.to_lowercase();
    let classifier = classify_intent(normalized, manifest).await;

    let (intent, source, fallback_used, classifier_error) = match classifier {
        Ok(intent) if intent.confidence >= MIN_CLASSIFIER_CONFIDENCE => {
            (intent, "semantic_classifier".to_string(), false, None)
        }
        Ok(intent) => {
            if let Some(fallback) = fallback_intent(&lower, normalized) {
                (
                    fallback,
                    "heuristic_fallback_low_confidence".to_string(),
                    true,
                    Some(format!(
                        "classifier confidence {:.2} below threshold for {}",
                        intent.confidence,
                        intent.kind.as_str()
                    )),
                )
            } else {
                let mut guarded = SemanticIntent {
                    kind: SemanticIntentKind::NormalChat,
                    target: CapabilityTarget::Unknown,
                    action: SemanticAction::None,
                    params: json!({}),
                    screen: empty_screen_request(),
                    confidence: intent.confidence,
                    language: intent.language,
                    rationale: intent.rationale,
                };
                let source = if looks_like_governed_action_request(&lower) {
                    guarded.kind = SemanticIntentKind::ToolActionRequest;
                    guarded.action = SemanticAction::Unknown;
                    guarded.rationale = Some(
                        "classifier confidence was too low for a governed action; asking for grounded clarification".into(),
                    );
                    "heuristic_guard_low_confidence".to_string()
                } else {
                    "semantic_classifier_low_confidence".to_string()
                };
                let fallback_used = source.starts_with("heuristic_guard");
                (guarded, source, fallback_used, None)
            }
        }
        Err(error) => {
            if let Some(fallback) = fallback_intent(&lower, normalized) {
                (
                    fallback,
                    "heuristic_fallback_classifier_error".to_string(),
                    true,
                    Some(error),
                )
            } else if looks_like_governed_action_request(&lower) {
                (
                    SemanticIntent {
                        kind: SemanticIntentKind::ToolActionRequest,
                        target: CapabilityTarget::Unknown,
                        action: SemanticAction::Unknown,
                        params: json!({}),
                        screen: empty_screen_request(),
                        confidence: 0.35,
                        language: None,
                        rationale: Some(
                            "classifier unavailable; guarded action clarification instead of normal chat".into(),
                        ),
                    },
                    "heuristic_guard_classifier_error".to_string(),
                    true,
                    Some(error),
                )
            } else {
                (
                    SemanticIntent {
                        kind: SemanticIntentKind::NormalChat,
                        target: CapabilityTarget::Unknown,
                        action: SemanticAction::None,
                        params: json!({}),
                        screen: empty_screen_request(),
                        confidence: 0.0,
                        language: None,
                        rationale: Some(
                            "classifier unavailable and no safety fallback matched".into(),
                        ),
                    },
                    "classifier_unavailable".to_string(),
                    false,
                    Some(error),
                )
            }
        }
    };

    let language = response_language(&intent, normalized);
    let mut diagnostic = ConversationRouteDiagnostic {
        message_excerpt: normalized.chars().take(160).collect(),
        classifier_source: source,
        intent: intent.kind.as_str().into(),
        target: Some(intent.target.as_str().into()),
        action: Some(intent.action.as_str().into()),
        tool_name: None,
        extracted_params: Some(intent.params.clone()),
        confidence: Some(intent.confidence),
        routed_to: "normal_llm".into(),
        grounded: false,
        fallback_used,
        submit_action_called: false,
        action_id: None,
        action_status: None,
        approval_created: false,
        audit_expected: false,
        rationale: intent.rationale.clone(),
        error: classifier_error,
    };

    let route = route_intent(
        runtime,
        manifest,
        normalized,
        &lower,
        &intent,
        language,
        &mut diagnostic,
    )
    .await;
    Ok(ConversationRouteResult { route, diagnostic })
}

async fn route_intent(
    runtime: &DesktopAgentRuntime,
    manifest: &CapabilityManifest,
    normalized: &str,
    lower: &str,
    intent: &SemanticIntent,
    language: ResponseLanguage,
    diagnostic: &mut ConversationRouteDiagnostic,
) -> ConversationRoute {
    match intent.kind {
        SemanticIntentKind::CapabilityQuestion => {
            diagnostic.routed_to = "grounded_capability_response".into();
            diagnostic.grounded = true;
            ConversationRoute::DirectResponse(capability_response(
                manifest,
                &intent.target,
                language,
            ))
        }
        SemanticIntentKind::ScreenQuestion => {
            diagnostic.routed_to = "grounded_screen_state_response".into();
            diagnostic.grounded = true;
            ConversationRoute::DirectResponse(screen_state_response(manifest, language))
        }
        SemanticIntentKind::ApprovalStatusQuestion => {
            diagnostic.routed_to = "grounded_approval_response".into();
            diagnostic.grounded = true;
            ConversationRoute::DirectResponse(approval_status_response(runtime, manifest, language))
        }
        SemanticIntentKind::ScreenAnalysisRequest => {
            route_screen_analysis(
                runtime, manifest, normalized, lower, intent, language, diagnostic,
            )
            .await
        }
        SemanticIntentKind::ToolActionRequest => {
            route_tool_action(runtime, manifest, normalized, intent, language, diagnostic)
        }
        SemanticIntentKind::NormalChat => ConversationRoute::Continue,
    }
}

async fn route_screen_analysis(
    runtime: &DesktopAgentRuntime,
    manifest: &CapabilityManifest,
    normalized: &str,
    lower: &str,
    intent: &SemanticIntent,
    language: ResponseLanguage,
    diagnostic: &mut ConversationRouteDiagnostic,
) -> ConversationRoute {
    diagnostic.routed_to = "screen_analysis_governance".into();
    diagnostic.grounded = true;

    if intent.screen.state_question == Some(true) {
        diagnostic.routed_to = "grounded_screen_state_response".into();
        return ConversationRoute::DirectResponse(screen_state_response(manifest, language));
    }

    if !manifest.screen.observation_supported {
        return ConversationRoute::DirectResponse(localized(
            language,
            "Screen observation is not available in this runtime yet.",
            "L'osservazione dello schermo non e' disponibile in questo runtime.",
        ));
    }
    if !manifest.screen.analysis_available {
        return ConversationRoute::DirectResponse(localized(
            language,
            "I can capture the screen, but no compatible local vision model is currently available for analysis.",
            "Posso acquisire lo schermo, ma al momento non c'e' un modello vision locale compatibile per analizzarlo.",
        ));
    }

    let capture_fresh = should_capture_fresh(manifest, lower, &intent.screen);
    if capture_fresh && !manifest.screen.observation_enabled {
        return ConversationRoute::DirectResponse(if manifest.screen.recent_capture_available {
            localized(
                language,
                "I can analyze the screen, but fresh capture is disabled because screen observation is off. Enable observation for a live capture, or ask me to reuse the recent capture.",
                "Posso analizzare lo schermo, ma una nuova acquisizione e' bloccata perche' l'osservazione e' disattivata. Abilitala per una cattura live, oppure chiedimi di riusare la cattura recente.",
            )
        } else {
            localized(
                language,
                "I can analyze the screen, but screen observation is currently disabled. Enable observation first so I can capture the current screen.",
                "Posso analizzare lo schermo, ma l'osservazione e' disattivata. Abilitala prima, cosi' posso acquisire la schermata attuale.",
            )
        });
    }

    match runtime
        .analyze_screen(ScreenAnalysisRequest {
            question: Some(normalized.to_string()),
            capture_fresh,
        })
        .await
    {
        Ok(result) => {
            diagnostic.routed_to = if capture_fresh {
                "screen_analysis_fresh_capture".into()
            } else {
                "screen_analysis_recent_capture".into()
            };
            ConversationRoute::ScreenAnalysis(ScreenAnalysisResultEnvelope {
                response_text: localized_owned(
                    language,
                    format!(
                        "I analyzed the current screen using {}. {}",
                        result.model, result.answer
                    ),
                    format!(
                        "Ho analizzato lo schermo con {}. {}",
                        result.model, result.answer
                    ),
                ),
            })
        }
        Err(error) => {
            diagnostic.error = Some(error.clone());
            ConversationRoute::DirectResponse(localized_owned(
                language,
                format!("I could not analyze the screen: {error}"),
                format!("Non sono riuscita ad analizzare lo schermo: {error}"),
            ))
        }
    }
}

fn route_tool_action(
    runtime: &DesktopAgentRuntime,
    manifest: &CapabilityManifest,
    normalized: &str,
    intent: &SemanticIntent,
    language: ResponseLanguage,
    diagnostic: &mut ConversationRouteDiagnostic,
) -> ConversationRoute {
    diagnostic.routed_to = "desktop_action_governance".into();
    diagnostic.grounded = true;

    let action = match build_action_request(intent, normalized) {
        Ok(action) => action,
        Err(message) => {
            diagnostic.routed_to = "desktop_action_missing_parameters".into();
            return ConversationRoute::DirectResponse(localized_owned(
                language,
                message.clone(),
                translate_missing_parameter(&message),
            ));
        }
    };
    diagnostic.tool_name = Some(action.tool_name.clone());
    diagnostic.extracted_params = Some(action.params.clone());

    if let Some(availability) = availability_for_tool(manifest, &action.tool_name) {
        if !availability.available || !availability.enabled {
            diagnostic.routed_to = "desktop_action_blocked_by_capability_state".into();
            return ConversationRoute::DirectResponse(tool_blocked_response(
                &action.tool_name,
                availability,
                language,
            ));
        }
    }

    diagnostic.submit_action_called = true;
    match runtime.submit_action(uuid::Uuid::new_v4().to_string(), action) {
        Ok(response) => {
            diagnostic.routed_to = "desktop_action_submitted".into();
            diagnostic.action_id = Some(response.action_id.clone());
            diagnostic.action_status = Some(action_status_label(&response.status).into());
            diagnostic.approval_created =
                matches!(response.status, DesktopActionStatus::ApprovalRequired);
            diagnostic.audit_expected = matches!(
                response.status,
                DesktopActionStatus::Executed | DesktopActionStatus::ApprovalRequired
            );
            ConversationRoute::ActionResponse(response)
        }
        Err(error) => {
            diagnostic.error = Some(error.clone());
            diagnostic.routed_to = "desktop_action_rejected_by_runtime".into();
            ConversationRoute::DirectResponse(localized_owned(
                language,
                format!("I could not run that action: {error}"),
                format!("Non posso eseguire quell'azione: {error}"),
            ))
        }
    }
}

fn action_status_label(status: &DesktopActionStatus) -> &'static str {
    match status {
        DesktopActionStatus::Executed => "executed",
        DesktopActionStatus::ApprovalRequired => "approval_required",
        DesktopActionStatus::Rejected => "rejected",
        DesktopActionStatus::Failed => "failed",
    }
}

fn build_action_request(
    intent: &SemanticIntent,
    original: &str,
) -> Result<DesktopActionRequest, String> {
    let params = effective_params(&intent.params);
    match intent.action {
        SemanticAction::BrowserSearch => {
            let query = param_str(&params, "query")
                .or_else(|| strip_any_prefix(original, &["search the web for", "search web for", "cerca sul web", "cerca online"]))
                .unwrap_or_else(|| original.to_string());
            Ok(DesktopActionRequest {
                tool_name: "browser.search".into(),
                params: json!({"query": query}),
                preview_only: false,
                reason: Some("User requested a web search".into()),
            })
        }
        SemanticAction::BrowserOpenUrl => {
            let url = param_str(&params, "url")
                .ok_or_else(|| "I need the URL to open.".to_string())?;
            if !is_http_url(&url) {
                return Err("I can open web URLs when they include http:// or https://.".into());
            }
            Ok(DesktopActionRequest {
                tool_name: "browser.open".into(),
                params: json!({"url": url}),
                preview_only: false,
                reason: Some("User requested opening a URL".into()),
            })
        }
        SemanticAction::FilesystemRead => {
            let path = param_str(&params, "path")
                .ok_or_else(|| "I need the file path to read.".to_string())?;
            Ok(DesktopActionRequest {
                tool_name: "filesystem.read_text".into(),
                params: json!({"path": path}),
                preview_only: false,
                reason: Some("User requested reading a file".into()),
            })
        }
        SemanticAction::FilesystemSearch => {
            let pattern = param_str(&params, "pattern")
                .or_else(|| param_str(&params, "query"))
                .ok_or_else(|| "I need the filename or pattern to search for.".to_string())?;
            let mut params = json!({"pattern": pattern, "max_results": 25});
            if let Some(root) = param_str(&params, "root") {
                params["root"] = json!(root);
            }
            Ok(DesktopActionRequest {
                tool_name: "filesystem.search".into(),
                params,
                preview_only: false,
                reason: Some("User requested searching files".into()),
            })
        }
        SemanticAction::FilesystemWrite => {
            let path = resolve_file_write_path(&params, original)
                .ok_or_else(|| "I need the target file path before writing.".to_string())?;
            let inferred_content = infer_file_content_from_text(original);
            let create_empty = is_create_empty_file_request(&params, original);
            let content = param_str(&params, "content")
                .or(inferred_content)
                .or_else(|| create_empty.then(String::new))
                .ok_or_else(|| "I need the content to write before creating or modifying a file.".to_string())?;
            let mode = param_str(&params, "mode").unwrap_or_else(|| "overwrite".into());
            Ok(DesktopActionRequest {
                tool_name: "filesystem.write_text".into(),
                params: json!({"path": path, "content": content, "mode": mode, "create_empty": create_empty}),
                preview_only: false,
                reason: Some(if create_empty {
                    "User requested creating an empty file".into()
                } else {
                    "User requested writing a file".into()
                }),
            })
        }
        SemanticAction::TerminalRun => {
            let command = param_str(&params, "command")
                .ok_or_else(|| "I need the terminal command to run.".to_string())?;
            let args = param_string_array_any(&params, &["args", "arguments"]);
            let mut params = json!({"command": command, "args": args});
            if let Some(cwd) = param_str(&params, "cwd") {
                params["cwd"] = json!(cwd);
            }
            Ok(DesktopActionRequest {
                tool_name: "terminal.run".into(),
                params,
                preview_only: false,
                reason: Some("User requested running a terminal command".into()),
            })
        }
        SemanticAction::DesktopLaunchApp => {
            let app_or_path = param_str_any(&params, &["path", "app", "app_name", "application", "name"])
                .ok_or_else(|| "I need the app name or executable path to launch.".to_string())?;
            let path = resolve_app_alias(&app_or_path).unwrap_or(app_or_path);
            let mut args = param_string_array_any(&params, &["args", "arguments"]);
            if args.is_empty() && wants_new_tab(original) {
                args.push("--new-tab".into());
            }
            Ok(DesktopActionRequest {
                tool_name: "desktop.launch_app".into(),
                params: json!({"path": path, "args": args}),
                preview_only: false,
                reason: Some("User requested launching a desktop app".into()),
            })
        }
        SemanticAction::None | SemanticAction::Unknown => Err(
            "I understood this as an action request, but I need a clearer target before running anything."
                .into(),
        ),
    }
}

fn capability_response(
    manifest: &CapabilityManifest,
    target: &CapabilityTarget,
    language: ResponseLanguage,
) -> String {
    match target {
        CapabilityTarget::Screen => screen_state_response(manifest, language),
        CapabilityTarget::Browser => tool_pair_response(
            "browser actions",
            "azioni browser",
            &manifest.browser_open,
            &manifest.browser_search,
            language,
        ),
        CapabilityTarget::Terminal => single_tool_response(
            "terminal commands",
            "comandi terminale",
            &manifest.terminal,
            language,
        ),
        CapabilityTarget::FilesystemRead => single_tool_response(
            "file reading",
            "lettura dei file",
            &manifest.filesystem_read,
            language,
        ),
        CapabilityTarget::FilesystemWrite => single_tool_response(
            "file writing",
            "scrittura dei file",
            &manifest.filesystem_write,
            language,
        ),
        CapabilityTarget::FilesystemSearch => single_tool_response(
            "file search",
            "ricerca nei file",
            &manifest.filesystem_search,
            language,
        ),
        CapabilityTarget::DesktopLaunch => single_tool_response(
            "desktop app launch",
            "apertura di applicazioni desktop",
            &manifest.desktop_launch,
            language,
        ),
        CapabilityTarget::Approval => approval_policy_response(manifest, language),
        CapabilityTarget::General | CapabilityTarget::Unknown => {
            general_capability_response(manifest, language)
        }
    }
}

fn screen_state_response(manifest: &CapabilityManifest, language: ResponseLanguage) -> String {
    if !manifest.screen.observation_supported {
        return localized(
            language,
            "Screen observation is not available in this runtime.",
            "L'osservazione dello schermo non e' disponibile in questo runtime.",
        );
    }

    if manifest.screen.observation_enabled && manifest.screen.analysis_available {
        if manifest.screen.recent_capture_available {
            let age = manifest
                .screen
                .recent_capture_age_ms
                .map(format_age)
                .unwrap_or_else(|| "recently".into());
            localized_owned(
                language,
                format!(
                    "Yes. Screen observation is enabled, I can capture a fresh screen, and I can also reuse the recent capture from {age}."
                ),
                format!(
                    "Si'. L'osservazione dello schermo e' attiva, posso acquisire una schermata nuova e posso anche riusare la cattura recente di {age}."
                ),
            )
        } else {
            localized(
                language,
                "Yes. Screen observation is enabled and a local vision model is available. I would take a fresh capture first because there is no recent capture yet.",
                "Si'. L'osservazione dello schermo e' attiva e un modello vision locale e' disponibile. Farei prima una nuova cattura perche' non c'e' ancora una cattura recente.",
            )
        }
    } else if !manifest.screen.observation_enabled && manifest.screen.analysis_available {
        if manifest.screen.recent_capture_available {
            localized(
                language,
                "I can analyze the screen, but live observation is currently disabled. I can reuse the recent capture, or you can enable observation for a fresh current capture.",
                "Posso analizzare lo schermo, ma l'osservazione live e' disattivata. Posso riusare la cattura recente, oppure puoi abilitarla per una nuova cattura attuale.",
            )
        } else {
            localized(
                language,
                "I can analyze the screen, but screen observation is currently disabled. Enable observation first so I can capture the current screen.",
                "Posso analizzare lo schermo, ma l'osservazione e' disattivata. Abilitala prima, cosi' posso acquisire la schermata attuale.",
            )
        }
    } else if manifest.screen.capture_available {
        localized(
            language,
            "I can capture the screen, but screen analysis is not ready because no compatible local vision model is available.",
            "Posso acquisire lo schermo, ma l'analisi non e' pronta perche' non c'e' un modello vision locale compatibile.",
        )
    } else {
        localized(
            language,
            "Screen capture is not currently available in this runtime.",
            "La cattura dello schermo non e' disponibile in questo runtime.",
        )
    }
}

fn approval_status_response(
    runtime: &DesktopAgentRuntime,
    manifest: &CapabilityManifest,
    language: ResponseLanguage,
) -> String {
    let approvals = runtime.pending_approvals();
    if approvals.is_empty() {
        return localized_owned(
            language,
            format!(
                "There are no pending approvals right now. High-risk actions {} require approval.",
                if manifest.approvals.approval_required_for_high_risk {
                    "do"
                } else {
                    "do not"
                }
            ),
            format!(
                "Non ci sono approvazioni in sospeso. Le azioni ad alto rischio {} approvazione.",
                if manifest.approvals.approval_required_for_high_risk {
                    "richiedono"
                } else {
                    "non richiedono"
                }
            ),
        );
    }

    let summary = approvals
        .iter()
        .map(|approval| format!("{} ({:?})", approval.tool_name, approval.risk_level))
        .collect::<Vec<_>>()
        .join(", ");
    localized_owned(
        language,
        format!("There are {} pending approval(s): {}. Nothing in that list has executed yet.", approvals.len(), summary),
        format!("Ci sono {} approvazioni in sospeso: {}. Nulla in quella lista e' stato ancora eseguito.", approvals.len(), summary),
    )
}

fn approval_policy_response(manifest: &CapabilityManifest, language: ResponseLanguage) -> String {
    localized_owned(
        language,
        format!(
            "Astra uses Rust-side approval governance. Pending approvals: {}. High-risk actions {} approval before execution.",
            manifest.approvals.pending_count,
            if manifest.approvals.approval_required_for_high_risk { "require" } else { "do not require" }
        ),
        format!(
            "Astra usa una governance approvazioni lato Rust. Approvazioni in sospeso: {}. Le azioni ad alto rischio {} approvazione prima dell'esecuzione.",
            manifest.approvals.pending_count,
            if manifest.approvals.approval_required_for_high_risk { "richiedono" } else { "non richiedono" }
        ),
    )
}

fn general_capability_response(
    manifest: &CapabilityManifest,
    language: ResponseLanguage,
) -> String {
    let ready = manifest
        .tools
        .iter()
        .filter(|tool| tool.enabled)
        .map(|tool| tool.tool_name.clone())
        .collect::<Vec<_>>()
        .join(", ");
    localized_owned(
        language,
        format!(
            "I can operate as a governed local desktop agent. Ready tools: {}. Screen observation is {}. Pending approvals: {}.",
            if ready.is_empty() {
                "none".into()
            } else {
                ready.clone()
            },
            if manifest.screen.observation_enabled { "enabled" } else { "disabled" },
            manifest.approvals.pending_count,
        ),
        format!(
            "Posso operare come agente desktop locale governato. Strumenti pronti: {}. Osservazione schermo: {}. Approvazioni in sospeso: {}.",
            if ready.is_empty() { "nessuno".into() } else { ready },
            if manifest.screen.observation_enabled { "attiva" } else { "disattivata" },
            manifest.approvals.pending_count,
        ),
    )
}

fn tool_pair_response(
    english_name: &str,
    italian_name: &str,
    primary: &CapabilityToolAvailability,
    secondary: &CapabilityToolAvailability,
    language: ResponseLanguage,
) -> String {
    if primary.enabled || secondary.enabled {
        if primary.requires_approval || secondary.requires_approval {
            localized_owned(
                language,
                format!("Yes. {english_name} are available, and some actions may require approval before execution."),
                format!("Si'. Le {italian_name} sono disponibili, e alcune possono richiedere approvazione prima dell'esecuzione."),
            )
        } else {
            localized_owned(
                language,
                format!("Yes. {english_name} are available and ready to use."),
                format!("Si'. Le {italian_name} sono disponibili e pronte all'uso."),
            )
        }
    } else if primary.available || secondary.available {
        localized_owned(
            language,
            format!("{english_name} exist in this runtime, but they are currently disabled by policy or permissions."),
            format!("Le {italian_name} esistono in questo runtime, ma sono disattivate da policy o permessi."),
        )
    } else {
        localized_owned(
            language,
            format!("{english_name} are not currently available in this runtime."),
            format!("Le {italian_name} non sono disponibili in questo runtime."),
        )
    }
}

fn single_tool_response(
    english_name: &str,
    italian_name: &str,
    availability: &CapabilityToolAvailability,
    language: ResponseLanguage,
) -> String {
    match capability_label(availability) {
        "not available" => localized_owned(
            language,
            format!("{english_name} are not currently available in this runtime."),
            format!("{italian_name} non e' disponibile in questo runtime."),
        ),
        "available but disabled" => localized_owned(
            language,
            format!("{english_name} are available in Astra, but currently disabled by policy or permissions."),
            format!("{italian_name} e' disponibile in Astra, ma ora e' disattivata da policy o permessi."),
        ),
        "available and approval-gated" => localized_owned(
            language,
            format!("Yes. {english_name} are available, but may require your approval before execution."),
            format!("Si'. {italian_name} e' disponibile, ma puo' richiedere la tua approvazione prima dell'esecuzione."),
        ),
        _ => localized_owned(
            language,
            format!("Yes. {english_name} are available and ready to use."),
            format!("Si'. {italian_name} e' disponibile e pronta all'uso."),
        ),
    }
}

fn tool_blocked_response(
    tool_name: &str,
    availability: &CapabilityToolAvailability,
    language: ResponseLanguage,
) -> String {
    let reason = availability
        .disabled_reason
        .clone()
        .unwrap_or_else(|| "policy or permissions".into());
    localized_owned(
        language,
        format!("{tool_name} exists, but it is not enabled right now because of {reason}."),
        format!("{tool_name} esiste, ma ora non e' abilitato per: {reason}."),
    )
}

fn should_capture_fresh(
    manifest: &CapabilityManifest,
    lower: &str,
    screen: &SemanticScreenRequest,
) -> bool {
    if screen.reuse_recent == Some(true) && manifest.screen.recent_capture_available {
        return false;
    }
    if screen.capture_fresh == Some(true) {
        return true;
    }
    if screen.capture_fresh == Some(false) && manifest.screen.recent_capture_available {
        return false;
    }
    if !manifest.screen.recent_capture_available {
        return true;
    }

    let current_markers = [
        "right now",
        "current screen",
        "what am i looking at",
        "this screen",
        "here",
        "adesso",
        "ora",
        "schermo attuale",
        "questa schermata",
        "cosa sto vedendo",
        "qui",
    ];
    current_markers.iter().any(|marker| lower.contains(marker))
}

fn response_language(intent: &SemanticIntent, message: &str) -> ResponseLanguage {
    if let Some(language) = &intent.language {
        if language.starts_with("it") || language.contains("italian") {
            return ResponseLanguage::Italian;
        }
        if language.starts_with("en") || language.contains("english") {
            return ResponseLanguage::English;
        }
    }

    let lower = message.to_lowercase();
    let italian_markers = [
        "puoi",
        "schermo",
        "approv",
        "file",
        "browser",
        "terminale",
        "cerca",
        "leggi",
        "apri",
        "analizza",
        "vedendo",
        "adesso",
        "esegui",
    ];
    if italian_markers.iter().any(|marker| lower.contains(marker)) {
        ResponseLanguage::Italian
    } else {
        ResponseLanguage::English
    }
}

fn availability_for_tool<'a>(
    manifest: &'a CapabilityManifest,
    tool_name: &str,
) -> Option<&'a CapabilityToolAvailability> {
    match tool_name {
        "filesystem.read_text" => Some(&manifest.filesystem_read),
        "filesystem.write_text" => Some(&manifest.filesystem_write),
        "filesystem.search" => Some(&manifest.filesystem_search),
        "terminal.run" => Some(&manifest.terminal),
        "browser.open" => Some(&manifest.browser_open),
        "browser.search" => Some(&manifest.browser_search),
        "desktop.launch_app" => Some(&manifest.desktop_launch),
        _ => None,
    }
}

fn fallback_intent(lower: &str, original: &str) -> Option<SemanticIntent> {
    if is_pending_approval_question(lower) {
        return Some(intent(
            SemanticIntentKind::ApprovalStatusQuestion,
            CapabilityTarget::Approval,
            SemanticAction::None,
            json!({}),
            0.60,
            "approval fallback",
        ));
    }
    if is_screen_analysis_request(lower) {
        return Some(intent(
            SemanticIntentKind::ScreenAnalysisRequest,
            CapabilityTarget::Screen,
            SemanticAction::None,
            json!({}),
            0.62,
            "screen analysis fallback",
        ));
    }
    if asks_about_screen(lower) {
        return Some(intent(
            SemanticIntentKind::ScreenQuestion,
            CapabilityTarget::Screen,
            SemanticAction::None,
            json!({}),
            0.58,
            "screen capability fallback",
        ));
    }
    if let Some(params) = infer_file_write_params_from_text(original) {
        return Some(intent(
            SemanticIntentKind::ToolActionRequest,
            CapabilityTarget::FilesystemWrite,
            SemanticAction::FilesystemWrite,
            params,
            0.66,
            "file write action fallback",
        ));
    }
    if let Some(query) = strip_any_prefix(
        original,
        &[
            "search the web for",
            "search web for",
            "cerca sul web",
            "cerca online",
        ],
    ) {
        return Some(intent(
            SemanticIntentKind::ToolActionRequest,
            CapabilityTarget::Browser,
            SemanticAction::BrowserSearch,
            json!({"query": query}),
            0.58,
            "web search action fallback",
        ));
    }
    if looks_like_chrome_launch_request(lower) {
        return Some(intent(
            SemanticIntentKind::ToolActionRequest,
            CapabilityTarget::DesktopLaunch,
            SemanticAction::DesktopLaunchApp,
            json!({
                "app": "google chrome",
                "args": if wants_new_tab(original) { vec!["--new-tab"] } else { Vec::<&str>::new() },
            }),
            0.64,
            "chrome launch fallback",
        ));
    }
    if lower.starts_with("open browser") || lower.starts_with("apri il browser") {
        return Some(intent(
            SemanticIntentKind::ToolActionRequest,
            CapabilityTarget::DesktopLaunch,
            SemanticAction::DesktopLaunchApp,
            json!({
                "app": "browser",
                "args": if wants_new_tab(original) { vec!["--new-tab"] } else { Vec::<&str>::new() },
            }),
            0.58,
            "browser launch fallback",
        ));
    }
    if let Some(url) = strip_any_prefix(original, &["open", "apri"]) {
        if is_http_url(&url) {
            return Some(intent(
                SemanticIntentKind::ToolActionRequest,
                CapabilityTarget::Browser,
                SemanticAction::BrowserOpenUrl,
                json!({"url": url}),
                0.58,
                "open URL fallback",
            ));
        }
    }
    if asks_about_browser(lower) {
        return Some(intent(
            SemanticIntentKind::CapabilityQuestion,
            CapabilityTarget::Browser,
            SemanticAction::None,
            json!({}),
            0.55,
            "browser capability fallback",
        ));
    }
    if asks_about_terminal(lower) {
        return Some(intent(
            SemanticIntentKind::CapabilityQuestion,
            CapabilityTarget::Terminal,
            SemanticAction::None,
            json!({}),
            0.55,
            "terminal capability fallback",
        ));
    }
    if asks_about_file_read(lower) {
        return Some(intent(
            SemanticIntentKind::CapabilityQuestion,
            CapabilityTarget::FilesystemRead,
            SemanticAction::None,
            json!({}),
            0.55,
            "file read capability fallback",
        ));
    }
    if asks_about_file_write(lower) {
        return Some(intent(
            SemanticIntentKind::CapabilityQuestion,
            CapabilityTarget::FilesystemWrite,
            SemanticAction::None,
            json!({}),
            0.55,
            "file write capability fallback",
        ));
    }
    if asks_about_file_search(lower) {
        return Some(intent(
            SemanticIntentKind::CapabilityQuestion,
            CapabilityTarget::FilesystemSearch,
            SemanticAction::None,
            json!({}),
            0.55,
            "file search capability fallback",
        ));
    }
    if asks_about_desktop_control(lower) {
        return Some(intent(
            SemanticIntentKind::CapabilityQuestion,
            CapabilityTarget::DesktopLaunch,
            SemanticAction::None,
            json!({}),
            0.55,
            "desktop capability fallback",
        ));
    }

    None
}

fn intent(
    kind: SemanticIntentKind,
    target: CapabilityTarget,
    action: SemanticAction,
    params: Value,
    confidence: f32,
    rationale: &str,
) -> SemanticIntent {
    SemanticIntent {
        kind,
        target,
        action,
        params,
        screen: empty_screen_request(),
        confidence,
        language: None,
        rationale: Some(rationale.into()),
    }
}

fn empty_screen_request() -> SemanticScreenRequest {
    SemanticScreenRequest {
        capture_fresh: None,
        reuse_recent: None,
        state_question: None,
    }
}

fn effective_params(params: &Value) -> Value {
    params
        .get("parameters")
        .filter(|value| value.is_object())
        .cloned()
        .unwrap_or_else(|| params.clone())
}

fn param_str(params: &Value, key: &str) -> Option<String> {
    params
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn param_str_any(params: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| param_str(params, key))
}

fn param_string_array_any(params: &Value, keys: &[&str]) -> Vec<String> {
    keys.iter()
        .find_map(|key| {
            params.get(key).and_then(Value::as_array).map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect::<Vec<_>>()
            })
        })
        .unwrap_or_default()
}

fn resolve_file_write_path(params: &Value, original: &str) -> Option<String> {
    if let Some(path) = param_str(params, "path") {
        return Some(resolve_desktop_relative_path(&path, original));
    }

    let file_name = param_str_any(params, &["file_name", "filename", "name"])
        .or_else(|| infer_file_name_from_text(original))?;
    Some(resolve_desktop_relative_path(&file_name, original))
}

fn resolve_desktop_relative_path(path_or_name: &str, original: &str) -> String {
    let trimmed = path_or_name.trim().trim_matches('"').trim_matches('\'');
    if trimmed.contains('\\') || trimmed.contains('/') || has_windows_drive_prefix(trimmed) {
        return trimmed.to_string();
    }

    if mentions_desktop(original) {
        if let Some(desktop) = desktop_dir() {
            return desktop
                .join(ensure_txt_extension(trimmed))
                .display()
                .to_string();
        }
    }

    ensure_txt_extension(trimmed)
}

fn desktop_dir() -> Option<std::path::PathBuf> {
    std::env::var("USERPROFILE")
        .ok()
        .map(|home| std::path::PathBuf::from(home).join("Desktop"))
}

fn has_windows_drive_prefix(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() >= 3 && bytes[1] == b':' && (bytes[2] == b'\\' || bytes[2] == b'/')
}

fn ensure_txt_extension(value: &str) -> String {
    let clean = value.trim().trim_matches('.');
    if clean.to_ascii_lowercase().ends_with(".txt") {
        clean.to_string()
    } else {
        format!("{clean}.txt")
    }
}

fn is_create_empty_file_request(params: &Value, original: &str) -> bool {
    if params
        .get("create_empty")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return true;
    }

    param_str(params, "content").is_none()
        && infer_file_content_from_text(original).is_none()
        && looks_like_file_create_request(&original.to_lowercase())
}

fn infer_file_write_params_from_text(original: &str) -> Option<Value> {
    let lower = original.to_lowercase();
    if !looks_like_file_write_action_request(&lower) {
        return None;
    }

    let mut params = serde_json::Map::new();
    if let Some(file_name) = infer_file_name_from_text(original) {
        params.insert("file_name".into(), json!(file_name));
    }

    if let Some(content) = infer_file_content_from_text(original) {
        params.insert("content".into(), json!(content));
        params.insert("create_empty".into(), json!(false));
    } else {
        params.insert("create_empty".into(), json!(true));
    }

    Some(Value::Object(params))
}

fn infer_file_content_from_text(original: &str) -> Option<String> {
    let markers = [
        "contenente",
        "che contiene",
        "con scritto",
        "con dentro",
        "e scrivici",
        "scrivici",
        "containing",
        "with content",
        "with text",
        "that says",
        "saying",
        "and write",
        "write into it",
    ];

    markers.iter().find_map(|marker| {
        extract_after_marker(original, marker)
            .and_then(|tail| materialize_content_request(tail, original))
    })
}

fn extract_after_marker<'a>(value: &'a str, marker: &str) -> Option<&'a str> {
    let lower = value.to_ascii_lowercase();
    let marker = marker.to_ascii_lowercase();
    let start = lower.find(&marker)? + marker.len();
    value.get(start..)
}

fn materialize_content_request(raw: &str, original: &str) -> Option<String> {
    if let Some(quoted) = extract_quoted_content(raw) {
        return Some(quoted);
    }

    let description = clean_content_tail(raw);
    if description.is_empty() {
        return None;
    }

    let lower = description.to_lowercase();
    if lower.contains("frase d'amore")
        || lower.contains("frase d\u{2019}amore")
        || lower.contains("love sentence")
        || lower.contains("love phrase")
    {
        return Some(if looks_italian_request(original) {
            "Ti amo piu' di quanto le parole possano dire.".into()
        } else {
            "Love is the quiet light I carry with me.".into()
        });
    }

    strip_content_noun_prefix(&description).or(Some(description))
}

fn extract_quoted_content(value: &str) -> Option<String> {
    let trimmed = value.trim_start();
    for quote in ['"', '`'] {
        if !trimmed.starts_with(quote) {
            continue;
        }
        let rest = &trimmed[quote.len_utf8()..];
        if let Some(end) = rest.find(quote) {
            let content = rest[..end].trim();
            if !content.is_empty() {
                return Some(content.to_string());
            }
        }
    }
    None
}

fn clean_content_tail(value: &str) -> String {
    let mut text = value
        .trim()
        .trim_start_matches(|ch: char| matches!(ch, ':' | '-' | '>' | '"' | '\'' | '`'))
        .trim()
        .to_string();

    for suffix in [" per favore", " please"] {
        let lower = text.to_lowercase();
        if lower.ends_with(suffix) {
            let new_len = text.len().saturating_sub(suffix.len());
            text.truncate(new_len);
            text = text.trim().to_string();
        }
    }

    text.trim_end_matches(|ch: char| matches!(ch, '"' | '\'' | '`'))
        .trim()
        .to_string()
}

fn strip_content_noun_prefix(value: &str) -> Option<String> {
    let lower = value.to_lowercase();
    for prefix in ["la frase ", "il testo ", "the sentence ", "the text "] {
        if lower.starts_with(prefix) {
            let content = value[prefix.len()..].trim();
            if !content.is_empty() {
                return Some(content.to_string());
            }
        }
    }
    None
}

fn infer_file_name_from_text(original: &str) -> Option<String> {
    let normalized = original
        .replace(',', " ")
        .replace(':', " ")
        .replace(';', " ")
        .replace('"', " ")
        .replace('\'', " ");
    let tokens = normalized.split_whitespace().collect::<Vec<_>>();

    if let Some(token) = tokens.iter().find(|token| {
        let lower = token.to_ascii_lowercase();
        lower.ends_with(".txt") && lower != ".txt" && lower != "txt"
    }) {
        return Some(sanitize_file_name_token(token));
    }

    for marker in ["chiamato", "called", "named", "nome", "name"] {
        if let Some(index) = tokens
            .iter()
            .position(|token| token.eq_ignore_ascii_case(marker))
        {
            if let Some(next) = tokens.get(index + 1) {
                return Some(ensure_txt_extension(&sanitize_file_name_token(next)));
            }
        }
    }

    None
}

fn sanitize_file_name_token(token: &str) -> String {
    token
        .trim()
        .trim_matches(|ch: char| matches!(ch, '.' | ',' | ';' | ':' | '"' | '\''))
        .to_string()
}

fn looks_like_file_create_request(lower: &str) -> bool {
    let create = lower.contains("crea ")
        || lower.starts_with("crea")
        || lower.contains("crei ")
        || lower.contains("mi crei")
        || lower.contains("creami")
        || lower.contains("create")
        || lower.contains("make")
        || lower.contains("fammi");
    let file = lower.contains("file") || lower.contains(".txt") || lower.contains("testo");
    create && file
}

fn looks_like_file_write_action_request(lower: &str) -> bool {
    let file = lower.contains("file") || lower.contains(".txt") || lower.contains("testo");
    let write = lower.contains("scrivi")
        || lower.contains("scrivici")
        || lower.contains("write")
        || lower.contains("contenente")
        || lower.contains("containing");
    file && (looks_like_file_create_request(lower) || write)
}

fn looks_like_governed_action_request(lower: &str) -> bool {
    if looks_like_file_write_action_request(lower)
        || looks_like_chrome_launch_request(lower)
        || lower.contains("http://")
        || lower.contains("https://")
    {
        return true;
    }

    let action_verb = [
        "apri",
        "aprimi",
        "open",
        "launch",
        "avvia",
        "esegui",
        "run",
        "scrivi",
        "write",
        "cerca sul web",
        "search the web",
    ]
    .iter()
    .any(|marker| lower.contains(marker));
    let governed_target = [
        "file",
        ".txt",
        "browser",
        "chrome",
        "terminale",
        "terminal",
        "powershell",
        "cmd",
        "desktop",
        "cartella",
        "folder",
        "url",
    ]
    .iter()
    .any(|marker| lower.contains(marker));

    action_verb && governed_target
}

fn looks_italian_request(value: &str) -> bool {
    let lower = value.to_lowercase();
    [
        "crea ",
        "crei",
        "creami",
        "scrivi",
        "scrivici",
        "frase",
        "amore",
        "chiamato",
        "contenente",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn mentions_desktop(value: &str) -> bool {
    let lower = value.to_lowercase();
    lower.contains("desktop") || lower.contains("scrivania")
}

fn resolve_app_alias(app_or_path: &str) -> Option<String> {
    let normalized = app_or_path
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .to_ascii_lowercase()
        .replace('-', " ");

    let is_chrome = matches!(
        normalized.as_str(),
        "browser" | "chrome" | "google chrome" | "googlechrome"
    );
    is_chrome.then(|| browser_executable().to_string())
}

fn wants_new_tab(value: &str) -> bool {
    let lower = value.to_lowercase();
    lower.contains("new tab")
        || lower.contains("nuova scheda")
        || lower.contains("scheda nuova")
        || lower.contains("tab nuova")
        || lower.contains("--new-tab")
}

fn looks_like_chrome_launch_request(lower: &str) -> bool {
    let mentions_chrome = lower.contains("chrome") || lower.contains("google chrome");
    let launch_verb = lower.contains("apri")
        || lower.contains("aprimi")
        || lower.contains("open")
        || lower.contains("launch")
        || lower.contains("avvia");
    mentions_chrome && launch_verb
}

fn strip_any_prefix(value: &str, prefixes: &[&str]) -> Option<String> {
    let trimmed = value.trim();
    let lower = trimmed.to_lowercase();
    prefixes
        .iter()
        .find_map(|prefix| {
            lower.strip_prefix(prefix).map(|_| {
                trimmed
                    .chars()
                    .skip(prefix.chars().count())
                    .collect::<String>()
                    .trim()
                    .trim_matches(':')
                    .trim()
                    .to_string()
            })
        })
        .filter(|value| !value.is_empty())
}

fn is_http_url(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

fn format_age(age_ms: u64) -> String {
    if age_ms < 1_000 {
        "less than a second ago".into()
    } else if age_ms < 60_000 {
        format!("{} seconds ago", age_ms / 1_000)
    } else {
        format!("{} minutes ago", age_ms / 60_000)
    }
}

fn localized(language: ResponseLanguage, english: &str, italian: &str) -> String {
    match language {
        ResponseLanguage::English => english.into(),
        ResponseLanguage::Italian => italian.into(),
    }
}

fn localized_owned(language: ResponseLanguage, english: String, italian: String) -> String {
    match language {
        ResponseLanguage::English => english,
        ResponseLanguage::Italian => italian,
    }
}

fn translate_missing_parameter(message: &str) -> String {
    match message {
        "I need the URL to open." => "Mi serve l'URL da aprire.".into(),
        "I can open web URLs when they include http:// or https://." => {
            "Posso aprire URL web quando includono http:// o https://.".into()
        }
        "I need the file path to read." => "Mi serve il percorso del file da leggere.".into(),
        "I need the filename or pattern to search for." => {
            "Mi serve il nome o pattern del file da cercare.".into()
        }
        "I need the target file path before writing." => {
            "Mi serve il percorso del file di destinazione prima di scrivere.".into()
        }
        "I need the content to write before creating or modifying a file." => {
            "Mi serve il contenuto da scrivere prima di creare o modificare un file.".into()
        }
        "I need the terminal command to run." => "Mi serve il comando terminale da eseguire.".into(),
        "I need the app name or executable path to launch." => {
            "Mi serve il nome dell'app o il percorso dell'eseguibile da aprire.".into()
        }
        _ => "Ho capito che vuoi un'azione, ma mi serve un obiettivo piu' chiaro prima di eseguire qualcosa.".into(),
    }
}

fn asks_about_screen(lower: &str) -> bool {
    lower.contains("see the screen")
        || lower.contains("can you see the screen")
        || lower.contains("vedere lo schermo")
        || lower.contains("vedi lo schermo")
        || lower.contains("screen observation")
        || lower.contains("osservazione schermo")
}
fn asks_about_browser(lower: &str) -> bool {
    lower.contains("open the browser")
        || lower.contains("use the browser")
        || lower.contains("search the web")
        || lower.contains("browser")
        || lower.contains("navigare")
}
fn asks_about_terminal(lower: &str) -> bool {
    lower.contains("terminal")
        || lower.contains("shell")
        || lower.contains("command line")
        || lower.contains("run commands")
        || lower.contains("terminale")
}
fn asks_about_file_read(lower: &str) -> bool {
    lower.contains("read file") || lower.contains("read files") || lower.contains("leggere file")
}
fn asks_about_file_write(lower: &str) -> bool {
    lower.contains("write file")
        || lower.contains("write files")
        || lower.contains("modify file")
        || lower.contains("scrivere file")
        || lower.contains("modificare file")
}
fn asks_about_file_search(lower: &str) -> bool {
    lower.contains("search files") || lower.contains("find files") || lower.contains("cercare file")
}
fn asks_about_desktop_control(lower: &str) -> bool {
    lower.contains("open applications")
        || lower.contains("launch app")
        || lower.contains("aprire programmi")
        || lower.contains("desktop control")
}
fn is_pending_approval_question(lower: &str) -> bool {
    lower.contains("pending approval")
        || lower.contains("pending approvals")
        || lower.contains("need my approval")
        || lower.contains("approvazioni")
        || lower.contains("approval for that")
}
fn is_screen_analysis_request(lower: &str) -> bool {
    lower.contains("what am i looking at")
        || lower.contains("cosa sto vedendo")
        || lower.contains("what's wrong here")
        || lower.contains("what is wrong here")
        || lower.contains("what should i click")
        || lower.contains("what does this error mean")
        || lower.contains("help me on this screen")
        || lower.contains("analyze the screen")
        || lower.contains("analizza lo schermo")
}
fn browser_executable() -> &'static str {
    if cfg!(target_os = "windows") {
        let candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ];
        candidates
            .iter()
            .find(|candidate| std::path::Path::new(candidate).exists())
            .copied()
            .unwrap_or("chrome.exe")
    } else {
        "xdg-open"
    }
}

#[cfg(test)]
mod tests {
    use super::{build_action_request, fallback_intent};
    use crate::semantic_intent::{CapabilityTarget, SemanticAction, SemanticIntentKind};
    use serde_json::json;

    #[test]
    fn chrome_new_tab_fallback_builds_launch_action_with_args() {
        let original = "ciao, aprimi una scheda nuova di google chrome";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected chrome fallback");

        assert_eq!(intent.kind, SemanticIntentKind::ToolActionRequest);
        assert_eq!(intent.target, CapabilityTarget::DesktopLaunch);
        assert_eq!(intent.action, SemanticAction::DesktopLaunchApp);

        let action = build_action_request(&intent, original).expect("expected launch action");
        assert_eq!(action.tool_name, "desktop.launch_app");
        assert_eq!(
            action
                .params
                .get("args")
                .and_then(|value| value.as_array())
                .unwrap()[0],
            json!("--new-tab")
        );
        assert!(action
            .params
            .get("path")
            .and_then(|value| value.as_str())
            .unwrap()
            .to_ascii_lowercase()
            .contains("chrome"));
    }

    #[test]
    fn create_text_file_fallback_builds_empty_write_action() {
        let original = "creami un file test.txt sul desktop";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected file create fallback");

        assert_eq!(intent.kind, SemanticIntentKind::ToolActionRequest);
        assert_eq!(intent.target, CapabilityTarget::FilesystemWrite);
        assert_eq!(intent.action, SemanticAction::FilesystemWrite);

        let action = build_action_request(&intent, original).expect("expected write action");
        assert_eq!(action.tool_name, "filesystem.write_text");
        assert_eq!(
            action
                .params
                .get("content")
                .and_then(|value| value.as_str()),
            Some("")
        );
        assert_eq!(
            action
                .params
                .get("create_empty")
                .and_then(|value| value.as_bool()),
            Some(true)
        );
        assert!(action
            .params
            .get("path")
            .and_then(|value| value.as_str())
            .unwrap()
            .to_ascii_lowercase()
            .ends_with("test.txt"));
    }

    #[test]
    fn file_write_fallback_extracts_italian_content_request() {
        let original = "mi crei un file sul desktop chiamato text.txt contenente una frase d'amore";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected file write fallback");

        assert_eq!(intent.kind, SemanticIntentKind::ToolActionRequest);
        assert_eq!(intent.target, CapabilityTarget::FilesystemWrite);
        assert_eq!(intent.action, SemanticAction::FilesystemWrite);

        let action = build_action_request(&intent, original).expect("expected write action");
        assert_eq!(action.tool_name, "filesystem.write_text");
        assert_eq!(
            action
                .params
                .get("content")
                .and_then(|value| value.as_str()),
            Some("Ti amo piu' di quanto le parole possano dire.")
        );
        assert_eq!(
            action
                .params
                .get("create_empty")
                .and_then(|value| value.as_bool()),
            Some(false)
        );
        assert!(action
            .params
            .get("path")
            .and_then(|value| value.as_str())
            .unwrap()
            .to_ascii_lowercase()
            .ends_with("text.txt"));
    }

    #[test]
    fn file_write_fallback_extracts_english_content_request() {
        let original = "create text.txt on desktop containing a love sentence";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected file write fallback");
        let action = build_action_request(&intent, original).expect("expected write action");

        assert_eq!(action.tool_name, "filesystem.write_text");
        assert_eq!(
            action
                .params
                .get("content")
                .and_then(|value| value.as_str()),
            Some("Love is the quiet light I carry with me.")
        );
        assert_eq!(
            action
                .params
                .get("create_empty")
                .and_then(|value| value.as_bool()),
            Some(false)
        );
    }

    #[test]
    fn classifier_file_write_without_content_uses_inferred_content() {
        let original = "crea un file text.txt sul desktop e scrivici una frase d'amore";
        let intent = crate::semantic_intent::SemanticIntent {
            kind: SemanticIntentKind::ToolActionRequest,
            target: CapabilityTarget::FilesystemWrite,
            action: SemanticAction::FilesystemWrite,
            params: json!({"file_name": "text.txt"}),
            screen: super::empty_screen_request(),
            confidence: 0.9,
            language: Some("it".into()),
            rationale: None,
        };

        let action = build_action_request(&intent, original).expect("expected write action");
        assert_eq!(
            action
                .params
                .get("content")
                .and_then(|value| value.as_str()),
            Some("Ti amo piu' di quanto le parole possano dire.")
        );
        assert_eq!(
            action
                .params
                .get("create_empty")
                .and_then(|value| value.as_bool()),
            Some(false)
        );
    }

    #[test]
    fn create_txt_file_named_after_extension_marker_uses_name() {
        let original = "creami un nuovo file .txt sul desktop chiamato test";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected file create fallback");
        let action = build_action_request(&intent, original).expect("expected write action");

        assert!(action
            .params
            .get("path")
            .and_then(|value| value.as_str())
            .unwrap()
            .to_ascii_lowercase()
            .ends_with("test.txt"));
    }

    #[test]
    fn legacy_classifier_params_are_normalized() {
        let intent = crate::semantic_intent::SemanticIntent {
            kind: SemanticIntentKind::ToolActionRequest,
            target: CapabilityTarget::DesktopLaunch,
            action: SemanticAction::DesktopLaunchApp,
            params: json!({
                "parameters": {
                    "app_name": "google-chrome",
                    "arguments": ["--new-tab"]
                }
            }),
            screen: super::empty_screen_request(),
            confidence: 0.9,
            language: Some("it".into()),
            rationale: None,
        };

        let action = build_action_request(&intent, "aprimi una scheda nuova di chrome")
            .expect("expected normalized launch action");
        assert_eq!(action.tool_name, "desktop.launch_app");
        assert_eq!(
            action
                .params
                .get("args")
                .and_then(|value| value.as_array())
                .unwrap()[0],
            json!("--new-tab")
        );
    }
}
