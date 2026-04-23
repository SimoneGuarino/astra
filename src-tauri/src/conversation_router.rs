use serde_json::{json, Value};

use crate::{
    action_resolution::{
        ActionDomain, ActionOperation, ActionResolution, QueryMode, ResolutionSource,
    },
    assistant_context::capability_label,
    desktop_agent::DesktopAgentRuntime,
    desktop_agent_types::{
        CapabilityManifest, CapabilityToolAvailability, ConversationRouteDiagnostic,
        DesktopActionRequest, DesktopActionResponse, DesktopActionStatus, ScreenAnalysisRequest,
        ScreenAnalysisResult,
    },
    screen_workflow::{render_screen_workflow_run_response, resolve_screen_workflow},
    semantic_intent::{
        classify_intent, CapabilityTarget, SemanticAction, SemanticIntent, SemanticIntentKind,
        SemanticScreenRequest,
    },
    workflow_continuation::{
        is_contextual_followup_message, render_continuation_refusal, WorkflowContinuationResolution,
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
    pub analysis: Option<ScreenAnalysisResult>,
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
    let followup_language = response_language_from_message(normalized);
    if is_contextual_followup_message(normalized) {
        if let Some(route_result) =
            route_workflow_continuation(runtime, manifest, normalized, followup_language).await
        {
            return Ok(route_result);
        }
    }

    let classifier = classify_intent(normalized, manifest).await;

    let (intent, source, fallback_used, classifier_error) = match classifier {
        Ok(intent) if intent.confidence >= MIN_CLASSIFIER_CONFIDENCE => {
            if should_override_normal_chat_with_action(&intent) {
                if let Some(fallback) = actionable_fallback_intent(&lower, normalized) {
                    (
                        fallback,
                        "heuristic_override_action_request".to_string(),
                        true,
                        Some(format!(
                            "classifier selected {} for an actionable desktop request",
                            intent.kind.as_str()
                        )),
                    )
                } else {
                    (intent, "semantic_classifier".to_string(), false, None)
                }
            } else {
                (intent, "semantic_classifier".to_string(), false, None)
            }
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

    if is_model_contextual_followup_intent(&intent) {
        if let Some(route_result) = route_model_assisted_workflow_continuation(
            runtime,
            manifest,
            normalized,
            followup_language,
            &intent,
        )
        .await
        {
            return Ok(route_result);
        }
    }

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

async fn route_workflow_continuation(
    runtime: &DesktopAgentRuntime,
    manifest: &CapabilityManifest,
    normalized: &str,
    language: ResponseLanguage,
) -> Option<ConversationRouteResult> {
    match runtime.resolve_followup_continuation(manifest, normalized)? {
        WorkflowContinuationResolution::Workflow(workflow) => {
            let run = runtime.execute_screen_workflow(workflow).await;
            let diagnostic = ConversationRouteDiagnostic {
                message_excerpt: normalized.chars().take(160).collect(),
                classifier_source: "recent_workflow_continuation".into(),
                intent: "workflow_followup".into(),
                target: Some("recent_workflow".into()),
                action: Some("continue_screen_workflow".into()),
                tool_name: Some("screen.workflow".into()),
                extracted_params: Some(run.diagnostic_value()),
                confidence: run
                    .workflow
                    .continuation
                    .as_ref()
                    .map(|continuation| continuation.followup.confidence),
                routed_to: "recent_workflow_continuation_executed".into(),
                grounded: true,
                fallback_used: false,
                submit_action_called: false,
                action_id: None,
                action_status: Some(run.status.as_str().into()),
                approval_created: false,
                audit_expected: true,
                rationale: Some(
                    "Resolved the request against Rust-owned recent workflow context before normal chat routing."
                        .into(),
                ),
                error: None,
            };
            Some(ConversationRouteResult {
                route: ConversationRoute::DirectResponse(render_screen_workflow_run_response(
                    &run,
                    matches!(language, ResponseLanguage::Italian),
                )),
                diagnostic,
            })
        }
        WorkflowContinuationResolution::Refusal(refusal) => {
            let diagnostic = ConversationRouteDiagnostic {
                message_excerpt: normalized.chars().take(160).collect(),
                classifier_source: "recent_workflow_continuation".into(),
                intent: "workflow_followup".into(),
                target: Some("recent_workflow".into()),
                action: Some(format!("{:?}", refusal.followup.action_kind)),
                tool_name: Some("screen.workflow".into()),
                extracted_params: Some(refusal.diagnostic_value()),
                confidence: Some(refusal.followup.confidence),
                routed_to: "recent_workflow_continuation_refused".into(),
                grounded: true,
                fallback_used: false,
                submit_action_called: false,
                action_id: None,
                action_status: Some(format!("{:?}", refusal.policy.status)),
                approval_created: false,
                audit_expected: false,
                rationale: Some(
                    "The request was recognized as a continuation, but policy did not allow execution."
                        .into(),
                ),
                error: None,
            };
            Some(ConversationRouteResult {
                route: ConversationRoute::DirectResponse(render_continuation_refusal(
                    &refusal,
                    matches!(language, ResponseLanguage::Italian),
                )),
                diagnostic,
            })
        }
    }
}

async fn route_model_assisted_workflow_continuation(
    runtime: &DesktopAgentRuntime,
    manifest: &CapabilityManifest,
    normalized: &str,
    language: ResponseLanguage,
    intent: &SemanticIntent,
) -> Option<ConversationRouteResult> {
    let params = effective_params(&intent.params);
    match runtime.resolve_followup_continuation_with_model_params(
        manifest,
        normalized,
        &params,
        intent.confidence,
    )? {
        WorkflowContinuationResolution::Workflow(workflow) => {
            let run = runtime.execute_screen_workflow(workflow).await;
            let diagnostic = ConversationRouteDiagnostic {
                message_excerpt: normalized.chars().take(160).collect(),
                classifier_source: "model_assisted_workflow_continuation".into(),
                intent: "workflow_followup".into(),
                target: Some("recent_workflow".into()),
                action: Some("continue_screen_workflow".into()),
                tool_name: Some("screen.workflow".into()),
                extracted_params: Some(run.diagnostic_value()),
                confidence: Some(intent.confidence),
                routed_to: "model_assisted_workflow_continuation_executed".into(),
                grounded: true,
                fallback_used: false,
                submit_action_called: false,
                action_id: None,
                action_status: Some(run.status.as_str().into()),
                approval_created: false,
                audit_expected: true,
                rationale: Some(
                    "Classifier supplied contextual follow-up roles; Rust normalized, merged, grounded, and executed them."
                        .into(),
                ),
                error: None,
            };
            Some(ConversationRouteResult {
                route: ConversationRoute::DirectResponse(render_screen_workflow_run_response(
                    &run,
                    matches!(language, ResponseLanguage::Italian),
                )),
                diagnostic,
            })
        }
        WorkflowContinuationResolution::Refusal(refusal) => {
            let diagnostic = ConversationRouteDiagnostic {
                message_excerpt: normalized.chars().take(160).collect(),
                classifier_source: "model_assisted_workflow_continuation".into(),
                intent: "workflow_followup".into(),
                target: Some("recent_workflow".into()),
                action: Some(format!("{:?}", refusal.followup.action_kind)),
                tool_name: Some("screen.workflow".into()),
                extracted_params: Some(refusal.diagnostic_value()),
                confidence: Some(intent.confidence),
                routed_to: "model_assisted_workflow_continuation_refused".into(),
                grounded: true,
                fallback_used: false,
                submit_action_called: false,
                action_id: None,
                action_status: Some(format!("{:?}", refusal.policy.status)),
                approval_created: false,
                audit_expected: false,
                rationale: Some(
                    "Classifier supplied contextual follow-up roles, but Rust policy refused execution."
                        .into(),
                ),
                error: None,
            };
            Some(ConversationRouteResult {
                route: ConversationRoute::DirectResponse(render_continuation_refusal(
                    &refusal,
                    matches!(language, ResponseLanguage::Italian),
                )),
                diagnostic,
            })
        }
    }
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
            route_tool_action(runtime, manifest, normalized, intent, language, diagnostic).await
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
            let response_text = localized_owned(
                language,
                format!(
                    "I analyzed the current screen using {}. {}",
                    result.model, result.answer
                ),
                format!(
                    "Ho analizzato lo schermo con {}. {}",
                    result.model, result.answer
                ),
            );
            ConversationRoute::ScreenAnalysis(ScreenAnalysisResultEnvelope {
                response_text,
                analysis: Some(result),
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

async fn route_tool_action(
    runtime: &DesktopAgentRuntime,
    manifest: &CapabilityManifest,
    normalized: &str,
    intent: &SemanticIntent,
    language: ResponseLanguage,
    diagnostic: &mut ConversationRouteDiagnostic,
) -> ConversationRoute {
    diagnostic.routed_to = "desktop_action_governance".into();
    diagnostic.grounded = true;

    let resolution = resolve_action_resolution(intent, normalized);
    if let Some(resolution) = resolution.as_ref() {
        diagnostic.extracted_params = Some(resolution.diagnostic_value());
        if let Some(rationale) = resolution.rationale.as_ref() {
            diagnostic.rationale = Some(rationale.clone());
        }
        if matches!(
            resolution.operation,
            ActionOperation::ScreenGuidedBrowserWorkflow
                | ActionOperation::ScreenGuidedFollowupAction
                | ActionOperation::ScreenGuidedNavigationWorkflow
        ) {
            diagnostic.routed_to = "screen_grounded_workflow_planned".into();
            if let Some(workflow) = resolve_screen_workflow(resolution, manifest, normalized) {
                let run = runtime.execute_screen_workflow(workflow).await;
                diagnostic.routed_to = "screen_grounded_workflow_executed".into();
                diagnostic.extracted_params = Some(run.diagnostic_value());
                return ConversationRoute::DirectResponse(render_screen_workflow_run_response(
                    &run,
                    matches!(language, ResponseLanguage::Italian),
                ));
            }
        }
    }

    let action_result = if let Some(resolution) = resolution.as_ref() {
        build_action_request_from_resolution(resolution, normalized)
    } else {
        build_action_request(intent, normalized)
    };

    let action = match action_result {
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

fn resolve_action_resolution(intent: &SemanticIntent, original: &str) -> Option<ActionResolution> {
    if !matches!(intent.kind, SemanticIntentKind::ToolActionRequest) {
        return None;
    }

    let params = effective_params(&intent.params);
    let lower = original.to_lowercase();
    let source = if param_str(&params, "operation").is_some()
        || params.get("entities").and_then(Value::as_object).is_some()
    {
        ResolutionSource::ModelAssisted
    } else if intent
        .rationale
        .as_deref()
        .map(|value| value.contains("fallback"))
        .unwrap_or(false)
    {
        ResolutionSource::HeuristicFallback
    } else {
        ResolutionSource::RustNormalizer
    };

    let operation = model_operation(&params)
        .or_else(|| infer_resolution_operation(intent, &lower, &params))
        .or_else(|| infer_textual_resolution_operation(&lower))?;
    let domain = model_domain(&params).unwrap_or_else(|| domain_for_operation(&operation, intent));
    let mut resolution =
        ActionResolution::new(operation.clone(), domain, intent.confidence, source);

    resolution.provider = param_str_any_deep(&params, &["provider", "search_provider", "engine"])
        .or_else(|| infer_search_provider(&lower));
    resolution.query_mode = param_str_any_deep(&params, &["query_mode"])
        .as_deref()
        .and_then(QueryMode::from_str)
        .or_else(|| infer_query_mode(original, &operation));
    resolution.entities = normalized_resolution_entities(&params, original, &operation);
    resolution.post_processing = normalized_post_processing(&params, &operation);
    resolution.requires_screen_context = params
        .get("requires_screen_context")
        .and_then(Value::as_bool)
        .unwrap_or(matches!(
            operation,
            ActionOperation::ScreenGuidedBrowserWorkflow
        ));
    resolution.workflow_steps = param_string_array_any(&params, &["workflow_steps", "steps"])
        .into_iter()
        .collect();
    if matches!(operation, ActionOperation::ScreenGuidedBrowserWorkflow)
        && resolution.workflow_steps.is_empty()
    {
        resolution.workflow_steps = vec![
            "locate_existing_browser_tab".into(),
            "focus_search_input".into(),
            "enter_query".into(),
            "submit_search".into(),
            "open_first_result".into(),
        ];
    }
    resolution.ambiguity = param_str(&params, "ambiguity");
    resolution.rationale = Some(format!(
        "{} resolution for {}",
        resolution.source.as_str(),
        resolution.operation.as_str()
    ));

    Some(resolution)
}

fn build_action_request_from_resolution(
    resolution: &ActionResolution,
    original: &str,
) -> Result<DesktopActionRequest, String> {
    match resolution.operation {
        ActionOperation::ReadFile | ActionOperation::ReadAndSummarizeFile => {
            let path = resolve_file_read_path(&resolution.entities, original)
                .ok_or_else(|| "I need the file path to read.".to_string())?;
            let post_processing = if matches!(
                resolution.operation,
                ActionOperation::ReadAndSummarizeFile
            ) {
                let mut post_processing = resolution.post_processing.clone();
                if !post_processing.is_object() {
                    post_processing = json!({});
                }
                post_processing["mode"] = json!("summary");
                post_processing
            } else {
                resolution.post_processing.clone()
            };

            Ok(DesktopActionRequest {
                tool_name: "filesystem.read_text".into(),
                params: json!({
                    "path": path,
                    "operation": resolution.operation.as_str(),
                    "post_processing": post_processing,
                }),
                preview_only: false,
                reason: Some(match resolution.operation {
                    ActionOperation::ReadAndSummarizeFile => {
                        "User requested reading and summarizing a file".into()
                    }
                    _ => "User requested reading a file".into(),
                }),
            })
        }
        ActionOperation::SearchFile => {
            let pattern = entity_str_any(resolution, &["pattern", "query", "filename", "file_name"])
                .ok_or_else(|| "I need the filename or pattern to search for.".to_string())?;
            let mut params = json!({"pattern": pattern, "max_results": 25});
            if let Some(root) = entity_str_any(resolution, &["root", "location_hint"]) {
                if root != "desktop" && root != "scrivania" {
                    params["root"] = json!(root);
                }
            }
            Ok(DesktopActionRequest {
                tool_name: "filesystem.search".into(),
                params,
                preview_only: false,
                reason: Some("User requested searching files".into()),
            })
        }
        ActionOperation::WriteFile => {
            let path = resolve_file_write_path(&resolution.entities, original)
                .ok_or_else(|| "I need the target file path before writing.".to_string())?;
            let content = entity_str_any(resolution, &["content", "text"])
                .or_else(|| infer_file_content_from_text(original))
                .ok_or_else(|| {
                    "I need the content to write before creating or modifying a file.".to_string()
                })?;
            let mode = entity_str_any(resolution, &["mode"]).unwrap_or_else(|| "overwrite".into());
            Ok(DesktopActionRequest {
                tool_name: "filesystem.write_text".into(),
                params: json!({"path": path, "content": content, "mode": mode, "create_empty": false}),
                preview_only: false,
                reason: Some("User requested writing a file".into()),
            })
        }
        ActionOperation::BrowserSearch => {
            let provider = resolution
                .provider
                .as_deref()
                .unwrap_or("web")
                .to_ascii_lowercase();
            let query =
                entity_str_any(resolution, &["query_candidate", "query", "search_query"]).or_else(
                    || {
                        let provider = if provider == "youtube" || provider == "you tube" {
                            SearchProvider::YouTube
                        } else {
                            SearchProvider::Google
                        };
                        extract_search_query(original, provider)
                    },
                );
            let query = query.ok_or_else(|| "I need a search query.".to_string())?;
            if !is_valid_search_query(&query) {
                return Err("I need a search query.".into());
            }
            if provider == "youtube" || provider == "you tube" {
                return Ok(DesktopActionRequest {
                    tool_name: "browser.open".into(),
                    params: json!({
                        "url": youtube_search_url(&query),
                        "query": query,
                        "provider": "youtube",
                        "query_mode": resolution.query_mode.as_ref().map(QueryMode::as_str),
                    }),
                    preview_only: false,
                    reason: Some("User requested a YouTube search".into()),
                });
            }
            Ok(DesktopActionRequest {
                tool_name: "browser.search".into(),
                params: json!({
                    "query": query,
                    "provider": provider,
                    "query_mode": resolution.query_mode.as_ref().map(QueryMode::as_str),
                }),
                preview_only: false,
                reason: Some("User requested a web search".into()),
            })
        }
        ActionOperation::BrowserOpen => {
            let url = entity_str_any(resolution, &["url"])
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
        ActionOperation::DesktopLaunchApp => {
            let app_or_path =
                entity_str_any(resolution, &["path", "app", "app_name", "application", "name"])
                    .ok_or_else(|| {
                        "I need the app name or executable path to launch.".to_string()
                    })?;
            let path = resolve_app_alias(&app_or_path).unwrap_or(app_or_path);
            Ok(DesktopActionRequest {
                tool_name: "desktop.launch_app".into(),
                params: json!({"path": path, "args": Vec::<String>::new()}),
                preview_only: false,
                reason: Some("User requested launching a desktop app".into()),
            })
        }
        ActionOperation::ScreenGuidedBrowserWorkflow
        | ActionOperation::ScreenGuidedFollowupAction
        | ActionOperation::ScreenGuidedNavigationWorkflow => Err(
            "Screen-guided workflows are recognized, but interactive UI control is not available in this runtime yet."
                .into(),
        ),
        ActionOperation::Unknown => Err(
            "I understood this as an action request, but I need a clearer target before running anything."
                .into(),
        ),
    }
}

fn model_operation(params: &Value) -> Option<ActionOperation> {
    param_str(params, "operation")
        .as_deref()
        .and_then(ActionOperation::from_str)
}

fn model_domain(params: &Value) -> Option<ActionDomain> {
    match param_str(params, "domain")
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_")
        .as_str()
    {
        "filesystem" | "file" => Some(ActionDomain::Filesystem),
        "browser" | "web" => Some(ActionDomain::Browser),
        "desktop" => Some(ActionDomain::Desktop),
        "terminal" => Some(ActionDomain::Terminal),
        "browser_screen_interaction" | "screen_browser" => {
            Some(ActionDomain::BrowserScreenInteraction)
        }
        "screen_interaction" | "screen_navigation" => Some(ActionDomain::Screen),
        "screen" => Some(ActionDomain::Screen),
        "unknown" => Some(ActionDomain::Unknown),
        _ => None,
    }
}

fn infer_resolution_operation(
    intent: &SemanticIntent,
    lower: &str,
    params: &Value,
) -> Option<ActionOperation> {
    if looks_like_screen_guided_browser_workflow(lower) {
        return Some(ActionOperation::ScreenGuidedBrowserWorkflow);
    }
    if looks_like_screen_guided_followup_action(lower) {
        return Some(ActionOperation::ScreenGuidedFollowupAction);
    }
    if looks_like_screen_guided_navigation_workflow(lower) {
        return Some(ActionOperation::ScreenGuidedNavigationWorkflow);
    }

    match intent.action {
        SemanticAction::FilesystemRead => Some(if looks_like_file_summary_request(lower) {
            ActionOperation::ReadAndSummarizeFile
        } else {
            ActionOperation::ReadFile
        }),
        SemanticAction::FilesystemSearch => Some(ActionOperation::SearchFile),
        SemanticAction::FilesystemWrite => Some(ActionOperation::WriteFile),
        SemanticAction::BrowserSearch => Some(ActionOperation::BrowserSearch),
        SemanticAction::BrowserOpenUrl => {
            if param_str(params, "url")
                .map(|url| url.contains("youtube.com/results") || url.contains("google.com/search"))
                .unwrap_or(false)
            {
                Some(ActionOperation::BrowserSearch)
            } else {
                Some(ActionOperation::BrowserOpen)
            }
        }
        SemanticAction::DesktopLaunchApp => Some(ActionOperation::DesktopLaunchApp),
        SemanticAction::TerminalRun | SemanticAction::None | SemanticAction::Unknown => None,
    }
}

fn infer_textual_resolution_operation(lower: &str) -> Option<ActionOperation> {
    if looks_like_screen_guided_browser_workflow(lower) {
        return Some(ActionOperation::ScreenGuidedBrowserWorkflow);
    }
    if looks_like_screen_guided_followup_action(lower) {
        return Some(ActionOperation::ScreenGuidedFollowupAction);
    }
    if looks_like_screen_guided_navigation_workflow(lower) {
        return Some(ActionOperation::ScreenGuidedNavigationWorkflow);
    }
    if looks_like_file_read_or_summary_request(lower) {
        return Some(if looks_like_file_summary_request(lower) {
            ActionOperation::ReadAndSummarizeFile
        } else {
            ActionOperation::ReadFile
        });
    }
    if looks_like_search_request(lower) {
        return Some(ActionOperation::BrowserSearch);
    }
    None
}

fn domain_for_operation(operation: &ActionOperation, intent: &SemanticIntent) -> ActionDomain {
    match operation {
        ActionOperation::ReadFile
        | ActionOperation::ReadAndSummarizeFile
        | ActionOperation::WriteFile
        | ActionOperation::SearchFile => ActionDomain::Filesystem,
        ActionOperation::BrowserSearch | ActionOperation::BrowserOpen => ActionDomain::Browser,
        ActionOperation::DesktopLaunchApp => ActionDomain::Desktop,
        ActionOperation::ScreenGuidedBrowserWorkflow => ActionDomain::BrowserScreenInteraction,
        ActionOperation::ScreenGuidedFollowupAction
        | ActionOperation::ScreenGuidedNavigationWorkflow => ActionDomain::Screen,
        ActionOperation::Unknown => match intent.target {
            CapabilityTarget::Browser => ActionDomain::Browser,
            CapabilityTarget::FilesystemRead
            | CapabilityTarget::FilesystemWrite
            | CapabilityTarget::FilesystemSearch => ActionDomain::Filesystem,
            CapabilityTarget::DesktopLaunch => ActionDomain::Desktop,
            CapabilityTarget::Terminal => ActionDomain::Terminal,
            CapabilityTarget::Screen => ActionDomain::Screen,
            _ => ActionDomain::Unknown,
        },
    }
}

fn normalized_resolution_entities(
    params: &Value,
    original: &str,
    operation: &ActionOperation,
) -> Value {
    let mut entities = params
        .get("entities")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    for key in [
        "path",
        "file_name",
        "filename",
        "name",
        "query",
        "query_candidate",
        "search_query",
        "url",
        "content",
        "mode",
        "location_hint",
        "root",
        "pattern",
        "app",
        "app_name",
        "browser_app",
    ] {
        if let Some(value) = param_str(params, key) {
            entities.entry(key).or_insert_with(|| json!(value));
        }
    }

    if matches!(
        operation,
        ActionOperation::ReadFile
            | ActionOperation::ReadAndSummarizeFile
            | ActionOperation::WriteFile
            | ActionOperation::SearchFile
    ) {
        if !has_any_entity(
            &Value::Object(entities.clone()),
            &["path", "file_name", "filename"],
        ) {
            if let Some(file_name) = infer_file_name_from_text(original) {
                entities.insert("filename".into(), json!(file_name));
            }
        }
        if mentions_desktop(original)
            && !has_any_entity(&Value::Object(entities.clone()), &["location_hint", "root"])
        {
            entities.insert("location_hint".into(), json!("desktop"));
        }
    }

    if matches!(operation, ActionOperation::BrowserSearch) {
        let provider = param_str_any_deep(params, &["provider", "search_provider", "engine"])
            .or_else(|| infer_search_provider(&original.to_lowercase()))
            .unwrap_or_else(|| "web".into());
        entities
            .entry("provider")
            .or_insert_with(|| json!(provider.clone()));
        if !has_any_entity(
            &Value::Object(entities.clone()),
            &["query_candidate", "query"],
        ) {
            let search_provider = if provider.eq_ignore_ascii_case("youtube")
                || provider.eq_ignore_ascii_case("you tube")
            {
                SearchProvider::YouTube
            } else {
                SearchProvider::Google
            };
            if let Some(query) = extract_search_query(original, search_provider) {
                entities.insert("query_candidate".into(), json!(query));
            }
        }
    }

    Value::Object(entities)
}

fn normalized_post_processing(params: &Value, operation: &ActionOperation) -> Value {
    let mut post_processing = params
        .get("post_processing")
        .filter(|value| value.is_object())
        .cloned()
        .unwrap_or_else(|| json!({}));
    if matches!(operation, ActionOperation::ReadAndSummarizeFile) {
        post_processing["mode"] = json!("summary");
        post_processing["summary_style"] = post_processing
            .get("summary_style")
            .cloned()
            .unwrap_or_else(|| json!("concise"));
    }
    post_processing
}

fn resolve_file_read_path(entities: &Value, original: &str) -> Option<String> {
    if let Some(path) = param_str(entities, "path") {
        return Some(resolve_desktop_relative_path(&path, original));
    }

    let file_name = param_str_any(entities, &["file_name", "filename", "name"])
        .or_else(|| infer_file_name_from_text(original))?;
    Some(resolve_desktop_relative_path(&file_name, original))
}

fn entity_str_any(resolution: &ActionResolution, keys: &[&str]) -> Option<String> {
    param_str_any(&resolution.entities, keys)
}

fn param_str_any_deep(params: &Value, keys: &[&str]) -> Option<String> {
    param_str_any(params, keys).or_else(|| {
        params
            .get("entities")
            .and_then(|entities| param_str_any(entities, keys))
    })
}

fn has_any_entity(entities: &Value, keys: &[&str]) -> bool {
    keys.iter().any(|key| param_str(entities, key).is_some())
}

fn infer_query_mode(original: &str, operation: &ActionOperation) -> Option<QueryMode> {
    if !matches!(operation, ActionOperation::BrowserSearch) {
        return None;
    }
    let lower = original.to_lowercase();
    if original.contains('"') || original.contains('\'') || lower.contains("esatto") {
        return Some(QueryMode::Precise);
    }
    if lower.contains("una canzone")
        || lower.contains("un brano")
        || lower.contains("un video")
        || lower.contains("qualcosa")
        || lower.contains("something")
        || lower.contains("a song")
        || lower.contains("a video")
    {
        return Some(QueryMode::Semantic);
    }
    Some(QueryMode::Precise)
}

fn infer_search_provider(lower: &str) -> Option<String> {
    let mut provider_scope = lower
        .replace("google chrome", " ")
        .replace("chrome", " ")
        .replace("microsoft edge", " ")
        .replace("firefox", " ")
        .replace("safari", " ");
    if looks_like_edge_browser_reference(lower) {
        provider_scope = provider_scope.replace("edge", " ");
    }
    if provider_scope.contains("youtube") || provider_scope.contains("you tube") {
        Some("youtube".into())
    } else if provider_scope.contains("google") {
        Some("google".into())
    } else if provider_scope.contains("web")
        || provider_scope.contains("internet")
        || provider_scope.contains("online")
    {
        Some("web".into())
    } else {
        None
    }
}

fn looks_like_edge_browser_reference(lower: &str) -> bool {
    lower.contains("microsoft edge")
        || [
            "browser edge",
            "edge browser",
            "app edge",
            "apri edge",
            "aprimi edge",
            "open edge",
            "in edge",
            "su edge",
            "scheda edge",
            "tab edge",
            "finestra edge",
            "window edge",
        ]
        .iter()
        .any(|marker| lower.contains(marker))
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
            if args.is_empty() && wants_new_window(original) {
                args.push("--new-window".into());
            } else if args.is_empty() && wants_new_tab(original) {
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

    response_language_from_message(message)
}

fn response_language_from_message(message: &str) -> ResponseLanguage {
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
        "clicca",
        "primo",
        "secondo",
        "continua",
        "torna",
        "scrivi",
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

fn should_override_normal_chat_with_action(intent: &SemanticIntent) -> bool {
    matches!(
        intent.kind,
        SemanticIntentKind::NormalChat | SemanticIntentKind::CapabilityQuestion
    )
}

fn is_model_contextual_followup_intent(intent: &SemanticIntent) -> bool {
    if !matches!(intent.kind, SemanticIntentKind::ToolActionRequest) {
        return false;
    }
    let params = effective_params(&intent.params);
    let operation = param_str(&params, "operation")
        .or_else(|| {
            params
                .get("entities")
                .and_then(|entities| param_str(entities, "operation"))
        })
        .unwrap_or_default()
        .replace('-', "_")
        .to_ascii_lowercase();

    matches!(
        operation.as_str(),
        "screen_guided_followup_action" | "screen_guided_navigation_workflow"
    )
}

fn actionable_fallback_intent(lower: &str, original: &str) -> Option<SemanticIntent> {
    fallback_intent(lower, original).filter(|intent| {
        matches!(
            intent.kind,
            SemanticIntentKind::ToolActionRequest | SemanticIntentKind::ScreenAnalysisRequest
        )
    })
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
    if let Some(workflow) = infer_screen_guided_browser_workflow_intent(lower, original) {
        return Some(workflow);
    }
    if let Some(followup) = infer_screen_guided_followup_intent(lower, original) {
        return Some(followup);
    }
    if let Some(navigation) = infer_screen_guided_navigation_intent(lower, original) {
        return Some(navigation);
    }
    if let Some(file_read) = infer_file_read_or_summary_intent(lower, original) {
        return Some(file_read);
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
    if let Some(search) = infer_browser_search_action(lower, original) {
        return Some(search);
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
    if let Some(url) = infer_youtube_chrome_search_url(lower, original) {
        return Some(intent(
            SemanticIntentKind::ToolActionRequest,
            CapabilityTarget::DesktopLaunch,
            SemanticAction::DesktopLaunchApp,
            json!({
                "app": "google chrome",
                "args": vec!["--new-window".to_string(), url],
            }),
            0.67,
            "youtube chrome search fallback",
        ));
    }
    if looks_like_chrome_launch_request(lower) {
        return Some(intent(
            SemanticIntentKind::ToolActionRequest,
            CapabilityTarget::DesktopLaunch,
            SemanticAction::DesktopLaunchApp,
            json!({
                "app": "google chrome",
                "args": if wants_new_window(original) {
                    vec!["--new-window"]
                } else if wants_new_tab(original) {
                    vec!["--new-tab"]
                } else {
                    Vec::<&str>::new()
                },
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
                "args": if wants_new_window(original) {
                    vec!["--new-window"]
                } else if wants_new_tab(original) {
                    vec!["--new-tab"]
                } else {
                    Vec::<&str>::new()
                },
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

fn infer_file_read_or_summary_intent(lower: &str, original: &str) -> Option<SemanticIntent> {
    if !looks_like_file_read_or_summary_request(lower) {
        return None;
    }

    let operation = if looks_like_file_summary_request(lower) {
        "read_and_summarize_file"
    } else {
        "read_file"
    };
    let mut entities = serde_json::Map::new();
    if let Some(file_name) = infer_file_name_from_text(original) {
        entities.insert("filename".into(), json!(file_name));
    }
    if mentions_desktop(original) {
        entities.insert("location_hint".into(), json!("desktop"));
    }

    Some(intent(
        SemanticIntentKind::ToolActionRequest,
        CapabilityTarget::FilesystemRead,
        SemanticAction::FilesystemRead,
        json!({
            "operation": operation,
            "domain": "filesystem",
            "entities": Value::Object(entities),
            "post_processing": if operation == "read_and_summarize_file" {
                json!({"mode": "summary", "summary_style": "concise"})
            } else {
                json!({})
            },
        }),
        0.68,
        "file read action resolution fallback",
    ))
}

fn infer_screen_guided_browser_workflow_intent(
    lower: &str,
    original: &str,
) -> Option<SemanticIntent> {
    if !looks_like_screen_guided_browser_workflow(lower) {
        return None;
    }

    let provider = infer_search_provider(lower).unwrap_or_else(|| "web".into());
    let query_provider = if provider == "youtube" {
        SearchProvider::YouTube
    } else {
        SearchProvider::Google
    };
    let query = extract_search_query(original, query_provider);

    Some(intent(
        SemanticIntentKind::ToolActionRequest,
        CapabilityTarget::Browser,
        SemanticAction::Unknown,
        json!({
            "operation": "screen_guided_browser_workflow",
            "domain": "browser_screen_interaction",
            "provider": provider.clone(),
            "query_mode": query.as_ref().map(|_| "semantic"),
            "entities": {
                "provider": provider.clone(),
                "query_candidate": query,
            },
            "workflow_steps": [
                "locate_existing_browser_tab",
                "focus_search_input",
                "enter_query",
                "submit_search",
                "open_first_result"
            ],
            "requires_screen_context": true,
        }),
        0.56,
        "screen-guided browser workflow resolution fallback",
    ))
}

fn infer_screen_guided_followup_intent(lower: &str, _original: &str) -> Option<SemanticIntent> {
    if !looks_like_screen_guided_followup_action(lower) {
        return None;
    }

    if looks_like_screen_guided_typing_followup(lower) {
        let query = extract_screen_typing_value(_original);
        return Some(intent(
            SemanticIntentKind::ToolActionRequest,
            CapabilityTarget::Screen,
            SemanticAction::Unknown,
            json!({
                "operation": "screen_guided_followup_action",
                "domain": "screen_interaction",
                "entities": {
                    "query_candidate": query,
                    "requires_recent_focus_target": true,
                },
                "workflow_steps": ["enter_text"],
                "requires_screen_context": true,
            }),
            0.52,
            "screen-guided typing follow-up resolution fallback",
        ));
    }

    Some(intent(
        SemanticIntentKind::ToolActionRequest,
        CapabilityTarget::Screen,
        SemanticAction::Unknown,
        json!({
            "operation": "screen_guided_followup_action",
            "domain": "screen_interaction",
            "entities": {
                "selection_strategy": if lower.contains("primo") || lower.contains("first") {
                    "first_visible_result"
                } else {
                    "referenced_visible_element"
                }
            },
            "workflow_steps": if lower.contains("primo") || lower.contains("first") {
                json!(["open_ranked_result"])
            } else {
                json!(["click_visible_element"])
            },
            "requires_screen_context": true,
        }),
        0.54,
        "screen-guided follow-up action resolution fallback",
    ))
}

fn infer_screen_guided_navigation_intent(lower: &str, _original: &str) -> Option<SemanticIntent> {
    if !looks_like_screen_guided_navigation_workflow(lower) {
        return None;
    }

    Some(intent(
        SemanticIntentKind::ToolActionRequest,
        CapabilityTarget::Screen,
        SemanticAction::Unknown,
        json!({
            "operation": "screen_guided_navigation_workflow",
            "domain": "screen_navigation",
            "workflow_steps": ["navigate_back"],
            "requires_screen_context": true,
        }),
        0.52,
        "screen-guided navigation workflow resolution fallback",
    ))
}

fn extract_screen_typing_value(original: &str) -> Option<String> {
    [
        "ora scrivi",
        "adesso scrivi",
        "scrivi qui",
        "scrivi nel campo",
        "now type",
        "type here",
    ]
    .iter()
    .find_map(|marker| extract_after_marker(original, marker))
    .map(trim_query_edges)
    .filter(|value| !value.is_empty())
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

fn looks_like_file_read_or_summary_request(lower: &str) -> bool {
    let file = lower.contains("file")
        || lower.contains(".txt")
        || lower.contains("documento")
        || lower.contains("contenuto del");
    if file && looks_like_file_summary_request(lower) {
        return true;
    }

    let read = lower.contains("leggimi")
        || lower.contains("leggi ")
        || lower.starts_with("leggi")
        || lower.contains("read ")
        || lower.contains("mostrami")
        || lower.contains("contenuto");
    file && (read || looks_like_file_summary_request(lower))
        && !looks_like_file_write_action_request(lower)
}

fn looks_like_file_summary_request(lower: &str) -> bool {
    (lower.contains("riassunt")
        || lower.contains("sintesi")
        || lower.contains("summar")
        || lower.contains("summary"))
        && (lower.contains("file") || lower.contains(".txt") || lower.contains("documento"))
}

fn looks_like_screen_guided_browser_workflow(lower: &str) -> bool {
    let browser_context = lower.contains("tab")
        || lower.contains("scheda")
        || lower.contains("chrome")
        || lower.contains("browser");
    let search_context = lower.contains("youtube")
        || lower.contains("google")
        || lower.contains("cerca")
        || lower.contains("cerchi")
        || lower.contains("search");
    let follow_through = lower.contains("primo risultato")
        || lower.contains("first result")
        || lower.contains("aprimi il primo")
        || lower.contains("open the first");

    browser_context && search_context && follow_through
}

fn looks_like_screen_guided_followup_action(lower: &str) -> bool {
    if looks_like_screen_guided_typing_followup(lower) {
        return true;
    }

    let click_or_open = lower.contains("clicca")
        || lower.contains("click")
        || lower.contains("premi")
        || lower.contains("apri")
        || lower.contains("open");
    let visible_reference = lower.contains("primo risultato")
        || lower.contains("first result")
        || lower.contains("quello")
        || lower.contains("quella")
        || lower.contains("che vedi")
        || lower.contains("visible");

    click_or_open && visible_reference
}

fn looks_like_screen_guided_typing_followup(lower: &str) -> bool {
    let typing = lower.contains("ora scrivi")
        || lower.contains("adesso scrivi")
        || lower.contains("scrivi qui")
        || lower.contains("scrivi nel campo")
        || lower.contains("type here")
        || lower.contains("now type");
    typing && !looks_like_file_write_action_request(lower)
}

fn looks_like_screen_guided_navigation_workflow(lower: &str) -> bool {
    (lower.contains("torna")
        || lower.contains("back")
        || lower.contains("indietro")
        || lower.contains("go back"))
        && (lower.contains("schermata")
            || lower.contains("screen")
            || lower.contains("pagina")
            || lower.contains("prima")
            || lower.contains("previous"))
}

fn looks_like_governed_action_request(lower: &str) -> bool {
    if looks_like_file_write_action_request(lower)
        || looks_like_file_read_or_summary_request(lower)
        || looks_like_screen_guided_browser_workflow(lower)
        || looks_like_screen_guided_followup_action(lower)
        || looks_like_screen_guided_navigation_workflow(lower)
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
        "cerchi",
        "cercami",
        "sirchi",
        "serchi",
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
        "google",
        "youtube",
        "web",
        "internet",
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

fn wants_new_window(value: &str) -> bool {
    let lower = value.to_lowercase();
    lower.contains("new window")
        || lower.contains("nuova finestra")
        || lower.contains("finestra nuova")
        || lower.contains("--new-window")
}

fn infer_browser_search_action(lower: &str, original: &str) -> Option<SemanticIntent> {
    if !looks_like_search_request(lower) {
        return None;
    }

    let mentions_chrome = lower.contains("chrome") || lower.contains("google chrome");
    let new_window = wants_new_window(original);
    let youtube = lower.contains("youtube") || lower.contains("you tube");
    let google = lower.contains("google")
        || lower.contains("web")
        || lower.contains("internet")
        || lower.contains("online")
        || lower.contains("sul web");

    if youtube {
        let query = extract_search_query(original, SearchProvider::YouTube)?;
        let url = youtube_search_url(&query);
        if mentions_chrome {
            return Some(desktop_launch_search_intent(
                url,
                new_window,
                "youtube chrome search fallback",
            ));
        }
        return Some(intent(
            SemanticIntentKind::ToolActionRequest,
            CapabilityTarget::Browser,
            SemanticAction::BrowserOpenUrl,
            json!({"url": url}),
            0.66,
            "youtube search URL fallback",
        ));
    }

    if google || lower.contains("cerca") || lower.contains("cerchi") || lower.contains("sirchi") {
        let query = extract_search_query(original, SearchProvider::Google)?;
        if mentions_chrome {
            return Some(desktop_launch_search_intent(
                google_search_url(&query),
                new_window,
                "google chrome search fallback",
            ));
        }
        return Some(intent(
            SemanticIntentKind::ToolActionRequest,
            CapabilityTarget::Browser,
            SemanticAction::BrowserSearch,
            json!({"query": query}),
            0.64,
            "google search action fallback",
        ));
    }

    None
}

fn desktop_launch_search_intent(url: String, new_window: bool, rationale: &str) -> SemanticIntent {
    let first_arg = if new_window {
        "--new-window"
    } else {
        "--new-tab"
    };
    intent(
        SemanticIntentKind::ToolActionRequest,
        CapabilityTarget::DesktopLaunch,
        SemanticAction::DesktopLaunchApp,
        json!({
            "app": "google chrome",
            "args": vec![first_arg.to_string(), url],
        }),
        0.67,
        rationale,
    )
}

#[derive(Debug, Clone, Copy)]
enum SearchProvider {
    Google,
    YouTube,
}

fn looks_like_search_request(lower: &str) -> bool {
    let search_verb = [
        "cerca", "cercami", "cerchi", "sirchi", "serchi", "search", "trova", "trovi", "trovami",
        "find",
    ]
    .iter()
    .any(|marker| lower.contains(marker));
    let search_target = [
        "google", "youtube", "you tube", "web", "internet", "online", "chrome",
    ]
    .iter()
    .any(|marker| lower.contains(marker));
    search_verb && search_target
}

fn extract_search_query(original: &str, provider: SearchProvider) -> Option<String> {
    let query = trim_query_edges(&strip_wake_prefix(original));

    let markers: &[&str] = match provider {
        SearchProvider::Google => &[
            "apri google e cerca",
            "google e cerca",
            "mi cerchi su google",
            "mi sirchi su google",
            "mi serchi su google",
            "cercami su google",
            "cerca su google",
            "search google for",
            "apri chrome in una nuova finestra e cerca",
            "apri chrome e cerca",
            "cerca sul web",
            "cerca online",
            "cercami",
            "cerca",
            "cerchi",
            "sirchi",
            "serchi",
        ],
        SearchProvider::YouTube => &[
            "mi cerchi su youtube",
            "mi sirchi su youtube",
            "mi serchi su youtube",
            "cercami su youtube",
            "cerca su youtube",
            "cerca youtube",
            "apri youtube e cerca",
            "youtube e cerca",
            "mi trovi su youtube",
            "trovami su youtube",
            "trova su youtube",
            "search youtube for",
            "search on youtube for",
        ],
    };

    let mut candidates = Vec::new();
    for marker in markers {
        if let Some(index) = find_case_insensitive(&query, marker) {
            let after = index + marker.len();
            candidates.push(query[after..].trim().to_string());
        }
    }

    if let Some(before_domain) = extract_search_query_before_domain(&query, provider) {
        candidates.push(before_domain);
    }
    candidates.push(query);

    candidates
        .into_iter()
        .find_map(|candidate| clean_search_query_candidate(&candidate, provider))
}

fn extract_search_query_before_domain(query: &str, provider: SearchProvider) -> Option<String> {
    let domain_markers: &[&str] = match provider {
        SearchProvider::Google => &[
            " su google",
            " on google",
            " sul web",
            " su internet",
            " online",
            " google",
        ],
        SearchProvider::YouTube => &[
            " su youtube",
            " su you tube",
            " on youtube",
            " on you tube",
            " youtube",
            " you tube",
        ],
    };

    domain_markers
        .iter()
        .filter_map(|marker| find_case_insensitive(query, marker))
        .min()
        .and_then(|index| {
            let candidate = query[..index].trim();
            (!candidate.is_empty()).then(|| candidate.to_string())
        })
}

fn clean_search_query_candidate(candidate: &str, provider: SearchProvider) -> Option<String> {
    let mut query = trim_query_edges(candidate);
    if query.is_empty() {
        return None;
    }

    query = strip_conversational_query_prefixes(&query);
    query = strip_search_command_prefixes(&query);
    query = strip_provider_location_prefixes(&query, provider);
    query = truncate_search_query_suffixes(&query, provider);
    query = strip_conversational_query_prefixes(&query);
    query = strip_search_command_prefixes(&query);
    query = strip_provider_location_prefixes(&query, provider);
    query = trim_query_edges(&query);

    if matches!(provider, SearchProvider::YouTube) {
        query = normalize_youtube_media_query(&query);
    }

    query = trim_query_edges(&query);
    is_valid_search_query(&query).then_some(query)
}

fn strip_conversational_query_prefixes(value: &str) -> String {
    let mut query = trim_query_edges(value);
    loop {
        let before = query.clone();
        for prefix in [
            "ciao",
            "ehi",
            "hey",
            "hi",
            "hello",
            "per favore",
            "per piacere",
            "please",
            "scusa",
            "scusami",
            "astra",
            "astrami",
        ] {
            if let Some(stripped) = strip_case_insensitive_prefix(&query, prefix) {
                query = trim_query_edges(stripped);
                break;
            }
        }
        if query == before {
            return query;
        }
    }
}

fn truncate_search_query_suffixes(query: &str, provider: SearchProvider) -> String {
    let mut query = query.to_string();
    for marker in [
        ", su una finestra",
        " su una finestra",
        ", in una finestra",
        " in una finestra",
        ", su google chrome",
        " su google chrome",
        ", su chrome",
        " su chrome",
        " sul web",
        " su internet",
        " online",
    ] {
        if let Some(index) = find_case_insensitive(&query, marker) {
            query.truncate(index);
        }
    }

    let provider_suffixes: &[&str] = match provider {
        SearchProvider::Google => &[
            " su google",
            " on google",
            " google",
            " sul web",
            " su internet",
            " online",
        ],
        SearchProvider::YouTube => &[
            " su youtube",
            " su you tube",
            " on youtube",
            " on you tube",
            " youtube",
            " you tube",
        ],
    };
    for marker in provider_suffixes {
        if let Some(index) = find_case_insensitive(&query, marker) {
            query.truncate(index);
        }
    }

    trim_query_edges(&query)
}

fn strip_search_command_prefixes(value: &str) -> String {
    let mut query = trim_query_edges(value);
    loop {
        let before = query.clone();
        for prefix in [
            "mi puoi cercare",
            "potresti cercare",
            "puoi cercare",
            "mi cerchi",
            "mi sirchi",
            "mi serchi",
            "mi trovi",
            "trovami",
            "cercami",
            "cerca",
            "cerchi",
            "sirchi",
            "serchi",
            "trova",
            "trovi",
            "search for",
            "search",
            "find",
            "apri google e cerca",
            "apri youtube e cerca",
            "open google and search",
            "open youtube and search",
        ] {
            if let Some(stripped) = strip_case_insensitive_prefix(&query, prefix) {
                query = trim_query_edges(stripped);
                break;
            }
        }
        if query == before {
            return query;
        }
    }
}

fn strip_provider_location_prefixes(value: &str, provider: SearchProvider) -> String {
    let mut query = trim_query_edges(value);
    let prefixes: &[&str] = match provider {
        SearchProvider::Google => &[
            "su google",
            "on google",
            "google",
            "sul web",
            "su internet",
            "online",
        ],
        SearchProvider::YouTube => &[
            "su youtube",
            "su you tube",
            "on youtube",
            "on you tube",
            "youtube",
            "you tube",
        ],
    };

    loop {
        let before = query.clone();
        for prefix in prefixes {
            if let Some(stripped) = strip_case_insensitive_prefix(&query, prefix) {
                query = trim_query_edges(stripped);
                break;
            }
        }
        if query == before {
            return query;
        }
    }
}

fn normalize_youtube_media_query(value: &str) -> String {
    let query = trim_query_edges(value);

    for (prefix, content_word) in [
        ("una canzone di ", "canzone"),
        ("un brano di ", "brano"),
        ("un video di ", "video"),
        ("a song by ", "song"),
        ("song by ", "song"),
        ("a video by ", "video"),
        ("video by ", "video"),
    ] {
        if let Some(rest) = strip_case_insensitive_prefix(&query, prefix) {
            let rest = trim_query_edges(rest);
            if is_valid_search_query(&rest) {
                return format!("{rest} {content_word}");
            }
        }
    }

    for prefix in [
        "la canzone ",
        "il brano ",
        "la traccia ",
        "canzone ",
        "brano ",
        "song ",
    ] {
        if let Some(rest) = strip_case_insensitive_prefix(&query, prefix) {
            let rest = trim_query_edges(rest);
            if is_valid_search_query(&rest) {
                return rest;
            }
        }
    }

    query
}

fn trim_query_edges(value: &str) -> String {
    value
        .trim()
        .trim_matches(|ch: char| matches!(ch, ',' | '.' | ':' | ';' | '?' | '!' | '"' | '\''))
        .trim()
        .to_string()
}

fn is_valid_search_query(value: &str) -> bool {
    let trimmed = value.trim();
    !trimmed.is_empty()
        && trimmed.chars().any(|ch| ch.is_alphanumeric())
        && !matches!(
            trimmed.to_ascii_lowercase().as_str(),
            "su" | "on" | "youtube" | "you tube" | "google" | "web" | "online"
        )
}

fn strip_wake_prefix(value: &str) -> String {
    let trimmed = value.trim();
    let lower = trimmed.to_lowercase();
    if lower == "astra" {
        return String::new();
    }
    for prefix in ["astra,", "astra ", "astra:", "astrami,", "astrami "] {
        if lower.starts_with(prefix) {
            return trimmed[prefix.len()..].trim_start().to_string();
        }
    }
    trimmed.to_string()
}

fn google_search_url(query: &str) -> String {
    format!(
        "https://www.google.com/search?q={}",
        url_encode_query(query)
    )
}

fn youtube_search_url(query: &str) -> String {
    format!(
        "https://www.youtube.com/results?search_query={}",
        url_encode_query(query)
    )
}

fn infer_youtube_chrome_search_url(lower: &str, original: &str) -> Option<String> {
    let mentions_youtube = lower.contains("youtube") || lower.contains("you tube");
    let mentions_chrome = lower.contains("chrome") || lower.contains("google chrome");
    let asks_search = lower.contains("cerca")
        || lower.contains("cerchi")
        || lower.contains("search")
        || lower.contains("trova");
    if !(mentions_youtube && mentions_chrome && asks_search) {
        return None;
    }

    let query = extract_search_query(original, SearchProvider::YouTube)?;

    Some(format!(
        "https://www.youtube.com/results?search_query={}",
        url_encode_query(&query)
    ))
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

fn strip_case_insensitive_prefix<'a>(value: &'a str, prefix: &str) -> Option<&'a str> {
    let trimmed = value.trim_start();
    let lower = trimmed.to_lowercase();
    lower
        .starts_with(prefix)
        .then(|| trimmed[prefix.len()..].trim_start())
}

fn find_case_insensitive(value: &str, needle: &str) -> Option<usize> {
    value.to_lowercase().find(&needle.to_lowercase())
}

fn url_encode_query(value: &str) -> String {
    value
        .bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            b' ' => "+".to_string(),
            _ => format!("%{:02X}", b),
        })
        .collect::<String>()
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
        || lower.contains("what do you see")
        || lower.contains("what can you see")
        || lower.contains("cosa sto vedendo")
        || lower.contains("cosa vedi")
        || lower.contains("che cosa vedi")
        || lower.contains("cosa c'e sullo schermo")
        || lower.contains("cosa c'è sullo schermo")
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
    use super::{
        build_action_request, extract_search_query, fallback_intent, resolve_action_resolution,
        SearchProvider,
    };
    use crate::action_resolution::{ActionOperation, QueryMode};
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
    fn youtube_chrome_search_fallback_builds_confirmed_launch_action() {
        let original =
            "mi cerchi su youtube la canzone stella stellina, su una finestra nuova di google chrome";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected youtube chrome fallback");

        assert_eq!(intent.kind, SemanticIntentKind::ToolActionRequest);
        assert_eq!(intent.target, CapabilityTarget::DesktopLaunch);
        assert_eq!(intent.action, SemanticAction::DesktopLaunchApp);

        let action = build_action_request(&intent, original).expect("expected launch action");
        assert_eq!(action.tool_name, "desktop.launch_app");
        let args = action
            .params
            .get("args")
            .and_then(|value| value.as_array())
            .expect("args");
        assert_eq!(args[0], json!("--new-window"));
        assert!(args[1]
            .as_str()
            .unwrap()
            .contains("youtube.com/results?search_query=stella+stellina"));
    }

    #[test]
    fn google_search_fallback_handles_italian_wake_prefix() {
        let original = "astra mi cerchi su google coca cola";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected google search fallback");
        let action = build_action_request(&intent, original).expect("expected browser search");

        assert_eq!(action.tool_name, "browser.search");
        assert_eq!(
            action.params.get("query").and_then(|value| value.as_str()),
            Some("coca cola")
        );
    }

    #[test]
    fn google_search_fallback_handles_noisy_stt_sirchi() {
        let original = "astra, mi sirchi su google Coca-Cola";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected noisy search fallback");
        let action = build_action_request(&intent, original).expect("expected browser search");

        assert_eq!(action.tool_name, "browser.search");
        assert_eq!(
            action.params.get("query").and_then(|value| value.as_str()),
            Some("Coca-Cola")
        );
    }

    #[test]
    fn open_google_and_search_fallback_executes_search() {
        let original = "apri google e cerca coca cola";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected google search fallback");
        let action = build_action_request(&intent, original).expect("expected browser search");

        assert_eq!(action.tool_name, "browser.search");
        assert_eq!(
            action.params.get("query").and_then(|value| value.as_str()),
            Some("coca cola")
        );
    }

    #[test]
    fn search_web_suffix_fallback_executes_search() {
        let original = "cercami coca cola sul web";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected web search fallback");
        let action = build_action_request(&intent, original).expect("expected browser search");

        assert_eq!(action.tool_name, "browser.search");
        assert_eq!(
            action.params.get("query").and_then(|value| value.as_str()),
            Some("coca cola")
        );
    }

    #[test]
    fn youtube_search_without_chrome_opens_youtube_results() {
        let original = "cerca su youtube stella stellina";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected youtube search fallback");
        let action = build_action_request(&intent, original).expect("expected browser open");

        assert_eq!(action.tool_name, "browser.open");
        assert!(action
            .params
            .get("url")
            .and_then(|value| value.as_str())
            .unwrap()
            .contains("youtube.com/results?search_query=stella+stellina"));
    }

    #[test]
    fn youtube_search_extracts_query_before_domain_without_punctuation() {
        let original = "Astra, mi cerchi una canzone di Shiva su YouTube?";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected youtube search fallback");
        let action = build_action_request(&intent, original).expect("expected browser open");
        let url = action
            .params
            .get("url")
            .and_then(|value| value.as_str())
            .expect("url");

        assert_eq!(action.tool_name, "browser.open");
        assert!(url.contains("youtube.com/results?search_query="));
        assert!(url.contains("Shiva+canzone"));
        assert!(!url.contains("search_query=%3F"));
    }

    #[test]
    fn youtube_search_keeps_noisy_stt_entity_as_useful_query() {
        let original = "mi cerchi una canzone di sciva su youtube";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected youtube search fallback");
        let action = build_action_request(&intent, original).expect("expected browser open");
        let url = action
            .params
            .get("url")
            .and_then(|value| value.as_str())
            .expect("url");

        assert_eq!(action.tool_name, "browser.open");
        assert!(url.contains("youtube.com/results?search_query=sciva+canzone"));
        assert!(!url.contains("search_query=%3F"));
    }

    #[test]
    fn youtube_search_direct_query_before_domain() {
        let original = "cercami shiva su youtube";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected youtube search fallback");
        let action = build_action_request(&intent, original).expect("expected browser open");
        let url = action
            .params
            .get("url")
            .and_then(|value| value.as_str())
            .expect("url");

        assert_eq!(action.tool_name, "browser.open");
        assert!(url.contains("youtube.com/results?search_query=shiva"));
    }

    #[test]
    fn youtube_search_wrapped_with_greeting_extracts_clean_query() {
        let original = "ciao, cercami shiva su youtube";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected youtube search fallback");
        let action = build_action_request(&intent, original).expect("expected browser open");
        let url = action
            .params
            .get("url")
            .and_then(|value| value.as_str())
            .expect("url");

        assert_eq!(action.tool_name, "browser.open");
        assert!(url.contains("youtube.com/results?search_query=shiva"));
        assert!(!url.contains("ciao"));
        assert!(!url.contains("cercami"));
    }

    #[test]
    fn youtube_search_wrapped_with_polite_prefix_extracts_clean_query() {
        let original = "per favore cercami shiva su youtube";
        let query =
            extract_search_query(original, SearchProvider::YouTube).expect("expected query");

        assert_eq!(query, "shiva");
    }

    #[test]
    fn youtube_search_wrapped_with_greeting_and_imperative_extracts_clean_query() {
        let original = "ehi, cerca shiva su youtube";
        let query =
            extract_search_query(original, SearchProvider::YouTube).expect("expected query");

        assert_eq!(query, "shiva");
    }

    #[test]
    fn youtube_search_mi_cerchi_phrase_still_extracts_clean_query() {
        let original = "mi cerchi shiva su youtube?";
        let query =
            extract_search_query(original, SearchProvider::YouTube).expect("expected query");

        assert_eq!(query, "shiva");
    }

    #[test]
    fn youtube_search_after_open_youtube_and_search() {
        let original = "apri youtube e cerca una canzone di shiva";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected youtube search fallback");
        let action = build_action_request(&intent, original).expect("expected browser open");
        let url = action
            .params
            .get("url")
            .and_then(|value| value.as_str())
            .expect("url");

        assert_eq!(action.tool_name, "browser.open");
        assert!(url.contains("youtube.com/results?search_query=shiva+canzone"));
    }

    #[test]
    fn youtube_search_brano_phrase_keeps_media_context() {
        let original = "cercami su youtube un brano di shiva";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected youtube search fallback");
        let action = build_action_request(&intent, original).expect("expected browser open");
        let url = action
            .params
            .get("url")
            .and_then(|value| value.as_str())
            .expect("url");

        assert_eq!(action.tool_name, "browser.open");
        assert!(url.contains("youtube.com/results?search_query=shiva+brano"));
    }

    #[test]
    fn youtube_search_mi_trovi_phrase_uses_query_before_domain() {
        let original = "mi trovi una canzone di shiva su youtube";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected youtube search fallback");
        let action = build_action_request(&intent, original).expect("expected browser open");
        let url = action
            .params
            .get("url")
            .and_then(|value| value.as_str())
            .expect("url");

        assert_eq!(action.tool_name, "browser.open");
        assert!(url.contains("youtube.com/results?search_query=shiva+canzone"));
    }

    #[test]
    fn file_read_resolution_builds_filesystem_read_action() {
        let original = "leggimi il contenuto del file test.txt sul desktop";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected file read fallback");
        let resolution =
            resolve_action_resolution(&intent, original).expect("expected typed resolution");
        let action =
            super::build_action_request_from_resolution(&resolution, original).expect("action");

        assert_eq!(resolution.operation, ActionOperation::ReadFile);
        assert_eq!(action.tool_name, "filesystem.read_text");
        assert!(action
            .params
            .get("path")
            .and_then(|value| value.as_str())
            .unwrap()
            .ends_with("test.txt"));
    }

    #[test]
    fn file_summary_resolution_uses_read_tool_with_summary_post_processing() {
        let original = "fammi un riassunto del contenuto del file test.txt sul desktop";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected file summary fallback");
        let resolution =
            resolve_action_resolution(&intent, original).expect("expected typed resolution");
        let action =
            super::build_action_request_from_resolution(&resolution, original).expect("action");

        assert_eq!(resolution.operation, ActionOperation::ReadAndSummarizeFile);
        assert_eq!(intent.action, SemanticAction::FilesystemRead);
        assert_eq!(action.tool_name, "filesystem.read_text");
        assert_eq!(
            action
                .params
                .get("post_processing")
                .and_then(|value| value.get("mode"))
                .and_then(|value| value.as_str()),
            Some("summary")
        );
    }

    #[test]
    fn semantic_youtube_search_resolution_uses_semantic_query_mode() {
        let original = "cercami una canzone di Shiva su YouTube";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected youtube search fallback");
        let resolution =
            resolve_action_resolution(&intent, original).expect("expected typed resolution");
        let action =
            super::build_action_request_from_resolution(&resolution, original).expect("action");
        let url = action
            .params
            .get("url")
            .and_then(|value| value.as_str())
            .expect("url");

        assert_eq!(resolution.operation, ActionOperation::BrowserSearch);
        assert_eq!(resolution.query_mode, Some(QueryMode::Semantic));
        assert_eq!(resolution.provider.as_deref(), Some("youtube"));
        assert!(url.contains("youtube.com/results?search_query=Shiva+canzone"));
    }

    #[test]
    fn precise_youtube_search_resolution_preserves_precise_mode() {
        let original = "cerca su youtube stella stellina";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected youtube search fallback");
        let resolution =
            resolve_action_resolution(&intent, original).expect("expected typed resolution");

        assert_eq!(resolution.operation, ActionOperation::BrowserSearch);
        assert_eq!(resolution.query_mode, Some(QueryMode::Precise));
    }

    #[test]
    fn screen_guided_browser_workflow_is_represented_not_executed_as_simple_search() {
        let original = "vai sulla tab che ho aperta di google chrome dedicata a youtube, cercami una canzone di shiva e aprimi il primo risultato";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected workflow fallback");
        let resolution =
            resolve_action_resolution(&intent, original).expect("expected typed resolution");

        assert_eq!(
            resolution.operation,
            ActionOperation::ScreenGuidedBrowserWorkflow
        );
        assert!(resolution.requires_screen_context);
        assert!(!resolution.workflow_steps.is_empty());
    }

    #[test]
    fn chrome_new_window_google_search_uses_desktop_launch() {
        let original = "apri chrome in una nuova finestra e cerca coca cola";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected chrome search fallback");
        let action = build_action_request(&intent, original).expect("expected chrome launch");

        assert_eq!(action.tool_name, "desktop.launch_app");
        let args = action
            .params
            .get("args")
            .and_then(|value| value.as_array())
            .expect("args");
        assert_eq!(args[0], json!("--new-window"));
        assert!(args[1]
            .as_str()
            .unwrap()
            .contains("google.com/search?q=coca+cola"));
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
    fn short_italian_screen_question_uses_screen_analysis_fallback() {
        let original = "cosa vedi?";
        let lower = original.to_lowercase();
        let intent = fallback_intent(&lower, original).expect("expected screen analysis fallback");

        assert_eq!(intent.kind, SemanticIntentKind::ScreenAnalysisRequest);
        assert_eq!(intent.target, CapabilityTarget::Screen);
        assert_eq!(intent.action, SemanticAction::None);
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
