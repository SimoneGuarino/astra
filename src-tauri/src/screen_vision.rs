use crate::{
    desktop_agent_types::{
        PageEvidenceSource, PageSemanticEvidence, ScreenAnalysisResult, VisionAvailability,
    },
    structured_vision::{parse_structured_vision_candidates, StructuredVisionExtraction},
};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use reqwest::Client;
use serde_json::{json, Value};
use std::{env, fs};
use uuid::Uuid;

const OLLAMA_BASE_URL: &str = "http://127.0.0.1:11434";

#[derive(Clone)]
pub struct ScreenVisionRuntime {
    client: Client,
}

impl ScreenVisionRuntime {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }

    pub async fn availability(&self) -> VisionAvailability {
        let installed_models = self.fetch_installed_models().await.unwrap_or_default();
        let candidates = vision_candidates();
        let selected_model = select_vision_model(&installed_models);
        VisionAvailability {
            available: selected_model.is_some(),
            selected_model,
            candidates,
        }
    }

    pub async fn analyze(
        &self,
        image_path: &str,
        captured_at: u64,
        provider: &str,
        question: Option<String>,
    ) -> Result<ScreenAnalysisResult, String> {
        let installed_models = self.fetch_installed_models().await.unwrap_or_default();
        let model = select_vision_model(&installed_models)
            .ok_or_else(|| "No compatible Ollama vision model found. Install one of: qwen2.5vl, llava, llava-phi3, moondream".to_string())?;

        let image_bytes = fs::read(image_path)
            .map_err(|error| format!("screen analysis failed to read capture: {error}"))?;
        let image_b64 = BASE64_STANDARD.encode(image_bytes);

        let question = question
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "Describe what is visible on this screen, identify the active context, the main UI regions, and any likely issue or next action for the user.".to_string());

        let system = "You are Astra Vision, the screen-awareness subsystem of a local desktop assistant. Analyze screenshots conservatively and precisely. Describe only what is visible or strongly inferable. When helpful, structure the answer in short sections: context, notable elements, possible issue, next step. If the screen looks like code, IDE, terminal, browser, settings, or an error state, say so clearly. Do not invent hidden content. Keep the response concise but useful.";

        let payload = json!({
            "model": model,
            "stream": false,
            "options": {
                "temperature": 0.2,
                "num_ctx": 8192,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": question, "images": [image_b64]}
            ]
        });

        let response = self
            .client
            .post(format!("{OLLAMA_BASE_URL}/api/chat"))
            .json(&payload)
            .send()
            .await
            .map_err(|error| format!("Ollama vision request failed: {error}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!("Ollama vision HTTP error {status}: {body}"));
        }

        let body: Value = response
            .json()
            .await
            .map_err(|error| format!("Ollama vision parse failed: {error}"))?;

        let answer = body
            .get("message")
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "Ollama vision returned an empty response".to_string())?
            .to_string();

        Ok(ScreenAnalysisResult {
            analysis_id: Uuid::new_v4().to_string(),
            request_id: Uuid::new_v4().to_string(),
            captured_at,
            image_path: image_path.to_string(),
            model,
            provider: provider.to_string(),
            question,
            answer,
            ui_candidates: Vec::new(),
            structured_candidates_error: None,
        })
    }

    pub async fn extract_ui_candidates(
        &self,
        image_path: &str,
        captured_at: u64,
        provider: &str,
        requested_roles: &[String],
        app_hint: Option<&str>,
        provider_hint: Option<&str>,
    ) -> Result<StructuredVisionExtraction, String> {
        let installed_models = self.fetch_installed_models().await.unwrap_or_default();
        let model = select_vision_model(&installed_models)
            .ok_or_else(|| "No compatible Ollama vision model found. Install one of: qwen2.5vl, llava, llava-phi3, moondream".to_string())?;

        let image_bytes = fs::read(image_path).map_err(|error| {
            format!("structured screen candidate extraction failed to read capture: {error}")
        })?;
        let image_b64 = BASE64_STANDARD.encode(image_bytes);
        let requested_roles = if requested_roles.is_empty() {
            "search_input, ranked_result, button, link, text_input".to_string()
        } else {
            requested_roles.join(", ")
        };
        let app_hint = app_hint.unwrap_or("unknown");
        let provider_hint = provider_hint.unwrap_or("unknown");

        let system = concat!(
            "You are Astra Vision Target Grounding. Return only strict JSON, no markdown. ",
            "Your task is to propose visible UI element candidates for safe desktop automation. ",
            "Only include an element if you can provide an explicit rectangular region in screen pixel coordinates. ",
            "Use coordinate_space=\"screen\". Do not infer hidden elements. Do not include vague prose-only candidates. ",
            "Allowed roles: search_input, ranked_result, button, link, text_input. ",
            "Each candidate must include candidate_id, role, region {x,y,width,height,coordinate_space}, confidence 0..1, ",
            "supports_focus, supports_click, optional label, optional browser_app_hint/app_hint, optional content_provider_hint/provider_hint, optional page_kind_hint, optional rank, and rationale. ",
            "Also include optional page_evidence with browser_app_hint, content_provider_hint, page_kind_hint, query_hint, result_list_visible, confidence, and uncertainty. ",
            "content_provider_hint/provider_hint means the visible site or content provider such as youtube, google, github, or amazon. ",
            "Never copy observation backend or screenshot backend identifiers into provider_hint/content_provider_hint. ",
            "If uncertain, return {\"page_evidence\":{\"confidence\":0.0,\"uncertainty\":[\"inconclusive\"]},\"ui_candidates\":[]}."
        );
        let user = format!(
            "Extract UI candidates for requested roles: {requested_roles}. Browser/app hint: {app_hint}. Expected content provider hint: {provider_hint}. Observation backend: {provider} (technical capture metadata only; it is not the website/provider). Return JSON with keys page_evidence and ui_candidates."
        );

        let payload = json!({
            "model": model,
            "stream": false,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "num_ctx": 8192,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user, "images": [image_b64]}
            ]
        });

        let response = self
            .client
            .post(format!("{OLLAMA_BASE_URL}/api/chat"))
            .json(&payload)
            .send()
            .await
            .map_err(|error| format!("Ollama structured vision request failed: {error}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!(
                "Ollama structured vision HTTP error {status}: {body}"
            ));
        }

        let body: Value = response
            .json()
            .await
            .map_err(|error| format!("Ollama structured vision parse failed: {error}"))?;
        let content = body
            .get("message")
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "Ollama structured vision returned an empty response".to_string())?;

        let mut extraction = parse_structured_vision_candidates(content, captured_at)?;
        let page_evidence = extraction
            .page_evidence
            .get_or_insert_with(|| PageSemanticEvidence {
                browser_app_hint: None,
                content_provider_hint: None,
                page_kind_hint: None,
                query_hint: None,
                result_list_visible: None,
                confidence: 0.0,
                evidence_sources: vec![PageEvidenceSource::CaptureMetadata],
                capture_backend: Some(provider.to_string()),
                observation_source: Some("structured_vision".into()),
                uncertainty: vec!["vision_model_returned_no_page_evidence".into()],
            });
        page_evidence.capture_backend = page_evidence
            .capture_backend
            .clone()
            .or_else(|| Some(provider.to_string()));
        page_evidence.observation_source = page_evidence
            .observation_source
            .clone()
            .or_else(|| Some("structured_vision".into()));
        Ok(extraction)
    }

    async fn fetch_installed_models(&self) -> Result<Vec<String>, String> {
        let response = self
            .client
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
}

fn vision_candidates() -> Vec<String> {
    env::var("ASTRA_MODEL_VISION_CANDIDATES")
        .unwrap_or_else(|_| {
            "qwen2.5vl:7b,qwen2.5vl:3b,llava:7b,llava:13b,llava-phi3:3.8b,moondream:latest"
                .to_string()
        })
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>()
}

fn select_vision_model(installed_models: &[String]) -> Option<String> {
    let candidates = vision_candidates();

    if installed_models.is_empty() {
        return candidates.first().cloned();
    }

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
