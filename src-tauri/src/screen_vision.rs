use crate::desktop_agent_types::{ScreenAnalysisResult, VisionAvailability};
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
        Self { client: Client::new() }
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

        let response = self.client
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
        })
    }

    async fn fetch_installed_models(&self) -> Result<Vec<String>, String> {
        let response = self.client
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
        .unwrap_or_else(|_| "qwen2.5vl:7b,qwen2.5vl:3b,llava:7b,llava:13b,llava-phi3:3.8b,moondream:latest".to_string())
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
