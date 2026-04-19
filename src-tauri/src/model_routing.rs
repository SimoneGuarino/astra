use reqwest::Client;
use serde_json::{json, Value};
use std::env;

use crate::conversation_history::ConversationMessage;

const OLLAMA_BASE_URL: &str = "http://127.0.0.1:11434";
const DEFAULT_HISTORY_MESSAGES: usize = 10;

#[derive(Debug, Clone)]
pub struct ResolvedOllamaRequest {
    pub model: String,
    pub system_prompt: String,
    pub messages: Vec<Value>,
    pub options: Value,
}

pub async fn resolve_ollama_request(
    message: &str,
    source: &str,
    history: &[ConversationMessage],
    assistant_context: Option<&str>,
) -> Result<ResolvedOllamaRequest, String> {
    let source_kind = RequestSource::from_source(source);
    let installed_models = fetch_installed_models().await.unwrap_or_default();
    let model = select_model(message, source_kind, &installed_models);
    let system_prompt = build_system_prompt(source_kind, message, assistant_context);
    let messages = build_messages(&system_prompt, history, message);
    let options = build_options(source_kind, message);

    Ok(ResolvedOllamaRequest {
        model,
        system_prompt,
        messages,
        options,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestSource {
    Typed,
    Voice,
}

impl RequestSource {
    fn from_source(source: &str) -> Self {
        match source {
            "voice_session" => Self::Voice,
            _ => Self::Typed,
        }
    }
}

async fn fetch_installed_models() -> Result<Vec<String>, String> {
    let client = Client::new();
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

    let installed = body
        .get("models")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.get("name").and_then(Value::as_str))
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();

    Ok(installed)
}

fn select_model(message: &str, source: RequestSource, installed_models: &[String]) -> String {
    let reasoning = looks_reasoning_heavy(message);
    let candidates = match (source, reasoning) {
        (RequestSource::Voice, true) => env_candidates(
            "ASTRA_MODEL_VOICE_REASONING_CANDIDATES",
            "gpt-oss:20b,qwen3:30b,qwen3:32b,llama3.3:70b,qwen3:14b,qwen3:8b",
        ),
        (RequestSource::Voice, false) => env_candidates(
            "ASTRA_MODEL_VOICE_CANDIDATES",
            "gpt-oss:20b,qwen3:14b,qwen3:8b,llama3.1:8b",
        ),
        (RequestSource::Typed, true) => env_candidates(
            "ASTRA_MODEL_REASONING_CANDIDATES",
            "gpt-oss:20b,qwen3:30b,qwen3:32b,llama3.3:70b,qwen3:14b",
        ),
        (RequestSource::Typed, false) => env_candidates(
            "ASTRA_MODEL_CHAT_CANDIDATES",
            "gpt-oss:20b,qwen3:14b,qwen3:8b,llama3.1:8b",
        ),
    };

    select_first_available(&candidates, installed_models)
        .or_else(|| candidates.first().cloned())
        .unwrap_or_else(|| "gpt-oss:20b".to_string())
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

fn env_candidates(key: &str, fallback: &str) -> Vec<String> {
    env::var(key)
        .unwrap_or_else(|_| fallback.to_string())
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn looks_reasoning_heavy(message: &str) -> bool {
    let lower = message.to_lowercase();
    let keywords = [
        "codice", "code", "bug", "debug", "refactor", "architecture", "architettura",
        "ottimizza", "optimize", "analyze", "analizza", "implement", "implementa",
        "progetta", "sviluppa", "backend", "frontend", "typescript", "rust", "python",
        "go", "sql", "database", "algoritmo", "performance", "scalabilita", "enterprise",
        "spiega nel dettaglio", "analisi", "design", "tradeoff", "trade-off",
    ];

    message.chars().count() > 180 || keywords.iter().any(|keyword| lower.contains(keyword))
}

fn build_system_prompt(source: RequestSource, message: &str, assistant_context: Option<&str>) -> String {
    let reasoning = looks_reasoning_heavy(message);
    match source {
        RequestSource::Voice => {
            let brevity = if reasoning {
                "Se il tema è tecnico, resta accurata ma usa frasi corte e facili da ascoltare."
            } else {
                "Per richieste semplici, rispondi in 1-3 frasi brevi."
            };
            let mut prompt = format!(
                "Sei Astra, un'assistente AI locale. Devi parlare in italiano molto naturale, caldo, rapido e conversazionale, con una voce percepita simile a un assistente premium. Non usare markdown. Non usare elenchi puntati salvo richiesta esplicita. Evita meta-commenti, ripetizioni, filler inutili e spiegazioni prolisse. Usa aperture brevi solo quando aiutano il ritmo, per esempio: 'Certo,', 'Sì,', 'Va bene,'. Se serve fare una domanda, fanne una sola e molto breve. {brevity} Mantieni precisione tecnica quando serve, ma con resa orale pulita."
            );
            if let Some(context) = assistant_context.filter(|value| !value.trim().is_empty()) {
                prompt.push_str("

");
                prompt.push_str(context);
            }
            prompt
        }
        RequestSource::Typed => {
            let detail = if reasoning {
                "Quando il tema è tecnico o complesso, ragiona bene, dai una risposta solida e strutturata, con trade-off chiari."
            } else {
                "Per richieste semplici, sii diretta, fluida e naturale."
            };
            let mut prompt = format!(
                "Sei Astra, un'assistente AI locale molto competente. Rispondi in italiano naturale, chiaro e professionale. Evita ripetizioni e tono robotico. Se utile, usa una struttura leggibile, ma senza gonfiare la risposta. {detail}"
            );
            if let Some(context) = assistant_context.filter(|value| !value.trim().is_empty()) {
                prompt.push_str("

");
                prompt.push_str(context);
            }
            prompt
        }
    }
}

fn build_messages(system_prompt: &str, history: &[ConversationMessage], message: &str) -> Vec<Value> {
    let mut messages = Vec::with_capacity(history.len() + 2);
    messages.push(json!({"role": "system", "content": system_prompt}));

    let history_len = env::var("ASTRA_HISTORY_MESSAGES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_HISTORY_MESSAGES);

    let start = history.len().saturating_sub(history_len);
    for item in history.iter().skip(start) {
        messages.push(json!({
            "role": item.role,
            "content": item.content,
        }));
    }

    messages.push(json!({"role": "user", "content": message}));
    messages
}

fn build_options(source: RequestSource, message: &str) -> Value {
    let reasoning = looks_reasoning_heavy(message);
    match source {
        RequestSource::Voice => json!({
            "temperature": if reasoning { 0.45 } else { 0.62 },
            "top_p": 0.9,
            "repeat_penalty": 1.08,
            "num_predict": if reasoning { 220 } else { 120 },
        }),
        RequestSource::Typed => json!({
            "temperature": if reasoning { 0.28 } else { 0.42 },
            "top_p": 0.9,
            "repeat_penalty": 1.07,
            "num_predict": if reasoning { 900 } else { 360 },
        }),
    }
}
