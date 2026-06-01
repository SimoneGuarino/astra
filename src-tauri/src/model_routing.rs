use reqwest::Client;
use serde_json::{json, Value};
use std::env;

use crate::conversation_history::ConversationMessage;

pub const DEFAULT_OLLAMA_BASE_URL: &str = "http://127.0.0.1:11434";
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

pub async fn resolve_active_ollama_model(message: &str, source: &str) -> String {
    let source_kind = RequestSource::from_source(source);
    let installed_models = fetch_installed_models().await.unwrap_or_default();
    select_model(message, source_kind, &installed_models)
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
        .get(ollama_endpoint("/api/tags"))
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

pub fn resolve_ollama_base_url() -> String {
    let astra_url = env::var("ASTRA_OLLAMA_BASE_URL").ok();
    let ollama_host = env::var("OLLAMA_HOST").ok();
    resolve_ollama_base_url_from(astra_url.as_deref(), ollama_host.as_deref())
}

pub fn resolve_ollama_base_url_from(astra_url: Option<&str>, ollama_host: Option<&str>) -> String {
    astra_url
        .and_then(normalize_ollama_base_url)
        .or_else(|| ollama_host.and_then(normalize_ollama_base_url))
        .unwrap_or_else(|| DEFAULT_OLLAMA_BASE_URL.to_string())
}

pub fn ollama_endpoint(path: &str) -> String {
    format!(
        "{}/{}",
        resolve_ollama_base_url(),
        path.trim_start_matches('/')
    )
}

pub fn sanitize_ollama_endpoint_label(value: &str) -> String {
    let Some(redacted) = strip_url_credentials(value) else {
        return "configured endpoint".to_string();
    };
    if is_local_ollama_endpoint(&redacted) {
        redacted
    } else {
        "configured endpoint".to_string()
    }
}

fn normalize_ollama_base_url(value: &str) -> Option<String> {
    let trimmed = value.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return None;
    }
    let with_scheme = if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    };
    Some(with_scheme.trim_end_matches('/').to_string())
}

fn strip_url_credentials(value: &str) -> Option<String> {
    let trimmed = value.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return None;
    }
    let (scheme, rest) = trimmed.split_once("://")?;
    let host_and_path = rest.rsplit_once('@').map(|(_, host)| host).unwrap_or(rest);
    Some(format!("{scheme}://{host_and_path}"))
}

fn is_local_ollama_endpoint(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    lower.starts_with("http://127.0.0.1")
        || lower.starts_with("http://localhost")
        || lower.starts_with("http://[::1]")
        || lower.starts_with("https://127.0.0.1")
        || lower.starts_with("https://localhost")
        || lower.starts_with("https://[::1]")
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
        "codice",
        "code",
        "bug",
        "debug",
        "refactor",
        "architecture",
        "architettura",
        "ottimizza",
        "optimize",
        "analyze",
        "analizza",
        "implement",
        "implementa",
        "progetta",
        "sviluppa",
        "backend",
        "frontend",
        "typescript",
        "rust",
        "python",
        "go",
        "sql",
        "database",
        "algoritmo",
        "performance",
        "scalabilita",
        "enterprise",
        "spiegami",
        "come funziona",
        "com'e nata",
        "com'è nata",
        "come è nata",
        "come e nata",
        "perche",
        "perché",
        "origine",
        "storia",
        "processo",
        "cosa c'era",
        "come si forma",
        "come nasce",
        "approfondisci",
        "nel dettaglio",
        "spiega nel dettaglio",
        "analisi",
        "design",
        "tradeoff",
        "trade-off",
    ];

    message.chars().count() > 180 || keywords.iter().any(|keyword| lower.contains(keyword))
}

fn build_system_prompt(
    source: RequestSource,
    message: &str,
    assistant_context: Option<&str>,
) -> String {
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
                prompt.push_str(
                    "

",
                );
                prompt.push_str(context);
            }
            prompt
        }
        RequestSource::Typed => {
            let detail = if reasoning {
                "Quando il tema è tecnico o complesso, ragiona bene, dai una risposta solida e strutturata, con trade-off chiari."
            } else {
                "Per richieste semplici, sii diretta, fluida e naturale, ma comunque completa."
            };
            let mut prompt = format!(
                "Sei Astra, un'assistente AI locale molto competente. Rispondi in italiano naturale, chiaro e professionale. Per chat scritta, dai risposte utili e complete di default. Sii breve solo se l'utente chiede esplicitamente una risposta breve o un riassunto. Se usi elenchi numerati, completa tutti i punti introdotti e non fermarti mai su un numero nudo. Evita ripetizioni e tono robotico. Se utile, usa una struttura leggibile. {detail}"
            );
            if let Some(context) = assistant_context.filter(|value| !value.trim().is_empty()) {
                prompt.push_str(
                    "

",
                );
                prompt.push_str(context);
            }
            prompt
        }
    }
}

fn build_messages(
    system_prompt: &str,
    history: &[ConversationMessage],
    message: &str,
) -> Vec<Value> {
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
            "num_predict": if reasoning {
                num_predict_from_env("ASTRA_VOICE_REASONING_NUM_PREDICT", 260, 160, 800)
            } else {
                num_predict_from_env("ASTRA_VOICE_NUM_PREDICT", 140, 80, 600)
            },
        }),
        RequestSource::Typed => json!({
            "temperature": if reasoning { 0.28 } else { 0.42 },
            "top_p": 0.9,
            "repeat_penalty": 1.07,
            "num_predict": if reasoning {
                num_predict_from_env("ASTRA_TYPED_REASONING_NUM_PREDICT", 1400, 1200, 2400)
            } else {
                num_predict_from_env("ASTRA_TYPED_NUM_PREDICT", 800, 600, 1800)
            },
        }),
    }
}

fn num_predict_from_env(key: &str, default: i64, min: i64, max: i64) -> i64 {
    num_predict_from_override(env::var(key).ok().as_deref(), default, min, max)
}

fn num_predict_from_override(value: Option<&str>, default: i64, min: i64, max: i64) -> i64 {
    value
        .and_then(|value| value.trim().parse::<i64>().ok())
        .unwrap_or(default)
        .clamp(min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_astra_ollama_base_url_first() {
        assert_eq!(
            resolve_ollama_base_url_from(
                Some("http://localhost:11435/"),
                Some("http://localhost:11436")
            ),
            "http://localhost:11435"
        );
    }

    #[test]
    fn resolves_ollama_host_second() {
        assert_eq!(
            resolve_ollama_base_url_from(None, Some("localhost:11436/")),
            "http://localhost:11436"
        );
    }

    #[test]
    fn resolves_default_ollama_base_url() {
        assert_eq!(
            resolve_ollama_base_url_from(None, None),
            DEFAULT_OLLAMA_BASE_URL
        );
    }

    #[test]
    fn sanitizes_local_endpoint_and_strips_credentials() {
        assert_eq!(
            sanitize_ollama_endpoint_label("http://user:secret@localhost:11434/"),
            "http://localhost:11434"
        );
    }

    #[test]
    fn hides_non_local_endpoint_label() {
        assert_eq!(
            sanitize_ollama_endpoint_label("https://user:secret@example.com/ollama"),
            "configured endpoint"
        );
    }

    #[test]
    fn reasoning_detection_catches_conceptual_questions() {
        assert!(looks_reasoning_heavy("spiegami come è nata la Terra"));
        assert!(looks_reasoning_heavy(
            "prima dell'impatto iniziale cosa c'era?"
        ));
        assert!(looks_reasoning_heavy("come funziona la gravità?"));
    }

    #[test]
    fn typed_num_predict_defaults_are_not_too_small() {
        let ordinary = build_options(RequestSource::Typed, "chi sei?");
        let reasoning = build_options(RequestSource::Typed, "spiegami come è nata la Terra");

        assert!(ordinary["num_predict"].as_i64().unwrap() >= 600);
        assert!(reasoning["num_predict"].as_i64().unwrap() >= 1200);
    }

    #[test]
    fn num_predict_overrides_are_clamped() {
        assert_eq!(num_predict_from_override(Some("50"), 800, 600, 1800), 600);
        assert_eq!(
            num_predict_from_override(Some("9999"), 800, 600, 1800),
            1800
        );
        assert_eq!(num_predict_from_override(Some("900"), 800, 600, 1800), 900);
        assert_eq!(num_predict_from_override(Some("bad"), 800, 600, 1800), 800);
    }
}
