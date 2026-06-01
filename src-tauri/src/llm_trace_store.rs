use chrono::Utc;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    sync::{Arc, Mutex},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmTraceLevel {
    Off,
    Metadata,
    Redacted,
    Full,
}

impl LlmTraceLevel {
    pub fn from_env() -> Self {
        match std::env::var("ASTRA_LLM_TRACE_LEVEL")
            .unwrap_or_else(|_| "metadata".to_string())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "off" | "false" | "0" => Self::Off,
            "full" => Self::Full,
            "redacted" => Self::Redacted,
            _ => Self::Metadata,
        }
    }

    fn include_raw_prompt(self) -> bool {
        matches!(self, Self::Redacted | Self::Full)
    }

    fn include_raw_response(self) -> bool {
        matches!(self, Self::Redacted | Self::Full)
    }

    fn redact(self, value: &str) -> String {
        match self {
            Self::Full => value.to_string(),
            Self::Redacted => redact_sensitive_text(value),
            Self::Off | Self::Metadata => String::new(),
        }
    }
}

#[derive(Clone)]
pub struct LlmTraceStore {
    root: PathBuf,
    lock: Arc<Mutex<()>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LlmTraceRecord {
    pub schema_version: u32,
    pub timestamp: String,
    pub request_id: Option<String>,
    pub stage: String,
    pub attempt_kind: String,
    pub model: String,
    pub endpoint_label: Option<String>,
    pub used_json_mode: bool,
    pub duration_ms: Option<u64>,
    pub http_status: Option<u16>,
    pub prompt_char_count: usize,
    pub prompt_hash: String,
    pub response_body_len: Option<usize>,
    pub response_content_len: Option<usize>,
    pub response_hash: Option<String>,
    pub message_present: Option<bool>,
    pub done: Option<bool>,
    pub done_reason: Option<String>,
    pub total_duration: Option<u64>,
    pub load_duration: Option<u64>,
    pub prompt_eval_count: Option<u64>,
    pub prompt_eval_duration: Option<u64>,
    pub eval_count: Option<u64>,
    pub eval_duration: Option<u64>,
    pub parse_result: Option<String>,
    pub failure_class: Option<String>,
    pub repair_attempted: bool,
    pub repair_succeeded: bool,
    pub fallback_kind: Option<String>,
    pub raw_prompt_included: bool,
    pub raw_response_included: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_prompt: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_response: Option<String>,
}

impl LlmTraceStore {
    pub fn new(project_root: PathBuf) -> Self {
        let root = project_root.join(".astra").join("diagnostics").join("llm");
        let _ = fs::create_dir_all(&root);
        Self {
            root,
            lock: Arc::new(Mutex::new(())),
        }
    }

    pub fn append(&self, record: &LlmTraceRecord) {
        if LlmTraceLevel::from_env() == LlmTraceLevel::Off {
            return;
        }
        let _guard = self.lock.lock().expect("llm trace mutex poisoned");
        let day = Utc::now().format("%Y-%m-%d").to_string();
        let stage_dir = self.root.join(&record.stage);
        let _ = fs::create_dir_all(&stage_dir);
        let path = stage_dir.join(format!("{day}.jsonl"));
        if let Ok(line) = serde_json::to_string(record) {
            let _ = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .and_then(|mut file| writeln!(&mut file, "{line}"));
        }
    }
}

pub fn sha256_hex(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
}

pub fn build_trace_prompt_payload(messages: &[Value], level: LlmTraceLevel) -> Option<Value> {
    if !level.include_raw_prompt() {
        return None;
    }
    Some(Value::Array(
        messages
            .iter()
            .map(|message| redact_json_value(message, level))
            .collect(),
    ))
}

pub fn build_trace_response_payload(response: &str, level: LlmTraceLevel) -> Option<String> {
    level
        .include_raw_response()
        .then(|| level.redact(response))
}

fn redact_json_value(value: &Value, level: LlmTraceLevel) -> Value {
    match value {
        Value::String(text) => Value::String(level.redact(text)),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| redact_json_value(item, level))
                .collect(),
        ),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), redact_json_value(value, level)))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn redact_sensitive_text(value: &str) -> String {
    value
        .split_whitespace()
        .map(|token| {
            let lower = token.to_ascii_lowercase();
            if token.contains('@') && token.contains('.') {
                "[redacted-email]".to_string()
            } else if lower.contains("c:\\")
                || lower.contains("/users/")
                || lower.contains("\\users\\")
                || lower.contains("/home/")
            {
                "[redacted-path]".to_string()
            } else {
                token.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
