use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub text: String,
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub vector: Vec<f32>,
    pub model: String,
}

pub trait EmbeddingProvider: Send + Sync {
    fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, String>;
    fn provider_kind(&self) -> &'static str;
    fn default_model(&self) -> String;
    fn dimensions_hint(&self) -> Option<usize> { None }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryEmbeddingProviderKind {
    StableHash,
    Ollama,
}

impl MemoryEmbeddingProviderKind {
    pub fn from_env() -> Self {
        match std::env::var("ASTRA_MEMORY_EMBEDDING_PROVIDER")
            .unwrap_or_else(|_| "stable_hash".to_string())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "ollama" | "ollama_embed" | "ollama_embeddings" => Self::Ollama,
            _ => Self::StableHash,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::StableHash => "stable_hash_local",
            Self::Ollama => "ollama_embeddings",
        }
    }
}

pub fn build_embedding_provider() -> Box<dyn EmbeddingProvider> {
    match MemoryEmbeddingProviderKind::from_env() {
        MemoryEmbeddingProviderKind::Ollama => Box::new(OllamaEmbeddingProvider::from_env()),
        MemoryEmbeddingProviderKind::StableHash => Box::new(StableHashEmbeddingProvider::default()),
    }
}

/// Deterministic local embedding provider used as a safe fallback and as the
/// first governed vector backend. It is intentionally not presented as a
/// semantic model: it gives the Memory Graph a stable vector contract, local
/// indexing, replayability and test coverage while keeping real providers
/// behind the same adapter boundary.
#[derive(Debug, Clone)]
pub struct StableHashEmbeddingProvider {
    dimensions: usize,
    model_name: String,
}

impl Default for StableHashEmbeddingProvider {
    fn default() -> Self {
        Self {
            dimensions: env_usize("ASTRA_MEMORY_EMBEDDING_DIMENSIONS", 384).clamp(64, 4096),
            model_name: std::env::var("ASTRA_MEMORY_EMBEDDING_MODEL")
                .unwrap_or_else(|_| "stable-local-hash-v1".to_string()),
        }
    }
}

impl StableHashEmbeddingProvider {
    pub fn new(dimensions: usize, model_name: impl Into<String>) -> Self {
        Self {
            dimensions: dimensions.clamp(64, 4096),
            model_name: model_name.into(),
        }
    }

    pub fn dimensions(&self) -> usize { self.dimensions }
}

impl EmbeddingProvider for StableHashEmbeddingProvider {
    fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, String> {
        let text = request.text.trim();
        if text.is_empty() {
            return Err("embedding text is empty".into());
        }
        let model = request.model.unwrap_or_else(|| self.model_name.clone());
        Ok(EmbeddingResponse {
            vector: stable_hash_embedding(text, self.dimensions),
            model,
        })
    }

    fn provider_kind(&self) -> &'static str { "stable_hash_local" }
    fn default_model(&self) -> String { self.model_name.clone() }
    fn dimensions_hint(&self) -> Option<usize> { Some(self.dimensions) }
}

/// Real semantic embedding provider backed by Ollama's local HTTP API.
///
/// This provider is deliberately kept behind the same trait as the deterministic
/// fallback so the Memory Graph remains provider-agnostic. Rust still owns the
/// source of truth, validation and persistence; Ollama only produces vectors.
#[derive(Debug, Clone)]
pub struct OllamaEmbeddingProvider {
    endpoint: String,
    model_name: String,
    timeout_secs: u64,
}

impl OllamaEmbeddingProvider {
    pub fn from_env() -> Self {
        let endpoint = std::env::var("ASTRA_MEMORY_EMBEDDING_OLLAMA_ENDPOINT")
            .or_else(|_| std::env::var("OLLAMA_HOST"))
            .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string());
        let model_name = std::env::var("ASTRA_MEMORY_EMBEDDING_MODEL")
            .unwrap_or_else(|_| "nomic-embed-text".to_string());
        let timeout_secs = env_usize("ASTRA_MEMORY_EMBEDDING_TIMEOUT_SECS", 45) as u64;
        Self { endpoint, model_name, timeout_secs }
    }

    fn embeddings_url(&self) -> String {
        format!("{}/api/embeddings", self.endpoint.trim_end_matches('/'))
    }
}

#[derive(Debug, Deserialize)]
struct OllamaEmbeddingsResponse {
    #[serde(default)]
    embedding: Vec<f32>,
}

impl EmbeddingProvider for OllamaEmbeddingProvider {
    fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, String> {
        let text = request.text.trim();
        if text.is_empty() {
            return Err("embedding text is empty".into());
        }
        let model = request.model.unwrap_or_else(|| self.model_name.clone());
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(self.timeout_secs.max(1)))
            .build()
            .map_err(|error| format!("ollama embedding client build failed: {error}"))?;
        let response = client
            .post(self.embeddings_url())
            .json(&serde_json::json!({
                "model": model,
                "prompt": text,
            }))
            .send()
            .map_err(|error| format!("ollama embedding request failed: {error}"))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().unwrap_or_default();
            return Err(format!(
                "ollama embedding request failed with status {status}: {}",
                body.chars().take(600).collect::<String>()
            ));
        }
        let parsed = response
            .json::<OllamaEmbeddingsResponse>()
            .map_err(|error| format!("ollama embedding response parse failed: {error}"))?;
        if parsed.embedding.is_empty() {
            return Err("ollama embedding response contained an empty vector".into());
        }
        Ok(EmbeddingResponse { vector: parsed.embedding, model })
    }

    fn provider_kind(&self) -> &'static str { "ollama_embeddings" }
    fn default_model(&self) -> String { self.model_name.clone() }
}

pub fn stable_hash_embedding(text: &str, dimensions: usize) -> Vec<f32> {
    let dimensions = dimensions.clamp(64, 4096);
    let mut vector = vec![0.0f32; dimensions];
    for token in tokenize_embedding_text(text) {
        let mut hasher = Sha256::new();
        hasher.update(token.as_bytes());
        let digest = hasher.finalize();
        let index = u64::from_le_bytes([
            digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
        ]) as usize % dimensions;
        let sign = if digest[8] % 2 == 0 { 1.0 } else { -1.0 };
        vector[index] += sign;
    }
    normalize_vector(&mut vector);
    vector
}

pub fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.is_empty() || right.is_empty() || left.len() != right.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;
    for (l, r) in left.iter().zip(right.iter()) {
        dot += l * r;
        left_norm += l * l;
        right_norm += r * r;
    }
    if left_norm <= f32::EPSILON || right_norm <= f32::EPSILON {
        return 0.0;
    }
    dot / (left_norm.sqrt() * right_norm.sqrt())
}

fn normalize_vector(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        return;
    }
    for value in vector {
        *value /= norm;
    }
}

fn tokenize_embedding_text(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .map(str::trim)
        .filter(|token| token.chars().count() >= 2)
        .map(|token| token.to_lowercase())
        .take(4096)
        .collect()
}

fn env_usize(key: &str, fallback: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(fallback)
}
