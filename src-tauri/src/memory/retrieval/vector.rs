use crate::memory::types::MemoryNodeKind;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryVectorHit {
    pub node_id: String,
    pub chunk_id: String,
    pub score: f32,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryVectorFilter {
    #[serde(default)]
    pub node_kinds: Vec<MemoryNodeKind>,
    #[serde(default)]
    pub min_confidence: Option<f32>,
}

pub trait VectorMemoryIndex: Send + Sync {
    fn upsert_embedding(
        &self,
        _node_id: &str,
        _chunk_id: &str,
        _embedding: &[f32],
        _model: &str,
    ) -> Result<(), String>;
    fn search(
        &self,
        _query_embedding: &[f32],
        _filter: MemoryVectorFilter,
        _limit: usize,
    ) -> Result<Vec<MemoryVectorHit>, String>;
    fn delete_node(&self, _node_id: &str) -> Result<(), String>;
}

#[derive(Debug, Default)]
pub struct NoopVectorMemoryIndex;

impl VectorMemoryIndex for NoopVectorMemoryIndex {
    fn upsert_embedding(
        &self,
        _node_id: &str,
        _chunk_id: &str,
        _embedding: &[f32],
        _model: &str,
    ) -> Result<(), String> {
        Ok(())
    }

    fn search(
        &self,
        _query_embedding: &[f32],
        _filter: MemoryVectorFilter,
        _limit: usize,
    ) -> Result<Vec<MemoryVectorHit>, String> {
        Ok(Vec::new())
    }

    fn delete_node(&self, _node_id: &str) -> Result<(), String> {
        Ok(())
    }
}
