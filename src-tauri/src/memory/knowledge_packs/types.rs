use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgePackBuildRequest {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub dry_run: bool,
    #[serde(default = "default_snapshot_limit")]
    pub snapshot_limit: usize,
    #[serde(default = "default_max_packs")]
    pub max_packs: usize,
    #[serde(default = "default_max_nodes_per_pack")]
    pub max_nodes_per_pack: usize,
    #[serde(default = "default_min_nodes_per_pack")]
    pub min_nodes_per_pack: usize,
    #[serde(default = "default_max_pack_content_members")]
    pub max_pack_content_members: usize,
    #[serde(default = "default_min_pack_score")]
    pub min_pack_score: f32,
    #[serde(default = "default_true")]
    pub persist_packs: bool,
    #[serde(default)]
    pub include_unverified: bool,
    #[serde(default = "default_true")]
    pub include_low_confidence: bool,
    #[serde(default)]
    pub include_source_documents: bool,
    #[serde(default)]
    pub seed_domains: Vec<String>,
    #[serde(default)]
    pub blocked_domains: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
}

fn default_true() -> bool { true }
fn default_snapshot_limit() -> usize { 500 }
fn default_max_packs() -> usize { 12 }
fn default_max_nodes_per_pack() -> usize { 32 }
fn default_min_nodes_per_pack() -> usize { 3 }
fn default_max_pack_content_members() -> usize { 18 }
fn default_min_pack_score() -> f32 { 0.34 }

impl Default for KnowledgePackBuildRequest {
    fn default() -> Self {
        Self {
            enabled: true,
            dry_run: false,
            snapshot_limit: default_snapshot_limit(),
            max_packs: default_max_packs(),
            max_nodes_per_pack: default_max_nodes_per_pack(),
            min_nodes_per_pack: default_min_nodes_per_pack(),
            max_pack_content_members: default_max_pack_content_members(),
            min_pack_score: default_min_pack_score(),
            persist_packs: true,
            include_unverified: false,
            include_low_confidence: true,
            include_source_documents: false,
            seed_domains: Vec::new(),
            blocked_domains: Vec::new(),
            metadata: Value::Null,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgePackKindCount {
    pub kind: String,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgePackMember {
    pub node_id: String,
    pub title: String,
    pub kind: String,
    pub confidence: f32,
    pub salience: f32,
    pub verification_status: String,
    pub score: f32,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub signals: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgePackSummary {
    pub domain_slug: String,
    pub title: String,
    pub summary: String,
    pub content: String,
    pub score: f32,
    pub confidence: f32,
    pub salience: f32,
    pub member_count: usize,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub kind_counts: Vec<KnowledgePackKindCount>,
    #[serde(default)]
    pub members: Vec<KnowledgePackMember>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgePackBuildReceipt {
    pub accepted: bool,
    pub reason: String,
    pub started_at: i64,
    pub completed_at: i64,
    pub dry_run: bool,
    pub snapshot_nodes: usize,
    pub packs_built: usize,
    pub packs_persisted: usize,
    #[serde(default)]
    pub created_node_ids: Vec<String>,
    #[serde(default)]
    pub created_edge_ids: Vec<String>,
    #[serde(default)]
    pub packs: Vec<KnowledgePackSummary>,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub recommendations: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
}
