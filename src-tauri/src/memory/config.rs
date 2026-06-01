use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub root: PathBuf,
    pub sqlite_path: PathBuf,
    pub journal_dir: PathBuf,
    pub max_query_limit: usize,
    pub max_activation_depth: usize,
    pub max_activation_nodes: usize,
}

impl MemoryConfig {
    pub fn new(project_root: PathBuf) -> Self {
        let root = project_root.join(".astra").join("memory");
        let graph_dir = root.join("graph");
        let sqlite_path = graph_dir.join("astra_memory.sqlite");
        let journal_dir = root.join("journal");
        Self {
            root,
            sqlite_path,
            journal_dir,
            max_query_limit: env_usize("ASTRA_MEMORY_MAX_QUERY_LIMIT", 50),
            max_activation_depth: env_usize("ASTRA_MEMORY_MAX_ACTIVATION_DEPTH", 3),
            max_activation_nodes: env_usize("ASTRA_MEMORY_MAX_ACTIVATION_NODES", 60),
        }
    }
}

fn env_usize(key: &str, fallback: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(fallback)
}
