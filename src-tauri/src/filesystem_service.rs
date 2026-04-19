use serde_json::json;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone)]
pub struct FilesystemService {
    allowed_roots: Vec<PathBuf>,
}

impl FilesystemService {
    pub fn new(allowed_roots: &[String]) -> Self {
        Self {
            allowed_roots: allowed_roots.iter().map(PathBuf::from).collect(),
        }
    }

    pub fn ensure_allowed(&self, path: &Path) -> Result<PathBuf, String> {
        let normalized = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .map_err(|e| e.to_string())?
                .join(path)
        };
        for root in &self.allowed_roots {
            if normalized.starts_with(root) {
                return Ok(normalized);
            }
        }
        Err(format!(
            "Path is outside allowed roots: {}",
            normalized.display()
        ))
    }

    pub fn read_text(&self, path: &str) -> Result<serde_json::Value, String> {
        let resolved = self.ensure_allowed(Path::new(path))?;
        let content = fs::read_to_string(&resolved).map_err(|e| e.to_string())?;
        Ok(json!({"path": resolved.display().to_string(), "content": content}))
    }

    pub fn write_text(
        &self,
        path: &str,
        content: &str,
        mode: &str,
    ) -> Result<serde_json::Value, String> {
        let resolved = self.ensure_allowed(Path::new(path))?;
        if let Some(parent) = resolved.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        match mode {
            "append" => {
                use std::io::Write;
                let mut file = fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&resolved)
                    .map_err(|e| e.to_string())?;
                file.write_all(content.as_bytes())
                    .map_err(|e| e.to_string())?;
            }
            _ => {
                fs::write(&resolved, content).map_err(|e| e.to_string())?;
            }
        }
        Ok(
            json!({"path": resolved.display().to_string(), "bytes_written": content.len(), "mode": mode}),
        )
    }

    pub fn search(
        &self,
        root: Option<&str>,
        pattern: &str,
        max_results: usize,
    ) -> Result<serde_json::Value, String> {
        let root_path = if let Some(root) = root {
            self.ensure_allowed(Path::new(root))?
        } else {
            self.allowed_roots
                .first()
                .cloned()
                .ok_or_else(|| "No allowed roots configured".to_string())?
        };
        let needle = pattern.to_lowercase();
        let mut results = Vec::new();
        self.walk(&root_path, &needle, max_results, &mut results)?;
        Ok(json!({"root": root_path.display().to_string(), "matches": results}))
    }

    fn walk(
        &self,
        root: &Path,
        needle: &str,
        max_results: usize,
        results: &mut Vec<String>,
    ) -> Result<(), String> {
        if results.len() >= max_results {
            return Ok(());
        }
        for entry in fs::read_dir(root).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.is_dir() {
                self.walk(&path, needle, max_results, results)?;
            } else if path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|name| name.to_lowercase().contains(needle))
                .unwrap_or(false)
            {
                results.push(path.display().to_string());
                if results.len() >= max_results {
                    break;
                }
            }
        }
        Ok(())
    }
}
