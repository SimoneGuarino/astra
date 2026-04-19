use serde_json::json;
use std::{path::{Path, PathBuf}, process::Command};

#[derive(Clone)]
pub struct TerminalRunner {
    allowed_commands: Vec<String>,
    allowed_roots: Vec<PathBuf>,
}

impl TerminalRunner {
    pub fn new(allowed_commands: &[String], allowed_roots: &[String]) -> Self {
        Self {
            allowed_commands: allowed_commands.iter().map(|s| s.to_ascii_lowercase()).collect(),
            allowed_roots: allowed_roots.iter().map(PathBuf::from).collect(),
        }
    }

    pub fn run(&self, command: &str, args: &[String], cwd: Option<&str>) -> Result<serde_json::Value, String> {
        let normalized = command.trim().to_ascii_lowercase();
        if !self.allowed_commands.iter().any(|allowed| allowed == &normalized) {
            return Err(format!("Command is not allowlisted: {command}"));
        }

        let working_dir = if let Some(cwd) = cwd {
            let candidate = PathBuf::from(cwd);
            self.ensure_allowed_dir(&candidate)?
        } else {
            self.allowed_roots.first().cloned().ok_or_else(|| "No allowed roots configured".to_string())?
        };

        let output = if cfg!(target_os = "windows") {
            Command::new("cmd")
                .arg("/C")
                .arg(command)
                .args(args)
                .current_dir(&working_dir)
                .output()
        } else {
            Command::new(command)
                .args(args)
                .current_dir(&working_dir)
                .output()
        }
        .map_err(|e| e.to_string())?;

        Ok(json!({
            "command": command,
            "args": args,
            "cwd": working_dir.display().to_string(),
            "status": output.status.code(),
            "stdout": String::from_utf8_lossy(&output.stdout),
            "stderr": String::from_utf8_lossy(&output.stderr),
        }))
    }

    fn ensure_allowed_dir(&self, path: &Path) -> Result<PathBuf, String> {
        let normalized = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir().map_err(|e| e.to_string())?.join(path)
        };
        for root in &self.allowed_roots {
            if normalized.starts_with(root) {
                return Ok(normalized);
            }
        }
        Err(format!("Working directory is outside allowed roots: {}", normalized.display()))
    }
}
