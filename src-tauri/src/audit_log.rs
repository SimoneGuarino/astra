use crate::desktop_agent_types::DesktopAuditEvent;
use std::{fs, path::PathBuf, sync::{Arc, Mutex}};

#[derive(Clone)]
pub struct AuditLogStore {
    path: PathBuf,
    lock: Arc<Mutex<()>>,
}

impl AuditLogStore {
    pub fn new(project_root: &PathBuf) -> Self {
        let path = project_root
            .join(".astra")
            .join("audit")
            .join("desktop_agent_audit.jsonl");
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        Self { path, lock: Arc::new(Mutex::new(())) }
    }

    pub fn append(&self, event: &DesktopAuditEvent) {
        let _guard = self.lock.lock().expect("audit log mutex poisoned");
        if let Ok(line) = serde_json::to_string(event) {
            let _ = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)
                .and_then(|mut file| std::io::Write::write_all(&mut file, format!("{line}\n").as_bytes()));
        }
    }

    pub fn tail(&self, limit: usize) -> Vec<DesktopAuditEvent> {
        let _guard = self.lock.lock().expect("audit log mutex poisoned");
        let Ok(content) = fs::read_to_string(&self.path) else {
            return Vec::new();
        };
        let mut events: Vec<DesktopAuditEvent> = content
            .lines()
            .filter_map(|line| serde_json::from_str::<DesktopAuditEvent>(line).ok())
            .collect();
        if events.len() > limit {
            events = events.split_off(events.len() - limit);
        }
        events
    }
}
