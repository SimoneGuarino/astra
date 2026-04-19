use crate::desktop_agent_types::PendingApproval;
use std::{collections::HashMap, fs, path::PathBuf, sync::{Arc, Mutex}};

#[derive(Clone)]
pub struct PendingApprovalsStore {
    path: PathBuf,
    lock: Arc<Mutex<()>>,
}

impl PendingApprovalsStore {
    pub fn new(project_root: &PathBuf) -> Self {
        let path = project_root
            .join(".astra")
            .join("state")
            .join("pending_approvals.json");
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        Self { path, lock: Arc::new(Mutex::new(())) }
    }

    pub fn load(&self) -> HashMap<String, PendingApproval> {
        let _guard = self.lock.lock().expect("pending approvals store mutex poisoned");
        let Ok(raw) = fs::read_to_string(&self.path) else {
            return HashMap::new();
        };
        serde_json::from_str::<Vec<PendingApproval>>(&raw)
            .unwrap_or_default()
            .into_iter()
            .map(|approval| (approval.action_id.clone(), approval))
            .collect()
    }

    pub fn save(&self, approvals: impl IntoIterator<Item = PendingApproval>) -> Result<(), String> {
        let _guard = self.lock.lock().expect("pending approvals store mutex poisoned");
        let values: Vec<PendingApproval> = approvals.into_iter().collect();
        let raw = serde_json::to_string_pretty(&values).map_err(|e| e.to_string())?;
        fs::write(&self.path, raw).map_err(|e| e.to_string())
    }
}
