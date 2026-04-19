use crate::desktop_agent_types::Permission;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionProfile {
    pub allowed: Vec<Permission>,
}

impl PermissionProfile {
    pub fn default_local_agent() -> Self {
        Self {
            allowed: vec![
                Permission::FilesystemRead,
                Permission::FilesystemWrite,
                Permission::FilesystemSearch,
                Permission::TerminalSafe,
                Permission::BrowserRead,
                Permission::BrowserAction,
                Permission::DesktopObserve,
                Permission::DesktopControl,
            ],
        }
    }

    pub fn allows(&self, permission: &Permission) -> bool {
        self.allowed.contains(permission)
    }
}
