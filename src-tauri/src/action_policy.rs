use crate::{
    desktop_agent_types::{DesktopPolicySnapshot, Permission, RiskLevel},
    permissions::PermissionProfile,
};
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopAgentPolicy {
    pub allowed_roots: Vec<String>,
    pub terminal_allowed_commands: Vec<String>,
    pub approval_required_for_high_risk: bool,
    pub browser_enabled: bool,
    pub desktop_control_enabled: bool,
    #[serde(default = "default_accessibility_snapshot_enabled")]
    pub accessibility_snapshot_enabled: bool,
}

impl DesktopAgentPolicy {
    pub fn load_or_default(project_root: &PathBuf) -> Self {
        let config_path = project_root
            .join(".astra")
            .join("desktop_agent_policy.json");
        if let Ok(raw) = fs::read_to_string(&config_path) {
            if let Ok(parsed) = serde_json::from_str::<DesktopAgentPolicy>(&raw) {
                return parsed;
            }
        }

        let cwd = project_root.display().to_string();
        let desktop = std::env::var("USERPROFILE")
            .map(|home| format!("{home}\\Desktop"))
            .unwrap_or_else(|_| cwd.clone());
        Self {
            allowed_roots: vec![cwd, desktop],
            terminal_allowed_commands: vec![
                "git".into(),
                "cargo".into(),
                "npm".into(),
                "pnpm".into(),
                "yarn".into(),
                "python".into(),
                "py".into(),
                "dir".into(),
                "type".into(),
                "rg".into(),
                "ls".into(),
            ],
            approval_required_for_high_risk: true,
            browser_enabled: true,
            desktop_control_enabled: true,
            accessibility_snapshot_enabled: default_accessibility_snapshot_enabled(),
        }
    }

    pub fn snapshot(&self, permissions: &PermissionProfile) -> DesktopPolicySnapshot {
        DesktopPolicySnapshot {
            allowed_roots: self.allowed_roots.clone(),
            terminal_allowed_commands: self.terminal_allowed_commands.clone(),
            allowed_permissions: permissions.allowed.clone(),
            approval_required_for_high_risk: self.approval_required_for_high_risk,
            browser_enabled: self.browser_enabled,
            desktop_control_enabled: self.desktop_control_enabled,
            accessibility_snapshot_enabled: self.accessibility_snapshot_enabled,
        }
    }

    pub fn accessibility_snapshot_enabled(&self) -> bool {
        self.accessibility_snapshot_enabled && cfg!(target_os = "windows")
    }

    pub fn requires_approval(&self, risk_level: &RiskLevel) -> bool {
        self.approval_required_for_high_risk && matches!(risk_level, RiskLevel::High)
    }

    pub fn permission_enabled(&self, permission: &Permission) -> bool {
        match permission {
            Permission::BrowserRead | Permission::BrowserAction => self.browser_enabled,
            Permission::DesktopControl => self.desktop_control_enabled,
            _ => true,
        }
    }
}

fn default_accessibility_snapshot_enabled() -> bool {
    cfg!(target_os = "windows")
}
