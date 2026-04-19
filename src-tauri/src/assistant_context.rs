use crate::desktop_agent_types::{CapabilityManifest, Permission};

pub fn build_capability_context(manifest: &CapabilityManifest) -> String {
    let permissions = manifest
        .permissions
        .allowed_permissions
        .iter()
        .map(permission_name)
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        concat!(
            "Astra local capability context:
",
            "- Filesystem read: {}
",
            "- Filesystem write: {}
",
            "- Filesystem search: {}
",
            "- Terminal actions: {}
",
            "- Browser open: {}
",
            "- Browser search: {}
",
            "- Desktop app launch: {}
",
            "- Screen observation supported: {}
",
            "- Screen observation enabled: {}
",
            "- Screen capture available: {}
",
            "- Screen analysis available: {}
",
            "- Vision model available: {}
",
            "- Recent screen capture available: {}
",
            "- Pending approvals: {}
",
            "- Approval required for high-risk actions: {}
",
            "- Browser policy enabled: {}
",
            "- Desktop control policy enabled: {}
",
            "- Allowed filesystem roots: {}
",
            "- Allowed terminal commands: {}
",
            "- Enabled permissions: {}
",
            "Behavior rules:
",
            "1. Do not deny capabilities that are available in this context.
",
            "2. Be explicit about whether a capability is enabled, disabled, unavailable, or approval-gated.
",
            "3. Never say an action is already done when approval is still pending.
",
            "4. For screen questions, explain whether observation is enabled, whether a fresh capture may be needed, and whether a vision model is available.
",
            "5. When tool execution results are provided by the runtime, ground your answer in those results instead of speaking generically."
        ),
        capability_label(&manifest.filesystem_read),
        capability_label(&manifest.filesystem_write),
        capability_label(&manifest.filesystem_search),
        capability_label(&manifest.terminal),
        capability_label(&manifest.browser_open),
        capability_label(&manifest.browser_search),
        capability_label(&manifest.desktop_launch),
        yes_no(manifest.screen.observation_supported),
        yes_no(manifest.screen.observation_enabled),
        yes_no(manifest.screen.capture_available),
        yes_no(manifest.screen.analysis_available),
        manifest.screen.vision_model_name.clone().unwrap_or_else(|| yes_no(manifest.screen.vision_model_available).into()),
        yes_no(manifest.screen.recent_capture_available),
        manifest.approvals.pending_count,
        yes_no(manifest.approvals.approval_required_for_high_risk),
        yes_no(manifest.permissions.browser_enabled),
        yes_no(manifest.permissions.desktop_control_enabled),
        join_or_none(&manifest.permissions.allowed_roots),
        join_or_none(&manifest.permissions.terminal_allowed_commands),
        if permissions.is_empty() { "none".into() } else { permissions },
    )
}

fn capability_label(value: &crate::desktop_agent_types::CapabilityToolAvailability) -> &'static str {
    if !value.available { "not available" }
    else if !value.enabled { "available but disabled" }
    else if value.requires_approval { "available and approval-gated" }
    else { "available and ready" }
}

fn permission_name(permission: &Permission) -> &'static str {
    match permission {
        Permission::FilesystemRead => "filesystem_read",
        Permission::FilesystemWrite => "filesystem_write",
        Permission::FilesystemSearch => "filesystem_search",
        Permission::TerminalSafe => "terminal_safe",
        Permission::TerminalDangerous => "terminal_dangerous",
        Permission::BrowserRead => "browser_read",
        Permission::BrowserAction => "browser_action",
        Permission::DesktopObserve => "desktop_observe",
        Permission::DesktopControl => "desktop_control",
    }
}
fn yes_no(value: bool) -> &'static str { if value { "yes" } else { "no" } }
fn join_or_none(values: &[String]) -> String { if values.is_empty() { "none".into() } else { values.join(", ") } }
