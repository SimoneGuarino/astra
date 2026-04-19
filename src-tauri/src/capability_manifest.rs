use crate::desktop_agent_types::{
    CapabilityApprovalState, CapabilityManifest, CapabilityPermissionState, CapabilityScreenState,
    CapabilityToolAvailability, DesktopPolicySnapshot, Permission, ScreenObservationStatus,
    ToolDescriptor, VisionAvailability,
};

pub fn build_capability_manifest(
    tools: &[ToolDescriptor],
    policy: &DesktopPolicySnapshot,
    screen_status: &ScreenObservationStatus,
    pending_approvals: usize,
    vision: &VisionAvailability,
) -> CapabilityManifest {
    let tool_names = tools.iter().map(|tool| tool.tool_name.clone()).collect::<Vec<_>>();

    let filesystem_read = tool_summary(tools, "filesystem.read_text", policy, Permission::FilesystemRead, false);
    let filesystem_write = tool_summary(tools, "filesystem.write_text", policy, Permission::FilesystemWrite, true);
    let filesystem_search = tool_summary(tools, "filesystem.search", policy, Permission::FilesystemSearch, false);
    let terminal = tool_summary(tools, "terminal.run", policy, Permission::TerminalSafe, true);
    let browser_open = tool_summary(tools, "browser.open", policy, Permission::BrowserAction, false);
    let browser_search = tool_summary(tools, "browser.search", policy, Permission::BrowserRead, false);
    let desktop_launch = tool_summary(tools, "desktop.launch_app", policy, Permission::DesktopControl, true);

    let screen = CapabilityScreenState {
        observation_supported: screen_status.provider != "not_supported",
        observation_enabled: screen_status.enabled,
        capture_available: screen_status.provider != "not_supported",
        analysis_available: vision.available,
        vision_model_available: vision.available,
        vision_model_name: vision.selected_model.clone(),
        recent_capture_available: screen_status.last_capture_path.is_some(),
        last_capture_path: screen_status.last_capture_path.clone(),
        last_frame_at: screen_status.last_frame_at,
        provider: screen_status.provider.clone(),
        note: screen_status.note.clone(),
    };

    CapabilityManifest {
        tool_names,
        filesystem_read,
        filesystem_write,
        filesystem_search,
        terminal,
        browser_open,
        browser_search,
        desktop_launch,
        screen,
        approvals: CapabilityApprovalState {
            pending_count: pending_approvals,
            approval_required_for_high_risk: policy.approval_required_for_high_risk,
        },
        permissions: CapabilityPermissionState {
            allowed_permissions: policy.allowed_permissions.clone(),
            browser_enabled: policy.browser_enabled,
            desktop_control_enabled: policy.desktop_control_enabled,
            allowed_roots: policy.allowed_roots.clone(),
            terminal_allowed_commands: policy.terminal_allowed_commands.clone(),
        },
    }
}

fn tool_summary(
    tools: &[ToolDescriptor],
    tool_name: &str,
    policy: &DesktopPolicySnapshot,
    permission: Permission,
    approval_gated: bool,
) -> CapabilityToolAvailability {
    let available = tools.iter().any(|tool| tool.tool_name == tool_name);
    let enabled = available && policy.allowed_permissions.iter().any(|value| value == &permission);
    CapabilityToolAvailability {
        available,
        enabled,
        requires_approval: approval_gated && policy.approval_required_for_high_risk,
    }
}
