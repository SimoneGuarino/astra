use crate::desktop_agent_types::{
    CapabilityApprovalState, CapabilityManifest, CapabilityPendingApprovalSummary,
    CapabilityPermissionState, CapabilityRuntimeState, CapabilityScreenState,
    CapabilityToolAvailability, CapabilityToolState, DesktopPolicySnapshot, PendingApproval,
    Permission, ScreenObservationStatus, ToolDescriptor, VisionAvailability,
};

pub fn build_capability_manifest(
    tools: &[ToolDescriptor],
    policy: &DesktopPolicySnapshot,
    screen_status: &ScreenObservationStatus,
    pending_approvals: &[PendingApproval],
    vision: &VisionAvailability,
) -> CapabilityManifest {
    let generated_at = now_ms();
    let tool_names = tools
        .iter()
        .map(|tool| tool.tool_name.clone())
        .collect::<Vec<_>>();
    let tool_states = tools
        .iter()
        .map(|tool| tool_state(tool, policy))
        .collect::<Vec<_>>();
    let enabled_tool_names = tool_states
        .iter()
        .filter(|tool| tool.enabled)
        .map(|tool| tool.tool_name.clone())
        .collect::<Vec<_>>();
    let disabled_tool_names = tool_states
        .iter()
        .filter(|tool| !tool.enabled)
        .map(|tool| tool.tool_name.clone())
        .collect::<Vec<_>>();

    let filesystem_read = tool_summary(
        tools,
        "filesystem.read_text",
        policy,
        Permission::FilesystemRead,
        false,
    );
    let filesystem_write = tool_summary(
        tools,
        "filesystem.write_text",
        policy,
        Permission::FilesystemWrite,
        true,
    );
    let filesystem_search = tool_summary(
        tools,
        "filesystem.search",
        policy,
        Permission::FilesystemSearch,
        false,
    );
    let terminal = tool_summary(
        tools,
        "terminal.run",
        policy,
        Permission::TerminalSafe,
        true,
    );
    let browser_open = tool_summary(
        tools,
        "browser.open",
        policy,
        Permission::BrowserAction,
        false,
    );
    let browser_search = tool_summary(
        tools,
        "browser.search",
        policy,
        Permission::BrowserRead,
        false,
    );
    let desktop_launch = tool_summary(
        tools,
        "desktop.launch_app",
        policy,
        Permission::DesktopControl,
        true,
    );

    let screen = CapabilityScreenState {
        observation_supported: screen_status.provider != "not_supported",
        observation_enabled: screen_status.enabled,
        capture_available: screen_status.provider != "not_supported",
        analysis_available: vision.available,
        vision_model_available: vision.available,
        vision_model_name: vision.selected_model.clone(),
        recent_capture_available: screen_status.last_capture_path.is_some(),
        recent_capture_age_ms: screen_status
            .last_frame_at
            .map(|captured_at| generated_at.saturating_sub(captured_at)),
        fresh_capture_available: screen_status.provider != "not_supported",
        fresh_capture_requires_observation_enabled: true,
        accessibility_snapshot_enabled: policy.accessibility_snapshot_enabled,
        last_capture_path: screen_status.last_capture_path.clone(),
        last_frame_at: screen_status.last_frame_at,
        provider: screen_status.provider.clone(),
        note: screen_status.note.clone(),
    };

    CapabilityManifest {
        version: "astra_capability_manifest_v2".into(),
        generated_at,
        tool_names,
        enabled_tool_names,
        disabled_tool_names,
        tools: tool_states,
        filesystem_read,
        filesystem_write,
        filesystem_search,
        terminal,
        browser_open,
        browser_search,
        desktop_launch,
        screen,
        approvals: CapabilityApprovalState {
            pending_count: pending_approvals.len(),
            approval_required_for_high_risk: policy.approval_required_for_high_risk,
            pending_actions: pending_approvals
                .iter()
                .map(|approval| CapabilityPendingApprovalSummary {
                    action_id: approval.action_id.clone(),
                    tool_name: approval.tool_name.clone(),
                    risk_level: approval.risk_level.clone(),
                    reason: approval.reason.clone(),
                    requested_at: approval.requested_at,
                })
                .collect(),
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

fn tool_state(tool: &ToolDescriptor, policy: &DesktopPolicySnapshot) -> CapabilityToolState {
    let missing_permissions = tool
        .required_permissions
        .iter()
        .filter(|permission| {
            !policy
                .allowed_permissions
                .iter()
                .any(|allowed| allowed == *permission)
        })
        .map(permission_name)
        .collect::<Vec<_>>();

    let disabled_by_policy = tool
        .required_permissions
        .iter()
        .filter(|permission| !permission_enabled_by_policy(policy, permission))
        .map(permission_name)
        .collect::<Vec<_>>();

    let enabled = missing_permissions.is_empty() && disabled_by_policy.is_empty();
    let requires_approval = tool.requires_confirmation
        || (policy.approval_required_for_high_risk
            && matches!(
                tool.default_risk,
                crate::desktop_agent_types::RiskLevel::High
            ));
    let state = if !enabled {
        CapabilityRuntimeState::Disabled
    } else if requires_approval {
        CapabilityRuntimeState::ApprovalGated
    } else {
        CapabilityRuntimeState::Ready
    };
    let disabled_reason = if missing_permissions.is_empty() && disabled_by_policy.is_empty() {
        None
    } else {
        let mut reasons = Vec::new();
        if !missing_permissions.is_empty() {
            reasons.push(format!(
                "missing permissions: {}",
                missing_permissions.join(", ")
            ));
        }
        if !disabled_by_policy.is_empty() {
            reasons.push(format!(
                "disabled by policy: {}",
                disabled_by_policy.join(", ")
            ));
        }
        Some(reasons.join("; "))
    };

    CapabilityToolState {
        tool_name: tool.tool_name.clone(),
        category: tool.category.clone(),
        description: tool.description.clone(),
        required_permissions: tool.required_permissions.clone(),
        default_risk: tool.default_risk.clone(),
        requires_confirmation: tool.requires_confirmation,
        available: true,
        enabled,
        requires_approval,
        state,
        disabled_reason,
    }
}

fn tool_summary(
    tools: &[ToolDescriptor],
    tool_name: &str,
    policy: &DesktopPolicySnapshot,
    permission: Permission,
    approval_gated: bool,
) -> CapabilityToolAvailability {
    let descriptor = tools.iter().find(|tool| tool.tool_name == tool_name);
    let available = descriptor.is_some();
    let permission_allowed = policy
        .allowed_permissions
        .iter()
        .any(|value| value == &permission);
    let policy_enabled = permission_enabled_by_policy(policy, &permission);
    let enabled = available && permission_allowed && policy_enabled;
    let requires_approval = descriptor
        .map(|tool| {
            tool.requires_confirmation
                || (policy.approval_required_for_high_risk
                    && matches!(
                        tool.default_risk,
                        crate::desktop_agent_types::RiskLevel::High
                    ))
        })
        .unwrap_or(approval_gated && policy.approval_required_for_high_risk);
    let state = if !available {
        CapabilityRuntimeState::Unavailable
    } else if !enabled {
        CapabilityRuntimeState::Disabled
    } else if requires_approval {
        CapabilityRuntimeState::ApprovalGated
    } else {
        CapabilityRuntimeState::Ready
    };
    let disabled_reason = if available && !enabled {
        if !permission_allowed {
            Some(format!(
                "permission {} is not enabled",
                permission_name(&permission)
            ))
        } else if !policy_enabled {
            Some(format!(
                "permission {} is disabled by policy",
                permission_name(&permission)
            ))
        } else {
            Some("disabled by policy or permissions".into())
        }
    } else {
        None
    };

    CapabilityToolAvailability {
        available,
        enabled,
        requires_approval,
        state,
        disabled_reason,
    }
}

fn permission_enabled_by_policy(policy: &DesktopPolicySnapshot, permission: &Permission) -> bool {
    match permission {
        Permission::BrowserRead | Permission::BrowserAction => policy.browser_enabled,
        Permission::DesktopControl => policy.desktop_control_enabled,
        _ => true,
    }
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

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}
