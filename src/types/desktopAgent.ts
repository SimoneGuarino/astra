export type DesktopPermission =
    | "filesystem_read"
    | "filesystem_write"
    | "filesystem_search"
    | "terminal_safe"
    | "terminal_dangerous"
    | "browser_read"
    | "browser_action"
    | "desktop_observe"
    | "desktop_control";

export type DesktopRiskLevel = "low" | "medium" | "high";

export type ToolDescriptor = {
    tool_name: string;
    category: string;
    description: string;
    required_permissions: DesktopPermission[];
    default_risk: DesktopRiskLevel;
    requires_confirmation: boolean;
};

export type DesktopActionRequest = {
    tool_name: string;
    params: Record<string, unknown>;
    preview_only?: boolean;
    reason?: string | null;
};

export type DesktopActionStatus = "executed" | "approval_required" | "rejected" | "failed";

export type DesktopActionResponse = {
    action_id: string;
    request_id: string;
    tool_name: string;
    status: DesktopActionStatus;
    message?: string | null;
    result?: unknown;
    risk_level?: DesktopRiskLevel | null;
};

export type PendingApproval = {
    action_id: string;
    request_id: string;
    tool_name: string;
    params: Record<string, unknown>;
    risk_level: DesktopRiskLevel;
    reason: string;
    requested_at: number;
};

export type DesktopPolicySnapshot = {
    allowed_roots: string[];
    terminal_allowed_commands: string[];
    allowed_permissions: DesktopPermission[];
    approval_required_for_high_risk: boolean;
    browser_enabled: boolean;
    desktop_control_enabled: boolean;
};

export type DesktopAuditEvent = {
    audit_id: string;
    action_id: string;
    request_id: string;
    tool_name: string;
    stage: string;
    status: string;
    timestamp: number;
    risk_level: DesktopRiskLevel;
    details: unknown;
};

export type ScreenObservationStatus = {
    enabled: boolean;
    provider: string;
    last_frame_at: number | null;
    last_error: string | null;
    last_capture_path: string | null;
    capture_count: number;
    note: string;
};

export type ScreenCaptureResult = {
    capture_id: string;
    captured_at: number;
    image_path: string;
    width: number | null;
    height: number | null;
    bytes: number;
    provider: string;
};


export type ScreenAnalysisRequest = {
    question?: string | null;
    capture_fresh?: boolean;
};

export type ScreenAnalysisResult = {
    analysis_id: string;
    request_id: string;
    captured_at: number;
    image_path: string;
    model: string;
    provider: string;
    question: string;
    answer: string;
};


export type CapabilityToolAvailability = {
    available: boolean;
    enabled: boolean;
    requires_approval: boolean;
};

export type CapabilityScreenState = {
    observation_supported: boolean;
    observation_enabled: boolean;
    capture_available: boolean;
    analysis_available: boolean;
    vision_model_available: boolean;
    vision_model_name: string | null;
    recent_capture_available: boolean;
    last_capture_path: string | null;
    last_frame_at: number | null;
    provider: string;
    note: string;
};

export type CapabilityApprovalState = {
    pending_count: number;
    approval_required_for_high_risk: boolean;
};

export type CapabilityPermissionState = {
    allowed_permissions: DesktopPermission[];
    browser_enabled: boolean;
    desktop_control_enabled: boolean;
    allowed_roots: string[];
    terminal_allowed_commands: string[];
};

export type CapabilityManifest = {
    tool_names: string[];
    filesystem_read: CapabilityToolAvailability;
    filesystem_write: CapabilityToolAvailability;
    filesystem_search: CapabilityToolAvailability;
    terminal: CapabilityToolAvailability;
    browser_open: CapabilityToolAvailability;
    browser_search: CapabilityToolAvailability;
    desktop_launch: CapabilityToolAvailability;
    screen: CapabilityScreenState;
    approvals: CapabilityApprovalState;
    permissions: CapabilityPermissionState;
};
