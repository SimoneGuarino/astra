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

export type PageSemanticEvidence = {
    browser_app_hint?: string | null;
    content_provider_hint?: string | null;
    page_kind_hint?: string | null;
    query_hint?: string | null;
    result_list_visible?: boolean | null;
    confidence: number;
    evidence_sources?: string[];
    capture_backend?: string | null;
    observation_source?: string | null;
    uncertainty?: string[];
};

export type SemanticScreenFrame = {
    frame_id: string;
    captured_at: number;
    image_path?: string | null;
    page_evidence: PageSemanticEvidence;
    scene_summary: string;
    visible_entities?: unknown[];
    visible_result_items?: unknown[];
    actionable_controls?: unknown[];
    legacy_target_candidates?: unknown[];
    uncertainty?: unknown[];
};

export type GoalLoopStatus =
    | "running"
    | "goal_achieved"
    | "needs_execution"
    | "needs_perception"
    | "needs_clarification"
    | "refused"
    | "browser_handoff_failed"
    | "scroll_required_but_unsupported"
    | "budget_exhausted"
    | "verification_failed";

export type GoalVerificationRecord = {
    iteration: number;
    status: string;
    confidence: number;
    reason: string;
    frame_id?: string | null;
};

export type PerceptionRequestMode = "target_focus" | "visible_page_refinement";

export type VisibleGroundingGap =
    | "weak_item_typing"
    | "missing_click_region"
    | "missing_title"
    | "missing_ranking"
    | "repeated_visible_content"
    | "partial_semantic_signals";

export type ExecutableFallbackSource =
    | "actionable_control"
    | "legacy_target_candidate";

export type VisibleRefinementStrategy =
    | "none"
    | "target_region"
    | "visible_cluster"
    | "full_frame";

export type OffscreenInferenceStage =
    | "not_applicable"
    | "deferred_pending_refinement"
    | "eligible_after_refinement"
    | "confirmed_after_refinement";

export type VisibleActionabilityDiagnostic = {
    status?:
        | "visible_executable"
        | "visible_target_needs_click_region"
        | "visible_under_grounded"
        | "no_relevant_visible_content"
        | "likely_offscreen"
        | "unknown";
    relevant_visible_content?: boolean;
    target_visible_evidence?: boolean;
    refinement_eligible?: boolean;
    refinement_strategy?: VisibleRefinementStrategy;
    offscreen_inference_stage?: OffscreenInferenceStage;
    gaps?: VisibleGroundingGap[];
    result_item_count?: number;
    weak_result_count?: number;
    missing_click_region_count?: number;
    missing_title_count?: number;
    missing_ranking_count?: number;
    entity_signal_count?: number;
    actionable_control_signal_count?: number;
    legacy_candidate_signal_count?: number;
    visible_refinement_attempts?: number;
    safe_fallback_available?: boolean;
    fallback_source_used?: ExecutableFallbackSource | null;
};

export type PerceptionRoutingDecision =
    | "target_region_anchor"
    | "regionless_target_visible"
    | "visible_page_under_grounded";

export type ExecutableCoordinateInterpretation =
    | "screen_validated"
    | "window_relative_translated"
    | "rejected_suspicious_geometry"
    | "rejected_outside_browser_surface"
    | "rejected_outside_screen_bounds"
    | "rejected_unsupported_coordinate_space"
    | "rejected_untrusted_geometry"
    | "unknown";

export type BrowserRecoveryStatus =
    | "not_needed"
    | "attempted"
    | "reacquired"
    | "failed";

export type ExecutableGeometryDiagnostic = {
    raw_region?: unknown | null;
    interpreted_region?: unknown | null;
    raw_coordinate_space?: string | null;
    interpretation?: ExecutableCoordinateInterpretation;
    validation_passed?: boolean;
    translation_applied?: boolean;
    screen_bounds?: unknown | null;
    browser_window_bounds?: unknown | null;
    final_x?: number | null;
    final_y?: number | null;
    reason?: string | null;
};

export type PlannerStepExecutionRecord = {
    step_id: string;
    status: string;
    primitive: string;
    message: string;
    selected_target_candidate?: unknown | null;
    geometry?: ExecutableGeometryDiagnostic | null;
    fresh_capture_required?: boolean;
    fresh_capture_used?: boolean;
    target_signature?: string | null;
};

export type FocusedPerceptionRequest = {
    request_id: string;
    iteration: number;
    reason: string;
    mode?: PerceptionRequestMode;
    routing_decision?: PerceptionRoutingDecision;
    refinement_strategy?: VisibleRefinementStrategy;
    target_item_id?: string | null;
    target_entity_id?: string | null;
    region?: unknown | null;
    target_region_anchor_present?: boolean;
};

export type BrowserPageSemanticKind =
    | "search_results"
    | "watch_page"
    | "web_page"
    | "unknown";

export type BrowserHandoffVerificationDecision =
    | "goal_satisfied"
    | "normalized_page_kind"
    | "supporting_evidence"
    | "rejected"
    | "not_required";

export type BrowserHandoffVerificationDiagnostic = {
    raw_page_kind_hint?: string | null;
    normalized_page_kind?: BrowserPageSemanticKind;
    decision?: BrowserHandoffVerificationDecision;
    accepted?: boolean;
    provider_matches?: boolean;
    query_hint_present?: boolean;
    result_list_visible?: boolean;
    visible_result_item_count?: number;
    visible_entity_signal_count?: number;
    legacy_candidate_signal_count?: number;
    scene_summary_result_hint?: boolean;
    supporting_signal_count?: number;
    reason?: string | null;
};

export type BrowserVisualHandoffRecord = {
    iteration: number;
    status:
        | "not_required"
        | "visually_verified"
        | "activation_unsupported"
        | "activation_failed"
        | "page_not_verified"
        | "semantic_frame_unavailable";
    activation_status?:
        | "not_attempted"
        | "executed"
        | "unsupported"
        | "failed";
    failure_reason?:
        | "browser_handoff_required"
        | "browser_activation_unsupported"
        | "browser_activation_failed"
        | "browser_page_not_verified"
        | "semantic_frame_unavailable"
        | "permission_denied"
        | "unknown"
        | null;
    app_hint?: string | null;
    provider_hint?: string | null;
    page_kind_hint?: string | null;
    verification?: BrowserHandoffVerificationDiagnostic | null;
    activation_attempted: boolean;
    page_verified: boolean;
    frame_id?: string | null;
    confidence?: number | null;
    attempts: number;
    reason?: string | null;
};

export type ConfidenceSignalState =
    | "missing"
    | "explicit_zero"
    | "low"
    | "supported";

export type ExecutableTargetConfidenceDiagnostic = {
    raw_item_confidence?: number | null;
    raw_region_confidence?: number | null;
    raw_page_confidence?: number | null;
    raw_planner_confidence?: number | null;
    item_confidence_state?: ConfidenceSignalState;
    region_confidence_state?: ConfidenceSignalState;
    page_confidence_state?: ConfidenceSignalState;
    planner_confidence_state?: ConfidenceSignalState;
    structural_semantic_confidence: number;
    semantic_confidence: number;
    region_confidence: number;
    page_confidence: number;
    derived_confidence: number;
    required_threshold: number;
    confidence_was_derived?: boolean;
    fallback_source?: "actionable_control" | "legacy_target_candidate" | null;
    accepted?: boolean;
    reason?: string | null;
};

export type PlannerDecisionDiagnostic = {
    iteration: number;
    source: "rust_deterministic" | "model_assisted" | "model_assisted_fallback";
    accepted: boolean;
    fallback_used: boolean;
    planner_confidence: number;
    strategy?: string | null;
    reason?: string | null;
    decision_status?:
        | "accepted"
        | "normalized"
        | "downgraded"
        | "rejected"
        | "fallback_used";
    rejection_code?:
        | "model_unavailable"
        | "low_confidence"
        | "malformed_output"
        | "unsupported_primitive"
        | "missing_target"
        | "fabricated_target"
        | "missing_click_region"
        | "provider_mismatch"
        | "ambiguous_target"
        | "focused_perception_requested"
        | "visible_page_refinement_requested"
        | "likely_offscreen_target"
        | "scroll_required_but_unsupported"
        | "deterministic_fallback_advised"
        | "unknown"
        | null;
    visibility_assessment?:
        | "visible_grounded"
        | "visible_ambiguous"
        | "visible_under_grounded"
        | "visible_target_needs_click_region"
        | "focused_perception_needed"
        | "likely_offscreen"
        | "not_visible"
        | "unknown";
    scroll_intent?:
        | "not_needed"
        | "likely_needed"
        | "required_but_unsupported"
        | "future_primitive_required";
    visible_actionability?: VisibleActionabilityDiagnostic;
    target_confidence?: ExecutableTargetConfidenceDiagnostic | null;
    normalized?: boolean;
    downgraded?: boolean;
    focused_perception_needed?: boolean;
    replan_needed?: boolean;
};

export type GoalLoopRun = {
    run_id: string;
    goal: unknown;
    status: GoalLoopStatus;
    iteration_count: number;
    retry_budget: number;
    retries_used: number;
    current_strategy?: string | null;
    fallback_strategy_state?: string | null;
    frames?: SemanticScreenFrame[];
    planner_steps?: unknown[];
    planner_diagnostics?: PlannerDecisionDiagnostic[];
    executed_steps?: PlannerStepExecutionRecord[];
    verification_history?: GoalVerificationRecord[];
    focused_perception_requests?: FocusedPerceptionRequest[];
    browser_handoff_history?: BrowserVisualHandoffRecord[];
    browser_handoff?: BrowserVisualHandoffRecord | null;
    focused_perception_used: boolean;
    visible_refinement_used?: boolean;
    stale_capture_reuse_prevented?: boolean;
    browser_recovery_used?: boolean;
    browser_recovery_status?: BrowserRecoveryStatus;
    repeated_click_protection_triggered?: boolean;
    selected_target_candidate?: unknown | null;
    verifier_status?: string | null;
    failure_reason?: string | null;
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
    ui_candidates?: unknown[];
    structured_candidates_error?: string | null;
    semantic_frame?: SemanticScreenFrame | null;
};


export type CapabilityToolAvailability = {
    available: boolean;
    enabled: boolean;
    requires_approval: boolean;
    state: CapabilityRuntimeState;
    disabled_reason?: string | null;
};

export type CapabilityRuntimeState =
    | "unavailable"
    | "disabled"
    | "approval_gated"
    | "ready";

export type CapabilityToolState = {
    tool_name: string;
    category: string;
    description: string;
    required_permissions: DesktopPermission[];
    default_risk: DesktopRiskLevel;
    requires_confirmation: boolean;
    available: boolean;
    enabled: boolean;
    requires_approval: boolean;
    state: CapabilityRuntimeState;
    disabled_reason?: string | null;
};

export type CapabilityScreenState = {
    observation_supported: boolean;
    observation_enabled: boolean;
    capture_available: boolean;
    analysis_available: boolean;
    vision_model_available: boolean;
    vision_model_name: string | null;
    recent_capture_available: boolean;
    recent_capture_age_ms: number | null;
    fresh_capture_available: boolean;
    fresh_capture_requires_observation_enabled: boolean;
    last_capture_path: string | null;
    last_frame_at: number | null;
    provider: string;
    note: string;
};

export type CapabilityApprovalState = {
    pending_count: number;
    approval_required_for_high_risk: boolean;
    pending_actions: CapabilityPendingApprovalSummary[];
};

export type CapabilityPendingApprovalSummary = {
    action_id: string;
    tool_name: string;
    risk_level: DesktopRiskLevel;
    reason: string;
    requested_at: number;
};

export type CapabilityPermissionState = {
    allowed_permissions: DesktopPermission[];
    browser_enabled: boolean;
    desktop_control_enabled: boolean;
    allowed_roots: string[];
    terminal_allowed_commands: string[];
};

export type CapabilityManifest = {
    version: string;
    generated_at: number;
    tool_names: string[];
    enabled_tool_names: string[];
    disabled_tool_names: string[];
    tools: CapabilityToolState[];
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

export type ConversationRouteDiagnostic = {
    message_excerpt: string;
    classifier_source: string;
    intent: string;
    target: string | null;
    action: string | null;
    tool_name?: string | null;
    extracted_params?: unknown;
    confidence: number | null;
    routed_to: string;
    grounded: boolean;
    fallback_used: boolean;
    submit_action_called: boolean;
    action_id?: string | null;
    action_status?: string | null;
    approval_created: boolean;
    audit_expected: boolean;
    rationale: string | null;
    error: string | null;
};
