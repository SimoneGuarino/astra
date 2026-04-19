import { useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import type {
    CapabilityManifest,
    DesktopActionRequest,
    DesktopActionResponse,
    DesktopAuditEvent,
    DesktopPolicySnapshot,
    PendingApproval,
    ScreenAnalysisRequest,
    ScreenAnalysisResult,
    ScreenCaptureResult,
    ScreenObservationStatus,
    ToolDescriptor,
} from "../types/desktopAgent";

export function useDesktopAgent() {
    const listTools = useCallback(() => invoke<ToolDescriptor[]>("list_desktop_tools"), []);

    const getCapabilityManifest = useCallback(
        () => invoke<CapabilityManifest>("get_capability_manifest"),
        []
    );

    const getPolicySnapshot = useCallback(
        () => invoke<DesktopPolicySnapshot>("get_desktop_policy_snapshot"),
        []
    );

    const getPendingApprovals = useCallback(
        () => invoke<PendingApproval[]>("get_pending_desktop_approvals"),
        []
    );

    const getRecentAuditEvents = useCallback(
        (limit = 50) =>
            invoke<DesktopAuditEvent[]>("get_recent_desktop_audit_events", { limit }),
        []
    );

    const executeAction = useCallback(
        (payload: DesktopActionRequest) =>
            invoke<DesktopActionResponse>("execute_desktop_action", { payload }),
        []
    );

    const approveAction = useCallback(
        (actionId: string, note?: string) =>
            invoke<DesktopActionResponse>("approve_desktop_action", {
                payload: { action_id: actionId, note: note ?? null },
            }),
        []
    );

    const rejectAction = useCallback(
        (actionId: string, note?: string) =>
            invoke<void>("reject_desktop_action", {
                payload: { action_id: actionId, note: note ?? null },
            }),
        []
    );

    const getScreenObservationStatus = useCallback(
        () => invoke<ScreenObservationStatus>("get_screen_observation_status"),
        []
    );

    const setScreenObservationEnabled = useCallback(
        (enabled: boolean) => invoke<ScreenObservationStatus>("set_screen_observation_enabled", { enabled }),
        []
    );

    const captureScreenSnapshot = useCallback(
        () => invoke<ScreenCaptureResult>("capture_screen_snapshot"),
        []
    );

    const analyzeScreenContext = useCallback(
        (payload: ScreenAnalysisRequest) =>
            invoke<ScreenAnalysisResult>("analyze_screen_context", { payload }),
        []
    );

    return {
        analyzeScreenContext,
        approveAction,
        captureScreenSnapshot,
        executeAction,
        getCapabilityManifest,
        getPendingApprovals,
        getPolicySnapshot,
        getRecentAuditEvents,
        getScreenObservationStatus,
        listTools,
        rejectAction,
        setScreenObservationEnabled,
    };
}
