import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { Button } from "../ui/buttons/Button";
import { useMeeting } from "../hooks/useMeeting";
import type { CapabilityManifest, CapabilityToolState } from "../types/desktopAgent";
import type {
    ActionItem,
    CallInfo,
    CaptureBackend,
    ConsentState,
    DecisionLogEntry,
    MeetingConfig,
    MeetingDataClearPreview,
    MeetingDiagnostic,
    MeetingFinalizationStatus,
    MeetingIntelligenceResult,
    MeetingLiveCapabilitySnapshot,
    MeetingRecallResponse,
    MeetingSession,
    MeetingSessionArchiveDocument,
    MeetingSessionExportResponse,
    MeetingSessionListItem,
    MeetingSessionSearchResult,
    MeetingSessionState,
    MeetingStatus,
    SpeakerLabel,
    TranscriptEntry,
} from "../types/meeting";
import { CLEAR_MEETING_DATA_CONFIRMATION_PHRASE } from "../types/meeting";

type MeetingDebugPanelProps = {
    capabilities: CapabilityManifest | null;
};

type MeetingUiState =
    | "not_started"
    | "manual_active"
    | "blocked_permissions"
    | "ready_to_record"
    | "recording"
    | "degraded"
    | "transcribing"
    | "paused"
    | "failed"
    | "failed_recoverable"
    | "completed";

type MeetingUiSummary = {
    state: MeetingUiState;
    title: string;
    description: string;
    nextAction?: string;
    blockingReasons: string[];
    isRecording: boolean;
    isTranscribing: boolean;
};

type MeetingStartReadiness = {
    canRequestStart: boolean;
    requiresApprovalOrConfirmation: boolean;
    hardBlockers: string[];
    softGates: string[];
    statusLabel: string;
    primaryHelpText: string;
};

type CaptureReadiness = {
    backendAvailable: boolean;
    audioToolAvailable: boolean;
    segmentToolAvailable: boolean;
    audioPermissionReady: boolean;
    segmentPermissionReady: boolean;
    approvalRequired: boolean;
    blockedReason: string | null;
    blockedTools: string[];
};

const DEFAULT_PLATFORM = "teams";
type CaptureMode = "system_audio" | "microphone" | "both";

function formatStatus(status?: MeetingStatus | null): string {
    if (!status) return "unknown";
    if (typeof status === "string") return status;
    if ("failed" in status) return `failed: ${status.failed}`;
    return `error: ${status.error}`;
}

function statusKind(status?: MeetingStatus | null): string {
    if (!status) return "unknown";
    if (typeof status === "string") return status;
    if ("failed" in status) return "failed";
    return "error";
}

function transcriptSourceLabel(source?: string | null): string {
    switch (source) {
        case "microphone":
            return "MIC";
        case "system_audio":
            return "SYSTEM";
        case "manual":
            return "MANUAL";
        case "imported_file":
            return "FILE";
        default:
            return "UNKNOWN";
    }
}

function generatorLabel(intelligence?: MeetingIntelligenceResult | null): string {
    const generator = intelligence?.diagnostics.generator;
    if (!generator) return "not generated";
    switch (generator.type) {
        case "local_llm":
            return `local model (${generator.model})`;
        case "hybrid":
            return "hybrid";
        case "rule_based":
        default:
            return intelligence?.diagnostics.fallback_used ? "rule-based fallback" : "rule-based";
    }
}

function evidenceLabel(ids?: string[] | null): string {
    const count = ids?.length ?? 0;
    return count === 1 ? "Evidence: 1 transcript segment" : `Evidence: ${count} transcript segments`;
}

function confidenceLabel(confidence?: number | null): string {
    if (typeof confidence !== "number") return "confidence unknown";
    return `${Math.round(confidence * 100)}% confidence`;
}

function languageLabel(language?: string | null): string {
    switch (language) {
        case "italian":
            return "Italian";
        case "english":
            return "English";
        case "mixed":
            return "Mixed";
        case "unknown":
        default:
            return "Unknown";
    }
}

function sessionTypeLabel(sessionType?: string | null): string {
    switch (sessionType) {
        case "technical_debugging":
            return "Technical debugging";
        case "planning":
            return "Planning";
        case "decision_review":
            return "Decision review";
        case "support_call":
            return "Support call";
        case "work_meeting":
            return "Work meeting";
        case "general":
        default:
            return "General";
    }
}

function truncateEvidenceText(text: string, maxLength = 170): string {
    const compact = text.replace(/\s+/g, " ").trim();
    if (compact.length <= maxLength) return compact;
    return `${compact.slice(0, maxLength).trim()}...`;
}

function EvidencePreview({
    ids,
    transcriptBySegmentId,
    expanded,
    onToggle,
}: {
    ids?: string[] | null;
    transcriptBySegmentId: Map<string, TranscriptEntry>;
    expanded: boolean;
    onToggle: () => void;
}) {
    const evidenceIds = ids ?? [];
    const entries = evidenceIds
        .map((id) => transcriptBySegmentId.get(id))
        .filter((entry): entry is TranscriptEntry => Boolean(entry));

    return (
        <div className="meeting-evidence">
            <div className="meeting-evidence__summary">
                <span>{evidenceLabel(evidenceIds)}</span>
                {entries.length ? (
                    <button type="button" className="meeting-evidence__toggle" onClick={onToggle}>
                        {expanded ? "Hide evidence" : "Show evidence"}
                    </button>
                ) : evidenceIds.length ? (
                    <span className="desktop-agent-muted">IDs: {evidenceIds.slice(0, 4).join(", ")}</span>
                ) : null}
            </div>
            {expanded && entries.length ? (
                <ul className="meeting-evidence__list">
                    {entries.slice(0, 5).map((entry) => (
                        <li key={entry.segment_id}>
                            <strong>[{transcriptSourceLabel(entry.source)}] {entry.speaker_label ?? entry.speaker_id ?? "Unknown"}</strong>
                            <span>{truncateEvidenceText(entry.text)}</span>
                        </li>
                    ))}
                    {entries.length > 5 ? <li>{entries.length - 5} more evidence segment(s)</li> : null}
                </ul>
            ) : null}
        </div>
    );
}

function formatDurationMs(durationMs?: number | null): string {
    if (typeof durationMs !== "number") return "not measured";
    if (durationMs < 1000) return `${durationMs}ms`;
    return `${(durationMs / 1000).toFixed(1)}s`;
}

function formatTimelineTime(timestampMs?: number | null): string {
    if (typeof timestampMs !== "number") return "time unknown";
    const totalSeconds = Math.floor(timestampMs / 1000);
    const minutes = Math.floor(totalSeconds / 60).toString().padStart(2, "0");
    const seconds = (totalSeconds % 60).toString().padStart(2, "0");
    return `${minutes}:${seconds}`;
}

function makeLocalId(): string {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
        return crypto.randomUUID();
    }
    return `local-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function statusFailureReason(status?: MeetingStatus | null): string | null {
    if (!status || typeof status === "string") return null;
    if ("failed" in status) return status.failed;
    return status.error;
}

function toolState(tool?: CapabilityToolState): string {
    if (!tool) return "not registered";
    if (!tool.available) return tool.disabled_reason ?? "unavailable";
    return tool.state;
}

function toolReady(tool?: CapabilityToolState): boolean {
    return tool?.available === true && tool.enabled === true;
}

function permissionStatus(tool?: CapabilityToolState): string {
    if (!tool) return "not registered";
    if (!tool.available) return "unavailable";
    if (!tool.enabled) return "blocked";
    if (tool.state === "approval_gated") return "approval required";
    return "ready";
}

function pushUnique(values: string[], value: string | null | undefined) {
    if (value && !values.includes(value)) values.push(value);
}

function manualMeetingConfig(platform: string, backend: CaptureBackend): MeetingConfig {
    return {
        platform,
        capture_backend: backend,
        transcription_model: "local",
        sample_rate: 16_000,
        diarization_enabled: false,
        privacy_mode: "default",
        session_mode: "manual",
        live_transcription_enabled: false,
        capture_options: {
            system_audio: false,
            microphone: false,
            segment_transcription: false,
        },
    };
}

function realCaptureConfig(platform: string, backend: CaptureBackend, mode: CaptureMode): MeetingConfig {
    const systemAudio = mode === "system_audio" || mode === "both";
    const microphone = mode === "microphone" || mode === "both";
    return {
        platform,
        capture_backend: backend === "default" ? "wasapi" : backend,
        transcription_model: "local",
        sample_rate: 16_000,
        diarization_enabled: false,
        privacy_mode: "default",
        session_mode: "real_capture",
        live_transcription_enabled: true,
        capture_options: {
            system_audio: systemAudio,
            microphone,
            segment_transcription: true,
        },
    };
}

function normalizePlatformKey(value: string): string {
    const collapsed = value
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, " ")
        .trim()
        .replace(/\s+/g, " ");
    if (["teams", "microsoft teams", "ms teams", "msteams"].includes(collapsed)) {
        return "teams";
    }
    return collapsed.replace(/\s/g, "_");
}

function errorText(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null;
}

function summarizeOperationResult(result: unknown): string {
    if (result == null) return JSON.stringify({ ok: true });
    if (!isRecord(result)) return JSON.stringify(result);

    if ("persisted_entries_removed" in result) {
        return JSON.stringify({
            runtime_state_cleared: result.runtime_state_cleared,
            persisted_entries_removed: result.persisted_entries_removed,
            storage_path: result.storage_path,
            capture_stop_attempted: result.capture_stop_attempted,
            capture_stop_succeeded: result.capture_stop_succeeded,
            capture_stop_error_kind: result.capture_stop_error_kind ?? null,
            clear_aborted: result.clear_aborted,
        });
    }

    if ("persisted_entries" in result) {
        return JSON.stringify({
            runtime_state_present: result.runtime_state_present,
            persisted_entries: result.persisted_entries,
            storage_path: result.storage_path,
        });
    }

    if ("transcript_added" in result) {
        return JSON.stringify({
            transcript_added: result.transcript_added,
            transcript_index: result.transcript_index,
            transcript_source: result.transcript_source,
            segment_id: result.segment_id,
            text_length: result.text_length,
            audio_file_extension: result.audio_file_extension,
            file_size_bytes: result.file_size_bytes,
            stt_boundary: result.stt_boundary,
            source_audio_path_redacted: result.source_audio_path_redacted,
            managed_audio_path_redacted: result.managed_audio_path_redacted,
            cleanup_requested: result.cleanup_requested,
            cleanup_performed: result.cleanup_performed,
            cleanup_error: result.cleanup_error ?? null,
        });
    }

    if ("sessions" in result && Array.isArray(result.sessions)) {
        return JSON.stringify({
            sessions: result.sessions.length,
            next_cursor: result.next_cursor ?? null,
            diagnostics: Array.isArray(result.diagnostics) ? result.diagnostics.length : 0,
        });
    }

    if ("results" in result && Array.isArray(result.results)) {
        return JSON.stringify({
            results: result.results.length,
            searched_session_count: result.searched_session_count,
            matched_session_count: result.matched_session_count,
            truncated: result.truncated,
            corrupt_archive_count: result.corrupt_archive_count,
        });
    }

    if ("archive" in result && isRecord(result.archive)) {
        const archive = result.archive;
        const state = isRecord(archive.state) ? archive.state : {};
        const transcript = Array.isArray(state.transcript) ? state.transcript.length : 0;
        return JSON.stringify({
            session_id: archive.session_id,
            transcript,
            intelligence_present: Boolean(state.intelligence),
        });
    }

    if ("content_length" in result) {
        return JSON.stringify({
            session_id: result.session_id,
            format: result.format,
            filename: result.filename,
            content_length: result.content_length,
            diagnostics: Array.isArray(result.diagnostics) ? result.diagnostics.length : 0,
        });
    }

    if ("session_id" in result && "transcript" in result) {
        const transcript = Array.isArray(result.transcript) ? result.transcript.length : 0;
        const actionItems = Array.isArray(result.action_items) ? result.action_items.length : 0;
        const decisions = Array.isArray(result.decisions) ? result.decisions.length : 0;
        return JSON.stringify({
            session_id: result.session_id,
            platform: result.platform,
            transcript,
            action_items: actionItems,
            decisions,
        });
    }

    if ("session_id" in result) {
        return JSON.stringify({
            session_id: result.session_id,
            platform: result.platform,
            status: result.status,
            session_mode: result.session_mode,
        });
    }

    return JSON.stringify(result);
}

function deadlineToIso(value: string): string | null {
    if (!value) return null;
    const parsed = new Date(`${value}T23:59:00`);
    return Number.isNaN(parsed.getTime()) ? null : parsed.toISOString();
}

function formatEntryTime(value: string): string {
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return value;
    return parsed.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
    });
}

function formatSessionDate(value?: string | null): string {
    if (!value) return "unknown";
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return value;
    return parsed.toLocaleString([], {
        month: "short",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
    });
}

function artifactKindLabel(kind: string): string {
    return kind
        .replace(/^intelligence_/, "")
        .replace(/_/g, " ")
        .replace(/\b\w/g, (character) => character.toUpperCase());
}

function recallRelationLabel(relation?: string | null): string {
    switch (relation) {
        case "linked_screen_context":
            return "linked screen context";
        case "temporal_screen_context":
            return "nearest screen context";
        case "same_session_screen_context":
            return "same-session screen context";
        case "direct_match":
            return "direct evidence match";
        default:
            return "evidence match";
    }
}

function sttCompletenessLabel(status?: string | null): string {
    switch (status) {
        case "complete":
            return "STT complete";
        case "complete_no_speech":
            return "STT complete/no speech";
        case "captured_untranscribed":
        case "incomplete_pending_queue":
            return "STT incomplete - pending queue";
        case "incomplete_drain_timeout":
            return "STT incomplete - drain timed out";
        case "incomplete_in_flight":
            return "STT incomplete - in flight";
        case "incomplete_failed_segments":
            return "STT incomplete - failed segments";
        case "incomplete_timeouts":
            return "STT incomplete - timeout";
        case "unavailable":
            return "STT unavailable";
        case "incomplete":
            return "STT incomplete";
        default:
            return status ? `STT ${status}` : "STT unknown";
    }
}

function hasIncompleteTranscriptionMetadata(metadata: unknown): boolean {
    return (
        isRecord(metadata) &&
        metadata.meeting_segment_transcription_incomplete === true
    );
}

function relativeTimestamp(value?: string | null): string {
    if (!value) return "never";
    const parsed = new Date(value).getTime();
    if (Number.isNaN(parsed)) return "unknown";
    const diffSeconds = Math.max(0, Math.round((Date.now() - parsed) / 1000));
    if (diffSeconds < 10) return "just now";
    if (diffSeconds < 60) return `${diffSeconds}s ago`;
    const diffMinutes = Math.round(diffSeconds / 60);
    if (diffMinutes < 60) return `${diffMinutes}m ago`;
    const diffHours = Math.round(diffMinutes / 60);
    return `${diffHours}h ago`;
}

function callLooksLikeGoogleMeet(callInfo: CallInfo | null): boolean {
    if (!callInfo?.is_active_call) return false;
    const haystack = `${callInfo.platform} ${callInfo.process_name} ${callInfo.window_title}`.toLowerCase();
    return haystack.includes("google meet") || haystack.includes("meet.google") || haystack.includes("google_meet");
}

function transcriptEmptyCopy(uiState: MeetingUiState, sessionMode?: string | null): string {
    if (uiState === "recording" || uiState === "transcribing") {
        return "Recording has started; waiting for the first audio segment.";
    }
    if (sessionMode === "manual") {
        return "Manual session is active. Add a manual transcript or transcribe a .wav file.";
    }
    return "Astra is not recording.";
}

function speakerFallbackForSource(source?: string | null): string {
    switch (source) {
        case "microphone":
            return "You";
        case "system_audio":
            return "Speaker 1";
        case "manual":
            return "Manual";
        case "imported_file":
            return "Imported";
        default:
            return "Unknown";
    }
}

function speakerDisplayName(entry: TranscriptEntry): string {
    return entry.speaker_label?.trim() || entry.speaker?.trim() || speakerFallbackForSource(entry.source);
}

function transcriptOrderValue(entry: TranscriptEntry): number {
    if (typeof entry.start_ms === "number") return entry.start_ms;
    const timestamp = new Date(entry.timestamp).getTime();
    if (Number.isFinite(timestamp)) return timestamp;
    const createdAt = new Date(entry.created_at).getTime();
    return Number.isFinite(createdAt) ? createdAt : 0;
}

export function MeetingDebugPanel({ capabilities }: MeetingDebugPanelProps) {
    const meeting = useMeeting();
    const transcriptRef = useRef<HTMLElement | null>(null);
    const [platform, setPlatform] = useState(DEFAULT_PLATFORM);
    const [consent, setConsent] = useState<ConsentState | null>(null);
    const [activeSession, setActiveSession] = useState<MeetingSession | null>(null);
    const [activeState, setActiveState] = useState<MeetingSessionState | null>(null);
    const [lastCompletedState, setLastCompletedState] = useState<MeetingSessionState | null>(null);
    const [callInfo, setCallInfo] = useState<CallInfo | null>(null);
    const [devices, setDevices] = useState<string[]>([]);
    const [backend, setBackend] = useState<CaptureBackend>("default");
    const [liveCapabilities, setLiveCapabilities] = useState<MeetingLiveCapabilitySnapshot | null>(null);
    const [lastResult, setLastResult] = useState<string | null>(null);
    const [lastError, setLastError] = useState<string | null>(null);
    const [refreshWarnings, setRefreshWarnings] = useState<Record<string, string>>({});
    const [isBusy, setIsBusy] = useState(false);
    const [captureMode, setCaptureMode] = useState<CaptureMode>("both");
    const [transcriptSpeaker, setTranscriptSpeaker] = useState("unknown");
    const [transcriptText, setTranscriptText] = useState("");
    const [transcriptConfidence, setTranscriptConfidence] = useState("0.95");
    const [audioPath, setAudioPath] = useState("");
    const [audioSpeaker, setAudioSpeaker] = useState("unknown");
    const [cleanupAudioFile, setCleanupAudioFile] = useState(true);
    const [actionDescription, setActionDescription] = useState("");
    const [actionAssignee, setActionAssignee] = useState("");
    const [actionDeadline, setActionDeadline] = useState("");
    const [decisionText, setDecisionText] = useState("");
    const [decisionRationale, setDecisionRationale] = useState("");
    const [decisionMadeBy, setDecisionMadeBy] = useState("");
    const [speakerRenameId, setSpeakerRenameId] = useState("");
    const [speakerRenameName, setSpeakerRenameName] = useState("");
    const [clearPreview, setClearPreview] = useState<MeetingDataClearPreview | null>(null);
    const [showClearConfirmation, setShowClearConfirmation] = useState(false);
    const [clearPhrase, setClearPhrase] = useState("");
    const [isGeneratingIntelligence, setIsGeneratingIntelligence] = useState(false);
    const [isStoppingSession, setIsStoppingSession] = useState(false);
    const [isAttachingScreenContext, setIsAttachingScreenContext] = useState(false);
    const [expandedEvidenceKeys, setExpandedEvidenceKeys] = useState<Record<string, boolean>>({});
    const [sessionMemory, setSessionMemory] = useState<MeetingSessionListItem[]>([]);
    const [sessionMemoryDiagnostics, setSessionMemoryDiagnostics] = useState<MeetingDiagnostic[]>([]);
    const [sessionSearchQuery, setSessionSearchQuery] = useState("");
    const [sessionSearchResults, setSessionSearchResults] = useState<MeetingSessionSearchResult[]>([]);
    const [recallQuery, setRecallQuery] = useState("");
    const [recallResponse, setRecallResponse] = useState<MeetingRecallResponse | null>(null);
    const [openedArchive, setOpenedArchive] = useState<MeetingSessionArchiveDocument | null>(null);
    const [archiveExport, setArchiveExport] = useState<MeetingSessionExportResponse | null>(null);
    const [isSessionMemoryBusy, setIsSessionMemoryBusy] = useState(false);
    const [isRecallBusy, setIsRecallBusy] = useState(false);

    const meetingTools = useMemo(
        () => capabilities?.tools.filter((tool) => tool.category === "meeting") ?? [],
        [capabilities]
    );
    const toolByName = useCallback(
        (name: string) => meetingTools.find((tool) => tool.tool_name === name),
        [meetingTools]
    );

    const normalizedPlatform = useMemo(() => normalizePlatformKey(platform), [platform]);
    const displayedState = activeSession ? activeState : lastCompletedState;
    const stateKind = activeSession ? "active" : lastCompletedState ? "last completed" : "none";
    const hasActiveSession = activeSession !== null;
    const currentStatus = displayedState?.status ?? activeSession?.status ?? null;
    const currentStatusKind = statusKind(currentStatus);
    const sessionMode = activeSession?.session_mode ?? displayedState?.session.session_mode ?? null;
    const transcriptEntries = displayedState?.transcript ?? [];
    const displayedTranscriptEntries = useMemo(
        () =>
            [...transcriptEntries].sort((left, right) => {
                const order = transcriptOrderValue(right) - transcriptOrderValue(left);
                if (order !== 0) return order;
                return right.created_at.localeCompare(left.created_at);
            }),
        [transcriptEntries]
    );
    const transcriptBySegmentId = useMemo(
        () => new Map(transcriptEntries.map((entry) => [entry.segment_id, entry])),
        [transcriptEntries]
    );
    const archivedTranscriptBySegmentId = useMemo(
        () => new Map((openedArchive?.state.transcript ?? []).map((entry) => [entry.segment_id, entry])),
        [openedArchive]
    );
    const speakers = displayedState?.speakers ?? [];
    const notes = displayedState?.notes ?? [];
    const summaries = displayedState?.summary ?? [];
    const actionItems = displayedState?.action_items ?? [];
    const decisions = displayedState?.decisions ?? [];
    const intelligence = displayedState?.intelligence ?? null;
    const screenContexts = displayedState?.screen_contexts ?? [];
    const diagnostics = displayedState?.diagnostics ?? [];
    const metrics = liveCapabilities?.capture_health.metrics;
    const finalizationStatus: MeetingFinalizationStatus | null = liveCapabilities?.finalization_status ?? null;
    const systemHealth = liveCapabilities?.system_capture_health;
    const microphoneHealth = liveCapabilities?.microphone_capture_health;
    const systemMetrics = systemHealth?.metrics;
    const microphoneMetrics = microphoneHealth?.metrics;
    const segmentsWritten = metrics?.segments_written ?? 0;
    const segmentsTranscribed = metrics?.segments_transcribed ?? 0;
    const segmentsQueuedTotal = metrics?.segments_queued_total ?? metrics?.segments_queued ?? 0;
    const currentQueueDepth = metrics?.current_queue_depth ?? 0;
    const segmentsInFlight = metrics?.segments_in_flight ?? 0;
    const segmentsFailed = metrics?.segments_failed ?? metrics?.segment_transcription_failures_total ?? 0;
    const segmentTranscriptionTimeouts = metrics?.segment_transcription_timeouts ?? 0;
    const pendingSegments = currentQueueDepth + segmentsInFlight;
    const missingTranscriptSegments = Math.max(0, segmentsWritten - segmentsTranscribed);
    const capturedWithoutTranscript =
        segmentsWritten > 0 && segmentsTranscribed === 0 && transcriptEntries.length === 0;
    const drainStatus = metrics?.segment_transcription_drain_status ?? "idle";
    const drainTimedOut = metrics?.drain_timeout === true;
    const finalizingMeeting = Boolean(
        finalizationStatus &&
        !["idle", "completed", "completed_partial", "failed_recoverable", "failed"].includes(finalizationStatus.stage)
    );
    const finalizingStt = isStoppingSession || drainStatus === "running" || finalizationStatus?.stage === "draining_stt";
    const lastTranscriptAt = transcriptEntries.length
        ? displayedTranscriptEntries[0]?.timestamp
        : null;
    const lastSummaryTimestamp = displayedState?.summary.length
        ? displayedState.summary[displayedState.summary.length - 1]?.timestamp
        : null;
    const audioCaptureTool = toolByName("meeting.audio.capture");
    const systemAudioCaptureTool = toolByName("meeting.audio.capture.system");
    const microphoneCaptureTool = toolByName("meeting.audio.capture.microphone");
    const segmentTranscriptionTool = toolByName("meeting.transcription.segment");
    const screenContextTool = toolByName("meeting.screen_context.attach_current");
    const googleMeetDetected = callLooksLikeGoogleMeet(callInfo);

    const realCaptureReadiness = useMemo<CaptureReadiness>(() => {
        const backendAvailable = liveCapabilities?.windows_wasapi_capture.available === true;
        const selectedSourceCaptureTools = [
            ...(captureMode === "system_audio" || captureMode === "both"
                ? [{ label: "MeetingAudioCaptureSystem", tool: systemAudioCaptureTool }]
                : []),
            ...(captureMode === "microphone" || captureMode === "both"
                ? [{ label: "MeetingAudioCaptureMicrophone", tool: microphoneCaptureTool }]
                : []),
        ];
        const audioToolAvailable = audioCaptureTool?.available === true;
        const sourceToolsAvailable = selectedSourceCaptureTools.every(({ tool }) => tool?.available === true);
        const segmentToolAvailable = segmentTranscriptionTool?.available === true;
        const audioPermissionReady = toolReady(audioCaptureTool);
        const sourcePermissionsReady = selectedSourceCaptureTools.every(({ tool }) => toolReady(tool));
        const segmentPermissionReady = toolReady(segmentTranscriptionTool);
        const approvalRequired =
            audioCaptureTool?.requires_approval === true ||
            selectedSourceCaptureTools.some(
                ({ tool }) => tool?.requires_approval === true || tool?.state === "approval_gated"
            ) ||
            segmentTranscriptionTool?.requires_approval === true ||
            audioCaptureTool?.state === "approval_gated" ||
            segmentTranscriptionTool?.state === "approval_gated";
        const blockedTools: string[] = [];

        if (audioCaptureTool && (!audioToolAvailable || !audioCaptureTool.enabled)) {
            blockedTools.push("MeetingAudioCapture");
        }
        selectedSourceCaptureTools.forEach(({ label, tool }) => {
            if (!tool || !tool.available || !tool.enabled) {
                blockedTools.push(label);
            }
        });
        if (segmentTranscriptionTool && (!segmentToolAvailable || !segmentTranscriptionTool.enabled)) {
            blockedTools.push("MeetingTranscriptionSegment");
        }

        let blockedReason: string | null = null;
        if (!liveCapabilities) {
            blockedReason = "Live capture capability state has not been loaded.";
        } else if (!backendAvailable) {
            blockedReason = liveCapabilities.windows_wasapi_capture.reason ?? "WASAPI capture backend is unavailable.";
        } else if (!audioToolAvailable) {
            blockedReason = audioCaptureTool?.disabled_reason ?? "Meeting audio capture tool is unavailable.";
        } else if (!sourceToolsAvailable) {
            const blockedSource = selectedSourceCaptureTools.find(({ tool }) => tool?.available !== true);
            blockedReason = blockedSource?.tool?.disabled_reason ?? "Selected source capture tool is unavailable.";
        } else if (!segmentToolAvailable) {
            blockedReason = segmentTranscriptionTool?.disabled_reason ?? "Meeting segment transcription tool is unavailable.";
        } else if (!audioPermissionReady || !sourcePermissionsReady || !segmentPermissionReady) {
            blockedReason = "Required meeting capture permissions are disabled.";
        }

        return {
            backendAvailable,
            audioToolAvailable,
            segmentToolAvailable,
            audioPermissionReady,
            segmentPermissionReady,
            approvalRequired,
            blockedReason,
            blockedTools,
        };
    }, [
        audioCaptureTool,
        captureMode,
        liveCapabilities,
        microphoneCaptureTool,
        segmentTranscriptionTool,
        systemAudioCaptureTool,
    ]);

    const consentReady =
        consent?.given === true &&
        consent.global_enabled === true &&
        consent.per_app?.[normalizedPlatform] === true;

    const startReadiness = useMemo<MeetingStartReadiness>(() => {
        const hardBlockers: string[] = [];
        const softGates: string[] = [];
        if (consent === null) {
            pushUnique(hardBlockers, "Consent state not loaded");
        } else if (!consentReady) {
            pushUnique(hardBlockers, "Consent required");
        }
        if (!liveCapabilities) {
            pushUnique(hardBlockers, "Recording readiness not loaded");
        } else if (!realCaptureReadiness.backendAvailable) {
            pushUnique(hardBlockers, "WASAPI backend unavailable");
        }
        realCaptureReadiness.blockedTools.forEach((tool) => pushUnique(hardBlockers, tool));
        if (realCaptureReadiness.approvalRequired) {
            pushUnique(softGates, "Approval/confirmation required on click");
        }

        const canRequestStart = hardBlockers.length === 0;
        const requiresApprovalOrConfirmation = softGates.length > 0;
        const statusLabel = canRequestStart
            ? requiresApprovalOrConfirmation
                ? "Recording ready for governed start"
                : "Recording ready"
            : "Recording unavailable";
        const primaryHelpText = canRequestStart
            ? requiresApprovalOrConfirmation
                ? "Ready for governed start. Confirmation/audit will occur when you click Start."
                : "Ready to start capture."
            : "No pending approval exists because the request is blocked before approval creation.";

        return {
            canRequestStart,
            requiresApprovalOrConfirmation,
            hardBlockers,
            softGates,
            statusLabel,
            primaryHelpText,
        };
    }, [consent, consentReady, liveCapabilities, realCaptureReadiness]);

    const uiSummary = useMemo<MeetingUiSummary>(() => {
        const isRecording =
            activeSession?.capture_active === true ||
            liveCapabilities?.capture_health.state === "capturing" ||
            currentStatusKind === "capturing";
        const isTranscribing =
            isStoppingSession ||
            ((isRecording || currentStatusKind === "stopping") &&
                (currentStatusKind === "transcribing" ||
                    pendingSegments > 0 ||
                    drainStatus === "running"));
        const failureReason =
            statusFailureReason(currentStatus) ??
            liveCapabilities?.capture_health.last_error ??
            metrics?.last_backend_error_message ??
            metrics?.last_segment_transcription_error_kind ??
            null;
        const poisonQuarantined = diagnostics.some((diagnostic) =>
            diagnostic.code.includes("capture_controller_poisoned") ||
            diagnostic.code.includes("capture_controller_recovered")
        );
        const recoverableCaptureFailure =
            hasActiveSession &&
            sessionMode === "real_capture" &&
            activeSession?.capture_active !== true &&
            (currentStatusKind === "failed" ||
                liveCapabilities?.capture_health.state === "failed" ||
                liveCapabilities?.capture_health.status === "failed") &&
            (
                failureReason?.includes("capture") ||
                failureReason?.includes("poison") ||
                poisonQuarantined ||
                currentStatusKind === "failed"
            );

        if (liveCapabilities?.capture_summary_status === "degraded") {
            return {
                state: "degraded",
                title: "Partial capture active",
                description:
                    liveCapabilities.capture_summary_reason ??
                    `Active sources: ${liveCapabilities.active_sources.join(", ") || "unknown"}. Failed sources: ${liveCapabilities.failed_sources.join(", ") || "none"}.`,
                nextAction: "You can continue with the available source or stop/finalize the session.",
                blockingReasons: liveCapabilities.failed_sources,
                isRecording: true,
                isTranscribing,
            };
        }

        if (
            currentStatusKind === "failed" ||
            currentStatusKind === "error" ||
            liveCapabilities?.capture_health.state === "failed" ||
            liveCapabilities?.capture_health.status === "failed"
        ) {
            if (recoverableCaptureFailure) {
                return {
                    state: "failed_recoverable",
                    title: "Capture failed / recoverable",
                    description: poisonQuarantined
                        ? "Capture controller entered an inconsistent state and was quarantined. No audio data is being captured."
                        : failureReason
                          ? `Capture failed, but the session can be recovered. Reason: ${failureReason}`
                          : "Capture failed, but the session can be recovered.",
                    nextAction: "Force stop the session to archive what exists, or recover capture state before retrying.",
                    blockingReasons: failureReason ? [failureReason] : [],
                    isRecording: false,
                    isTranscribing,
                };
            }
            return {
                state: "failed",
                title: "Capture failed",
                description: failureReason ? `Reason: ${failureReason}` : "The meeting capture pipeline reported a failure.",
                nextAction: "Check advanced diagnostics or restart capture.",
                blockingReasons: failureReason ? [failureReason] : [],
                isRecording,
                isTranscribing,
            };
        }

        if (isStoppingSession || currentStatusKind === "stopping" || drainStatus === "running") {
            return {
                state: "transcribing",
                title: "Stopping / draining STT",
                description: `Astra is finalizing captured audio before archive/export. Queue: ${currentQueueDepth}; in flight: ${segmentsInFlight}; transcribed: ${segmentsTranscribed}/${segmentsWritten}.`,
                nextAction: "Wait for bounded STT drain to finish or time out.",
                blockingReasons: [],
                isRecording,
                isTranscribing: true,
            };
        }

        if (currentStatusKind === "paused" || liveCapabilities?.capture_health.state === "paused") {
            return {
                state: "paused",
                title: "Paused",
                description: "Audio capture is paused. Astra is not recording new audio right now.",
                nextAction: "Resume to continue capture, or stop the session.",
                blockingReasons: [],
                isRecording: false,
                isTranscribing: false,
            };
        }

        if (isTranscribing) {
            return {
                state: "transcribing",
                title: "Transcribing",
                description: `Astra is processing captured audio with the existing STT pipeline. Last transcript: ${relativeTimestamp(lastTranscriptAt)}.`,
                nextAction: pendingSegments > 0 || drainStatus === "running" ? "Wait for queued segments to finish." : "Keep recording or stop when finished.",
                blockingReasons: [],
                isRecording,
                isTranscribing,
            };
        }

        if (isRecording) {
            return {
                state: "recording",
                title: "Recording",
                description: `Astra is capturing the selected audio sources. Segments: ${segmentsWritten} written / ${segmentsTranscribed} transcribed.`,
                nextAction: "Keep the meeting audio playing, or stop when finished.",
                blockingReasons: [],
                isRecording,
                isTranscribing: false,
            };
        }

        if (hasActiveSession && sessionMode === "manual") {
            const description = googleMeetDetected
                ? "Astra is not listening to Google Meet. Browser or call detection is not recording."
                : "Astra is not listening to system audio. You can add notes manually or transcribe a .wav file.";
            return {
                state: "manual_active",
                title: "Manual session active - not recording",
                description,
                nextAction: startReadiness.hardBlockers.length
                    ? "Enable the blocked permissions, then start recording."
                    : startReadiness.requiresApprovalOrConfirmation
                      ? "Recording ready for governed start. Confirmation/audit will occur when you click Start."
                      : "Use manual tools, transcribe a .wav file, or start capture.",
                blockingReasons: startReadiness.hardBlockers,
                isRecording: false,
                isTranscribing: false,
            };
        }

        if (currentStatusKind === "completed") {
            return {
                state: "completed",
                title: "Completed",
                description: "The last meeting session has ended. Transcript and notes remain visible until cleared.",
                nextAction: "Review the transcript or clear meeting data when finished.",
                blockingReasons: startReadiness.hardBlockers,
                isRecording: false,
                isTranscribing: false,
            };
        }

        if (currentStatusKind === "stopped") {
            return {
                state: "completed",
                title: "Stopped",
                description: "The last meeting session has ended. Transcript and notes remain visible until cleared.",
                nextAction: "Review the transcript or clear meeting data when finished.",
                blockingReasons: startReadiness.hardBlockers,
                isRecording: false,
                isTranscribing: false,
            };
        }

        if (!startReadiness.canRequestStart) {
            return {
                state: startReadiness.hardBlockers.length ? "blocked_permissions" : "not_started",
                title: startReadiness.hardBlockers.length ? "Recording blocked" : "Not started",
                description: startReadiness.hardBlockers.length
                    ? "Astra cannot start WASAPI recording because required consent, backend, or permissions are not ready."
                    : "Astra is not recording or transcribing.",
                nextAction: startReadiness.hardBlockers.length
                    ? "Resolve the blocked items below, then start recording."
                    : "Grant consent and start a manual session or recording.",
                blockingReasons: startReadiness.hardBlockers,
                isRecording: false,
                isTranscribing: false,
            };
        }

        return {
            state: "ready_to_record",
            title: startReadiness.statusLabel,
            description: startReadiness.requiresApprovalOrConfirmation
                ? "This action is high-risk and will be confirmed/audited when you click Start. No pending approval exists yet."
                : "Consent is granted and the selected capture path is available.",
            nextAction: startReadiness.requiresApprovalOrConfirmation
                ? "Click Start recording to enter the governed confirmation path."
                : "Start recording.",
            blockingReasons: [],
            isRecording: false,
            isTranscribing: false,
        };
    }, [
        activeSession?.capture_active,
        consentReady,
        currentStatus,
        currentStatusKind,
        currentQueueDepth,
        diagnostics,
        googleMeetDetected,
        hasActiveSession,
        isStoppingSession,
        lastTranscriptAt,
        liveCapabilities,
        metrics,
        pendingSegments,
        segmentsInFlight,
        segmentsTranscribed,
        segmentsWritten,
        sessionMode,
        startReadiness,
    ]);

    const startRecordingDisabledReason = startReadiness.canRequestStart
        ? startReadiness.primaryHelpText
        : startReadiness.hardBlockers.join("; ") || "Recording unavailable";
    const canStartRealCapture = startReadiness.canRequestStart;

    const refreshMeeting = useCallback(async () => {
        const results = await Promise.allSettled([
            meeting.getConsentState(),
            meeting.getActiveSession(),
            meeting.getActiveState(),
            meeting.getLastCompletedState(),
            meeting.getAvailableAudioDevices(),
            meeting.autoDetectAudioBackend(),
            meeting.getLiveCapabilities(),
            meeting.listSessions({ limit: 8 }),
        ]);
        const warnings: Record<string, string> = {};

        const [
            nextConsent,
            nextSession,
            nextState,
            nextLastCompleted,
            nextDevices,
            nextBackend,
            nextLiveCapabilities,
            nextSessionMemory,
        ] = results;

        if (nextConsent.status === "fulfilled") setConsent(nextConsent.value);
        else warnings.consent = errorText(nextConsent.reason);

        if (nextSession.status === "fulfilled") setActiveSession(nextSession.value);
        else warnings.session = errorText(nextSession.reason);

        if (nextState.status === "fulfilled") setActiveState(nextState.value);
        else warnings.state = errorText(nextState.reason);

        if (nextLastCompleted.status === "fulfilled") setLastCompletedState(nextLastCompleted.value);
        else warnings.last_completed = errorText(nextLastCompleted.reason);

        if (nextDevices.status === "fulfilled") setDevices(nextDevices.value);
        else warnings.devices = errorText(nextDevices.reason);

        if (nextBackend.status === "fulfilled") setBackend(nextBackend.value);
        else warnings.backend = errorText(nextBackend.reason);

        if (nextLiveCapabilities.status === "fulfilled") setLiveCapabilities(nextLiveCapabilities.value);
        else warnings.live_capabilities = errorText(nextLiveCapabilities.reason);

        if (nextSessionMemory.status === "fulfilled") {
            setSessionMemory(nextSessionMemory.value.sessions);
            setSessionMemoryDiagnostics(nextSessionMemory.value.diagnostics);
        } else {
            warnings.session_memory = errorText(nextSessionMemory.reason);
        }

        setRefreshWarnings(warnings);
    }, [meeting]);

    useEffect(() => {
        void refreshMeeting();
    }, [refreshMeeting]);

    useEffect(() => {
        let cancelled = false;
        const unlisteners: Array<() => void> = [];
        const eventNames = [
            "meeting-finalization-updated",
            "meeting-session-updated",
            "meeting-transcript-updated",
            "meeting-artifacts-updated",
            "meeting-diagnostics-updated",
        ];

        const subscribe = async () => {
            for (const eventName of eventNames) {
                const unlisten = await listen(eventName, () => {
                    if (!cancelled) void refreshMeeting();
                });
                unlisteners.push(unlisten);
            }
        };

        void subscribe().catch((error) => {
            if (!cancelled) {
                setRefreshWarnings((current) => ({
                    ...current,
                    live_updates: errorText(error),
                }));
            }
        });

        return () => {
            cancelled = true;
            unlisteners.forEach((unlisten) => unlisten());
        };
    }, [refreshMeeting]);

    useEffect(() => {
        const intervalMs = hasActiveSession || uiSummary.isRecording || uiSummary.isTranscribing || finalizingMeeting ? 1000 : 5000;
        const timer = window.setInterval(() => {
            void refreshMeeting();
        }, intervalMs);
        return () => window.clearInterval(timer);
    }, [finalizingMeeting, hasActiveSession, refreshMeeting, uiSummary.isRecording, uiSummary.isTranscribing]);

    const runOperation = useCallback(
        async (label: string, operation: () => Promise<unknown>, refreshAfter = true) => {
            try {
                setIsBusy(true);
                setLastError(null);
                const result = await operation();
                setLastResult(`${label}: ${summarizeOperationResult(result)}`);
                if (refreshAfter) {
                    await refreshMeeting();
                }
                return true;
            } catch (err) {
                setLastResult(null);
                setLastError(`${label}: ${errorText(err)}`);
                return false;
            } finally {
                setIsBusy(false);
            }
        },
        [refreshMeeting]
    );

    const handleDetect = useCallback(
        () =>
            runOperation(
                "detect call",
                async () => {
                    const detected = await meeting.detectActiveCall();
                    setCallInfo(detected);
                    return detected ?? { detection_state: "idle" };
                },
                false
            ),
        [meeting, runOperation]
    );

    const handleAddTranscript = useCallback(async () => {
        const text = transcriptText.trim();
        if (!text) {
            setLastError("add transcript: text is required");
            return;
        }
        const parsedConfidence = Number.parseFloat(transcriptConfidence);
        const confidence = Number.isFinite(parsedConfidence)
            ? Math.max(0, Math.min(1, parsedConfidence))
            : 1;
        const entry: TranscriptEntry = {
            segment_id: makeLocalId(),
            session_id: activeSession?.session_id ?? "",
            source: "manual",
            timestamp: new Date().toISOString(),
            created_at: new Date().toISOString(),
            speaker: transcriptSpeaker.trim() || "unknown",
            speaker_id: null,
            text,
            confidence,
            start_ms: null,
            end_ms: null,
            stt_model: null,
            audio_backend: null,
        };
        const ok = await runOperation("add transcript", () => meeting.addTranscript(entry));
        if (ok) setTranscriptText("");
    }, [activeSession?.session_id, meeting, runOperation, transcriptConfidence, transcriptSpeaker, transcriptText]);

    const handleTranscribeAudioFile = useCallback(async () => {
        const path = audioPath.trim();
        if (!path) {
            setLastError("transcribe file: .wav file path is required");
            return;
        }
        const ok = await runOperation("transcribe file", () =>
            meeting.transcribeAudioFile({
                session_id: activeSession?.session_id ?? null,
                audio_path: path,
                speaker: audioSpeaker.trim() || null,
                cleanup_after_transcription: cleanupAudioFile,
            })
        );
        if (ok && cleanupAudioFile) setAudioPath("");
    }, [activeSession?.session_id, audioPath, audioSpeaker, cleanupAudioFile, meeting, runOperation]);

    const handleAddActionItem = useCallback(async () => {
        const description = actionDescription.trim();
        if (!description) {
            setLastError("add action item: description is required");
            return;
        }
        const item: ActionItem = {
            timestamp: new Date().toISOString(),
            description,
            assignee: actionAssignee.trim()
                ? { name: actionAssignee.trim(), speaker_id: null }
                : null,
            deadline: deadlineToIso(actionDeadline),
            status: "open",
        };
        const ok = await runOperation("add action item", () => meeting.addActionItem(item));
        if (ok) {
            setActionDescription("");
            setActionAssignee("");
            setActionDeadline("");
        }
    }, [actionAssignee, actionDeadline, actionDescription, meeting, runOperation]);

    const handleAddDecision = useCallback(async () => {
        const decision = decisionText.trim();
        if (!decision) {
            setLastError("add decision: decision is required");
            return;
        }
        const entry: DecisionLogEntry = {
            timestamp: new Date().toISOString(),
            decision,
            rationale: decisionRationale.trim(),
            made_by: decisionMadeBy.trim()
                ? { name: decisionMadeBy.trim(), speaker_id: null }
                : null,
        };
        const ok = await runOperation("add decision", () => meeting.addDecision(entry));
        if (ok) {
            setDecisionText("");
            setDecisionRationale("");
            setDecisionMadeBy("");
        }
    }, [decisionMadeBy, decisionRationale, decisionText, meeting, runOperation]);

    const handleSelectSpeakerForRename = useCallback((speaker: SpeakerLabel) => {
        setSpeakerRenameId(speaker.speaker_id);
        setSpeakerRenameName(speaker.display_name);
    }, []);

    const handleRenameSpeaker = useCallback(async () => {
        const speakerId = speakerRenameId.trim();
        const displayName = speakerRenameName.trim();
        if (!speakerId || !displayName) {
            setLastError("rename speaker: speaker and display name are required");
            return;
        }
        await runOperation("rename speaker", () =>
            meeting.renameSpeaker({
                speaker_id: speakerId,
                display_name: displayName,
            })
        );
    }, [meeting, runOperation, speakerRenameId, speakerRenameName]);

    const handlePreviewClearData = useCallback(
        () =>
            runOperation(
                "preview clear data",
                async () => {
                    const preview = await meeting.previewClearData();
                    setClearPreview(preview);
                    setClearPhrase("");
                    setShowClearConfirmation(true);
                    return preview;
                },
                false
            ),
        [meeting, runOperation]
    );

    const handleConfirmClearData = useCallback(async () => {
        const ok = await runOperation("clear meeting data", () =>
            meeting.clearData({
                scope: "all",
                confirmation_phrase: clearPhrase,
            })
        );
        if (ok) {
            setClearPreview(null);
            setClearPhrase("");
            setShowClearConfirmation(false);
        }
    }, [clearPhrase, meeting, runOperation]);

    const handleGenerateIntelligence = useCallback(async () => {
        try {
            setIsGeneratingIntelligence(true);
            await runOperation("generate intelligence", () =>
                meeting.generateIntelligence({
                    use_local_llm: true,
                    max_transcript_segments: 120,
                })
            );
        } finally {
            setIsGeneratingIntelligence(false);
        }
    }, [meeting, runOperation]);

    const handleStopSession = useCallback(async () => {
        try {
            setIsStoppingSession(true);
            await runOperation("request stop session", meeting.requestStopSession);
        } finally {
            setIsStoppingSession(false);
        }
    }, [meeting, runOperation]);

    const handleRetryFinalization = useCallback(
        () => runOperation("retry finalization", meeting.retryFinalization),
        [meeting, runOperation]
    );

    const handleRecoverFailedCapture = useCallback(
        () => runOperation("recover capture", meeting.recoverFailedCapture),
        [meeting, runOperation]
    );

    const handleForceFinalizeFailedCapture = useCallback(async () => {
        try {
            setIsStoppingSession(true);
            await runOperation("force stop session", meeting.forceFinalizeFailedCapture);
        } finally {
            setIsStoppingSession(false);
        }
    }, [meeting, runOperation]);

    const handleAttachCurrentScreen = useCallback(async () => {
        try {
            setIsAttachingScreenContext(true);
            await runOperation("attach screen context", () =>
                meeting.attachCurrentScreen({
                    store_screenshot: false,
                    capture_fresh: true,
                    attachment_mode: "current_moment",
                })
            );
        } finally {
            setIsAttachingScreenContext(false);
        }
    }, [meeting, runOperation]);

    const handleClearIntelligence = useCallback(
        () => runOperation("clear intelligence", () => meeting.clearIntelligence()),
        [meeting, runOperation]
    );

    const handleOpenArchivedSession = useCallback(async (sessionId: string) => {
        try {
            setIsSessionMemoryBusy(true);
            setLastError(null);
            const response = await meeting.readSessionArchive({
                session_id: sessionId,
                include_transcript: true,
                include_intelligence: true,
                include_diagnostics: true,
            });
            setOpenedArchive(response.archive);
            setArchiveExport(null);
            setLastResult(`open archived session: ${summarizeOperationResult(response)}`);
        } catch (error) {
            setLastResult(null);
            setLastError(`open archived session: ${errorText(error)}`);
        } finally {
            setIsSessionMemoryBusy(false);
        }
    }, [meeting]);

    const handleSearchSessionMemory = useCallback(async () => {
        const query = sessionSearchQuery.trim();
        if (!query) {
            setSessionSearchResults([]);
            return;
        }
        try {
            setIsSessionMemoryBusy(true);
            setLastError(null);
            const response = await meeting.searchSessions({ query, limit: 20 });
            setSessionSearchResults(response.results);
            setSessionMemoryDiagnostics(response.diagnostics);
            setLastResult(`search session memory: ${summarizeOperationResult(response)}`);
        } catch (error) {
            setLastResult(null);
            setLastError(`search session memory: ${errorText(error)}`);
        } finally {
            setIsSessionMemoryBusy(false);
        }
    }, [meeting, sessionSearchQuery]);

    const handleAnswerRecall = useCallback(async () => {
        const query = recallQuery.trim();
        if (!query) {
            setRecallResponse(null);
            return;
        }
        try {
            setIsRecallBusy(true);
            setLastError(null);
            const response = await meeting.answerRecall({
                query,
                limit: 12,
                include_transcript: true,
                include_intelligence: true,
                include_screen_context: true,
                use_local_llm: true,
            });
            setRecallResponse(response);
            setLastResult(`ask session memory: ${summarizeOperationResult(response)}`);
        } catch (error) {
            setRecallResponse(null);
            setLastResult(null);
            setLastError(`ask session memory: ${errorText(error)}`);
        } finally {
            setIsRecallBusy(false);
        }
    }, [meeting, recallQuery]);

    const handleExportArchivedSession = useCallback(async (sessionId: string, format: "markdown" | "json") => {
        try {
            setIsSessionMemoryBusy(true);
            setLastError(null);
            const response = await meeting.exportSessionArchive({ session_id: sessionId, format });
            setArchiveExport(response);
            setLastResult(`export archived session: ${summarizeOperationResult(response)}`);
        } catch (error) {
            setLastResult(null);
            setLastError(`export archived session: ${errorText(error)}`);
        } finally {
            setIsSessionMemoryBusy(false);
        }
    }, [meeting]);

    const handleReindexSessionMemory = useCallback(async () => {
        try {
            setIsSessionMemoryBusy(true);
            setLastError(null);
            const response = await meeting.reindexSessions();
            setSessionMemory(response.sessions);
            setSessionMemoryDiagnostics(response.diagnostics);
            setLastResult(`reindex session memory: ${summarizeOperationResult(response)}`);
        } catch (error) {
            setLastResult(null);
            setLastError(`reindex session memory: ${errorText(error)}`);
        } finally {
            setIsSessionMemoryBusy(false);
        }
    }, [meeting]);

    const handleCopyFollowUpDraft = useCallback(async () => {
        const draft = intelligence?.follow_up_draft;
        if (!draft) return;
        const text = `Subject: ${draft.subject}\n\n${draft.body}`;
        try {
            await navigator.clipboard.writeText(text);
            setLastResult("copy follow-up draft: copied to clipboard");
            setLastError(null);
        } catch (error) {
            setLastError(`copy follow-up draft: ${errorText(error)}`);
        }
    }, [intelligence]);

    const renderEvidence = useCallback(
        (key: string, ids?: string[] | null) => (
            <EvidencePreview
                ids={ids}
                transcriptBySegmentId={transcriptBySegmentId}
                expanded={expandedEvidenceKeys[key] === true}
                onToggle={() =>
                    setExpandedEvidenceKeys((prev) => ({
                        ...prev,
                        [key]: !prev[key],
                    }))
                }
            />
        ),
        [expandedEvidenceKeys, transcriptBySegmentId]
    );

    const scrollToTranscript = useCallback(() => {
        transcriptRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, []);

    const openedArchiveIntelligence = openedArchive?.state.intelligence ?? null;
    const openedArchiveSummary = openedArchiveIntelligence?.summary ?? null;
    const openedArchiveScreenContexts =
        openedArchive?.screen_contexts?.length
            ? openedArchive.screen_contexts
            : openedArchive?.state.screen_contexts ?? [];
    const openedArchiveTitle =
        openedArchive ? sessionMemory.find((item) => item.session_id === openedArchive.session_id)?.title ?? openedArchive.session_id : null;

    return (
        <div className="desktop-agent-section meeting-control-center">
            <section className={`desktop-agent-card meeting-status-card meeting-status-card--${uiSummary.state}`}>
                <div className="meeting-status-card__header">
                    <div>
                        <p className="meeting-section-kicker">Overview</p>
                        <h3>Meeting / Work Session</h3>
                        <p>Governed meeting capture and transcription</p>
                    </div>
                    <span className="meeting-status-pill">{uiSummary.title}</span>
                </div>

                <div className="meeting-status-main">
                    <div>
                        <p className="meeting-status-label">Status</p>
                        <h2>{uiSummary.title}</h2>
                        <p className="meeting-status-description">{uiSummary.description}</p>
                        {googleMeetDetected && !uiSummary.isRecording ? (
                            <p className="meeting-call-warning">
                                Google Meet may be open, but Astra is not recording. Start capture to listen to the selected audio sources.
                            </p>
                        ) : null}
                        {uiSummary.nextAction ? (
                            <p className="meeting-next-action"><strong>Next step:</strong> {uiSummary.nextAction}</p>
                        ) : null}
                    </div>
                    <div className="meeting-state-facts">
                        <span>Audio capture: <strong>{uiSummary.isRecording ? "active" : "inactive"}</strong></span>
                        <span>Transcription: <strong>{uiSummary.isTranscribing ? "active" : "inactive"}</strong></span>
                        <span>Session: <strong>{hasActiveSession ? uiSummary.title : stateKind}</strong></span>
                    </div>
                </div>

                <div className="meeting-runtime-status-grid">
                    <article>
                        <span>Session lifecycle</span>
                        <strong>{formatStatus(currentStatus)}</strong>
                    </article>
                    <article>
                        <span>Microphone capture</span>
                        <strong>{microphoneHealth?.state ?? liveCapabilities?.microphone_capture.state ?? "unknown"}</strong>
                        <small>
                            {microphoneMetrics?.segments_written ?? 0} written / {microphoneMetrics?.segments_transcribed ?? 0} transcribed
                            {liveCapabilities?.microphone_capture.reason ? ` - ${liveCapabilities.microphone_capture.reason}` : ""}
                        </small>
                    </article>
                    <article>
                        <span>System audio capture</span>
                        <strong>{systemHealth?.state ?? liveCapabilities?.system_audio_capture.state ?? "unknown"}</strong>
                        <small>
                            {systemMetrics?.segments_written ?? 0} written / {systemMetrics?.segments_transcribed ?? 0} transcribed
                            {liveCapabilities?.system_audio_capture.reason ? ` - ${liveCapabilities.system_audio_capture.reason}` : ""}
                        </small>
                    </article>
                    <article>
                        <span>Segment STT</span>
                        <strong>{uiSummary.isTranscribing ? "active" : liveCapabilities?.live_segment_transcription.state ?? "inactive"}</strong>
                        <small>{liveCapabilities?.live_segment_transcription.reason ?? "Segments are transcribed only after managed audio reaches the STT file bridge."}</small>
                    </article>
                    <article>
                        <span>Summary / notes</span>
                        <strong>{liveCapabilities?.live_summarization.state ?? "unknown"}</strong>
                        <small>{summaries.length} summaries / {notes.length} notes</small>
                    </article>
                    <article>
                        <span>Live UI updates</span>
                        <strong>{hasActiveSession || uiSummary.isRecording ? "1s active polling" : "5s idle polling"}</strong>
                        <small>Subscribed to meeting update events; polling catches background STT completions.</small>
                    </article>
                    <article>
                        <span>Speaker attribution</span>
                        <strong>source default</strong>
                        <small>{displayedState?.speaker_rename_count ?? 0} renames / diarization unsupported</small>
                    </article>
                </div>

                {uiSummary.blockingReasons.length ? (
                    <div className="meeting-blockers">
                        <strong>Blocked by:</strong>
                        <ul>
                            {uiSummary.blockingReasons.map((reason) => (
                                <li key={reason}>{reason}</li>
                            ))}
                        </ul>
                    </div>
                ) : null}

                <div className="meeting-primary-actions">
                    <label className="meeting-capture-mode">
                        <span>Capture mode</span>
                        <select
                            value={captureMode}
                            disabled={isBusy || hasActiveSession}
                            onChange={(event) => setCaptureMode(event.target.value as CaptureMode)}
                        >
                            <option value="both">System + microphone</option>
                            <option value="system_audio">System audio only</option>
                            <option value="microphone">Microphone only</option>
                        </select>
                    </label>
                    <Button variant="secondary" radius="full" size="xs" disabled={isBusy} onClick={() => void runOperation("grant consent", () => meeting.grantConsent(platform))}>
                        Grant consent
                    </Button>
                    <Button
                        variant="secondary"
                        radius="full"
                        size="xs"
                        disabled={isBusy}
                        onClick={() =>
                            void runOperation("start manual session", () =>
                                meeting.startSession(platform, manualMeetingConfig(platform, backend))
                            )
                        }
                    >
                        Start manual session
                    </Button>
                    <Button
                        variant="secondary"
                        radius="full"
                        size="xs"
                        disabled={isBusy || !canStartRealCapture}
                        title={startRecordingDisabledReason}
                        onClick={() =>
                            void runOperation("start recording", () =>
                                meeting.startSession(platform, realCaptureConfig(platform, backend, captureMode))
                            )
                        }
                    >
                        Start recording
                    </Button>
                    <Button variant="text" radius="full" size="xs" disabled={isBusy || !hasActiveSession} onClick={() => void runOperation("pause session", meeting.pauseSession)}>
                        Pause
                    </Button>
                    <Button variant="text" radius="full" size="xs" disabled={isBusy || !hasActiveSession} onClick={() => void runOperation("resume session", meeting.resumeSession)}>
                        Resume
                    </Button>
                    <Button variant="text" radius="full" size="xs" disabled={isBusy || !hasActiveSession} onClick={() => void handleStopSession()}>
                        {isStoppingSession ? "Finalizing..." : "Stop"}
                    </Button>
                    {uiSummary.state === "failed_recoverable" ? (
                        <>
                            <Button variant="secondary" radius="full" size="xs" disabled={isBusy || !hasActiveSession} onClick={() => void handleRecoverFailedCapture()}>
                                Recover capture
                            </Button>
                            <Button variant="secondary" radius="full" size="xs" disabled={isBusy || !hasActiveSession} onClick={() => void handleForceFinalizeFailedCapture()}>
                                Force stop session
                            </Button>
                        </>
                    ) : null}
                    <Button variant="text" radius="full" size="xs" onClick={scrollToTranscript}>
                        Open transcript
                    </Button>
                </div>

                {Object.entries(refreshWarnings).map(([scope, warning]) => (
                    <div key={scope} className="desktop-agent-error">{scope}: {warning}</div>
                ))}
                {lastError ? <div className="desktop-agent-error">{lastError}</div> : null}
                {lastResult ? <pre className="desktop-agent-json meeting-operation-result">{lastResult}</pre> : null}
            </section>

            <details className="desktop-agent-card meeting-section-card meeting-collapsible-section">
                <summary className="meeting-collapsible-summary">
                    <div>
                        <p className="meeting-section-kicker">Live recording</p>
                        <h3>Recording readiness</h3>
                        <p>Advanced start/capture readiness and STT drain details.</p>
                    </div>
                    <Button variant="text" radius="full" size="xs" disabled={isBusy} onClick={() => void refreshMeeting()}>
                        Refresh
                    </Button>
                </summary>
                <div className="meeting-readiness-grid">
                    <div>
                        <span>WASAPI backend</span>
                        <strong>{realCaptureReadiness.backendAvailable ? "ready" : "unavailable"}</strong>
                    </div>
                    <div>
                        <span>Audio capture permission</span>
                        <strong>{permissionStatus(audioCaptureTool)}</strong>
                    </div>
                    <div>
                        <span>Segment transcription permission</span>
                        <strong>{permissionStatus(segmentTranscriptionTool)}</strong>
                    </div>
                    <div>
                        <span>Approval required</span>
                        <strong>{realCaptureReadiness.approvalRequired ? "yes" : "no"}</strong>
                    </div>
                    <div>
                        <span>Result</span>
                        <strong>{startReadiness.statusLabel}</strong>
                    </div>
                    <div>
                        <span>Capture mode</span>
                        <strong>{captureMode === "both" ? "system + mic" : captureMode === "microphone" ? "microphone" : "system"}</strong>
                    </div>
                </div>
                <p className="desktop-agent-muted">{startReadiness.primaryHelpText}</p>
                {startReadiness.requiresApprovalOrConfirmation && startReadiness.canRequestStart ? (
                    <div className="meeting-governed-start-note">
                        <strong>Recording ready for governed start.</strong>
                        <p>This action is high-risk and will be confirmed/audited when you click Start. No pending approval exists yet.</p>
                    </div>
                ) : null}
                {startReadiness.hardBlockers.length ? (
                    <div className="meeting-blockers">
                        <strong>Disabled by policy or readiness:</strong>
                        <ul>
                            {startReadiness.hardBlockers.map((reason) => (
                                <li key={reason}>{reason}</li>
                            ))}
                        </ul>
                        <p>No pending approval exists because the request is blocked before approval creation.</p>
                    </div>
                ) : null}
                {realCaptureReadiness.blockedReason ? (
                    <p className="desktop-agent-muted">{realCaptureReadiness.blockedReason}</p>
                ) : null}
                {sessionMode === "manual" && !uiSummary.isRecording ? (
                    <p className="desktop-agent-muted">Manual session is active, but it is not listening to system audio.</p>
                ) : null}
                <div className="meeting-capture-summary">
                    <span>Segments written: <strong>{segmentsWritten}</strong></span>
                    <span>Segments transcribed: <strong>{segmentsTranscribed}</strong></span>
                    <span>System segments: <strong>{systemMetrics?.segments_written ?? 0}</strong></span>
                    <span>Mic segments: <strong>{microphoneMetrics?.segments_written ?? 0}</strong></span>
                    <span>Queued total: <strong>{segmentsQueuedTotal}</strong></span>
                    <span>Current queue: <strong>{currentQueueDepth}</strong></span>
                    <span>In flight: <strong>{segmentsInFlight}</strong></span>
                    <span>Failed: <strong>{segmentsFailed}</strong></span>
                    <span>Drain: <strong>{drainTimedOut ? "timed out" : drainStatus}</strong></span>
                    <span>Silence dropped: <strong>{metrics?.dropped_silence_segments ?? 0}</strong></span>
                </div>
                <div className="meeting-capture-summary">
                    <span>Finalization: <strong>{finalizationStatus?.stage ?? "idle"}</strong></span>
                    <span>Session: <strong>{finalizationStatus?.session_id ?? "none"}</strong></span>
                    <span>Pending: <strong>{finalizationStatus?.pending_segments ?? 0}</strong></span>
                    <span>Queue: <strong>{finalizationStatus?.queue_depth ?? 0}</strong></span>
                    <span>In flight: <strong>{finalizationStatus?.in_flight_segments ?? 0}</strong></span>
                    <span>Drain timeout: <strong>{finalizationStatus?.drain_timeout ? "yes" : "no"}</strong></span>
                    <span>Export: <strong>{finalizationStatus?.export_written ? "written" : "pending"}</strong></span>
                    <span>Archive: <strong>{finalizationStatus?.archive_written ? "written" : "pending"}</strong></span>
                    <span>Recoverable: <strong>{finalizationStatus?.recoverable ? "yes" : "no"}</strong></span>
                    {finalizationStatus?.recoverable ? (
                        <Button
                            variant="secondary"
                            radius="full"
                            size="xs"
                            disabled={isBusy}
                            onClick={() => void handleRetryFinalization()}
                        >
                            Retry finalization
                        </Button>
                    ) : null}
                </div>
                {capturedWithoutTranscript ? (
                    <div className="meeting-stt-warning">
                        <strong>Audio segments were captured, but no transcript has been produced yet.</strong>
                        <p>
                            STT may be pending, unavailable, failed, or still finalizing. Current queue: {currentQueueDepth};
                            in flight: {segmentsInFlight}; failures: {segmentsFailed}; timeouts: {segmentTranscriptionTimeouts}.
                        </p>
                    </div>
                ) : null}
                {finalizingStt ? (
                    <div className="meeting-stt-warning">
                        <strong>Stopping and draining STT queue.</strong>
                        <p>
                            Astra is waiting for bounded segment transcription finalization. Queue: {currentQueueDepth};
                            in flight: {segmentsInFlight}; transcribed: {segmentsTranscribed}/{segmentsWritten}.
                        </p>
                    </div>
                ) : null}
                {drainTimedOut ? (
                    <div className="meeting-stt-warning meeting-stt-warning--error">
                        <strong>STT drain timed out before all captured segments were transcribed.</strong>
                        <p>
                            The session was saved, but transcript/export may be missing {missingTranscriptSegments || segmentsInFlight || currentQueueDepth} segment(s).
                            Check STT worker/device diagnostics.
                        </p>
                    </div>
                ) : null}
                <div className="desktop-agent-inline-actions">
                    <Button
                        variant="secondary"
                        radius="full"
                        size="xs"
                        disabled={isBusy || !canStartRealCapture}
                        title={startRecordingDisabledReason}
                        onClick={() =>
                            void runOperation("start recording", () =>
                                meeting.startSession(platform, realCaptureConfig(platform, backend, captureMode))
                            )
                        }
                    >
                        Start capture
                    </Button>
                    <Button variant="text" radius="full" size="xs" disabled title={liveCapabilities?.live_transcription.reason ?? "Live STT unsupported"}>
                        Streaming STT unsupported
                    </Button>
                </div>
            </details>

            <details className="desktop-agent-card meeting-section-card meeting-screen-context meeting-collapsible-section">
                <summary className="meeting-collapsible-summary">
                    <div>
                        <p className="meeting-section-kicker">Screen Context</p>
                        <h3>Attached Screen Context</h3>
                        <p>Manual, governed screen observation attached to this work session. Screenshots are not stored by default.</p>
                    </div>
                    <span className="meeting-count-pill">{screenContexts.length} saved</span>
                </summary>
                <div className="desktop-agent-inline-actions">
                    <Button
                        variant="secondary"
                        radius="full"
                        size="xs"
                        disabled={isBusy || isAttachingScreenContext || !hasActiveSession}
                        title={
                            hasActiveSession
                                ? "Attach current screen without desktop control"
                                : "Start or reopen an active session before attaching screen context"
                        }
                        onClick={() => void handleAttachCurrentScreen()}
                    >
                        {isAttachingScreenContext ? "Attaching..." : "Attach current screen"}
                    </Button>
                    <span className="desktop-agent-muted">Permission: {permissionStatus(screenContextTool)}</span>
                </div>
                {screenContexts.length ? (
                    <div className="meeting-generated-list">
                        {[...screenContexts].slice(-3).reverse().map((context) => (
                            <article key={context.context_id} className="meeting-generated-item">
                                <div className="meeting-generated-block__header">
                                    <strong>{formatEntryTime(context.captured_at)}</strong>
                                    <span>{context.screenshot_ref ? "screenshot stored" : "screenshot not stored"}</span>
                                </div>
                                <p>{truncateEvidenceText(context.summary, 260)}</p>
                                <EvidencePreview
                                    ids={context.linked_transcript_segment_ids}
                                    transcriptBySegmentId={transcriptBySegmentId}
                                    expanded={expandedEvidenceKeys[`screen-context-${context.context_id}`] === true}
                                    onToggle={() =>
                                        setExpandedEvidenceKeys((prev) => ({
                                            ...prev,
                                            [`screen-context-${context.context_id}`]:
                                                !prev[`screen-context-${context.context_id}`],
                                        }))
                                    }
                                />
                                {context.diagnostics.length ? (
                                    <small className="desktop-agent-muted">
                                        Diagnostics: {context.diagnostics.map((diagnostic) => diagnostic.code).join(", ")}
                                    </small>
                                ) : null}
                            </article>
                        ))}
                    </div>
                ) : (
                    <div className="desktop-agent-empty">
                        Attach the current screen when visual context matters to the session. Astra stores a bounded local summary and links it to nearby transcript segments.
                    </div>
                )}
            </details>

            <section className="desktop-agent-card meeting-section-card meeting-session-memory">
                <div className="meeting-section-heading">
                    <div>
                        <p className="meeting-section-kicker">Session Memory</p>
                        <h3>Archived Sessions</h3>
                        <p>Local completed sessions, indexed for governed read/search/export.</p>
                    </div>
                    <span className="meeting-count-pill">{sessionMemory.length} recent</span>
                </div>

                <div className="meeting-session-memory__search">
                    <input
                        className="desktop-agent-input"
                        value={sessionSearchQuery}
                        onChange={(event) => setSessionSearchQuery(event.target.value)}
                        placeholder="Search transcripts and artifacts"
                        aria-label="Search archived meeting sessions"
                    />
                    <Button
                        variant="secondary"
                        radius="full"
                        size="xs"
                        disabled={isSessionMemoryBusy || !sessionSearchQuery.trim()}
                        onClick={() => void handleSearchSessionMemory()}
                    >
                        Search
                    </Button>
                    <Button
                        variant="text"
                        radius="full"
                        size="xs"
                        disabled={isSessionMemoryBusy}
                        onClick={() => void handleReindexSessionMemory()}
                    >
                        Reindex
                    </Button>
                </div>

                <div className="meeting-session-memory__recall">
                    <div className="meeting-generated-block__header">
                        <h4>Ask Session Memory</h4>
                        <span>{recallResponse?.status ?? "local evidence"}</span>
                    </div>
                    <div className="meeting-session-memory__search">
                        <input
                            className="desktop-agent-input"
                            value={recallQuery}
                            onChange={(event) => setRecallQuery(event.target.value)}
                            placeholder="Ask what was decided, discussed, or visible on screen"
                            aria-label="Ask archived meeting sessions"
                        />
                        <Button
                            variant="secondary"
                            radius="full"
                            size="xs"
                            disabled={isRecallBusy || !recallQuery.trim()}
                            onClick={() => void handleAnswerRecall()}
                        >
                            {isRecallBusy ? "Asking" : "Ask"}
                        </Button>
                    </div>
                    {recallResponse ? (
                        <div className="meeting-recall-answer">
                            <p>{recallResponse.answer}</p>
                            <div className="meeting-session-memory__diagnostics">
                                <span>Status: {recallResponse.status}</span>
                                <span>Generator: {recallResponse.diagnostics.generator}</span>
                                <span>Intent: {artifactKindLabel(recallResponse.diagnostics.recall_intent)}</span>
                                <span>Evidence: {recallResponse.evidence.length}</span>
                                {recallResponse.diagnostics.fallback_used ? <span>fallback</span> : null}
                            </div>
                            {recallResponse.evidence.length ? (
                                <div className="meeting-session-search-results">
                                    {recallResponse.evidence.slice(0, 5).map((item, index) => (
                                        <article key={`${item.session_id}-${item.matched_kind}-${index}`} className="meeting-session-search-result">
                                            <span>{artifactKindLabel(item.matched_kind)} - {item.session_title}</span>
                                            <strong>{item.title}</strong>
                                            <p>{item.snippet}</p>
                                            <small>
                                                Relation: {recallRelationLabel(item.relation)} -{" "}
                                                {item.screen_context_ids.length ? `Screen context: ${item.screen_context_ids.join(", ")} - ` : ""}
                                                Evidence: {item.evidence_segment_ids.join(", ") || "none"}
                                            </small>
                                            <Button
                                                variant="text"
                                                radius="full"
                                                size="xs"
                                                disabled={isSessionMemoryBusy}
                                                onClick={() => void handleOpenArchivedSession(item.session_id)}
                                            >
                                                Open session
                                            </Button>
                                        </article>
                                    ))}
                                </div>
                            ) : null}
                        </div>
                    ) : null}
                </div>

                {sessionMemoryDiagnostics.length ? (
                    <div className="meeting-session-memory__diagnostics">
                        {sessionMemoryDiagnostics.slice(0, 3).map((diagnostic) => (
                            <span key={`${diagnostic.code}-${diagnostic.created_at}`}>
                                {diagnostic.severity}: {diagnostic.code}
                            </span>
                        ))}
                    </div>
                ) : null}

                <div className="meeting-session-memory__grid">
                    <div className="meeting-session-memory__panel">
                        <div className="meeting-generated-block__header">
                            <h4>Recent Sessions</h4>
                            <span>{isSessionMemoryBusy ? "working" : "local"}</span>
                        </div>
                        {sessionMemory.length ? (
                            <div className="meeting-session-list">
                                {sessionMemory.map((item) => (
                                    <article key={item.session_id} className="meeting-session-list-item">
                                        <div>
                                            <strong>{item.title}</strong>
                                            <span>{formatSessionDate(item.started_at)} - {item.platform}</span>
                                            <p>{item.summary_preview || "No summary preview yet."}</p>
                                        </div>
                                        <div className="meeting-session-list-item__meta">
                                            <span>{item.transcript_count} transcript</span>
                                            <span>{item.intelligence_present ? "intelligence" : "no intelligence"}</span>
                                            <span>{item.screen_context_count} screen</span>
                                            <span>{sttCompletenessLabel(item.stt_completeness_status)}</span>
                                            <span>Drain: {item.drain_status || "unknown"}</span>
                                        </div>
                                        {item.stt_completeness_detail ? (
                                            <p className="desktop-agent-muted">{item.stt_completeness_detail}</p>
                                        ) : null}
                                        <div className="desktop-agent-inline-actions">
                                            <Button
                                                variant="text"
                                                radius="full"
                                                size="xs"
                                                disabled={isSessionMemoryBusy}
                                                onClick={() => void handleOpenArchivedSession(item.session_id)}
                                            >
                                                Open
                                            </Button>
                                            <Button
                                                variant="text"
                                                radius="full"
                                                size="xs"
                                                disabled={isSessionMemoryBusy}
                                                onClick={() => void handleExportArchivedSession(item.session_id, "markdown")}
                                            >
                                                Export MD
                                            </Button>
                                            <Button
                                                variant="text"
                                                radius="full"
                                                size="xs"
                                                disabled={isSessionMemoryBusy}
                                                onClick={() => void handleExportArchivedSession(item.session_id, "json")}
                                            >
                                                Export JSON
                                            </Button>
                                        </div>
                                    </article>
                                ))}
                            </div>
                        ) : (
                            <div className="desktop-agent-empty">No archived sessions yet. Completed meetings appear here after Stop.</div>
                        )}
                    </div>

                    <div className="meeting-session-memory__panel">
                        <div className="meeting-generated-block__header">
                            <h4>Search Results</h4>
                            <span>{sessionSearchResults.length}</span>
                        </div>
                        {sessionSearchResults.length ? (
                            <div className="meeting-session-search-results">
                                {sessionSearchResults.map((result, index) => (
                                    <article key={`${result.session_id}-${result.matched_kind}-${index}`} className="meeting-session-search-result">
                                        <span>{artifactKindLabel(result.matched_kind)} - {result.session_title}</span>
                                        <strong>{result.title}</strong>
                                        <p>{result.snippet}</p>
                                        <small>
                                            {result.speaker_display_name ? `${result.speaker_display_name} - ` : ""}
                                            {result.screen_context_id ? `Screen context: ${result.screen_context_id} - ` : ""}
                                            Evidence: {result.evidence_segment_ids.join(", ") || "none"}
                                        </small>
                                        <Button
                                            variant="text"
                                            radius="full"
                                            size="xs"
                                            disabled={isSessionMemoryBusy}
                                            onClick={() => void handleOpenArchivedSession(result.session_id)}
                                        >
                                            Open session
                                        </Button>
                                    </article>
                                ))}
                            </div>
                        ) : (
                            <div className="desktop-agent-empty">Search uses bounded lexical matching across transcripts and saved artifacts.</div>
                        )}
                    </div>
                </div>

                {openedArchive ? (
                    <article className="meeting-archive-view">
                        <div className="meeting-generated-block__header">
                            <div>
                                <h4>Archived Session: {openedArchiveTitle}</h4>
                                <span>{formatSessionDate(openedArchive.exported.started_at)} - {openedArchive.exported.platform}</span>
                            </div>
                            <span>{openedArchive.state.transcript.length} transcript entries</span>
                        </div>
                        {hasIncompleteTranscriptionMetadata(openedArchive.exported.metadata) ? (
                            <p className="meeting-intelligence-warning">
                                This archived export includes incomplete STT diagnostics. Check metadata before relying on the transcript.
                            </p>
                        ) : null}
                        {openedArchiveSummary ? (
                            <div className="meeting-generated-item">
                                <strong>Summary</strong>
                                <p>{openedArchiveSummary.text}</p>
                                <EvidencePreview
                                    ids={openedArchiveSummary.evidence_segment_ids}
                                    transcriptBySegmentId={archivedTranscriptBySegmentId}
                                    expanded={expandedEvidenceKeys[`archive-summary-${openedArchiveSummary.id}`] === true}
                                    onToggle={() =>
                                        setExpandedEvidenceKeys((prev) => ({
                                            ...prev,
                                            [`archive-summary-${openedArchiveSummary.id}`]:
                                                !prev[`archive-summary-${openedArchiveSummary.id}`],
                                        }))
                                    }
                                />
                            </div>
                        ) : null}
                        <div className="meeting-session-memory__archive-meta">
                            <span>Actions: <strong>{openedArchive.state.action_items.length + (openedArchiveIntelligence?.action_items.length ?? 0)}</strong></span>
                            <span>Decisions: <strong>{openedArchive.state.decisions.length + (openedArchiveIntelligence?.decisions.length ?? 0)}</strong></span>
                            <span>Speakers: <strong>{openedArchive.state.speakers.map((speaker) => speaker.display_name).join(", ") || "none"}</strong></span>
                            <span>STT: <strong>{sttCompletenessLabel(sessionMemory.find((item) => item.session_id === openedArchive.session_id)?.stt_completeness_status)}</strong></span>
                        </div>
                        {sessionMemory.find((item) => item.session_id === openedArchive.session_id)?.stt_completeness_detail ? (
                            <p className="desktop-agent-muted">
                                {sessionMemory.find((item) => item.session_id === openedArchive.session_id)?.stt_completeness_detail}
                            </p>
                        ) : null}
                        {openedArchiveScreenContexts.length ? (
                            <div className="meeting-generated-block">
                                <div className="meeting-generated-block__header">
                                    <h4>Screen Contexts</h4>
                                    <span>{openedArchiveScreenContexts.length}</span>
                                </div>
                                {openedArchiveScreenContexts.slice(0, 5).map((context) => (
                                    <article key={context.context_id} className="meeting-generated-item">
                                        <div className="meeting-generated-block__header">
                                            <strong>{formatEntryTime(context.captured_at)}</strong>
                                            <span>{context.screenshot_ref ? "screenshot stored" : "summary only"}</span>
                                        </div>
                                        <p>{truncateEvidenceText(context.summary, 240)}</p>
                                        <EvidencePreview
                                            ids={context.linked_transcript_segment_ids}
                                            transcriptBySegmentId={archivedTranscriptBySegmentId}
                                            expanded={expandedEvidenceKeys[`archive-screen-context-${context.context_id}`] === true}
                                            onToggle={() =>
                                                setExpandedEvidenceKeys((prev) => ({
                                                    ...prev,
                                                    [`archive-screen-context-${context.context_id}`]:
                                                        !prev[`archive-screen-context-${context.context_id}`],
                                                }))
                                            }
                                        />
                                    </article>
                                ))}
                                {openedArchiveScreenContexts.length > 5 ? (
                                    <p className="desktop-agent-muted">
                                        {openedArchiveScreenContexts.length - 5} more screen context attachment(s) in archive/export.
                                    </p>
                                ) : null}
                            </div>
                        ) : null}
                        <div className="meeting-session-memory__transcript-preview">
                            {openedArchive.state.transcript.slice(0, 5).map((entry) => (
                                <p key={entry.segment_id}>
                                    <strong>[{transcriptSourceLabel(entry.source)}] {speakerDisplayName(entry)}:</strong> {truncateEvidenceText(entry.text, 220)}
                                </p>
                            ))}
                            {openedArchive.state.transcript.length > 5 ? (
                                <p className="desktop-agent-muted">{openedArchive.state.transcript.length - 5} more transcript entries in archive/export.</p>
                            ) : null}
                        </div>
                    </article>
                ) : null}

                {archiveExport ? (
                    <article className="meeting-archive-export-preview">
                        <div className="meeting-generated-block__header">
                            <h4>{archiveExport.filename}</h4>
                            <span>{archiveExport.content_length} bytes</span>
                        </div>
                        {archiveExport.diagnostics.length ? (
                            <p className="meeting-intelligence-warning">
                                Export diagnostics: {archiveExport.diagnostics.map((diagnostic) => diagnostic.code).join(", ")}
                            </p>
                        ) : null}
                        <pre>{archiveExport.content.slice(0, 2400)}{archiveExport.content.length > 2400 ? "\n..." : ""}</pre>
                    </article>
                ) : null}
            </section>

            <section ref={transcriptRef} className="desktop-agent-card meeting-section-card">
                <div className="meeting-section-heading">
                    <div>
                        <p className="meeting-section-kicker">Transcript</p>
                        <h3>Live Transcript</h3>
                        <p>Newest entries are shown first; export and artifact evidence keep chronological storage.</p>
                    </div>
                    <span className="meeting-count-pill">{transcriptEntries.length} entries</span>
                </div>
                {transcriptEntries.length === 0 ? (
                    <div className="desktop-agent-empty">
                        <strong>No transcript yet.</strong>
                        <p>{transcriptEmptyCopy(uiSummary.state, sessionMode)}</p>
                    </div>
                ) : (
                    <div className="meeting-transcript-list">
                        {displayedTranscriptEntries.map((entry, index) => (
                            <article key={`${entry.timestamp}-${index}`} className="meeting-transcript-entry">
                                <span>[{formatEntryTime(entry.timestamp)}]</span>
                                <span className={`meeting-source-badge meeting-source-badge--${entry.source ?? "unknown"}`}>
                                    {transcriptSourceLabel(entry.source)}
                                </span>
                                <strong>{speakerDisplayName(entry)}:</strong>
                                <p>{entry.text}</p>
                                {entry.speaker_id ? (
                                    <button
                                        type="button"
                                        className="meeting-speaker-rename-button"
                                        onClick={() =>
                                            handleSelectSpeakerForRename({
                                                speaker_id: entry.speaker_id ?? "",
                                                display_name: speakerDisplayName(entry),
                                                source: entry.source ?? "unknown",
                                                confidence: entry.speaker_confidence ?? 0,
                                                attribution_method: entry.speaker_attribution_method ?? "unknown",
                                            })
                                        }
                                    >
                                        Rename
                                    </button>
                                ) : null}
                            </article>
                        ))}
                    </div>
                )}
                <div className="meeting-speaker-panel">
                    <div className="meeting-section-heading">
                        <div>
                            <p className="meeting-section-kicker">Speakers</p>
                            <h3>Speaker Labels</h3>
                            <p>Source-default labels are metadata. Rename labels when you know who they represent.</p>
                        </div>
                        <span className="meeting-count-pill">{speakers.length}</span>
                    </div>
                    {speakers.length ? (
                        <div className="meeting-speaker-list">
                            {speakers.map((speaker) => (
                                <button
                                    key={speaker.speaker_id}
                                    type="button"
                                    className="meeting-speaker-chip"
                                    onClick={() => handleSelectSpeakerForRename(speaker)}
                                >
                                    <strong>{speaker.display_name}</strong>
                                    <span>{transcriptSourceLabel(speaker.source)}</span>
                                </button>
                            ))}
                        </div>
                    ) : (
                        <p className="desktop-agent-muted">Speaker labels appear after a session starts.</p>
                    )}
                    <div className="meeting-speaker-rename-form">
                        <input
                            className="desktop-agent-input"
                            value={speakerRenameId}
                            onChange={(event) => setSpeakerRenameId(event.target.value)}
                            aria-label="Speaker id"
                            placeholder="speaker_id"
                        />
                        <input
                            className="desktop-agent-input"
                            value={speakerRenameName}
                            onChange={(event) => setSpeakerRenameName(event.target.value)}
                            aria-label="Speaker display name"
                            placeholder="Display name"
                        />
                        <Button
                            variant="secondary"
                            radius="full"
                            size="xs"
                            disabled={isBusy || !speakerRenameId.trim() || !speakerRenameName.trim()}
                            onClick={() => void handleRenameSpeaker()}
                        >
                            Rename speaker
                        </Button>
                    </div>
                    <p className="desktop-agent-muted">Diarization is not active: microphone defaults to You and system audio defaults to Speaker 1 until renamed.</p>
                </div>
            </section>

            <section className="desktop-agent-card meeting-section-card meeting-generated-intelligence">
                <div className="meeting-section-heading">
                    <div>
                        <p className="meeting-section-kicker">Derived artifacts</p>
                        <h3>Meeting Intelligence</h3>
                        <p>Generated from transcript evidence. Raw transcript remains the source of truth.</p>
                    </div>
                    <span className={`meeting-intelligence-status meeting-intelligence-status--${intelligence?.status ?? "idle"}`}>
                        {intelligence?.status ?? "idle"}
                    </span>
                </div>
                <div className="desktop-agent-inline-actions">
                    <Button
                        variant="secondary"
                        radius="full"
                        size="xs"
                        disabled={isBusy || isGeneratingIntelligence || transcriptEntries.length === 0}
                        onClick={() => void handleGenerateIntelligence()}
                    >
                        {intelligence ? "Regenerate intelligence" : "Generate intelligence"}
                    </Button>
                    <Button
                        variant="text"
                        radius="full"
                        size="xs"
                        disabled={isBusy || !intelligence}
                        onClick={() => void handleClearIntelligence()}
                    >
                        Clear generated intelligence
                    </Button>
                    {intelligence?.follow_up_draft ? (
                        <Button
                            variant="text"
                            radius="full"
                            size="xs"
                            disabled={isBusy}
                            onClick={() => void handleCopyFollowUpDraft()}
                        >
                            Copy follow-up draft
                        </Button>
                    ) : null}
                </div>
                {transcriptEntries.length === 0 ? (
                    <div className="desktop-agent-empty">
                        Add or capture transcript entries before generating meeting intelligence.
                    </div>
                ) : null}
                {intelligence ? (
                    <>
                        <div className="meeting-intelligence-meta">
                            <span>Generator: <strong>{generatorLabel(intelligence)}</strong></span>
                            <span>Provider: <strong>{intelligence.diagnostics.model_provider ?? "none"}</strong></span>
                            <span>Model: <strong>{intelligence.diagnostics.model_name ?? "none"}</strong></span>
                            <span>Endpoint: <strong>{intelligence.diagnostics.llm_endpoint ?? "not used"}</strong></span>
                            <span>Language: <strong>{languageLabel(intelligence.diagnostics.detected_language)}</strong></span>
                            <span>Output: <strong>{languageLabel(intelligence.diagnostics.output_language)}</strong></span>
                            <span>Session type: <strong>{sessionTypeLabel(intelligence.diagnostics.session_type)}</strong></span>
                            <span>Generation: <strong>{formatDurationMs(intelligence.diagnostics.total_generation_duration_ms ?? intelligence.diagnostics.llm_generation_duration_ms)}</strong></span>
                            <span>Segments: <strong>{intelligence.source_transcript_segment_count}</strong></span>
                            <span>LLM: <strong>{intelligence.diagnostics.llm_used ? "used" : "not used"}</strong></span>
                            <span>Fallback: <strong>{intelligence.diagnostics.fallback_used ? "yes" : "no"}</strong></span>
                            <span>Language retry: <strong>{intelligence.diagnostics.language_retry_attempted ? (intelligence.diagnostics.language_retry_succeeded ? "succeeded" : "attempted") : "not needed"}</strong></span>
                            <span>Input: <strong>{intelligence.diagnostics.input_segment_count || intelligence.source_transcript_segment_count}</strong></span>
                            <span>Audit: <strong>{intelligence.diagnostics.audit_redacted ? "redacted" : "not redacted"}</strong></span>
                            <span>Transcript logged: <strong>{intelligence.diagnostics.transcript_text_logged ? "yes" : "no"}</strong></span>
                        </div>
                        {intelligence.diagnostics.output_language_mismatch ? (
                            <p className="meeting-intelligence-warning">
                                Output language did not fully match the detected transcript language. Review the generated text or regenerate.
                            </p>
                        ) : null}
                        {intelligence.diagnostics.input_truncated ? (
                            <p className="meeting-intelligence-warning">
                                Local model input was truncated to {intelligence.diagnostics.input_segment_count} segment(s)
                                and {intelligence.diagnostics.input_char_count} character(s).
                            </p>
                        ) : null}
                        {intelligence.diagnostics.rejected_artifact_count || intelligence.diagnostics.invalid_evidence_ids ? (
                            <p className="meeting-intelligence-warning">
                                Evidence validation rejected {intelligence.diagnostics.rejected_artifact_count} artifact(s)
                                and ignored {intelligence.diagnostics.invalid_evidence_ids} invalid evidence id(s).
                            </p>
                        ) : null}
                        {intelligence.diagnostics.transcript_changed_during_generation ? (
                            <p className="meeting-intelligence-warning">
                                Transcript changed during generation. Regenerate to include the newest context.
                            </p>
                        ) : null}
                        {intelligence.diagnostics.model_unavailable_reason ? (
                            <p className="meeting-intelligence-warning">
                                Local model degraded: {intelligence.diagnostics.degraded_reason ?? intelligence.diagnostics.model_unavailable_reason}. Rule-based fallback is displayed.
                            </p>
                        ) : null}
                        {intelligence.diagnostics.warnings.length ? (
                            <div className="meeting-intelligence-warning-list">
                                {intelligence.diagnostics.warnings.map((warning) => (
                                    <p key={warning}>{warning}</p>
                                ))}
                            </div>
                        ) : null}

                        {intelligence.summary ? (
                            <article className="meeting-generated-block">
                                <div className="meeting-generated-block__header">
                                    <h4>Summary</h4>
                                    <span>{confidenceLabel(intelligence.summary.confidence)}</span>
                                </div>
                                <p>{intelligence.summary.text}</p>
                                {intelligence.summary.bullets.length ? (
                                    <ul>
                                        {intelligence.summary.bullets.map((bullet) => (
                                            <li key={bullet}>{bullet}</li>
                                        ))}
                                    </ul>
                                ) : null}
                                {renderEvidence(`summary-${intelligence.summary.id}`, intelligence.summary.evidence_segment_ids)}
                            </article>
                        ) : null}

                        <div className="meeting-generated-columns">
                            <article className="meeting-generated-block">
                                <div className="meeting-generated-block__header">
                                    <h4>Decisions</h4>
                                    <span>{intelligence.decisions.length}</span>
                                </div>
                                {intelligence.decisions.length ? intelligence.decisions.map((decision) => (
                                    <div key={decision.id} className="meeting-generated-item">
                                        <p>{decision.decision}</p>
                                        {decision.rationale ? <span>Rationale: {decision.rationale}</span> : null}
                                        {decision.made_by_display_name ? <span>By: {decision.made_by_display_name}</span> : null}
                                        <span>{confidenceLabel(decision.confidence)}</span>
                                        {renderEvidence(`decision-${decision.id}`, decision.evidence_segment_ids)}
                                    </div>
                                )) : <p className="desktop-agent-muted">No evidence-backed decisions detected.</p>}
                            </article>

                            <article className="meeting-generated-block">
                                <div className="meeting-generated-block__header">
                                    <h4>Action Items</h4>
                                    <span>{intelligence.action_items.length}</span>
                                </div>
                                {intelligence.action_items.length ? intelligence.action_items.map((item) => (
                                    <div key={item.id} className="meeting-generated-item">
                                        <p>{item.task}</p>
                                        <span>{item.assignee_display_name ? `Assignee: ${item.assignee_display_name}` : "Assignee: not detected"}</span>
                                        {item.due_date ? <span>Due: {item.due_date}</span> : null}
                                        <span>Status: {item.status}</span>
                                        <span>{confidenceLabel(item.confidence)}</span>
                                        {renderEvidence(`action-${item.id}`, item.evidence_segment_ids)}
                                    </div>
                                )) : <p className="desktop-agent-muted">No evidence-backed action items detected.</p>}
                            </article>
                        </div>

                        <div className="meeting-generated-columns">
                            <article className="meeting-generated-block">
                                <div className="meeting-generated-block__header">
                                    <h4>Open Questions</h4>
                                    <span>{intelligence.open_questions.length}</span>
                                </div>
                                {intelligence.open_questions.length ? intelligence.open_questions.map((question) => (
                                    <div key={question.id} className="meeting-generated-item">
                                        <p>{question.question}</p>
                                        {question.asked_by_display_name ? <span>Asked by: {question.asked_by_display_name}</span> : null}
                                        <span>{confidenceLabel(question.confidence)}</span>
                                        {renderEvidence(`question-${question.id}`, question.evidence_segment_ids)}
                                    </div>
                                )) : <p className="desktop-agent-muted">No unresolved questions detected.</p>}
                            </article>

                            <article className="meeting-generated-block">
                                <div className="meeting-generated-block__header">
                                    <h4>Risks / Blockers</h4>
                                    <span>{intelligence.risks.length}</span>
                                </div>
                                {intelligence.risks.length ? intelligence.risks.map((risk) => (
                                    <div key={risk.id} className="meeting-generated-item">
                                        <p>{risk.risk}</p>
                                        <span>Severity: {risk.severity}</span>
                                        <span>{confidenceLabel(risk.confidence)}</span>
                                        {renderEvidence(`risk-${risk.id}`, risk.evidence_segment_ids)}
                                    </div>
                                )) : <p className="desktop-agent-muted">No grounded risks detected.</p>}
                            </article>
                        </div>

                        <div className="meeting-generated-columns">
                            <article className="meeting-generated-block">
                                <div className="meeting-generated-block__header">
                                    <h4>Technical Recap</h4>
                                    <span>{intelligence.technical_recap?.evidence_segment_ids.length ?? 0}</span>
                                </div>
                                {intelligence.technical_recap?.bullets.length ? (
                                    <>
                                        <ul>
                                            {intelligence.technical_recap.bullets.map((bullet) => (
                                                <li key={bullet}>{bullet}</li>
                                            ))}
                                        </ul>
                                        {intelligence.technical_recap.mentioned_files.length ? (
                                            <span>Files: {intelligence.technical_recap.mentioned_files.join(", ")}</span>
                                        ) : null}
                                        {intelligence.technical_recap.mentioned_commands.length ? (
                                            <span>Commands: {intelligence.technical_recap.mentioned_commands.join(", ")}</span>
                                        ) : null}
                                        {intelligence.technical_recap.mentioned_errors.length ? (
                                            <span>Errors: {intelligence.technical_recap.mentioned_errors.join(", ")}</span>
                                        ) : null}
                                        {renderEvidence(`technical-${intelligence.technical_recap.id}`, intelligence.technical_recap.evidence_segment_ids)}
                                    </>
                                ) : <p className="desktop-agent-muted">No grounded technical details detected.</p>}
                            </article>

                            <article className="meeting-generated-block">
                                <div className="meeting-generated-block__header">
                                    <h4>Follow-up Draft</h4>
                                    <span>{intelligence.follow_up_draft ? "draft only" : "none"}</span>
                                </div>
                                {intelligence.follow_up_draft ? (
                                    <div className="meeting-followup-draft">
                                        <strong>{intelligence.follow_up_draft.subject}</strong>
                                        <pre>{intelligence.follow_up_draft.body}</pre>
                                        <span>{confidenceLabel(intelligence.follow_up_draft.confidence)}</span>
                                        {renderEvidence(`followup-${intelligence.follow_up_draft.id}`, intelligence.follow_up_draft.evidence_segment_ids)}
                                    </div>
                                ) : <p className="desktop-agent-muted">No follow-up draft generated.</p>}
                            </article>
                        </div>

                        <article className="meeting-generated-block">
                            <div className="meeting-generated-block__header">
                                <h4>Timeline</h4>
                                <span>{intelligence.timeline.length}</span>
                            </div>
                            {intelligence.timeline.length ? (
                                <div className="meeting-timeline-list">
                                    {intelligence.timeline.map((item) => (
                                        <div key={item.id} className="meeting-timeline-item">
                                            <span>{formatTimelineTime(item.timestamp_ms)}</span>
                                            <strong>{item.speaker_display_name ?? "Unknown"}</strong>
                                            <p>{item.detail || item.title}</p>
                                            {renderEvidence(`timeline-${item.id}`, item.evidence_segment_ids)}
                                        </div>
                                    ))}
                                </div>
                            ) : <p className="desktop-agent-muted">Timeline appears after generation.</p>}
                        </article>
                    </>
                ) : (
                    transcriptEntries.length ? (
                        <div className="desktop-agent-empty">
                            Generate meeting intelligence to create evidence-linked summary, decisions, action items, questions, risks, technical recap, follow-up draft, and timeline.
                        </div>
                    ) : null
                )}
            </section>

            <section className="meeting-intelligence-grid">
                <article className="desktop-agent-card meeting-section-card">
                    <div className="meeting-section-heading">
                        <div>
                            <p className="meeting-section-kicker">Notes</p>
                            <h3>Structured Notes</h3>
                        </div>
                        <span className="meeting-count-pill">{notes.length}</span>
                    </div>
                    {notes.length ? notes.slice(-6).map((note) => (
                        <div key={note.id} className="meeting-intelligence-item">
                            <p>{note.content}</p>
                            <span>Evidence: {note.evidence_segment_ids.join(", ") || "none"}</span>
                        </div>
                    )) : <div className="desktop-agent-empty">Notes are created only after transcript entries exist.</div>}
                </article>

                <article className="desktop-agent-card meeting-section-card">
                    <div className="meeting-section-heading">
                        <div>
                            <p className="meeting-section-kicker">Actions</p>
                            <h3>Action Items</h3>
                        </div>
                        <span className="meeting-count-pill">{actionItems.length}</span>
                    </div>
                    {actionItems.length ? actionItems.map((item, index) => (
                        <div key={item.id ?? `${item.timestamp}-${index}`} className="meeting-intelligence-item">
                            <strong>{item.title || item.description}</strong>
                            <p>{item.description}</p>
                            <span>{item.assignee?.name ?? "unassigned"} / {item.status}</span>
                            <span>Evidence: {(item.evidence_segment_ids ?? []).join(", ") || "manual"}</span>
                        </div>
                    )) : <div className="desktop-agent-empty">No action items derived from transcript yet.</div>}
                </article>

                <article className="desktop-agent-card meeting-section-card">
                    <div className="meeting-section-heading">
                        <div>
                            <p className="meeting-section-kicker">Decisions</p>
                            <h3>Decisions</h3>
                        </div>
                        <span className="meeting-count-pill">{decisions.length}</span>
                    </div>
                    {decisions.length ? decisions.map((decision, index) => (
                        <div key={decision.id ?? `${decision.timestamp}-${index}`} className="meeting-intelligence-item">
                            <strong>{decision.decision}</strong>
                            {decision.rationale ? <p>{decision.rationale}</p> : null}
                            <span>Evidence: {(decision.evidence_segment_ids ?? []).join(", ") || "manual"}</span>
                        </div>
                    )) : <div className="desktop-agent-empty">No decisions derived from transcript yet.</div>}
                </article>
            </section>

            <details className="meeting-manual-tools">
                <summary className="desktop-agent-card meeting-manual-tools-summary">
                    <div>
                        <p className="meeting-section-kicker">Advanced</p>
                        <h3>Manual fallback tools</h3>
                        <p>Use these only when audio/STT missed something or you need to import an existing .wav file.</p>
                    </div>
                    <span className="meeting-count-pill">closed by default</span>
                </summary>
                <div className="desktop-agent-card-grid">
                    <article className="desktop-agent-card meeting-form-card">
                        <h3>Manual transcript entry</h3>
                        <p className="desktop-agent-muted">Adds explicit text to the current session and keeps it tagged as manual.</p>
                        <input className="desktop-agent-input" value={transcriptSpeaker} onChange={(event) => setTranscriptSpeaker(event.target.value)} aria-label="Transcript speaker" placeholder="unknown" />
                        <textarea className="desktop-agent-textarea" value={transcriptText} onChange={(event) => setTranscriptText(event.target.value)} rows={3} aria-label="Transcript text" placeholder="Transcript text" />
                        <input className="desktop-agent-input" type="number" min="0" max="1" step="0.01" value={transcriptConfidence} onChange={(event) => setTranscriptConfidence(event.target.value)} aria-label="Transcript confidence" />
                        <div className="desktop-agent-inline-actions">
                            <Button variant="secondary" radius="full" size="xs" disabled={isBusy || !hasActiveSession} onClick={() => void handleAddTranscript()}>
                                Add transcript
                            </Button>
                        </div>
                    </article>

                    <article className="desktop-agent-card meeting-form-card">
                        <h3>Transcribe existing .wav file</h3>
                        <p className="desktop-agent-muted">Imports a local audio file into the current session through the governed file STT bridge.</p>
                        <input className="desktop-agent-input" value={audioPath} onChange={(event) => setAudioPath(event.target.value)} aria-label="Audio file path" placeholder=".wav file path" />
                        <input className="desktop-agent-input" value={audioSpeaker} onChange={(event) => setAudioSpeaker(event.target.value)} aria-label="File transcription speaker" placeholder="unknown" />
                        <label className="desktop-agent-toggle-row">
                            <input type="checkbox" checked={cleanupAudioFile} onChange={(event) => setCleanupAudioFile(event.target.checked)} />
                            <span>Cleanup managed copy</span>
                        </label>
                        <div className="desktop-agent-inline-actions">
                            <Button variant="secondary" radius="full" size="xs" disabled={isBusy || !hasActiveSession} onClick={() => void handleTranscribeAudioFile()}>
                                Transcribe file
                            </Button>
                        </div>
                    </article>

                    <article className="desktop-agent-card meeting-form-card">
                        <h3>Add action item</h3>
                        <textarea className="desktop-agent-textarea" value={actionDescription} onChange={(event) => setActionDescription(event.target.value)} rows={3} aria-label="Action item description" placeholder="Action item" />
                        <input className="desktop-agent-input" value={actionAssignee} onChange={(event) => setActionAssignee(event.target.value)} aria-label="Action item assignee" placeholder="optional assignee" />
                        <input className="desktop-agent-input" type="date" value={actionDeadline} onChange={(event) => setActionDeadline(event.target.value)} aria-label="Action item deadline" />
                        <div className="desktop-agent-inline-actions">
                            <Button variant="secondary" radius="full" size="xs" disabled={isBusy || !hasActiveSession} onClick={() => void handleAddActionItem()}>
                                Add action
                            </Button>
                        </div>
                    </article>

                    <article className="desktop-agent-card meeting-form-card">
                        <h3>Add decision</h3>
                        <textarea className="desktop-agent-textarea" value={decisionText} onChange={(event) => setDecisionText(event.target.value)} rows={2} aria-label="Decision text" placeholder="Decision" />
                        <textarea className="desktop-agent-textarea" value={decisionRationale} onChange={(event) => setDecisionRationale(event.target.value)} rows={2} aria-label="Decision rationale" placeholder="Rationale" />
                        <input className="desktop-agent-input" value={decisionMadeBy} onChange={(event) => setDecisionMadeBy(event.target.value)} aria-label="Decision made by" placeholder="optional made by" />
                        <div className="desktop-agent-inline-actions">
                            <Button variant="secondary" radius="full" size="xs" disabled={isBusy || !hasActiveSession} onClick={() => void handleAddDecision()}>
                                Add decision
                            </Button>
                        </div>
                    </article>
                </div>
            </details>

            <details className="desktop-agent-card meeting-section-card meeting-collapsible-section">
                <summary className="meeting-collapsible-summary">
                    <div>
                        <p className="meeting-section-kicker">Data & privacy</p>
                        <h3>Consent and clear data</h3>
                        <p>Consent controls and destructive data tools.</p>
                    </div>
                    <span className="meeting-count-pill">{consentReady ? "consent ready" : "consent required"}</span>
                </summary>
                <div className="meeting-privacy-grid">
                    <div>
                        <span>Platform key</span>
                        <input
                            className="desktop-agent-input"
                            value={platform}
                            onChange={(event) => setPlatform(event.target.value)}
                            aria-label="Meeting platform"
                        />
                    </div>
                    <div>
                        <span>Consent</span>
                        <strong>{consent?.given ? "granted" : "not granted"}</strong>
                    </div>
                    <div>
                        <span>Global consent</span>
                        <strong>{consent?.global_enabled ? "enabled" : "disabled"}</strong>
                    </div>
                    <div>
                        <span>Scoped app</span>
                        <strong>{consent?.per_app?.[normalizedPlatform] ? "allowed" : "not allowed"}</strong>
                    </div>
                </div>
                <div className="desktop-agent-inline-actions">
                    <Button variant="secondary" radius="full" size="xs" disabled={isBusy} onClick={() => void runOperation("read consent", meeting.getConsentState)}>
                        Read consent
                    </Button>
                    <Button variant="secondary" radius="full" size="xs" disabled={isBusy} onClick={() => void runOperation("grant consent", () => meeting.grantConsent(platform))}>
                        Grant consent
                    </Button>
                    <Button variant="text" radius="full" size="xs" disabled={isBusy} onClick={() => void runOperation("revoke consent", () => meeting.revokeConsent(platform))}>
                        Revoke consent
                    </Button>
                    <Button variant="secondary" radius="full" size="xs" disabled={isBusy} onClick={() => void handleDetect()}>
                        Detect active call
                    </Button>
                    <Button variant="danger" radius="full" size="xs" disabled={isBusy} onClick={() => void handlePreviewClearData()}>
                        Preview clear data
                    </Button>
                </div>
                <div className="meeting-call-status">
                    <p>Call detection: <strong>{callInfo?.detection_state ?? "not checked"}</strong></p>
                    <p>Platform: <strong>{callInfo?.platform || "unknown"}</strong></p>
                    <p>Process: <strong>{callInfo?.process_name || "unknown"}</strong></p>
                    <p>Active call: <strong>{callInfo?.is_active_call ? "yes" : "no"}</strong></p>
                </div>

                {showClearConfirmation ? (
                    <div className="meeting-clear-confirmation">
                        <h3>Confirm clear data</h3>
                        <p>Runtime state: <strong>{clearPreview?.runtime_state_present ? "present" : "empty"}</strong></p>
                        <p>Persisted entries: <strong>{clearPreview?.persisted_entries ?? 0}</strong></p>
                        <p className="desktop-agent-muted">{clearPreview?.storage_path ?? "Storage path unavailable."}</p>
                        <input
                            className="desktop-agent-input"
                            value={clearPhrase}
                            onChange={(event) => setClearPhrase(event.target.value)}
                            aria-label="Clear data confirmation phrase"
                            placeholder={CLEAR_MEETING_DATA_CONFIRMATION_PHRASE}
                        />
                        <div className="desktop-agent-inline-actions">
                            <Button
                                variant="danger"
                                radius="full"
                                size="xs"
                                disabled={isBusy || clearPhrase !== CLEAR_MEETING_DATA_CONFIRMATION_PHRASE}
                                onClick={() => void handleConfirmClearData()}
                            >
                                Confirm delete
                            </Button>
                            <Button
                                variant="text"
                                radius="full"
                                size="xs"
                                disabled={isBusy}
                                onClick={() => {
                                    setShowClearConfirmation(false);
                                    setClearPhrase("");
                                }}
                            >
                                Cancel
                            </Button>
                        </div>
                    </div>
                ) : null}
            </details>

            <details className="desktop-agent-card meeting-diagnostics">
                <summary>Advanced diagnostics</summary>
                <div className="desktop-agent-card-grid meeting-diagnostics-grid">
                    <article className="desktop-agent-card">
                        <h3>Session</h3>
                        <p>State: <strong>{stateKind}</strong></p>
                        <p>ID: <strong>{activeSession?.session_id || displayedState?.session.session_id || "none"}</strong></p>
                        <p>Status: <strong>{formatStatus(currentStatus)}</strong></p>
                        <p>Mode: <strong>{sessionMode ?? "none"}</strong></p>
                        <p>Capture active: <strong>{activeSession?.capture_active ? "yes" : "no"}</strong></p>
                        <p>UI live updates: <strong>events + bounded polling</strong></p>
                        <p>Transcript view: <strong>newest first</strong></p>
                        <p>Speaker renames: <strong>{displayedState?.speaker_rename_count ?? 0}</strong></p>
                        <p className="desktop-agent-muted">{activeSession?.capture_backend_status ?? "No active session."}</p>
                    </article>

                    <article className="desktop-agent-card">
                        <h3>Counts</h3>
                        <p>Transcript: <strong>{displayedState?.transcript.length ?? 0}</strong></p>
                        <p>Actions: <strong>{displayedState?.action_items.length ?? 0}</strong></p>
                        <p>Decisions: <strong>{displayedState?.decisions.length ?? 0}</strong></p>
                        <p>Notes: <strong>{displayedState?.notes.length ?? 0}</strong></p>
                        <p>Screen contexts: <strong>{screenContexts.length}</strong></p>
                        <p>Diagnostics: <strong>{diagnostics.length}</strong></p>
                        {diagnostics.slice(-3).map((diagnostic) => (
                            <p key={`${diagnostic.code}-${diagnostic.created_at}`} className="desktop-agent-muted">
                                {diagnostic.severity}: {diagnostic.code}
                            </p>
                        ))}
                    </article>

                    <article className="desktop-agent-card">
                        <h3>Backend</h3>
                        <p>Preferred: <strong>{backend}</strong></p>
                        <p>Devices: <strong>{devices.length}</strong></p>
                        <p className="desktop-agent-muted">{devices.join(", ") || "No devices reported."}</p>
                    </article>

                    <article className="desktop-agent-card">
                        <h3>Capture metrics</h3>
                        <p>Controller: <strong>{liveCapabilities?.capture_health.state ?? "unknown"}</strong></p>
                        <p>Health: <strong>{liveCapabilities?.capture_health.status ?? "unknown"}</strong></p>
                        <p>Handle: <strong>{liveCapabilities?.capture_health.active_handle_present ? "present" : "absent"}</strong></p>
                        <p>WASAPI: <strong>{liveCapabilities?.windows_wasapi_capture.state ?? "unknown"}</strong></p>
                        <p>System controller: <strong>{systemHealth?.state ?? "unknown"}</strong></p>
                        <p>Microphone controller: <strong>{microphoneHealth?.state ?? "unknown"}</strong></p>
                        <p>Endpoint: <strong>{metrics?.wasapi_endpoint_acquired ? "acquired" : "not acquired"}</strong></p>
                        <p>Mix format: <strong>{metrics?.wasapi_mix_format_detected ? `${metrics.wasapi_sample_rate ?? 0} Hz / ${metrics.wasapi_channel_count ?? 0} ch / ${metrics.wasapi_sample_format ?? "unknown"}` : "unknown"}</strong></p>
                        <p>Buffer frames: <strong>{metrics?.wasapi_buffer_frame_count ?? 0}</strong></p>
                        <p>Stream: <strong>{metrics?.wasapi_stream_started ? "started" : metrics?.wasapi_stream_initialized ? "initialized" : "not initialized"}</strong></p>
                        <p>Packets read: <strong>{metrics?.wasapi_packets_read ?? 0}</strong></p>
                        <p>Frames captured: <strong>{metrics?.frames_captured ?? 0}</strong></p>
                        <p>Frames converted: <strong>{metrics?.frames_converted ?? 0}</strong></p>
                        <p>Segments: <strong>{segmentsWritten}</strong></p>
                        <p>System segments: <strong>{systemMetrics?.segments_written ?? 0} / {systemMetrics?.segments_transcribed ?? 0}</strong></p>
                        <p>Microphone segments: <strong>{microphoneMetrics?.segments_written ?? 0} / {microphoneMetrics?.segments_transcribed ?? 0}</strong></p>
                        <p>Queued total: <strong>{segmentsQueuedTotal}</strong></p>
                        <p>Current queue: <strong>{currentQueueDepth}</strong></p>
                        <p>Dequeued: <strong>{metrics?.segments_dequeued_total ?? 0}</strong></p>
                        <p>In flight: <strong>{segmentsInFlight}</strong></p>
                        <p>Transcribed: <strong>{segmentsTranscribed}</strong></p>
                        <p>STT failures: <strong>{segmentsFailed}</strong></p>
                        <p>STT timeouts: <strong>{segmentTranscriptionTimeouts}</strong></p>
                        <p>Consecutive STT failures: <strong>{metrics?.segment_transcription_failures_consecutive ?? 0}</strong></p>
                        <p>Last STT error: <strong>{metrics?.last_segment_transcription_error_kind ?? "none"}</strong></p>
                        <p>Last STT started: <strong>{metrics?.last_transcription_started_segment_id ?? "none"}</strong></p>
                        <p>Last STT completed: <strong>{metrics?.last_transcription_completed_segment_id ?? "none"}</strong></p>
                        <p>Last STT failed: <strong>{metrics?.last_transcription_failed_segment_id ?? "none"}</strong></p>
                        <p>Drain: <strong>{drainTimedOut ? "timed out" : drainStatus}</strong></p>
                        <p>Drain started: <strong>{metrics?.drain_started_at ?? "none"}</strong></p>
                        <p>Drain completed: <strong>{metrics?.drain_completed_at ?? "none"}</strong></p>
                        <p>Dropped: <strong>{metrics?.segments_dropped ?? 0}</strong></p>
                        <p>Silence dropped: <strong>{metrics?.dropped_silence_segments ?? 0}</strong></p>
                        <p>Silence frames skipped: <strong>{metrics?.silence_frames_skipped ?? 0}</strong></p>
                        <p>VAD speech/silence: <strong>{metrics?.last_speech_ratio_bps ?? 0} / {metrics?.last_silence_ratio_bps ?? 0} bps</strong></p>
                        <p>Clipping count: <strong>{metrics?.audio_clipped_sample_count ?? 0}</strong></p>
                        <p>Audio peak/RMS: <strong>{metrics?.audio_peak_abs ?? 0} / {metrics?.audio_rms_bps ?? 0} bps</strong></p>
                        <p>Queue full: <strong>{metrics?.queue_full_events ?? 0}</strong></p>
                        <p>Max queue: <strong>{metrics?.max_queue_depth_seen ?? 0}</strong></p>
                        <p>Effective segment: <strong>{liveCapabilities?.capture_health.effective_pipeline.effective_segment_duration_ms ?? 0} ms</strong></p>
                        <p>Session cap: <strong>{liveCapabilities?.capture_health.effective_pipeline.effective_max_segments_per_session ?? 0}</strong></p>
                        <p>Failure threshold: <strong>{liveCapabilities?.capture_health.pipeline.max_consecutive_transcription_failures ?? 0}</strong></p>
                        <p>VAD config: <strong>{liveCapabilities?.capture_health.pipeline.vad_enabled ? `${liveCapabilities.capture_health.pipeline.vad_silence_threshold_pcm} pcm / ${liveCapabilities.capture_health.pipeline.vad_min_speech_ms} ms` : "disabled"}</strong></p>
                        <p>Write failures: <strong>{metrics?.segment_write_failures ?? 0}</strong></p>
                        <p>Backend error: <strong>{metrics?.last_backend_error_kind ?? "none"}</strong></p>
                        <p>Stop state: <strong>{liveCapabilities?.capture_health.last_error ?? "none"}</strong></p>
                        <p className="desktop-agent-muted">{liveCapabilities?.capture_health.last_segment_status ?? liveCapabilities?.audio_capture.reason ?? "Live capture status not loaded."}</p>
                    </article>

                    <article className="desktop-agent-card">
                        <h3>Capability raw details</h3>
                        <p>Manual mode: <strong>{liveCapabilities?.manual_session.state ?? "unknown"}</strong></p>
                        <p>File transcription: <strong>{toolState(toolByName("meeting.transcription.file"))}</strong></p>
                        <p>Audio capture: <strong>{toolState(toolByName("meeting.audio.capture"))}</strong></p>
                        <p>Windows WASAPI: <strong>{liveCapabilities?.windows_wasapi_capture.state ?? "unknown"}</strong></p>
                        <p>Live transcription: <strong>{toolState(toolByName("meeting.transcription.live"))}</strong></p>
                        <p>Segment STT tool: <strong>{toolState(toolByName("meeting.transcription.segment"))}</strong></p>
                        <p>Live segment STT: <strong>{liveCapabilities?.live_segment_transcription.state ?? "unknown"}</strong></p>
                        <p>Live streaming STT: <strong>{liveCapabilities?.live_streaming_stt.state ?? "unknown"}</strong></p>
                        <p>Chunk streaming: <strong>{liveCapabilities?.chunk_streaming.state ?? "unknown"}</strong></p>
                        <p>Diarization: <strong>{liveCapabilities?.diarization.state ?? "unknown"}</strong></p>
                        <p>Speaker attribution: <strong>source default</strong></p>
                        <p>Live summary: <strong>{liveCapabilities?.live_summarization.state ?? "unknown"}</strong></p>
                        <p>Last summary update: <strong>{lastSummaryTimestamp ?? "none"}</strong></p>
                        <p>Follow-up draft: <strong>{toolState(toolByName("meeting.followup.draft"))}</strong></p>
                        <p>Follow-up sending: <strong>{toolState(toolByName("meeting.followup.send"))}</strong></p>
                        <p>Clear data: <strong>{toolState(toolByName("meeting.clear_data"))}</strong></p>
                        <p className="desktop-agent-muted">Diarization: {liveCapabilities?.diarization.reason ?? "unavailable; captured segments use non-identifying segment metadata only."}</p>
                        <p className="desktop-agent-muted">Follow-up draft is copy-only and generated through governed Meeting Intelligence; email sending remains unavailable.</p>
                        <p className="desktop-agent-muted">Summary: {liveCapabilities?.live_summarization.reason ?? "live summary is unavailable unless a governed model adapter is connected."}</p>
                    </article>

                    <article className="desktop-agent-card">
                        <h3>STT boundary</h3>
                        <p>Adapter: <strong>{liveCapabilities?.stt_adapter.state ?? "unknown"}</strong></p>
                        <p>File transcription: <strong>{liveCapabilities?.stt_adapter.file_transcription.state ?? "unknown"}</strong></p>
                        <p>Live transcription: <strong>{liveCapabilities?.stt_adapter.live_transcription.state ?? "unknown"}</strong></p>
                        <p>Chunk streaming: <strong>{liveCapabilities?.stt_adapter.chunk_streaming.state ?? "unknown"}</strong></p>
                        <p>Boundary: <strong>{liveCapabilities?.stt_adapter.existing_boundary ?? "unknown"}</strong></p>
                        <p>Segment STT: <strong>{liveCapabilities?.live_segment_transcription.state ?? "unknown"}</strong></p>
                        <p>Streaming STT: <strong>{liveCapabilities?.live_streaming_stt.state ?? "unknown"}</strong></p>
                        <p>Chunk stream: <strong>{liveCapabilities?.stt_adapter.chunk_streaming_supported ? "supported" : "unsupported"}</strong></p>
                        <p>Placeholder text: <strong>{liveCapabilities?.stt_adapter.emits_placeholder_transcripts ? "possible" : "never"}</strong></p>
                        {liveCapabilities?.stt_adapter.reason ? (
                            <p className="desktop-agent-muted">Adapter: {liveCapabilities.stt_adapter.reason}</p>
                        ) : null}
                        <p className="desktop-agent-muted">File: {liveCapabilities?.stt_adapter.file_transcription.reason ?? "Ready when the existing file STT bridge is attached."}</p>
                        <p className="desktop-agent-muted">Live: {liveCapabilities?.stt_adapter.live_transcription.reason ?? "Live transcription remains unavailable."}</p>
                        <p className="desktop-agent-muted">Chunk: {liveCapabilities?.stt_adapter.chunk_streaming.reason ?? "Chunk streaming remains unavailable."}</p>
                    </article>

                    <article className="desktop-agent-card">
                        <h3>Hardware validation</h3>
                        <p>1. Enable MeetingAudioCapture and MeetingTranscriptionSegment.</p>
                        <p>2. Grant consent for the meeting platform.</p>
                        <p>3. Start capture and play system audio or speak into the microphone.</p>
                        <p>4. Wait one effective segment duration.</p>
                        <p>5. Confirm segments written and transcribed increase.</p>
                        <p>6. Confirm VAD does not drop speech.</p>
                        <p>7. Revoke consent and confirm capture stops.</p>
                    </article>

                    <article className="desktop-agent-card">
                        <h3>Registered meeting tools</h3>
                        <div className="desktop-agent-tool-list">
                            {meetingTools.map((tool) => (
                                <div key={tool.tool_name} className="desktop-agent-tool-row">
                                    <strong>{tool.tool_name}</strong>
                                    <span>{tool.available ? "available" : "unavailable"}</span>
                                </div>
                            ))}
                        </div>
                    </article>
                </div>
            </details>
        </div>
    );
}
