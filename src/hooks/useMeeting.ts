import { useCallback, useMemo } from "react";
import { invoke } from "@tauri-apps/api/core";
import type {
    ActionItem,
    CallInfo,
    CaptureBackend,
    ClearMeetingDataRequest,
    ConsentState,
    DecisionLogEntry,
    ExportedMeeting,
    MeetingAudioFileTranscriptionRequest,
    MeetingAudioFileTranscriptionResult,
    MeetingConfig,
    MeetingDataClearPreview,
    MeetingDataClearResult,
    MeetingDiagnostic,
    MeetingIntelligenceGenerationOptions,
    MeetingIntelligenceResult,
    MeetingLiveCapabilitySnapshot,
    MeetingSession,
    MeetingSessionState,
    NoteEntry,
    RenameSpeakerRequest,
    RenameSpeakerResult,
    SummaryEntry,
    TranscriptEntry,
} from "../types/meeting";

export function useMeeting() {
    const getConsentState = useCallback(
        () => invoke<ConsentState>("get_meeting_consent_state"),
        []
    );

    const grantConsent = useCallback(
        (appName: string) =>
            invoke<ConsentState>("grant_meeting_consent", { appName }),
        []
    );

    const revokeConsent = useCallback(
        (appName: string) =>
            invoke<ConsentState>("revoke_meeting_consent", { appName }),
        []
    );

    const startSession = useCallback(
        (platform: string, config: MeetingConfig) =>
            invoke<MeetingSession>("start_meeting_session", { platform, config }),
        []
    );

    const getActiveSession = useCallback(
        () => invoke<MeetingSession | null>("get_active_meeting_session"),
        []
    );

    const getActiveState = useCallback(
        () => invoke<MeetingSessionState>("get_active_meeting_state"),
        []
    );

    const getLastCompletedState = useCallback(
        () => invoke<MeetingSessionState | null>("get_last_completed_meeting_state"),
        []
    );

    const getLiveCapabilities = useCallback(
        () => invoke<MeetingLiveCapabilitySnapshot>("get_meeting_live_capabilities"),
        []
    );

    const listTranscript = useCallback(
        () => invoke<TranscriptEntry[]>("list_meeting_transcript"),
        []
    );

    const readNotes = useCallback(
        () => invoke<NoteEntry[]>("read_meeting_notes"),
        []
    );

    const readSummary = useCallback(
        () => invoke<SummaryEntry[]>("read_meeting_summary"),
        []
    );

    const readActionItems = useCallback(
        () => invoke<ActionItem[]>("read_meeting_action_items"),
        []
    );

    const readDecisions = useCallback(
        () => invoke<DecisionLogEntry[]>("read_meeting_decisions"),
        []
    );

    const readDiagnostics = useCallback(
        () => invoke<MeetingDiagnostic[]>("read_meeting_diagnostics"),
        []
    );

    const generateIntelligence = useCallback(
        (options: MeetingIntelligenceGenerationOptions) =>
            invoke<MeetingIntelligenceResult>("generate_meeting_intelligence", { options }),
        []
    );

    const readIntelligence = useCallback(
        () => invoke<MeetingIntelligenceResult | null>("read_meeting_intelligence"),
        []
    );

    const clearIntelligence = useCallback(
        () => invoke<void>("clear_meeting_intelligence"),
        []
    );

    const pauseSession = useCallback(
        () => invoke<void>("pause_meeting_session"),
        []
    );

    const resumeSession = useCallback(
        () => invoke<void>("resume_meeting_session"),
        []
    );

    const stopSession = useCallback(
        () => invoke<ExportedMeeting>("stop_meeting_session"),
        []
    );

    const addTranscript = useCallback(
        (entry: TranscriptEntry) =>
            invoke<void>("add_meeting_transcript", { entry }),
        []
    );

    const renameSpeaker = useCallback(
        (request: RenameSpeakerRequest) =>
            invoke<RenameSpeakerResult>("rename_meeting_speaker", { request }),
        []
    );

    const transcribeAudioFile = useCallback(
        (request: MeetingAudioFileTranscriptionRequest) =>
            invoke<MeetingAudioFileTranscriptionResult>("transcribe_meeting_audio_file", { request }),
        []
    );

    const addActionItem = useCallback(
        (item: ActionItem) => invoke<void>("add_meeting_action_item", { item }),
        []
    );

    const addDecision = useCallback(
        (entry: DecisionLogEntry) =>
            invoke<void>("add_meeting_decision", { entry }),
        []
    );

    const clearSession = useCallback(
        () => invoke<void>("clear_meeting_session"),
        []
    );

    const detectActiveCall = useCallback(
        () => invoke<CallInfo | null>("detect_active_call"),
        []
    );

    const getAvailableAudioDevices = useCallback(
        () => invoke<string[]>("get_available_audio_devices"),
        []
    );

    const autoDetectAudioBackend = useCallback(
        () => invoke<CaptureBackend>("auto_detect_audio_backend"),
        []
    );

    const previewClearData = useCallback(
        () => invoke<MeetingDataClearPreview>("preview_clear_meeting_data"),
        []
    );

    const clearData = useCallback(
        (request: ClearMeetingDataRequest) =>
            invoke<MeetingDataClearResult>("clear_meeting_data", { request }),
        []
    );

    return useMemo(
        () => ({
            addActionItem,
            addDecision,
            addTranscript,
            autoDetectAudioBackend,
            clearData,
            clearIntelligence,
            clearSession,
            detectActiveCall,
            getActiveSession,
            getActiveState,
            getAvailableAudioDevices,
            getConsentState,
            generateIntelligence,
            getLastCompletedState,
            getLiveCapabilities,
            grantConsent,
            listTranscript,
            pauseSession,
            previewClearData,
            readActionItems,
            readDecisions,
            readDiagnostics,
            readIntelligence,
            readNotes,
            readSummary,
            renameSpeaker,
            resumeSession,
            revokeConsent,
            startSession,
            stopSession,
            transcribeAudioFile,
        }),
        [
            addActionItem,
            addDecision,
            addTranscript,
            autoDetectAudioBackend,
            clearData,
            clearIntelligence,
            clearSession,
            detectActiveCall,
            getActiveSession,
            getActiveState,
            getAvailableAudioDevices,
            getConsentState,
            generateIntelligence,
            getLastCompletedState,
            getLiveCapabilities,
            grantConsent,
            listTranscript,
            pauseSession,
            previewClearData,
            readActionItems,
            readDecisions,
            readDiagnostics,
            readIntelligence,
            readNotes,
            readSummary,
            renameSpeaker,
            resumeSession,
            revokeConsent,
            startSession,
            stopSession,
            transcribeAudioFile,
        ]
    );
}
