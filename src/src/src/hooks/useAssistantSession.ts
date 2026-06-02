import { useCallback, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useAssistantEvents } from "./useAssistantEvents";
import {
    type AudioSegmentFailedEvent,
    type AudioSegmentReadyEvent,
    useAssistantAudio,
} from "./useAssistantAudio";
import { useVoiceInput } from "./useVoiceInput";
import { useVoiceSession } from "./useVoiceSession";
import type {
    AssistantActivityEvent,
    AssistantDeepSearchActivityEvent,
    AssistantErrorEvent,
    AssistantInterruptedEvent,
    AssistantOrchestratorDiagnostic,
    AssistantRequestFinishedEvent,
    AssistantRequestSettledEvent,
    AssistantRequestStartedEvent,
    AssistantRouterDiagnostic,
    AssistantStatus,
    AssistantDeepSearchOptions,
    AssistantToolSynthesisDiagnostic,
    ChatMessage,
    RequestMetricsSnapshot,
    SpeechSegmentQueuedEvent,
    StartChatResponse,
    StreamChunkEvent,
    VoiceSessionStateEvent,
    VoiceSessionTranscriptEvent,
    VoiceTurnMetricsSnapshot,
} from "../types/assistant";
import type { ConversationRouteDiagnostic } from "../types/desktopAgent";

const INITIAL_MESSAGES: ChatMessage[] = [
    /*{
        id: crypto.randomUUID(),
        role: "assistant",
        content: "Ciao. Sono pronta. Dimmi pure cosa vuoi fare.",
    },*/
];

const EMPTY_RESPONSE_FALLBACK =
    "Non ho ricevuto una risposta testuale dal modello. Riprova o cambia modello.";

const DEEP_SEARCH_STORAGE_KEY = "astra.assistant.deepSearch.enabled";

export function useAssistantSession() {
    const [messages, setMessages] = useState<ChatMessage[]>(INITIAL_MESSAGES);
    const [inputValue, setInputValue] = useState("");
    const [status, setStatus] = useState<AssistantStatus>("idle");
    const [activeModel, setActiveModel] = useState("unknown");
    const [isLoading, setIsLoading] = useState(false);
    const [autoSubmitVoice, setAutoSubmitVoice] = useState(true); // default to false to prevent unexpected behavior, can be toggled by user
    const [deepSearchEnabled, setDeepSearchEnabledState] = useState(() => {
        try {
            return window.localStorage.getItem(DEEP_SEARCH_STORAGE_KEY) === "true";
        } catch {
            return false;
        }
    });
    const [lastMetrics, setLastMetrics] = useState<RequestMetricsSnapshot | null>(null);
    const [lastVoiceMetrics, setLastVoiceMetrics] = useState<VoiceTurnMetricsSnapshot | null>(null);
    const [lastVoiceTranscript, setLastVoiceTranscript] =
        useState<VoiceSessionTranscriptEvent | null>(null);

    const activeAssistantMessageId = useRef<string | null>(null);
    const pendingAssistantMessageIdRef = useRef<string | null>(null);
    const activeRequestIdRef = useRef<string | null>(null);
    const assistantMessageByRequestRef = useRef<Map<string, string>>(new Map());
    const bufferedStreamChunksRef = useRef<Map<string, string>>(new Map());
    const bufferedFinishedEventsRef = useRef<Map<string, AssistantRequestFinishedEvent>>(new Map());
    const finishedRequestIdsRef = useRef<Set<string>>(new Set());
    const startedAudioSessionRequestIdsRef = useRef<Set<string>>(new Set());
    const pendingSpeechSegmentsRef = useRef<Set<string>>(new Set());
    const completedAudioSessionsRef = useRef<Set<string>>(new Set());
    const failedAudioSessionsRef = useRef<Set<string>>(new Set());
    const isStreamingRef = useRef(false);
    const isAudioSpeakingRef = useRef(false);
    const voiceRestStatusRef = useRef<AssistantStatus | null>(null);

    const setDeepSearchEnabled = useCallback((enabled: boolean) => {
        setDeepSearchEnabledState(enabled);
        try {
            window.localStorage.setItem(DEEP_SEARCH_STORAGE_KEY, String(enabled));
        } catch (error) {
            console.debug("Unable to persist Astra Deep Search toggle:", error);
        }
    }, []);

    const completeAudioSession = useCallback(async (requestId: string) => {
        if (completedAudioSessionsRef.current.has(requestId)) return;
        completedAudioSessionsRef.current.add(requestId);

        try {
            const hadFailures = failedAudioSessionsRef.current.has(requestId);
            await invoke("notify_audio_session_completed", {
                payload: {
                    request_id: requestId,
                    had_failures: hadFailures,
                },
            });
        } catch (error) {
            console.error("notify_audio_session_completed error:", error);
        } finally {
            failedAudioSessionsRef.current.delete(requestId);
        }
    }, []);

    const settleVisualStatus = useCallback(() => {
        if (isAudioSpeakingRef.current) {
            setStatus("speaking");
            return;
        }

        if (isStreamingRef.current) {
            setStatus("thinking");
            return;
        }

        if (pendingSpeechSegmentsRef.current.size > 0) {
            setStatus("settling");
            return;
        }

        setStatus(voiceRestStatusRef.current ?? "idle");
    }, []);

    const handleSpeakingStart = useCallback(() => {
        isAudioSpeakingRef.current = true;
        setStatus("speaking");
    }, []);

    const handleSpeakingEnd = useCallback(() => {
        isAudioSpeakingRef.current = false;
        settleVisualStatus();
    }, [settleVisualStatus]);

    const handleQueueIdle = useCallback(() => {
        isAudioSpeakingRef.current = false;
        settleVisualStatus();
    }, [settleVisualStatus]);

    const notifyAudioPlaybackStarted = useCallback((segment: AudioSegmentReadyEvent) => {
        void invoke("notify_audio_playback_started", {
            payload: playbackPayload(segment),
        }).catch((error) => console.error("notify_audio_playback_started error:", error));
    }, []);

    const notifyAudioPlaybackCompleted = useCallback((segment: AudioSegmentReadyEvent) => {
        void invoke("notify_audio_playback_completed", {
            payload: playbackPayload(segment),
        }).catch((error) => console.error("notify_audio_playback_completed error:", error));
    }, []);

    const {
        enqueueAudioSegment,
        hasPendingWork,
        markAudioSegmentFailed,
        startNewRequestAudioSession,
        stopAllAudio,
    } = useAssistantAudio({
        onSpeakingStart: handleSpeakingStart,
        onSpeakingEnd: handleSpeakingEnd,
        onQueueIdle: handleQueueIdle,
        onSegmentPlaybackStart: notifyAudioPlaybackStarted,
        onSegmentPlaybackComplete: notifyAudioPlaybackCompleted,
        onSessionPlaybackIdle: completeAudioSession,
    });

    const startAudioSessionOnce = useCallback(
        (requestId: string) => {
            if (startedAudioSessionRequestIdsRef.current.has(requestId)) return;
            startedAudioSessionRequestIdsRef.current.add(requestId);
            startNewRequestAudioSession(requestId);
        },
        [startNewRequestAudioSession]
    );

    const updateAssistantMessage = useCallback(
        (assistantMessageId: string, text: string, mode: "append" | "replace") => {
            if (!text) return;

            setMessages((prev) =>
                prev.map((msg) => {
                    if (msg.id !== assistantMessageId) return msg;
                    return {
                        ...msg,
                        content: mode === "replace" ? text : msg.content + text,
                    };
                })
            );
        },
        []
    );

    const finishRequestLifecycle = useCallback(
        (requestId: string) => {
            isStreamingRef.current = false;

            const assistantMessageId = assistantMessageByRequestRef.current.get(requestId);
            if (pendingAssistantMessageIdRef.current === assistantMessageId) {
                pendingAssistantMessageIdRef.current = null;
            }
            if (activeAssistantMessageId.current === assistantMessageId) {
                activeAssistantMessageId.current = null;
            }

            setIsLoading(false);

            if (
                !isAudioSpeakingRef.current &&
                !hasPendingWork() &&
                pendingSpeechSegmentsRef.current.size === 0
            ) {
                settleVisualStatus();
                void completeAudioSession(requestId);
            } else if (!isAudioSpeakingRef.current) {
                setStatus("settling");
            }
        },
        [completeAudioSession, hasPendingWork, settleVisualStatus]
    );

    const applyBufferedResponseEvents = useCallback(
        (requestId: string, assistantMessageId: string) => {
            const finished = bufferedFinishedEventsRef.current.get(requestId);
            if (finished) {
                bufferedFinishedEventsRef.current.delete(requestId);
                bufferedStreamChunksRef.current.delete(requestId);
                finishedRequestIdsRef.current.add(requestId);
                updateAssistantMessage(assistantMessageId, finalAssistantText(finished.full_text), "replace");
                finishRequestLifecycle(requestId);
                return;
            }

            const bufferedChunk = bufferedStreamChunksRef.current.get(requestId);
            if (bufferedChunk) {
                bufferedStreamChunksRef.current.delete(requestId);
                updateAssistantMessage(assistantMessageId, bufferedChunk, "append");
            }
        },
        [finishRequestLifecycle, updateAssistantMessage]
    );

    const bindRequestToAssistantMessage = useCallback(
        (requestId: string, assistantMessageId: string) => {
            assistantMessageByRequestRef.current.set(requestId, assistantMessageId);
            applyBufferedResponseEvents(requestId, assistantMessageId);
        },
        [applyBufferedResponseEvents]
    );

    const submitMessage = useCallback(
        async (
            messageOverride?: string,
            options: { inputModality?: "typed" | "voice"; audioResponse?: "auto" | "enabled" | "disabled"; deepSearch?: boolean } = {}
        ) => {
            const trimmed = (messageOverride ?? inputValue).trim();
            if (!trimmed) return;
            const inputModality = options.inputModality ?? "typed";
            const audioResponse = options.audioResponse ?? "auto";
            const deepSearchForTurn = options.deepSearch ?? deepSearchEnabled;
            const deepSearchPayload: AssistantDeepSearchOptions = {
                enabled: deepSearchForTurn,
                enable_web_discovery: true,
                include_general_web: true,
                include_academic_sources: true,
                document_ingestion: true,
                prefer_academic_landing_pages: false,
                enable_pdf_text_extraction: true,
                autonomous_loop: true,
                max_research_passes: 5,
                min_research_passes: 2,
                max_sources_per_pass: 8,
                min_new_information_gain: 0.08,
                min_coverage_score: 0.66,
                min_supported_claim_ratio: 0.55,
                enable_claim_graph: true,
                min_independent_sources_for_claim: 2,
                enable_contradiction_detection: true,
                enable_memory_promotion_policy: true,
                auto_promote_supported_claims: true,
                require_user_confirmation_for_system_verified: true,
                min_promotion_confidence: 0.62,
                min_promotion_independent_sources: 2,
                enable_source_reliability_scoring: true,
                min_reliable_source_score_for_promotion: 0.5,
                max_sources: 24,
                max_discovery_results_per_provider: 10,
                max_discovered_sources: 192,
                initial_query_count: 6,
                require_cross_source_verification: true,
            };

            stopAllAudio();
            startedAudioSessionRequestIdsRef.current.clear();
            pendingSpeechSegmentsRef.current.clear();
            isStreamingRef.current = true;
            isAudioSpeakingRef.current = false;

            const userMessage: ChatMessage = {
                id: crypto.randomUUID(),
                role: "user",
                content: trimmed,
            };
            const assistantMessageId = crypto.randomUUID();
            const clientRequestId = crypto.randomUUID();
            const assistantPlaceholder: ChatMessage = {
                id: assistantMessageId,
                role: "assistant",
                content: "",
                activity: {
                    current: {
                        id: `${clientRequestId}:queued`,
                        stage: deepSearchForTurn ? "deep_search_queued" : "queued",
                        title: deepSearchForTurn ? "Preparing Deep Search" : "Preparing response",
                        detail: deepSearchForTurn
                            ? "Astra is preparing governed web and memory research."
                            : "Astra is preparing the response pipeline.",
                        timestamp_ms: Date.now(),
                        metadata: { local: true },
                    },
                    steps: [],
                },
            };

            activeAssistantMessageId.current = assistantMessageId;
            pendingAssistantMessageIdRef.current = assistantMessageId;
            activeRequestIdRef.current = clientRequestId;
            bindRequestToAssistantMessage(clientRequestId, assistantMessageId);

            setMessages((prev) => [...prev, userMessage, assistantPlaceholder]);
            setInputValue("");
            setIsLoading(true);
            setStatus("thinking");

            try {
                const started = await invoke<StartChatResponse>("start_chat_message_stream", {
                    payload: {
                        client_request_id: clientRequestId,
                        message: trimmed,
                        input_modality: inputModality,
                        audio_response: audioResponse,
                        deep_search: deepSearchPayload,
                    },
                });

                const alreadyActive = activeRequestIdRef.current === started.request_id;
                bindRequestToAssistantMessage(started.request_id, assistantMessageId);
                const alreadyFinished = finishedRequestIdsRef.current.has(started.request_id);
                activeRequestIdRef.current = started.request_id;
                if (!alreadyFinished) {
                    activeAssistantMessageId.current = assistantMessageId;
                }
                completedAudioSessionsRef.current.delete(started.request_id);
                failedAudioSessionsRef.current.delete(started.request_id);
                setActiveModel(started.model);

                if (!alreadyActive && !alreadyFinished && started.audio_response_enabled) {
                    startAudioSessionOnce(started.request_id);
                }
            } catch (error) {
                console.error("start_chat_message_stream error:", error);
                const errorText = error instanceof Error ? error.message : String(error);

                setMessages((prev) =>
                    prev.map((msg) =>
                        msg.id === assistantMessageId
                            ? { ...msg, content: `Errore backend/Tauri/Ollama: ${errorText}` }
                            : msg
                    )
                );

                isStreamingRef.current = false;
                pendingSpeechSegmentsRef.current.clear();
                activeAssistantMessageId.current = null;
                pendingAssistantMessageIdRef.current = null;
                setIsLoading(false);
                settleVisualStatus();
            }
        },
        [bindRequestToAssistantMessage, deepSearchEnabled, inputValue, settleVisualStatus, startAudioSessionOnce, stopAllAudio]
    );

    const handleTranscript = useCallback(
        (text: string, shouldAutoSubmit: boolean) => {
            setInputValue(text);
            if (shouldAutoSubmit) {
                void submitMessage(text, { inputModality: "voice", audioResponse: "auto" });
            }
        },
        [submitMessage]
    );

    const voiceInput = useVoiceInput({
        autoSubmit: autoSubmitVoice,
        onListeningStart: () => setStatus("listening"),
        onListeningEnd: settleVisualStatus,
        onTranscript: handleTranscript,
        onError: (message) => {
            console.error("Voice input error:", message);
            settleVisualStatus();
        },
    });

    const voiceSession = useVoiceSession({
        onSessionListening: () => {
            voiceRestStatusRef.current = "passive";
            settleVisualStatus();
        },
        onSessionStopped: () => {
            voiceRestStatusRef.current = null;
            settleVisualStatus();
        },
        onError: (message) => {
            console.error("Voice session error:", message);
            voiceRestStatusRef.current = null;
            setStatus("idle");
        },
    });

    const handleRequestStarted = useCallback(
        ({ request_id, model, user_message, audio_response_enabled }: AssistantRequestStartedEvent) => {
            const alreadyActive = activeRequestIdRef.current === request_id;
            let assistantMessageId =
                pendingAssistantMessageIdRef.current ?? activeAssistantMessageId.current;

            if (!assistantMessageId && user_message?.trim()) {
                const userMessage: ChatMessage = {
                    id: crypto.randomUUID(),
                    role: "user",
                    content: user_message.trim(),
                };
                assistantMessageId = crypto.randomUUID();
                const assistantPlaceholder: ChatMessage = {
                    id: assistantMessageId,
                    role: "assistant",
                    content: "",
                };

                setMessages((prev) => [...prev, userMessage, assistantPlaceholder]);
            }

            activeRequestIdRef.current = request_id;
            if (assistantMessageId) {
                bindRequestToAssistantMessage(request_id, assistantMessageId);
            }
            const alreadyFinished = finishedRequestIdsRef.current.has(request_id);
            activeAssistantMessageId.current = alreadyFinished ? null : assistantMessageId;
            completedAudioSessionsRef.current.delete(request_id);
            failedAudioSessionsRef.current.delete(request_id);
            isStreamingRef.current = !alreadyFinished;
            isAudioSpeakingRef.current = false;
            pendingSpeechSegmentsRef.current.clear();

            setActiveModel(model);
            if (!alreadyFinished) {
                setIsLoading(true);
                setStatus("thinking");
            }

            if (!alreadyActive && !alreadyFinished && audio_response_enabled) {
                startAudioSessionOnce(request_id);
            }
        },
        [bindRequestToAssistantMessage, startAudioSessionOnce]
    );

    const handleStreamChunk = useCallback(({ request_id, chunk }: StreamChunkEvent) => {
        if (finishedRequestIdsRef.current.has(request_id)) return;

        const assistantId =
            assistantMessageByRequestRef.current.get(request_id) ??
            (activeRequestIdRef.current === request_id ? activeAssistantMessageId.current : null);

        if (!assistantId) {
            const previous = bufferedStreamChunksRef.current.get(request_id) ?? "";
            bufferedStreamChunksRef.current.set(request_id, previous + chunk);
            return;
        }

        updateAssistantMessage(assistantId, chunk, "append");
    }, [updateAssistantMessage]);

    const handleAudioReady = useCallback(
        (event: AudioSegmentReadyEvent) => {
            pendingSpeechSegmentsRef.current.delete(`${event.request_id}:${event.sequence}`);
            enqueueAudioSegment(event);
        },
        [enqueueAudioSegment]
    );

    const handleAudioFailed = useCallback(
        (event: AudioSegmentFailedEvent) => {
            pendingSpeechSegmentsRef.current.delete(`${event.request_id}:${event.sequence}`);
            failedAudioSessionsRef.current.add(event.request_id);
            markAudioSegmentFailed(event);
            settleVisualStatus();
        },
        [markAudioSegmentFailed, settleVisualStatus]
    );

    const handleSpeechQueued = useCallback((event: SpeechSegmentQueuedEvent) => {
        if (activeRequestIdRef.current !== event.request_id) return;

        pendingSpeechSegmentsRef.current.add(`${event.request_id}:${event.sequence}`);
        if (!isAudioSpeakingRef.current) {
            setStatus("settling");
        }
    }, []);

    const handleRequestFinished = useCallback(
        (event: AssistantRequestFinishedEvent) => {
            const assistantId = assistantMessageByRequestRef.current.get(event.request_id);
            if (!assistantId) {
                bufferedFinishedEventsRef.current.set(event.request_id, event);
                return;
            }

            bufferedStreamChunksRef.current.delete(event.request_id);
            finishedRequestIdsRef.current.add(event.request_id);
            updateAssistantMessage(assistantId, finalAssistantText(event.full_text), "replace");
            finishRequestLifecycle(event.request_id);
        },
        [finishRequestLifecycle, updateAssistantMessage]
    );

    const handleAssistantError = useCallback(
        ({ request_id, stage, message }: AssistantErrorEvent) => {
            if (activeRequestIdRef.current !== request_id && stage !== "stt") return;

            console.error(`Assistant ${stage} error:`, message);

            if (stage === "tts") {
                settleVisualStatus();
                return;
            }

            isStreamingRef.current = false;
            pendingSpeechSegmentsRef.current.clear();
            setIsLoading(false);

            const assistantId =
                activeAssistantMessageId.current ?? pendingAssistantMessageIdRef.current;

            if (assistantId) {
                setMessages((prev) =>
                    prev.map((msg) =>
                        msg.id === assistantId
                            ? {
                                  ...msg,
                                  content: msg.content
                                      ? `${msg.content}\n\nErrore: ${message}`
                                      : `Errore backend/Tauri/Ollama: ${message}`,
                              }
                            : msg
                    )
                );
            }

            activeAssistantMessageId.current = null;
            pendingAssistantMessageIdRef.current = null;
            settleVisualStatus();
        },
        [settleVisualStatus]
    );

    const handleAssistantInterrupted = useCallback(
        (event: AssistantInterruptedEvent) => {
            stopAllAudio();
            pendingSpeechSegmentsRef.current.clear();
            isStreamingRef.current = false;
            isAudioSpeakingRef.current = false;
            setIsLoading(false);
            if (event.reason === "replaced_by_new_request") {
                setStatus("thinking");
                return;
            }
            setStatus("listening");
        },
        [stopAllAudio]
    );

    const handleRequestSettled = useCallback(
        (_event: AssistantRequestSettledEvent) => {
            pendingSpeechSegmentsRef.current.clear();
            isStreamingRef.current = false;
            isAudioSpeakingRef.current = false;
            setIsLoading(false);
            settleVisualStatus();
        },
        [settleVisualStatus]
    );

    const handleVoiceSessionState = useCallback(
        (event: VoiceSessionStateEvent) => {
            voiceSession.applyStateEvent(event);

            voiceRestStatusRef.current = getVoiceRestStatus(event);

            if (event.state === "listening" || event.state === "interrupted") {
                setStatus("listening");
                return;
            }
            if (event.state === "processing") {
                setStatus("thinking");
                return;
            }
            if (event.state === "speaking") {
                setStatus("speaking");
                return;
            }
            if (event.state === "armed") {
                setStatus("armed");
                return;
            }
            if (
                event.state === "disabled" ||
                event.state === "passive" ||
                event.state === "cooldown"
            ) {
                settleVisualStatus();
            }
        },
        [settleVisualStatus, voiceSession.applyStateEvent]
    );

    const handleVoiceSessionTranscript = useCallback((event: VoiceSessionTranscriptEvent) => {
        setLastVoiceTranscript(event);
    }, []);

    const appendAssistantActivity = useCallback((requestId: string, step: Omit<NonNullable<ChatMessage["activity"]>["steps"][number], "id">) => {
        const assistantId =
            assistantMessageByRequestRef.current.get(requestId) ??
            (activeRequestIdRef.current === requestId ? activeAssistantMessageId.current : null);
        if (!assistantId) return;

        const normalizedStep = {
            id: `${requestId}:${step.stage}:${step.timestamp_ms ?? Date.now()}:${Math.random().toString(36).slice(2, 7)}`,
            ...step,
        };

        setMessages((prev) =>
            prev.map((msg) => {
                if (msg.id !== assistantId) return msg;
                const previousSteps = msg.activity?.steps ?? [];
                const steps = [...previousSteps, normalizedStep].slice(-24);
                return {
                    ...msg,
                    activity: {
                        current: normalizedStep,
                        steps,
                        expanded: msg.activity?.expanded ?? false,
                    },
                };
            })
        );
    }, []);

    const handleAssistantActivity = useCallback(
        (event: AssistantActivityEvent) => {
            appendAssistantActivity(event.request_id, {
                stage: event.stage,
                title: event.title,
                detail: event.detail,
                timestamp_ms: event.timestamp_ms ?? Date.now(),
                metadata: event.metadata ?? null,
            });
        },
        [appendAssistantActivity]
    );

    const handleDeepSearchActivity = useCallback(
        (event: AssistantDeepSearchActivityEvent) => {
            const detailParts = [
                typeof event.sources_accepted === "number" ? `${event.sources_accepted} sources accepted` : null,
                typeof event.candidate_sources_discovered === "number" ? `${event.candidate_sources_discovered} candidates` : null,
                typeof event.extracted_claims === "number" ? `${event.extracted_claims} claims` : null,
                typeof event.promoted_claims === "number" ? `${event.promoted_claims} promoted` : null,
                event.error ? `error: ${event.error}` : null,
            ].filter(Boolean);
            appendAssistantActivity(event.request_id, {
                stage: `deep_search_${event.status}`,
                title:
                    event.status === "started"
                        ? "Deep Search is exploring sources"
                        : event.status === "failed"
                          ? "Deep Search failed"
                          : "Deep Search updated",
                detail: detailParts.length ? detailParts.join(" · ") : "Astra updated the governed Deep Search pipeline.",
                timestamp_ms: Date.now(),
                metadata: event as unknown as Record<string, unknown>,
            });
        },
        [appendAssistantActivity]
    );

    const handleRouteDiagnostic = useCallback((event: ConversationRouteDiagnostic) => {
        console.info("Astra route diagnostic:", event);
    }, []);

    const handleRouterDiagnostic = useCallback((event: AssistantRouterDiagnostic) => {
        console.debug("Astra router diagnostic:", event);
    }, []);

    const handleOrchestratorDiagnostic = useCallback((event: AssistantOrchestratorDiagnostic) => {
        console.debug("Astra orchestrator diagnostic:", event);
    }, []);

    const handleToolSynthesisDiagnostic = useCallback((event: AssistantToolSynthesisDiagnostic) => {
        console.debug("Astra tool synthesis diagnostic:", event);
    }, []);

    const handleStatus = useCallback(
        (nextStatus: AssistantStatus) => {
            if (nextStatus === "idle") {
                isStreamingRef.current = false;
                if (
                    !isAudioSpeakingRef.current &&
                    !hasPendingWork() &&
                    pendingSpeechSegmentsRef.current.size === 0
                ) {
                    settleVisualStatus();
                }
                return;
            }

            if (nextStatus === "settling") {
                if (!isAudioSpeakingRef.current) {
                    setStatus("settling");
                }
                return;
            }

            if (nextStatus === "thinking" && isAudioSpeakingRef.current) {
                return;
            }

            setStatus(nextStatus);
        },
        [hasPendingWork, settleVisualStatus]
    );

    useAssistantEvents({
        onRequestStarted: handleRequestStarted,
        onStreamChunk: handleStreamChunk,
        onAudioReady: handleAudioReady,
        onAudioFailed: handleAudioFailed,
        onSpeechQueued: handleSpeechQueued,
        onRequestFinished: handleRequestFinished,
        onRequestSettled: handleRequestSettled,
        onAssistantError: handleAssistantError,
        onAssistantActivity: handleAssistantActivity,
        onDeepSearchActivity: handleDeepSearchActivity,
        onStatus: handleStatus,
        onModel: setActiveModel,
        onMetrics: setLastMetrics,
        onAssistantInterrupted: handleAssistantInterrupted,
        onVoiceSessionState: handleVoiceSessionState,
        onVoiceSessionTranscript: handleVoiceSessionTranscript,
        onVoiceTurnMetrics: setLastVoiceMetrics,
        onRouteDiagnostic: handleRouteDiagnostic,
        onOrchestratorDiagnostic: handleOrchestratorDiagnostic,
        onRouterDiagnostic: handleRouterDiagnostic,
        onToolSynthesisDiagnostic: handleToolSynthesisDiagnostic,
    });

    return {
        activeModel,
        autoSubmitVoice,
        deepSearchEnabled,
        inputValue,
        isLoading,
        lastMetrics,
        lastVoiceMetrics,
        lastVoiceTranscript,
        messages,
        setAutoSubmitVoice,
        setDeepSearchEnabled,
        setInputValue,
        status,
        stopAllAudio,
        submitMessage,
        voiceInput,
        voiceSession,
    };
}

function playbackPayload(segment: AudioSegmentReadyEvent) {
    return {
        request_id: segment.request_id,
        segment_id: segment.segment_id,
        sequence: segment.sequence,
        output_path: segment.output_path,
    };
}

function finalAssistantText(text: string) {
    return text.trim() ? text : EMPTY_RESPONSE_FALLBACK;
}

function getVoiceRestStatus(event: VoiceSessionStateEvent): AssistantStatus | null {
    if (event.state === "disabled") return null;
    if (event.state === "passive") return "passive";
    if (event.mode === "conversation") return "armed";
    return "passive";
}
