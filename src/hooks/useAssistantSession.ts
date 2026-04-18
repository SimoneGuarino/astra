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
    AssistantErrorEvent,
    AssistantInterruptedEvent,
    AssistantRequestFinishedEvent,
    AssistantRequestStartedEvent,
    AssistantStatus,
    ChatMessage,
    RequestMetricsSnapshot,
    SpeechSegmentQueuedEvent,
    StartChatResponse,
    StreamChunkEvent,
    VoiceSessionStateEvent,
    VoiceSessionTranscriptEvent,
    VoiceTurnMetricsSnapshot,
} from "../types/assistant";

const INITIAL_MESSAGES: ChatMessage[] = [
    {
        id: crypto.randomUUID(),
        role: "assistant",
        content: "Ciao. Sono pronta. Dimmi pure cosa vuoi fare.",
    },
];

export function useAssistantSession() {
    const [messages, setMessages] = useState<ChatMessage[]>(INITIAL_MESSAGES);
    const [inputValue, setInputValue] = useState("");
    const [status, setStatus] = useState<AssistantStatus>("idle");
    const [activeModel, setActiveModel] = useState("unknown");
    const [isLoading, setIsLoading] = useState(false);
    const [autoSubmitVoice, setAutoSubmitVoice] = useState(true); // default to false to prevent unexpected behavior, can be toggled by user
    const [lastMetrics, setLastMetrics] = useState<RequestMetricsSnapshot | null>(null);
    const [lastVoiceMetrics, setLastVoiceMetrics] = useState<VoiceTurnMetricsSnapshot | null>(null);
    const [lastVoiceTranscript, setLastVoiceTranscript] =
        useState<VoiceSessionTranscriptEvent | null>(null);

    const activeAssistantMessageId = useRef<string | null>(null);
    const pendingAssistantMessageIdRef = useRef<string | null>(null);
    const activeRequestIdRef = useRef<string | null>(null);
    const pendingSpeechSegmentsRef = useRef<Set<string>>(new Set());
    const completedAudioSessionsRef = useRef<Set<string>>(new Set());
    const isStreamingRef = useRef(false);
    const isAudioSpeakingRef = useRef(false);
    const voiceRestStatusRef = useRef<AssistantStatus | null>(null);

    const completeAudioSession = useCallback(async (requestId: string) => {
        if (completedAudioSessionsRef.current.has(requestId)) return;
        completedAudioSessionsRef.current.add(requestId);

        try {
            await invoke("notify_audio_session_completed", {
                payload: { request_id: requestId },
            });
        } catch (error) {
            console.error("notify_audio_session_completed error:", error);
        }
    }, []);

    const settleVisualStatus = useCallback(() => {
        if (isAudioSpeakingRef.current) {
            setStatus("speaking");
            return;
        }

        if (isStreamingRef.current || pendingSpeechSegmentsRef.current.size > 0) {
            setStatus("thinking");
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

    const submitMessage = useCallback(
        async (messageOverride?: string) => {
            const trimmed = (messageOverride ?? inputValue).trim();
            if (!trimmed) return;

            stopAllAudio();
            pendingSpeechSegmentsRef.current.clear();
            isStreamingRef.current = true;
            isAudioSpeakingRef.current = false;

            const userMessage: ChatMessage = {
                id: crypto.randomUUID(),
                role: "user",
                content: trimmed,
            };
            const assistantMessageId = crypto.randomUUID();
            const assistantPlaceholder: ChatMessage = {
                id: assistantMessageId,
                role: "assistant",
                content: "",
            };

            activeAssistantMessageId.current = assistantMessageId;
            pendingAssistantMessageIdRef.current = assistantMessageId;

            setMessages((prev) => [...prev, userMessage, assistantPlaceholder]);
            setInputValue("");
            setIsLoading(true);
            setStatus("thinking");

            try {
                const started = await invoke<StartChatResponse>("start_chat_message_stream", {
                    payload: { message: trimmed },
                });

                const alreadyActive = activeRequestIdRef.current === started.request_id;
                activeRequestIdRef.current = started.request_id;
                activeAssistantMessageId.current = assistantMessageId;
                completedAudioSessionsRef.current.delete(started.request_id);
                setActiveModel(started.model);

                if (!alreadyActive) {
                    startNewRequestAudioSession(started.request_id);
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
        [inputValue, settleVisualStatus, startNewRequestAudioSession, stopAllAudio]
    );

    const handleTranscript = useCallback(
        (text: string, shouldAutoSubmit: boolean) => {
            setInputValue(text);
            if (shouldAutoSubmit) {
                void submitMessage(text);
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
        ({ request_id, model, user_message }: AssistantRequestStartedEvent) => {
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
            activeAssistantMessageId.current = assistantMessageId;
            completedAudioSessionsRef.current.delete(request_id);
            isStreamingRef.current = true;
            isAudioSpeakingRef.current = false;
            pendingSpeechSegmentsRef.current.clear();

            setActiveModel(model);
            setIsLoading(true);
            setStatus("thinking");

            if (!alreadyActive) {
                startNewRequestAudioSession(request_id);
            }
        },
        [startNewRequestAudioSession]
    );

    const handleStreamChunk = useCallback(({ request_id, chunk }: StreamChunkEvent) => {
        if (activeRequestIdRef.current !== request_id) return;

        setMessages((prev) => {
            const assistantId = activeAssistantMessageId.current;
            if (!assistantId) return prev;

            return prev.map((msg) =>
                msg.id === assistantId ? { ...msg, content: msg.content + chunk } : msg
            );
        });
    }, []);

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
            markAudioSegmentFailed(event);
            settleVisualStatus();
        },
        [markAudioSegmentFailed, settleVisualStatus]
    );

    const handleSpeechQueued = useCallback((event: SpeechSegmentQueuedEvent) => {
        if (activeRequestIdRef.current !== event.request_id) return;

        pendingSpeechSegmentsRef.current.add(`${event.request_id}:${event.sequence}`);
        if (!isAudioSpeakingRef.current) {
            setStatus("thinking");
        }
    }, []);

    const handleRequestFinished = useCallback(
        ({ request_id }: AssistantRequestFinishedEvent) => {
            if (activeRequestIdRef.current !== request_id) return;

            isStreamingRef.current = false;
            pendingAssistantMessageIdRef.current = null;
            activeAssistantMessageId.current = null;
            setIsLoading(false);

            if (
                !isAudioSpeakingRef.current &&
                !hasPendingWork() &&
                pendingSpeechSegmentsRef.current.size === 0
            ) {
                settleVisualStatus();
                void completeAudioSession(request_id);
            }
        },
        [completeAudioSession, hasPendingWork, settleVisualStatus]
    );

    const handleAssistantError = useCallback(
        ({ request_id, stage, message }: AssistantErrorEvent) => {
            if (activeRequestIdRef.current !== request_id && stage !== "stt") return;

            console.error(`Assistant ${stage} error:`, message);

            if (stage === "tts") {
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
        (_event: AssistantInterruptedEvent) => {
            stopAllAudio();
            pendingSpeechSegmentsRef.current.clear();
            isStreamingRef.current = false;
            isAudioSpeakingRef.current = false;
            setIsLoading(false);
            setStatus("listening");
        },
        [stopAllAudio]
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
        onAssistantError: handleAssistantError,
        onStatus: handleStatus,
        onModel: setActiveModel,
        onMetrics: setLastMetrics,
        onAssistantInterrupted: handleAssistantInterrupted,
        onVoiceSessionState: handleVoiceSessionState,
        onVoiceSessionTranscript: handleVoiceSessionTranscript,
        onVoiceTurnMetrics: setLastVoiceMetrics,
    });

    return {
        activeModel,
        autoSubmitVoice,
        inputValue,
        isLoading,
        lastMetrics,
        lastVoiceMetrics,
        lastVoiceTranscript,
        messages,
        setAutoSubmitVoice,
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

function getVoiceRestStatus(event: VoiceSessionStateEvent): AssistantStatus | null {
    if (event.state === "disabled") return null;
    if (event.state === "passive") return "passive";
    if (event.mode === "conversation") return "armed";
    return "passive";
}
