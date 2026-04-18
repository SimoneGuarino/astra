import { useCallback, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { VoiceTranscriptionResponse } from "../types/assistant";

type UseVoiceInputParams = {
    autoSubmit: boolean;
    onListeningStart: () => void;
    onListeningEnd: () => void;
    onTranscript: (text: string, autoSubmit: boolean) => void;
    onError: (message: string) => void;
};

type VoiceRecordingState = "idle" | "recording" | "transcribing";

export function useVoiceInput({
    autoSubmit,
    onListeningStart,
    onListeningEnd,
    onTranscript,
    onError,
}: UseVoiceInputParams) {
    const [recordingState, setRecordingState] = useState<VoiceRecordingState>("idle");
    const recorderRef = useRef<MediaRecorder | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const chunksRef = useRef<BlobPart[]>([]);
    const discardCurrentRecordingRef = useRef(false);

    const stopTracks = useCallback(() => {
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
    }, []);

    const transcribeBlob = useCallback(
        async (blob: Blob) => {
            setRecordingState("transcribing");
            onListeningStart();

            try {
                const audioBytes = Array.from(new Uint8Array(await blob.arrayBuffer()));
                const response = await invoke<VoiceTranscriptionResponse>("transcribe_voice_input", {
                    payload: {
                        audio_bytes: audioBytes,
                        mime_type: blob.type || "audio/webm",
                        auto_submit: autoSubmit,
                    },
                });

                if (response.text.trim()) {
                    onTranscript(response.text.trim(), response.auto_submit);
                }
            } catch (error) {
                const message = error instanceof Error ? error.message : String(error);
                onError(message);
            } finally {
                setRecordingState("idle");
                onListeningEnd();
            }
        },
        [autoSubmit, onError, onListeningEnd, onListeningStart, onTranscript]
    );

    const stopRecording = useCallback(() => {
        const recorder = recorderRef.current;
        if (!recorder || recorder.state === "inactive") return;
        recorder.stop();
    }, []);

    const cancelRecording = useCallback(async () => {
        discardCurrentRecordingRef.current = true;
        stopRecording();
        stopTracks();
        setRecordingState("idle");
        onListeningEnd();

        try {
            await invoke("cancel_voice_input");
        } catch (error) {
            console.error("cancel_voice_input error:", error);
        }
    }, [onListeningEnd, stopRecording, stopTracks]);

    const startRecording = useCallback(async () => {
        if (recordingState !== "idle") return;

        if (!navigator.mediaDevices?.getUserMedia) {
            onError("Microfono non disponibile in questo ambiente.");
            return;
        }

        discardCurrentRecordingRef.current = false;
        chunksRef.current = [];
        setRecordingState("recording");
        onListeningStart();

        try {
            await invoke("cancel_active_response");
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mimeType = selectSupportedMimeType();
            const recorder = mimeType
                ? new MediaRecorder(stream, { mimeType })
                : new MediaRecorder(stream);

            streamRef.current = stream;
            recorderRef.current = recorder;

            recorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    chunksRef.current.push(event.data);
                }
            };

            recorder.onerror = (event) => {
                onError(`Errore registrazione microfono: ${event.error.message}`);
            };

            recorder.onstop = () => {
                const chunks = chunksRef.current;
                const shouldDiscard = discardCurrentRecordingRef.current;
                const type = recorder.mimeType || mimeType || "audio/webm";

                recorderRef.current = null;
                chunksRef.current = [];
                stopTracks();

                if (shouldDiscard || chunks.length === 0) {
                    setRecordingState("idle");
                    onListeningEnd();
                    return;
                }

                void transcribeBlob(new Blob(chunks, { type }));
            };

            recorder.start();
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            recorderRef.current = null;
            stopTracks();
            setRecordingState("idle");
            onListeningEnd();
            onError(`Avvio microfono fallito: ${message}`);
        }
    }, [
        onError,
        onListeningEnd,
        onListeningStart,
        recordingState,
        stopTracks,
        transcribeBlob,
    ]);

    const toggleRecording = useCallback(async () => {
        if (recordingState === "recording") {
            stopRecording();
            return;
        }

        if (recordingState === "idle") {
            await startRecording();
        }
    }, [recordingState, startRecording, stopRecording]);

    return {
        cancelRecording,
        isRecording: recordingState === "recording",
        isTranscribing: recordingState === "transcribing",
        recordingState,
        startRecording,
        stopRecording,
        toggleRecording,
    };
}

function selectSupportedMimeType() {
    const candidates = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4",
        "audio/ogg;codecs=opus",
    ];

    return candidates.find((candidate) => MediaRecorder.isTypeSupported(candidate)) ?? "";
}
