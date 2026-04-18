import type { KeyboardEventHandler } from "react";
import type { VoiceSessionTranscriptEvent } from "../types/assistant";
import { Button } from "../ui/buttons/Button";

import { LuSend } from "react-icons/lu";
import { CiMicrophoneOn } from "react-icons/ci";


type VoiceInputControls = {
    isRecording: boolean;
    isTranscribing: boolean;
    toggleRecording: () => Promise<void>;
};

type VoiceSessionControls = {
    inputLevel: number;
    isEnabled: boolean;
    isTransitioning: boolean;
    toggle: () => Promise<void>;
    voiceState: string;
};

type AssistantInputBarProps = {
    autoSubmitVoice: boolean;
    inputValue: string;
    isLoading: boolean;
    onSubmit: () => Promise<void>;
    setAutoSubmitVoice: (value: boolean) => void;
    setInputValue: (value: string) => void;
    lastVoiceTranscript: VoiceSessionTranscriptEvent | null;
    voiceInput: VoiceInputControls;
    voiceSession: VoiceSessionControls;
};

export function AssistantInputBar({
    //autoSubmitVoice,
    inputValue,
    isLoading,
    onSubmit,
    //setAutoSubmitVoice,
    setInputValue,
    lastVoiceTranscript,
    //voiceInput,
    voiceSession,
}: AssistantInputBarProps) {
    const handleKeyDown: KeyboardEventHandler<HTMLInputElement> = async (event) => {
        if (event.key === "Enter") {
            await onSubmit();
        }
    };

    /*const voiceLabel = voiceInput.isRecording
        ? "Stop"
        : voiceInput.isTranscribing
          ? "..."
          : "Mic";*/
    const liveLevel = Math.round(voiceSession.inputLevel * 100);
    const voiceFeedback = getVoiceFeedback(lastVoiceTranscript);

    return (
        <section className="input-panel">
            {voiceFeedback ? (
                <p className={`voice-session-feedback ${lastVoiceTranscript?.action}`}>
                    {voiceFeedback}
                </p>
            ) : null}

            <div className="flex gap-2 rounded-lg p-2 bg-gray-200 text:-gray-500 items-center">
                <input
                    type="text"
                    placeholder="Scrivi un messaggio..."
                    className="chat-input"
                    value={inputValue}
                    onChange={(event) => setInputValue(event.target.value)}
                    onKeyDown={handleKeyDown}
                />
                <Button variant="dark" radius="full" size="sm" onClick={onSubmit} disabled={!inputValue.trim()} loading={isLoading}>
                    {isLoading ? "Invia nuova" : <LuSend />}
                </Button>                
                <Button
                    variant="dark" radius="full" size="sm"
                    className={`${voiceSession.isEnabled ? "active" : ""} flex items-center gap-1`}
                    onClick={voiceSession.toggle}
                    disabled={voiceSession.isTransitioning}
                    title={`Sessione vocale: ${voiceSession.voiceState} - mic ${liveLevel}%`}
                >
                    {voiceSession.isTransitioning
                        ? "..."
                        : voiceSession.isEnabled
                          ? <><CiMicrophoneOn /> {liveLevel}</>
                          : <CiMicrophoneOn />}
                </Button>
                {/*<button
                    className={`voice-button ${voiceInput.isRecording ? "recording" : ""}`}
                    onClick={voiceInput.toggleRecording}
                    disabled={voiceInput.isTranscribing}
                >
                    {voiceLabel}
                </button>
                <button
                    className={`voice-auto-button ${autoSubmitVoice ? "active" : ""}`}
                    onClick={() => setAutoSubmitVoice(!autoSubmitVoice)}
                    title="Invia automaticamente dopo la trascrizione"
                >
                    Auto
                </button>*/}

            </div>
        </section>
    );
}

function getVoiceFeedback(event: VoiceSessionTranscriptEvent | null) {
    if (!event) return null;

    if (event.action === "ignored") {
        const heard = event.text.trim() || "...";
        return `Sentito: "${heard}" - manca la wake word Astra.`;
    }

    if (event.action === "armed") {
        return "Astra attiva. Puoi parlare senza ripetere la wake word.";
    }

    if (event.action === "responding") {
        const heard = event.text.trim() || event.response_text?.trim() || "...";
        return `Sentito: "${heard}"`;
    }

    return null;
}
