import "./App.css";
import { AssistantChat } from "./components/AssistantChat";
import { AssistantHeader } from "./components/AssistantHeader";
import { AssistantInputBar } from "./components/AssistantInputBar";
import AstraOrb from "./components/AstraOrb";
import { useAssistantSession } from "./hooks/useAssistantSession";
import { useAssistantVisualState } from "./hooks/useAssistantVisualState";
import { useWindowControls } from "./hooks/useWindowControls";

function App() {
    const session = useAssistantSession();
    const statusLabel = useAssistantVisualState(session.status);
    const windowControls = useWindowControls({
        onBeforeClose: session.stopAllAudio,
    });

    return (
        <main className="overlay-shell">
            <section className="assistant-overlay">
                <AssistantHeader
                    activeModel={session.activeModel}
                    isPinned={windowControls.isPinned}
                    onClose={windowControls.close}
                    onMinimize={windowControls.minimize}
                    onTogglePin={windowControls.togglePin}
                    startDrag={windowControls.startDrag}
                    status={session.status}
                    statusLabel={statusLabel}
                />

                <section className="orb-stage">
                    <AstraOrb status={session.status} />
                </section>

                <AssistantChat messages={session.messages} />

                <AssistantInputBar
                    autoSubmitVoice={session.autoSubmitVoice}
                    inputValue={session.inputValue}
                    isLoading={session.isLoading}
                    lastVoiceTranscript={session.lastVoiceTranscript}
                    onSubmit={() => session.submitMessage()}
                    setAutoSubmitVoice={session.setAutoSubmitVoice}
                    setInputValue={session.setInputValue}
                    voiceInput={session.voiceInput}
                    voiceSession={session.voiceSession}
                />
            </section>
        </main>
    );
}

export default App;
