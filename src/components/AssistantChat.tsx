import { useEffect, useRef } from "react";
import type { ChatMessage } from "../types/assistant";

type AssistantChatProps = {
    messages: ChatMessage[];
};

export function AssistantChat({ messages }: AssistantChatProps) {
    const chatAreaRef = useRef<HTMLElement | null>(null);

    useEffect(() => {
        if (!chatAreaRef.current) return;
        chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
    }, [messages]);

    return (
        <section className="chat-area" ref={chatAreaRef}>
            {messages.map((message) => (
                <div key={message.id} className={`message ${message.role}`}>
                    {message.content || "..."}
                </div>
            ))}
        </section>
    );
}
