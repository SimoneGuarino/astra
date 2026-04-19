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
                    {message.content ? <MessageContent content={message.content} /> : "..."}
                </div>
            ))}
        </section>
    );
}

type MessageBlock =
    | { type: "text"; text: string }
    | { type: "table"; headers: string[]; rows: string[][] };

function MessageContent({ content }: { content: string }) {
    const blocks = parseMessageBlocks(content);

    return (
        <>
            {blocks.map((block, index) => {
                if (block.type === "table") {
                    return (
                        <div className="message-table-wrap" key={`table-${index}`}>
                            <table className="message-table">
                                <thead>
                                    <tr>
                                        {block.headers.map((header, headerIndex) => (
                                            <th key={headerIndex}>{header}</th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody>
                                    {block.rows.map((row, rowIndex) => (
                                        <tr key={rowIndex}>
                                            {block.headers.map((_, cellIndex) => (
                                                <td key={cellIndex}>{row[cellIndex] ?? ""}</td>
                                            ))}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    );
                }

                return (
                    <div className="message-text-block" key={`text-${index}`}>
                        {block.text}
                    </div>
                );
            })}
        </>
    );
}

function parseMessageBlocks(content: string): MessageBlock[] {
    const lines = content.split(/\r?\n/);
    const blocks: MessageBlock[] = [];
    let textBuffer: string[] = [];
    let index = 0;

    const flushText = () => {
        const text = textBuffer.join("\n").trim();
        if (text) {
            blocks.push({ type: "text", text });
        }
        textBuffer = [];
    };

    while (index < lines.length) {
        if (isMarkdownTableStart(lines, index)) {
            flushText();
            const tableLines: string[] = [];
            while (index < lines.length && isPipeTableLine(lines[index])) {
                tableLines.push(lines[index]);
                index += 1;
            }
            const table = parseMarkdownTable(tableLines);
            if (table) {
                blocks.push(table);
            } else {
                textBuffer.push(...tableLines);
            }
            continue;
        }

        textBuffer.push(lines[index]);
        index += 1;
    }

    flushText();
    return blocks.length ? blocks : [{ type: "text", text: content }];
}

function isMarkdownTableStart(lines: string[], index: number) {
    return (
        isPipeTableLine(lines[index]) &&
        index + 1 < lines.length &&
        isMarkdownSeparatorLine(lines[index + 1])
    );
}

function isPipeTableLine(line: string) {
    return line.trim().startsWith("|") && line.trim().endsWith("|") && line.includes("|");
}

function isMarkdownSeparatorLine(line: string) {
    const cells = splitTableLine(line);
    return cells.length > 1 && cells.every((cell) => /^:?-{3,}:?$/.test(cell.trim()));
}

function parseMarkdownTable(lines: string[]): MessageBlock | null {
    if (lines.length < 3 || !isMarkdownSeparatorLine(lines[1])) return null;

    const headers = splitTableLine(lines[0]);
    const rows = lines
        .slice(2)
        .map(splitTableLine)
        .filter((row) => row.length > 0);

    if (headers.length < 2 || rows.length === 0) return null;
    return { type: "table", headers, rows };
}

function splitTableLine(line: string) {
    return line
        .trim()
        .replace(/^\|/, "")
        .replace(/\|$/, "")
        .split("|")
        .map((cell) => cell.trim());
}
