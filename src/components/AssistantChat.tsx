import { Fragment, useEffect, useState, type ReactNode } from "react";
import type { AssistantActivityState, ChatMessage } from "../types/assistant";

type AssistantChatProps = {
    messages: ChatMessage[];
    chatRef: React.RefObject<HTMLDivElement | null>;
};

export function AssistantChat({ messages, chatRef }: AssistantChatProps) {
    const [expandedActivityMessageId, setExpandedActivityMessageId] = useState<string | null>(null);

    useEffect(() => {
        if (!chatRef.current) return;
        chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }, [messages]);

    return (
        <>
            {messages.map((message) => (
                <div key={message.id} className={`message ${message.role}`}>
                    {message.content ? (
                        <MessageContent content={message.content} />
                    ) : message.role === "assistant" ? (
                        <AssistantActivityPreview
                            activity={message.activity}
                            isExpanded={expandedActivityMessageId === message.id}
                            onToggleExpanded={() =>
                                setExpandedActivityMessageId((current) =>
                                    current === message.id ? null : message.id
                                )
                            }
                        />
                    ) : (
                        ""
                    )}
                </div>
            ))}
        </>
    );
}

function AssistantActivityPreview({
    activity,
    isExpanded,
    onToggleExpanded,
}: {
    activity?: AssistantActivityState | null;
    isExpanded: boolean;
    onToggleExpanded: () => void;
}) {
    const current = activity?.current;
    const steps = activity?.steps ?? [];

    return (
        <div className="assistant-thinking-card">
            <div className="assistant-thinking-pulse" aria-hidden="true" />
            <div className="assistant-thinking-main">
                <div className="assistant-thinking-eyebrow">Astra sta lavorando</div>
                <div className="assistant-thinking-title">
                    {current?.title ?? "Preparazione risposta"}
                </div>
                <div className="assistant-thinking-detail">
                    {current?.detail ?? "Sto preparando contesto, memoria e risposta governata."}
                </div>
                {steps.length > 0 ? (
                    <button type="button" className="assistant-thinking-expand" onClick={onToggleExpanded}>
                        {isExpanded ? "Nascondi dettagli" : `Espandi passaggi (${steps.length})`}
                    </button>
                ) : null}
                {isExpanded ? (
                    <div className="assistant-thinking-details-panel">
                        {steps.map((step, index) => (
                            <div className="assistant-thinking-step" key={step.id}>
                                <span className="assistant-thinking-step-index">{index + 1}</span>
                                <div>
                                    <div className="assistant-thinking-step-title">{step.title}</div>
                                    {step.detail ? (
                                        <div className="assistant-thinking-step-detail">{step.detail}</div>
                                    ) : null}
                                </div>
                            </div>
                        ))}
                    </div>
                ) : null}
            </div>
        </div>
    );
}

type MessageBlock =
    | { type: "paragraph"; text: string }
    | { type: "heading"; level: 1 | 2 | 3; text: string }
    | { type: "list"; ordered: boolean; items: string[] }
    | { type: "code"; language: string | null; code: string }
    | { type: "blockquote"; text: string }
    | { type: "hr" }
    | { type: "source"; label: string }
    | { type: "note"; severity: "info" | "warning"; text: string }
    | { type: "evidence"; label: string }
    | { type: "table"; headers: string[]; rows: string[][] };

function MessageContent({ content }: { content: string }) {
    const blocks = parseMessageBlocks(content);

    return (
        <div className="message-markdown">
            {blocks.map((block, index) => {
                if (block.type === "heading") {
                    const children = renderInlineMarkdown(block.text);
                    if (block.level === 1) {
                        return <h2 className="message-heading message-heading-1" key={`heading-${index}`}>{children}</h2>;
                    }
                    if (block.level === 2) {
                        return <h3 className="message-heading message-heading-2" key={`heading-${index}`}>{children}</h3>;
                    }
                    return <h4 className="message-heading message-heading-3" key={`heading-${index}`}>{children}</h4>;
                }

                if (block.type === "paragraph") {
                    return (
                        <p className="message-paragraph" key={`paragraph-${index}`}>
                            {renderInlineMarkdown(block.text)}
                        </p>
                    );
                }

                if (block.type === "list") {
                    const ListTag = block.ordered ? "ol" : "ul";
                    return (
                        <ListTag className="message-list" key={`list-${index}`}>
                            {block.items.map((item, itemIndex) => (
                                <li key={itemIndex}>{renderInlineMarkdown(item)}</li>
                            ))}
                        </ListTag>
                    );
                }

                if (block.type === "code") {
                    return (
                        <pre className="message-code-block" key={`code-${index}`}>
                            <code data-language={block.language ?? undefined}>{block.code}</code>
                        </pre>
                    );
                }

                if (block.type === "blockquote") {
                    return (
                        <blockquote className="message-blockquote" key={`quote-${index}`}>
                            {renderInlineMarkdown(block.text)}
                        </blockquote>
                    );
                }

                if (block.type === "hr") {
                    return <hr className="message-hr" key={`hr-${index}`} />;
                }

                if (block.type === "source") {
                    return (
                        <div className="message-grounded-meta" key={`source-${index}`}>
                            <div className="message-source-card">
                                <span className="message-grounded-meta-label">Fonte</span>
                                <span>{renderInlineMarkdown(block.label)}</span>
                            </div>
                        </div>
                    );
                }

                if (block.type === "note") {
                    return (
                        <div
                            className={`message-note-callout${block.severity === "warning" ? " warning" : ""}`}
                            key={`note-${index}`}
                        >
                            <span className="message-grounded-meta-label">Nota</span>
                            <span>{renderInlineMarkdown(block.text)}</span>
                        </div>
                    );
                }

                if (block.type === "evidence") {
                    return (
                        <div className="message-grounded-meta" key={`evidence-${index}`}>
                            <div className="message-evidence-summary">
                                <span className="message-grounded-meta-label">Evidenze</span>
                                <span>{renderInlineMarkdown(block.label)}</span>
                            </div>
                        </div>
                    );
                }

                if (block.type === "table") {
                    return (
                        <div className="message-table-wrap" key={`table-${index}`}>
                            <table className="message-table">
                                <thead>
                                    <tr>
                                        {block.headers.map((header, headerIndex) => (
                                            <th key={headerIndex}>{renderInlineMarkdown(normalizeSafeHtmlLikeFragments(header))}</th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody>
                                    {block.rows.map((row, rowIndex) => (
                                        <tr key={rowIndex}>
                                            {block.headers.map((_, cellIndex) => (
                                                <td key={cellIndex}>
                                                    <TableCellContent content={row[cellIndex] ?? ""} />
                                                </td>
                                            ))}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    );
                }

                return null;
            })}
        </div>
    );
}

function parseMessageBlocks(content: string): MessageBlock[] {
    const lines = content.split(/\r?\n/);
    const blocks: MessageBlock[] = [];
    let paragraphBuffer: string[] = [];
    let index = 0;

    const flushParagraph = () => {
        const text = paragraphBuffer.join("\n").trim();
        if (text) {
            blocks.push({ type: "paragraph", text });
        }
        paragraphBuffer = [];
    };

    while (index < lines.length) {
        const rawLine = lines[index];
        const line = normalizeSafeHtmlLikeFragments(rawLine);
        const trimmed = line.trim();

        if (!trimmed) {
            flushParagraph();
            index += 1;
            continue;
        }

        const fence = trimmed.match(/^```([A-Za-z0-9_-]+)?\s*$/);
        if (fence) {
            flushParagraph();
            const language = fence[1] ?? null;
            const codeLines: string[] = [];
            index += 1;
            while (index < lines.length && !lines[index].trim().startsWith("```")) {
                codeLines.push(lines[index]);
                index += 1;
            }
            if (index < lines.length && lines[index].trim().startsWith("```")) {
                index += 1;
            }
            blocks.push({ type: "code", language, code: codeLines.join("\n") });
            continue;
        }

        if (isMarkdownTableStart(lines, index)) {
            flushParagraph();
            const tableLines: string[] = [];
            while (index < lines.length && isPipeTableLine(lines[index])) {
                tableLines.push(lines[index]);
                index += 1;
            }
            const table = parseMarkdownTable(tableLines);
            if (table) {
                blocks.push(table);
            } else {
                paragraphBuffer.push(...tableLines);
            }
            continue;
        }

        const groundedMeta = parseGroundedMetaLine(trimmed);
        if (groundedMeta === "skip") {
            flushParagraph();
            index += 1;
            continue;
        }
        if (groundedMeta) {
            flushParagraph();
            blocks.push(groundedMeta);
            index += 1;
            continue;
        }

        const heading = trimmed.match(/^(#{1,3})\s+(.+)$/);
        if (heading) {
            flushParagraph();
            blocks.push({
                type: "heading",
                level: heading[1].length as 1 | 2 | 3,
                text: heading[2].trim(),
            });
            index += 1;
            continue;
        }

        if (/^(?:-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
            flushParagraph();
            blocks.push({ type: "hr" });
            index += 1;
            continue;
        }

        if (trimmed.startsWith(">")) {
            flushParagraph();
            const quoteLines: string[] = [];
            while (index < lines.length && lines[index].trim().startsWith(">")) {
                quoteLines.push(lines[index].trim().replace(/^>\s?/, ""));
                index += 1;
            }
            blocks.push({ type: "blockquote", text: quoteLines.join("\n").trim() });
            continue;
        }

        const listItem = parseListItem(line);
        if (listItem) {
            flushParagraph();
            const ordered = listItem.ordered;
            const items = [listItem.text];
            index += 1;
            while (index < lines.length) {
                const nextItem = parseListItem(lines[index]);
                if (!nextItem || nextItem.ordered !== ordered) break;
                items.push(nextItem.text);
                index += 1;
            }
            blocks.push({ type: "list", ordered, items });
            continue;
        }

        paragraphBuffer.push(line);
        index += 1;
    }

    flushParagraph();
    return blocks.length ? blocks : [{ type: "paragraph", text: content }];
}

function TableCellContent({ content }: { content: string }) {
    const normalized = normalizeSafeHtmlLikeFragments(content);
    const blocks = parseMessageBlocks(normalized);
    return (
        <div className="message-table-cell-content">
            {blocks.map((block, index) => {
                if (block.type === "list") {
                    const ListTag = block.ordered ? "ol" : "ul";
                    return (
                        <ListTag className="message-list" key={`cell-list-${index}`}>
                            {block.items.map((item, itemIndex) => (
                                <li key={itemIndex}>{renderInlineMarkdown(item)}</li>
                            ))}
                        </ListTag>
                    );
                }
                if (block.type === "code") {
                    return (
                        <pre className="message-code-block" key={`cell-code-${index}`}>
                            <code data-language={block.language ?? undefined}>{block.code}</code>
                        </pre>
                    );
                }
                if (block.type === "blockquote") {
                    return (
                        <blockquote className="message-blockquote" key={`cell-quote-${index}`}>
                            {renderInlineMarkdown(block.text)}
                        </blockquote>
                    );
                }
                if (block.type === "hr") {
                    return <hr className="message-hr" key={`cell-hr-${index}`} />;
                }
                if (block.type === "source" || block.type === "note" || block.type === "evidence") {
                    return <Fragment key={`cell-meta-${index}`}>{renderInlineMarkdown(metaBlockText(block))}</Fragment>;
                }
                if (block.type === "heading" || block.type === "paragraph") {
                    return <Fragment key={`cell-text-${index}`}>{renderInlineMarkdown(block.text)}</Fragment>;
                }
                return <Fragment key={`cell-table-${index}`}>{renderInlineMarkdown(content)}</Fragment>;
            })}
        </div>
    );
}

function metaBlockText(block: Extract<MessageBlock, { type: "source" | "note" | "evidence" }>) {
    if (block.type === "source") return block.label;
    if (block.type === "note") return block.text;
    return block.label;
}

function parseGroundedMetaLine(line: string): MessageBlock | "skip" | null {
    const source = line.match(/^Fonte:\s*(.+)$/i);
    if (source) return { type: "source", label: source[1].trim() };

    const note = line.match(/^Nota:\s*(.+)$/i);
    if (note) {
        const text = note[1].trim();
        if (isInternalDiagnosticNote(text)) return "skip";
        return {
            type: "note",
            severity: noteLooksWarning(text) ? "warning" : "info",
            text,
        };
    }

    const evidence = line.match(/^Evidenze usate:\s*(.+)$/i);
    if (evidence) return { type: "evidence", label: sanitizeEvidenceLabel(evidence[1].trim()) };

    return null;
}

function noteLooksWarning(text: string) {
    return /(?:incomplete|timeout|warning|parzial|incomplet)/i.test(text);
}

function isInternalDiagnosticNote(text: string) {
    return /(?:context_answer_synthesizer_fallback|ToolResultFrame|ContextEvidenceSupport|metadata_only|raw_model_output)/i.test(
        text,
    );
}

function sanitizeEvidenceLabel(text: string) {
    return text.replace(/\bsegment:[0-9a-f]{8}-[0-9a-f-]{27,}\b/gi, "segmento transcript");
}

function normalizeSafeHtmlLikeFragments(text: string) {
    let output = text.replace(/<br\s*\/?>/gi, "\n");
    output = normalizeHtmlList(output, "ul", false);
    output = normalizeHtmlList(output, "ol", true);
    output = output.replace(/<li>([\s\S]*?)<\/li>/gi, (_, item: string) => `- ${item.trim()}`);
    output = output.replace(/<strong>([\s\S]*?)<\/strong>/gi, "**$1**");
    output = output.replace(/<b>([\s\S]*?)<\/b>/gi, "**$1**");
    output = output.replace(/<em>([\s\S]*?)<\/em>/gi, "*$1*");
    output = output.replace(/<i>([\s\S]*?)<\/i>/gi, "*$1*");
    output = output.replace(/<code>([\s\S]*?)<\/code>/gi, "`$1`");
    return output;
}

function normalizeHtmlList(text: string, tagName: "ul" | "ol", ordered: boolean) {
    const listPattern = new RegExp(`<${tagName}>([\\s\\S]*?)<\\/${tagName}>`, "gi");
    return text.replace(listPattern, (match, inner: string) => {
        const items = Array.from(inner.matchAll(/<li>([\s\S]*?)<\/li>/gi)).map((itemMatch) =>
            itemMatch[1].trim()
        );
        if (!items.length) return match;
        return items
            .map((item, index) => (ordered ? `${index + 1}. ${item}` : `- ${item}`))
            .join("\n");
    });
}

function parseListItem(line: string): { ordered: boolean; text: string } | null {
    const ordered = line.match(/^\s{0,3}\d+[.)]\s+(.+)$/);
    if (ordered) return { ordered: true, text: ordered[1].trim() };

    const unordered = line.match(/^\s{0,3}[-*+]\s+(.+)$/);
    if (unordered) return { ordered: false, text: unordered[1].trim() };

    return null;
}

function renderInlineMarkdown(text: string): ReactNode[] {
    const nodes: ReactNode[] = [];
    let cursor = 0;
    let key = 0;

    const pushText = (value: string) => {
        if (value) nodes.push(value);
    };

    while (cursor < text.length) {
        const next = findNextInlineToken(text, cursor);
        if (!next) {
            pushText(text.slice(cursor));
            break;
        }

        pushText(text.slice(cursor, next.index));

        if (next.token === "`") {
            const end = text.indexOf("`", next.index + 1);
            if (end === -1) {
                pushText(text.slice(next.index));
                break;
            }
            nodes.push(
                <code className="message-inline-code" key={`code-${key++}`}>
                    {text.slice(next.index + 1, end)}
                </code>
            );
            cursor = end + 1;
            continue;
        }

        if (next.token === "**") {
            const end = text.indexOf("**", next.index + 2);
            if (end === -1) {
                pushText(text.slice(next.index));
                break;
            }
            nodes.push(
                <strong key={`strong-${key++}`}>
                    {renderInlineMarkdown(text.slice(next.index + 2, end))}
                </strong>
            );
            cursor = end + 2;
            continue;
        }

        const end = text.indexOf("*", next.index + 1);
        if (end === -1) {
            pushText(text.slice(next.index));
            break;
        }
        nodes.push(
            <em key={`em-${key++}`}>
                {renderInlineMarkdown(text.slice(next.index + 1, end))}
            </em>
        );
        cursor = end + 1;
    }

    return nodes.map((node, index) =>
        typeof node === "string" ? <Fragment key={`text-${index}`}>{node}</Fragment> : node
    );
}

function findNextInlineToken(text: string, start: number): { token: "`" | "**" | "*"; index: number } | null {
    const candidates: Array<{ token: "`" | "**" | "*"; index: number }> = [];
    const code = text.indexOf("`", start);
    if (code !== -1) candidates.push({ token: "`", index: code });
    const bold = text.indexOf("**", start);
    if (bold !== -1) candidates.push({ token: "**", index: bold });
    const italic = findItalicToken(text, start);
    if (italic !== -1) candidates.push({ token: "*", index: italic });

    return candidates.sort((left, right) => left.index - right.index || right.token.length - left.token.length)[0] ?? null;
}

function findItalicToken(text: string, start: number) {
    let index = text.indexOf("*", start);
    while (index !== -1) {
        if (text[index + 1] !== "*" && text[index - 1] !== "*" && isUsefulItalicBoundary(text, index)) {
            return index;
        }
        index = text.indexOf("*", index + 1);
    }
    return -1;
}

function isUsefulItalicBoundary(text: string, index: number) {
    const next = text[index + 1];
    const previous = text[index - 1];
    return Boolean(next && !/\s/.test(next) && previous !== "\\");
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
