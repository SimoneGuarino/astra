from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

MemoryNodeKind = Literal[
    "conversation_turn",
    "work_session",
    "transcript_segment",
    "summary",
    "concept",
    "entity",
    "task",
    "tool_use",
    "error",
    "fix",
    "research_topic",
    "research_finding",
    "source_document",
    "code_pattern",
    "user_preference",
    "workflow",
    "claim",
    "decision",
    "procedure",
    "unknown",
]


@dataclass(frozen=True)
class EmbeddingWorkerRequest:
    text: str
    model: str | None = None


@dataclass(frozen=True)
class EmbeddingWorkerResponse:
    vector: list[float]
    model: str
    dimensions: int


@dataclass(frozen=True)
class MemoryExtractionCandidate:
    kind: MemoryNodeKind
    title: str
    summary: str
    confidence: float = 0.7
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
