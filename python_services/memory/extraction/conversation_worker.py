"""AstraOS conversation-memory extraction helper.

This worker is intentionally schema-oriented and side-effect free. Rust remains
source of truth for memory persistence, governance, validation, retention, and
activation. Python can be used by future pipelines to normalize model output or
pre-process conversation text before Rust consolidation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ConversationImportantPoint:
    title: str
    summary: str
    kind: Optional[str] = None
    confidence: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationEntity:
    name: str
    entity_type: Optional[str] = None
    summary: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationPreference:
    preference: str
    rationale: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationProcedure:
    title: str
    steps: List[str] = field(default_factory=list)
    rationale: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationDecision:
    title: str
    summary: str
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationMemoryPayload:
    user_message: str
    assistant_answer: str
    request_id: Optional[str] = None
    source: Optional[str] = None
    topic: Optional[str] = None
    summary: Optional[str] = None
    importance: Optional[float] = None
    confidence: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    important_points: List[ConversationImportantPoint] = field(default_factory=list)
    entities: List[ConversationEntity] = field(default_factory=list)
    preferences: List[ConversationPreference] = field(default_factory=list)
    procedures: List[ConversationProcedure] = field(default_factory=list)
    decisions: List[ConversationDecision] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_rust_payload(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_conversation_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an LLM-produced conversation-memory object.

    This does not persist memory. It only shapes data for the Rust command
    `consolidate_conversation_memory_bundle`.
    """

    def text(value: Any, max_chars: int) -> str:
        return str(value or "").strip()[:max_chars]

    normalized = {
        "request_id": payload.get("request_id"),
        "source": payload.get("source"),
        "user_message": text(payload.get("user_message"), 8000),
        "assistant_answer": text(payload.get("assistant_answer"), 12000),
        "topic": text(payload.get("topic"), 180) or None,
        "summary": text(payload.get("summary"), 3000) or None,
        "importance": payload.get("importance"),
        "confidence": payload.get("confidence"),
        "tags": [text(tag, 64).lower().replace(" ", "_") for tag in payload.get("tags", [])[:32] if text(tag, 64)],
        "important_points": payload.get("important_points", [])[:18],
        "entities": payload.get("entities", [])[:24],
        "preferences": payload.get("preferences", [])[:12],
        "procedures": payload.get("procedures", [])[:10],
        "decisions": payload.get("decisions", [])[:12],
        "metadata": payload.get("metadata", {}),
    }
    return normalized
