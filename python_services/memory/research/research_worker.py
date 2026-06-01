from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


def _as_list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


@dataclass(frozen=True)
class ResearchSource:
    title: str
    uri: str | None = None
    source_type: str | None = None
    summary: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResearchFinding:
    title: str
    summary: str
    evidence: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    confidence: float | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResearchClaim:
    claim: str
    rationale: str | None = None
    source_refs: list[str] = field(default_factory=list)
    confidence: float | None = None
    verification_status: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResearchProcedure:
    title: str
    steps: list[str] = field(default_factory=list)
    rationale: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResearchRecommendation:
    title: str
    summary: str
    actionability: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResearchMemoryBundle:
    topic: str
    objective: str | None = None
    query: str | None = None
    summary: str | None = None
    confidence: float | None = None
    verification_status: str | None = None
    tags: list[str] = field(default_factory=list)
    sources: list[ResearchSource] = field(default_factory=list)
    findings: list[ResearchFinding] = field(default_factory=list)
    claims: list[ResearchClaim] = field(default_factory=list)
    procedures: list[ResearchProcedure] = field(default_factory=list)
    recommendations: list[ResearchRecommendation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_rust_payload(self) -> dict[str, Any]:
        return asdict(self)


def normalize_research_bundle(payload: dict[str, Any]) -> ResearchMemoryBundle:
    """Normalize worker/deep-research output to the Rust-governed bundle schema.

    Python may help extraction and normalization, but Rust remains the source of
    truth for persistence, validation, privacy, graph relations, and governance.
    """

    findings: list[ResearchFinding] = []
    for index, item in enumerate(payload.get("findings", [])):
        if isinstance(item, dict):
            title = str(item.get("title") or f"Finding {index + 1}").strip()
            summary = str(item.get("summary") or item.get("text") or "").strip()
            if summary:
                findings.append(
                    ResearchFinding(
                        title=title,
                        summary=summary,
                        evidence=_as_list_of_strings(item.get("evidence")),
                        source_refs=_as_list_of_strings(item.get("source_refs")),
                        confidence=_safe_float_or_none(item.get("confidence")),
                        tags=_as_list_of_strings(item.get("tags")),
                        metadata=_as_dict(item.get("metadata")),
                    )
                )
        else:
            text = str(item).strip()
            if text:
                findings.append(ResearchFinding(title=f"Finding {index + 1}", summary=text))

    claims: list[ResearchClaim] = []
    for item in payload.get("claims", []):
        if isinstance(item, dict):
            claim = str(item.get("claim") or item.get("text") or "").strip()
            if claim:
                claims.append(
                    ResearchClaim(
                        claim=claim,
                        rationale=_optional_str(item.get("rationale")),
                        source_refs=_as_list_of_strings(item.get("source_refs")),
                        confidence=_safe_float_or_none(item.get("confidence")),
                        verification_status=_optional_str(item.get("verification_status")),
                        metadata=_as_dict(item.get("metadata")),
                    )
                )
        else:
            text = str(item).strip()
            if text:
                claims.append(ResearchClaim(claim=text))

    return ResearchMemoryBundle(
        topic=str(payload.get("topic") or "Untitled research").strip(),
        objective=_optional_str(payload.get("objective")),
        query=_optional_str(payload.get("query")),
        summary=_optional_str(payload.get("summary")),
        confidence=_safe_float_or_none(payload.get("confidence")),
        verification_status=_optional_str(payload.get("verification_status")),
        tags=_as_list_of_strings(payload.get("tags")),
        sources=[_normalize_source(item) for item in payload.get("sources", []) if isinstance(item, dict)],
        findings=findings,
        claims=claims,
        procedures=[_normalize_procedure(item, i) for i, item in enumerate(payload.get("procedures", [])) if isinstance(item, dict)],
        recommendations=[
            _normalize_recommendation(item, i)
            for i, item in enumerate(payload.get("recommendations", payload.get("recommended_actions", [])))
            if isinstance(item, dict) or str(item).strip()
        ],
        metadata=_as_dict(payload.get("metadata")),
    )


def _normalize_source(item: dict[str, Any]) -> ResearchSource:
    return ResearchSource(
        title=str(item.get("title") or item.get("uri") or "Research source").strip(),
        uri=_optional_str(item.get("uri")),
        source_type=_optional_str(item.get("source_type")),
        summary=_optional_str(item.get("summary")),
        confidence=_safe_float_or_none(item.get("confidence")),
        metadata=_as_dict(item.get("metadata")),
    )


def _normalize_procedure(item: dict[str, Any], index: int) -> ResearchProcedure:
    return ResearchProcedure(
        title=str(item.get("title") or f"Procedure {index + 1}").strip(),
        steps=_as_list_of_strings(item.get("steps")),
        rationale=_optional_str(item.get("rationale")),
        confidence=_safe_float_or_none(item.get("confidence")),
        metadata=_as_dict(item.get("metadata")),
    )


def _normalize_recommendation(item: Any, index: int) -> ResearchRecommendation:
    if isinstance(item, dict):
        return ResearchRecommendation(
            title=str(item.get("title") or f"Recommendation {index + 1}").strip(),
            summary=str(item.get("summary") or item.get("text") or "").strip(),
            actionability=_optional_str(item.get("actionability")),
            confidence=_safe_float_or_none(item.get("confidence")),
            metadata=_as_dict(item.get("metadata")),
        )
    text = str(item).strip()
    return ResearchRecommendation(title=f"Recommendation {index + 1}", summary=text)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
