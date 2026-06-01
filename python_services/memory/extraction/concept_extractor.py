from __future__ import annotations

import re

from python_services.memory.schemas.memory_payloads import MemoryExtractionCandidate


def extract_keyword_concepts(text: str, *, limit: int = 12) -> list[MemoryExtractionCandidate]:
    """Conservative local extractor used before LLM-based consolidation exists."""
    tokens = re.findall(r"[\wÀ-ÿ][\wÀ-ÿ\-]{3,}", text.lower())
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return [
        MemoryExtractionCandidate(
            kind="concept",
            title=token,
            summary=f"Concept candidate extracted from text frequency ({count} occurrences).",
            confidence=min(0.35 + count * 0.05, 0.75),
            tags=["auto_extracted", "keyword"],
        )
        for token, count in ranked
    ]
