from __future__ import annotations

"""AstraOS v0.6 Deep-Search extraction helper.

This worker is intentionally a normalization/extraction helper only. Rust owns
network policy, source acceptance, persistence, verification state, graph
relations, embeddings, audit and final consolidation.
"""

from dataclasses import dataclass, field, asdict
from hashlib import sha256
from html.parser import HTMLParser
from typing import Any

from .research_worker import normalize_research_bundle


@dataclass(frozen=True)
class DeepSearchDocument:
    url: str
    title: str
    text: str
    content_hash: str
    fetched_at: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []
        self.title: str | None = None
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text or self._skip_depth:
            return
        if self._in_title:
            self.title = text
        else:
            self.parts.append(text)


def normalize_html_document(url: str, html: str, *, fetched_at: int | None = None, metadata: dict[str, Any] | None = None) -> DeepSearchDocument:
    parser = _TextExtractor()
    parser.feed(html)
    text = "\n".join(parser.parts)
    content_hash = sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return DeepSearchDocument(
        url=url,
        title=parser.title or url,
        text=text,
        content_hash=content_hash,
        fetched_at=fetched_at,
        metadata=metadata or {},
    )


def build_foundation_bundle(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a deep-search extraction payload into Rust's ResearchMemoryBundle.

    Expected payload fields are intentionally close to the Rust boundary:
    topic, objective, query, sources, findings, claims, procedures,
    recommendations and metadata.
    """

    bundle = normalize_research_bundle(payload)
    data = bundle.to_rust_payload()
    metadata = dict(data.get("metadata") or {})
    metadata.update(
        {
            "python_worker": "memory.research.deep_search_worker",
            "schema_version": 1,
            "rust_remains_source_of_truth": True,
            "external_content_untrusted": True,
        }
    )
    data["metadata"] = metadata
    return data


def main() -> None:
    import json
    import sys

    payload = json.load(sys.stdin)
    json.dump(build_foundation_bundle(payload), sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
