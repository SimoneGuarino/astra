"""Astra memory embedding worker helper.

This module is intentionally optional: Rust remains the governed source of
truth for the Memory Graph and can already use Ollama directly. The worker keeps
a stable Python-side contract for future sentence-transformers or specialist
embedding providers without letting Python persist cognitive memory directly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = os.environ.get("ASTRA_MEMORY_PY_EMBEDDING_PROVIDER", "ollama")
    endpoint: str = os.environ.get("ASTRA_MEMORY_EMBEDDING_OLLAMA_ENDPOINT", os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    model: str = os.environ.get("ASTRA_MEMORY_EMBEDDING_MODEL", "nomic-embed-text")
    timeout_secs: int = int(os.environ.get("ASTRA_MEMORY_EMBEDDING_TIMEOUT_SECS", "45"))


def embed_text(text: str, config: EmbeddingConfig | None = None) -> dict[str, Any]:
    cfg = config or EmbeddingConfig()
    value = text.strip()
    if not value:
        raise ValueError("embedding text is empty")
    if cfg.provider.lower() != "ollama":
        raise ValueError(f"unsupported memory embedding provider: {cfg.provider}")

    url = cfg.endpoint.rstrip("/") + "/api/embeddings"
    payload = json.dumps({"model": cfg.model, "prompt": value}).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=cfg.timeout_secs) as response:  # nosec: local user-configured endpoint
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    vector = parsed.get("embedding") or []
    if not isinstance(vector, list) or not vector:
        raise ValueError("embedding provider returned an empty vector")
    return {
        "provider": "ollama",
        "model": cfg.model,
        "dimensions": len(vector),
        "vector": vector,
        "metadata_only": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Astra memory embedding worker")
    parser.add_argument("--text", default="", help="Text to embed. If omitted, stdin is used.")
    args = parser.parse_args()
    text = args.text or sys.stdin.read()
    try:
        print(json.dumps(embed_text(text), ensure_ascii=False))
        return 0
    except Exception as exc:  # pragma: no cover - CLI defensive boundary
        print(json.dumps({"error": str(exc), "metadata_only": True}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
