"""Shared deterministic helpers for the Day 08 RAG pipeline.

The production path can plug Weaviate/Jina/PageIndex behind these contracts.
For automated tests, these helpers keep retrieval local, repeatable, and free
from network/API-key failures.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"


def tokenize(text: str) -> list[str]:
    """Unicode-aware tokenization that works reasonably for Vietnamese text."""
    return re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)


def load_markdown_documents() -> list[dict]:
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if not md_file.is_file():
            continue
        content = md_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            continue
        rel_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = rel_path.parts[0] if len(rel_path.parts) > 1 else "unknown"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": str(rel_path).replace("\\", "/"),
                    "type": doc_type,
                },
            }
        )
    return documents


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
    chunks: list[str] = []
    start = 0

    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        end = hard_end
        window = text[start:hard_end]

        for sep in separators:
            idx = window.rfind(sep)
            if idx >= int(chunk_size * 0.55):
                end = start + idx + len(sep)
                break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


def load_chunks(chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    chunks: list[dict] = []
    for doc in load_markdown_documents():
        for idx, content in enumerate(chunk_text(doc["content"], chunk_size, chunk_overlap)):
            chunks.append(
                {
                    "content": content,
                    "metadata": {**doc["metadata"], "chunk_index": idx},
                }
            )
    return chunks


def cosine_from_counters(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def lexical_overlap_score(query: str, content: str) -> float:
    q_tokens = tokenize(query)
    c_tokens = tokenize(content)
    if not q_tokens or not c_tokens:
        return 0.0
    q_counts = Counter(q_tokens)
    c_counts = Counter(c_tokens)
    overlap = sum(min(q_counts[t], c_counts.get(t, 0)) for t in q_counts)
    coverage = overlap / max(len(q_tokens), 1)
    cosine = cosine_from_counters(q_counts, c_counts)
    return 0.7 * coverage + 0.3 * cosine

