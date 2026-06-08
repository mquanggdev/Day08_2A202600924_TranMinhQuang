"""Task 10 - citation-grounded generation via OpenRouter."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve

load_dotenv()


TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
MAX_TOKENS = 900

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1/chat/completions",
)
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "Day08-RAG-Pipeline-v2")

SYSTEM_PROMPT = """Bạn là trợ lý RAG cho pháp luật Việt Nam về ma túy và tin tức nghệ sĩ liên quan ma túy.
Chỉ sử dụng thông tin trong CONTEXT được cung cấp.
Mọi khẳng định thực tế phải có citation ngay sau câu, dạng [Nguồn, Năm] hoặc [Tên file, N/A].
Nếu context không đủ bằng chứng, trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có."
Không suy đoán điều luật, mức án, tình trạng tố tụng hoặc dữ kiện báo chí ngoài context."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Place strong evidence at the beginning and end of the context."""
    if len(chunks) <= 2:
        return list(chunks)

    reordered: list[dict] = []
    left = 0
    right = len(chunks) - 1
    place_front = True
    while left <= right:
        if place_front:
            reordered.append(chunks[left])
            left += 1
        else:
            reordered.append(chunks[right])
            right -= 1
        place_front = not place_front
    return reordered


def _citation_label(chunk: dict, index: int) -> str:
    metadata = chunk.get("metadata", {})
    source = metadata.get("source") or metadata.get("path") or f"Source {index}"
    year = metadata.get("year") or "N/A"
    return f"{source}, {year}"


def format_context(chunks: list[dict]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"Source {i}")
        year = metadata.get("year", "N/A")
        doc_type = metadata.get("type", "unknown")
        path = metadata.get("path", source)
        parts.append(
            f"[Document {i} | Citation: {source}, {year} | Type: {doc_type} | Path: {path}]\n"
            f"{chunk.get('content', '')}"
        )
    return "\n\n---\n\n".join(parts)


def _extractive_fallback(reordered: list[dict]) -> str:
    if not reordered:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    best = reordered[0]
    citation = _citation_label(best, 1)
    content = best.get("content", "").strip()
    if len(content) > 1200:
        content = content[:1200].rsplit(" ", 1)[0] + "..."
    return f"Dựa trên nguồn truy xuất được, nội dung liên quan nhất là: {content} [{citation}]."


def _call_openrouter(query: str, context: str) -> str:
    if not OPENROUTER_API_KEY:
        return ""
    import requests

    user_message = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_HTTP_REFERER,
        "X-Title": OPENROUTER_APP_TITLE,
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        # Low temperature/top_p keep legal/news answers factual and stable.
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
    }

    response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    answer = ""
    if reordered:
        try:
            answer = _call_openrouter(query, context)
        except Exception:
            answer = ""
    if not answer:
        answer = _extractive_fallback(reordered)

    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": reordered[0].get("source", "hybrid") if reordered else "none",
        "model": OPENROUTER_MODEL if OPENROUTER_API_KEY else "extractive-fallback",
    }


if __name__ == "__main__":
    result = generate_with_citation("Hình phạt tàng trữ ma túy là gì?")
    print(result["answer"])
