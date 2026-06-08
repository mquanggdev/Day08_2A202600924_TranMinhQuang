"""Task 7 - deterministic reranking.

Jina reranking can be added when API wiring is required. The default path is a
local lexical relevance reranker so tests and demos do not fail offline.
"""

from .retrieval_utils import lexical_overlap_score


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Local cross-encoder substitute: combine query overlap and prior score."""
    reranked: list[dict] = []
    for candidate in candidates:
        prior = float(candidate.get("score", 0.0))
        relevance = lexical_overlap_score(query, candidate.get("content", ""))
        item = candidate.copy()
        item["score"] = float(0.65 * relevance + 0.35 * prior)
        item.setdefault("metadata", {})
        reranked.append(item)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[: max(top_k, 0)]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Score-only MMR fallback for candidates without dense embeddings."""
    del query_embedding, lambda_param
    return sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)[: max(top_k, 0)]


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion over multiple ranked result lists."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = item.get("content", "")
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in items or item.get("score", 0.0) > items[key].get("score", 0.0):
                items[key] = item.copy()

    fused: list[dict] = []
    for content, score in scores.items():
        item = items[content].copy()
        item["score"] = float(score)
        item.setdefault("metadata", {})
        fused.append(item)

    fused.sort(key=lambda item: item["score"], reverse=True)
    return fused[: max(top_k, 0)]


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    if not candidates:
        return []
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        return rerank_mmr([], candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Dieu 248: Toi tang tru trai phep chat ma tuy", "score": 0.8, "metadata": {}},
        {"content": "Nghe si bi bat vi su dung ma tuy", "score": 0.7, "metadata": {}},
        {"content": "Python programming", "score": 0.6, "metadata": {}},
    ]
    for r in rerank("hinh phat ma tuy", dummy_candidates, top_k=2):
        print(f"[{r['score']:.3f}] {r['content']}")
