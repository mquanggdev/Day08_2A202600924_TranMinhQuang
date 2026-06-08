"""Task 5 - deterministic semantic-style search.

For pytest this uses local TF cosine similarity over chunks. The public
function contract matches a dense retriever, so a Weaviate/BGE-M3 backend can
replace the scorer without changing downstream code.
"""

from collections import Counter

from .retrieval_utils import cosine_from_counters, load_chunks, tokenize
from .task4_chunking_indexing import CHUNK_OVERLAP, CHUNK_SIZE


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    query_vec = Counter(tokenize(query))
    results: list[dict] = []

    for chunk in load_chunks(CHUNK_SIZE, CHUNK_OVERLAP):
        score = cosine_from_counters(query_vec, Counter(tokenize(chunk["content"])))
        if score > 0:
            results.append(
                {
                    "content": chunk["content"],
                    "score": float(score),
                    "metadata": chunk.get("metadata", {}),
                }
            )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[: max(top_k, 0)]


if __name__ == "__main__":
    for r in semantic_search("hinh phat ma tuy", top_k=5):
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
