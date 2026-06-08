"""Task 8 - PageIndex vectorless fallback contract.

The PageIndex SDK syntax is intentionally not guessed here. Until official SDK
usage is provided, this module exposes the required fallback interface by
searching the local standardized corpus and marking results as PageIndex.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from .retrieval_utils import lexical_overlap_score, load_chunks
from .task4_chunking_indexing import CHUNK_OVERLAP, CHUNK_SIZE

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """Placeholder hook for real PageIndex upload once SDK docs are supplied."""
    return {"uploaded": 0, "status": "sdk_docs_required"}


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Vectorless fallback search with the PageIndex result schema."""
    results: list[dict] = []
    for chunk in load_chunks(CHUNK_SIZE, CHUNK_OVERLAP):
        score = lexical_overlap_score(query, chunk["content"])
        if score <= 0:
            continue
        results.append(
            {
                "content": chunk["content"],
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
                "source": "pageindex",
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[: max(top_k, 0)]


if __name__ == "__main__":
    for r in pageindex_search("hinh phat ma tuy", top_k=3):
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
