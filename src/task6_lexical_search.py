"""Task 6 - BM25 lexical search with a dependency-free fallback."""

import math
from collections import Counter

from .retrieval_utils import load_chunks, tokenize
from .task4_chunking_indexing import CHUNK_OVERLAP, CHUNK_SIZE


def build_bm25_index(corpus: list[dict]):
    tokenized = [tokenize(doc.get("content", "")) for doc in corpus]
    return {"corpus": corpus, "tokenized": tokenized}


def _bm25_scores(query_tokens: list[str], tokenized_corpus: list[list[str]]) -> list[float]:
    if not query_tokens or not tokenized_corpus:
        return [0.0 for _ in tokenized_corpus]

    n_docs = len(tokenized_corpus)
    avgdl = sum(len(doc) for doc in tokenized_corpus) / max(n_docs, 1)
    doc_freq: Counter[str] = Counter()
    for doc in tokenized_corpus:
        doc_freq.update(set(doc))

    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    for doc in tokenized_corpus:
        tf = Counter(doc)
        doc_len = len(doc) or 1
        score = 0.0
        for term in query_tokens:
            if tf[term] == 0:
                continue
            idf = math.log(1 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * doc_len / max(avgdl, 1e-9))
            score += idf * (tf[term] * (k1 + 1)) / denom
        scores.append(float(score))
    return scores


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    corpus = load_chunks(CHUNK_SIZE, CHUNK_OVERLAP)
    index = build_bm25_index(corpus)
    scores = _bm25_scores(tokenize(query), index["tokenized"])

    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    results: list[dict] = []
    for idx, score in ranked[: max(top_k, 0)]:
        if score <= 0:
            continue
        doc = corpus[idx]
        results.append(
            {
                "content": doc["content"],
                "score": float(score),
                "metadata": doc.get("metadata", {}),
            }
        )
    return results


if __name__ == "__main__":
    for r in lexical_search("Dieu 248 ma tuy", top_k=5):
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
