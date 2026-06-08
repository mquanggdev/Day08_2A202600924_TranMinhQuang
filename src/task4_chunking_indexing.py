"""Task 4 - Chunking, embedding, and real Weaviate indexing.

This module follows the README architecture:
1. Load markdown from data/standardized/
2. Chunk with a recursive character splitter
3. Embed chunks
4. Index into Weaviate Cloud or local Docker

The Weaviate connection is real when WEAVIATE_URL is configured or a local
Docker instance is running. Tests can still import/use load/chunk functions
without requiring an active database.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

from dotenv import load_dotenv

from .retrieval_utils import chunk_text, load_markdown_documents, tokenize

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Recursive character chunking is robust for mixed legal/news markdown.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# README embedding choices:
# - OpenAI text-embedding-3-small: API-based, stable, 1536 dimensions.
# - sentence-transformers/all-MiniLM-L6-v2: lightweight local fallback.
# Jina is intentionally not used in Task 4; it belongs to optional reranking.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_EMBEDDING_BATCH_SIZE = int(os.getenv("OPENAI_EMBEDDING_BATCH_SIZE", "64"))
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LOCAL_EMBEDDING_DIM = int(os.getenv("LOCAL_EMBEDDING_DIM", "384"))
HASH_EMBEDDING_DIM = 384

VECTOR_STORE = "weaviate"
WEAVIATE_COLLECTION = os.getenv("WEAVIATE_COLLECTION", "DrugLawDocs")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "").strip()
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "").strip()
WEAVIATE_LOCAL_HOST = os.getenv("WEAVIATE_LOCAL_HOST", "localhost")
WEAVIATE_LOCAL_PORT = int(os.getenv("WEAVIATE_LOCAL_PORT", "8080"))
WEAVIATE_LOCAL_GRPC_PORT = int(os.getenv("WEAVIATE_LOCAL_GRPC_PORT", "50051"))


def load_documents() -> list[dict]:
    """Read markdown documents from data/standardized."""
    return load_markdown_documents()


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk documents into {'content', 'metadata'} records."""
    chunks: list[dict] = []
    for doc in documents:
        for idx, text in enumerate(chunk_text(doc.get("content", ""), CHUNK_SIZE, CHUNK_OVERLAP)):
            chunks.append(
                {
                    "content": text,
                    "metadata": {**doc.get("metadata", {}), "chunk_index": idx},
                }
            )
    return chunks


def _hash_embedding(text: str, dim: int = HASH_EMBEDDING_DIM) -> list[float]:
    """Deterministic lightweight embedding fallback for offline indexing."""
    vector = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[idx] += sign

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed chunks using README-approved models, then fallback safely."""
    if EMBEDDING_PROVIDER == "openai" and OPENAI_API_KEY and "xxx" not in OPENAI_API_KEY.lower():
        try:
            return _embed_chunks_openai(chunks)
        except Exception as exc:
            api_error = type(exc).__name__
    else:
        api_error = "not_configured"

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        embeddings = model.encode([c["content"] for c in chunks], normalize_embeddings=True)
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist()
            chunk["embedding_model"] = LOCAL_EMBEDDING_MODEL
        return chunks
    except Exception as exc:
        for chunk in chunks:
            chunk["embedding"] = _hash_embedding(chunk["content"])
            chunk["embedding_model"] = f"hash-fallback:api={api_error};local={type(exc).__name__}"
        return chunks


def _embed_chunks_openai(chunks: list[dict]) -> list[dict]:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    for start in range(0, len(chunks), OPENAI_EMBEDDING_BATCH_SIZE):
        batch = chunks[start : start + OPENAI_EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[item["content"] for item in batch],
        )
        data = sorted(response.data, key=lambda item: item.index)
        for chunk, item in zip(batch, data):
            chunk["embedding"] = item["embedding"]
            chunk["embedding_model"] = EMBEDDING_MODEL
    return chunks


def connect_weaviate():
    """Connect to Weaviate Cloud if configured, otherwise local Docker."""
    import weaviate
    from weaviate.classes.init import Auth

    if WEAVIATE_URL and "xxx" not in WEAVIATE_URL.lower():
        if not WEAVIATE_API_KEY or "xxx" in WEAVIATE_API_KEY.lower():
            raise RuntimeError("WEAVIATE_URL is set but WEAVIATE_API_KEY is missing or placeholder.")
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        )

    return weaviate.connect_to_local(
        host=WEAVIATE_LOCAL_HOST,
        port=WEAVIATE_LOCAL_PORT,
        grpc_port=WEAVIATE_LOCAL_GRPC_PORT,
    )


def _ensure_collection(client):
    from weaviate.classes.config import Configure, DataType, Property

    if client.collections.exists(WEAVIATE_COLLECTION):
        client.collections.delete(WEAVIATE_COLLECTION)

    return client.collections.create(
        name=WEAVIATE_COLLECTION,
        vector_config=Configure.Vectors.self_provided(),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="path", data_type=DataType.TEXT),
            Property(name="doc_type", data_type=DataType.TEXT),
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="embedding_model", data_type=DataType.TEXT),
        ],
    )


def index_to_vectorstore(chunks: list[dict]):
    """Create/recreate the Weaviate collection and batch import chunks."""
    if not chunks:
        return {"vector_store": VECTOR_STORE, "collection": WEAVIATE_COLLECTION, "indexed": 0}

    client = connect_weaviate()
    try:
        if not client.is_ready():
            raise RuntimeError("Weaviate client connected but server is not ready.")

        collection = _ensure_collection(client)
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                metadata = chunk.get("metadata", {})
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": metadata.get("source", ""),
                        "path": metadata.get("path", ""),
                        "doc_type": metadata.get("type", ""),
                        "chunk_index": int(metadata.get("chunk_index", 0)),
                        "embedding_model": chunk.get("embedding_model", ""),
                    },
                    vector=chunk["embedding"],
                )

        failed = getattr(collection.batch, "failed_objects", []) or []
        if failed:
            raise RuntimeError(f"Weaviate batch import failed for {len(failed)} objects.")

        return {
            "vector_store": VECTOR_STORE,
            "collection": WEAVIATE_COLLECTION,
            "indexed": len(chunks),
            "embedding_model": chunks[0].get("embedding_model", ""),
        }
    finally:
        client.close()


def run_pipeline():
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print(f"  Collection: {WEAVIATE_COLLECTION}")
    print("=" * 50)

    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks via {chunks[0].get('embedding_model') if chunks else 'n/a'}")

    summary = index_to_vectorstore(chunks)
    print(f"Indexed {summary['indexed']} chunks to {summary['collection']}")
    return summary


if __name__ == "__main__":
    run_pipeline()
