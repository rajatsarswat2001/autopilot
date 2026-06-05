"""
tools/rag_tools.py
─────────────────────────────────────────────────────────────────────────────
Qdrant vector store helpers for RAG (Retrieval-Augmented Generation).

Used by:
  • Research Agent   — stores research notes per video
  • Script Agent     — optionally retrieves similar past research
  • Compliance Agent — semantic similarity check against past scripts

Collection schema:
  collection: "research_notes"
  payload:    { topic, source_urls, text_chunk, video_id }
  vector:     384-dim from sentence-transformers/all-MiniLM-L6-v2 (fast CPU)

Falls back gracefully if Qdrant or sentence-transformers are unavailable.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import uuid
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

QDRANT_URL        = os.getenv("QDRANT_URL",        ":memory:")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY",    "")
EMBED_MODEL       = os.getenv("EMBED_MODEL",       "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_DIM        = 384
DEFAULT_COLLECTION = "research_notes"


# ─────────────────────────────────────────────────────────────────────────────
# Lazy-loaded singletons
# ─────────────────────────────────────────────────────────────────────────────

_embedder = None
_qdrant   = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_qdrant():
    global _qdrant
    if _qdrant is None:
        from qdrant_client import QdrantClient
        kwargs = {"location": ":memory:"} if QDRANT_URL == ":memory:" else {"url": QDRANT_URL}
        _qdrant = QdrantClient(
            **kwargs,
            api_key=QDRANT_API_KEY or None,
        )
    return _qdrant


def _embed(text: str) -> list[float]:
    return _get_embedder().encode(text, convert_to_numpy=True).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Collection management
# ─────────────────────────────────────────────────────────────────────────────

def ensure_collection(collection: str = DEFAULT_COLLECTION) -> bool:
    """Create the Qdrant collection if it doesn't exist. Returns True on success."""
    try:
        from qdrant_client.models import Distance, VectorParams
        client = _get_qdrant()
        existing = [c.name for c in client.get_collections().collections]
        if collection not in existing:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
            log.info("qdrant.collection_created", name=collection)
        return True
    except Exception as e:
        log.warning("qdrant.ensure_collection_failed", error=str(e))
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Write
# ─────────────────────────────────────────────────────────────────────────────

def upsert_document(
    collection: str,
    doc_id: str,
    text: str,
    metadata: Optional[dict] = None,
) -> bool:
    """
    Embed and upsert a text document into Qdrant.
    Returns True on success, False on any failure (non-fatal).
    """
    try:
        from qdrant_client.models import PointStruct
        ensure_collection(collection)
        vector = _embed(text[:2000])   # cap at 2k chars for speed
        payload = metadata or {}
        payload["text_chunk"] = text[:500]   # store first 500 chars as preview

        _get_qdrant().upsert(
            collection_name=collection,
            points=[PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)),
                vector=vector,
                payload=payload,
            )],
        )
        log.debug("qdrant.upserted", collection=collection, doc_id=doc_id)
        return True
    except Exception as e:
        log.warning("qdrant.upsert_failed", error=str(e))
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Read
# ─────────────────────────────────────────────────────────────────────────────

def semantic_search(
    collection: str,
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.60,
) -> list[dict]:
    """
    Search for similar documents in Qdrant.
    Returns list of payload dicts sorted by similarity.
    """
    try:
        vector = _embed(query)
        results = _get_qdrant().search(
            collection_name=collection,
            query_vector=vector,
            limit=top_k,
            score_threshold=score_threshold,
        )
        return [
            {"score": r.score, **r.payload}
            for r in results
        ]
    except Exception as e:
        log.warning("qdrant.search_failed", error=str(e))
        return []


def semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts (no Qdrant required).
    Returns 0.0–1.0.
    """
    try:
        import numpy as np
        va = _embed(text_a[:1000])
        vb = _embed(text_b[:1000])
        a  = np.array(va)
        b  = np.array(vb)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    except Exception:
        return 0.0
