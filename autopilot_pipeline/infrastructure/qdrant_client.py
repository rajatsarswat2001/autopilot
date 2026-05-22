"""
infrastructure/qdrant_client.py
─────────────────────────────────────────────────────────────────────────────
Qdrant client singleton factory with health check.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

QDRANT_URL     = os.getenv("QDRANT_URL",     "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

_client = None


def get_qdrant_client():
    """Return a shared QdrantClient. Raises ImportError if qdrant-client not installed."""
    global _client
    if _client is None:
        from qdrant_client import QdrantClient
        _client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY or None,
            timeout=10,
        )
        log.info("qdrant.client_initialized", url=QDRANT_URL)
    return _client


def health_check() -> bool:
    """Return True if Qdrant is reachable."""
    try:
        client = get_qdrant_client()
        client.get_collections()
        return True
    except Exception as e:
        log.warning("qdrant.health_check_failed", error=str(e))
        return False
