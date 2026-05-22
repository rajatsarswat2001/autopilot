"""
infrastructure/redis_streams.py
─────────────────────────────────────────────────────────────────────────────
Redis connection factory and stream utilities.

This module owns all raw Redis connection logic so the rest of the codebase
imports from here rather than calling redis.from_url() in multiple places.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

REDIS_URL       = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_MAX_CONNS = int(os.getenv("REDIS_MAX_CONNS", "10"))

_pool = None


def get_redis():
    """
    Return a Redis client from a shared connection pool.
    Returns a _NullRedis no-op client if redis is unavailable (dev mode).
    """
    global _pool
    try:
        import redis
        if _pool is None:
            _pool = redis.ConnectionPool.from_url(
                REDIS_URL, max_connections=REDIS_MAX_CONNS, decode_responses=True
            )
        client = redis.Redis(connection_pool=_pool)
        client.ping()
        return client
    except Exception as e:
        log.warning("redis.unavailable", error=str(e))
        return _NullRedis()


def stream_publish(stream: str, data: dict) -> str:
    """Publish to a Redis Stream. Returns message ID or empty string."""
    import json
    flat = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in data.items()}
    try:
        return get_redis().xadd(stream, flat) or ""
    except Exception as e:
        log.warning("redis_streams.publish_failed", stream=stream, error=str(e))
        return ""


def stream_read_latest(stream: str, count: int = 10) -> list[dict]:
    """Read the latest N messages from a stream (no consumer group)."""
    import json
    try:
        messages = get_redis().xrevrange(stream, count=count)
        result = []
        for msg_id, data in messages:
            parsed = {k: _try_json(v) for k, v in data.items()}
            parsed["_id"] = msg_id
            result.append(parsed)
        return result
    except Exception as e:
        log.warning("redis_streams.read_failed", stream=stream, error=str(e))
        return []


def _try_json(v: str):
    import json
    try:
        return json.loads(v)
    except Exception:
        return v


class _NullRedis:
    """No-op Redis client for offline dev mode."""
    def ping(self): return True
    def xadd(self, *a, **kw): return "0-0"
    def xrevrange(self, *a, **kw): return []
    def xgroup_create(self, *a, **kw): pass
    def xreadgroup(self, *a, **kw): return []
    def xack(self, *a, **kw): return 0
    def set(self, *a, **kw): return True
    def get(self, *a, **kw): return None
    def incr(self, *a, **kw): return 0
