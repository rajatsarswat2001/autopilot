"""
orchestration/event_bus.py
─────────────────────────────────────────────────────────────────────────────
Redis Streams event bus — the async backbone for distributing work to
independent workers without blocking the LangGraph supervisor.

Pattern:
  1. Supervisor dispatches jobs by publishing events to Redis Streams.
  2. Workers (workers/*.py) consume their stream, process, publish completion.
  3. Supervisor listens for completion events to update state.

This decouples heavy GPU workloads from the orchestration layer.

Stream names:
  autopilot:jobs:audio    → Audio Worker
  autopilot:jobs:visual   → Visual Worker
  autopilot:jobs:render   → Render Worker
  autopilot:events:done   → Supervisor listener (completions)

Usage (publish):
    bus = EventBus()
    job_id = bus.dispatch_audio_job(video_id, scene_manifest)

Usage (consume):
    for event in bus.consume("autopilot:jobs:audio", group="audio-workers"):
        process(event)
        bus.ack("autopilot:jobs:audio", "audio-workers", event["id"])
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Generator, Optional

import structlog

log = structlog.get_logger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

STREAM_AUDIO   = "autopilot:jobs:audio"
STREAM_VISUAL  = "autopilot:jobs:visual"
STREAM_RENDER  = "autopilot:jobs:render"
STREAM_DONE    = "autopilot:events:done"
STREAM_ERRORS  = "autopilot:events:errors"

CONSUMER_GROUPS = {
    STREAM_AUDIO:  "audio-workers",
    STREAM_VISUAL: "visual-workers",
    STREAM_RENDER: "render-workers",
    STREAM_DONE:   "supervisor",
}


class EventBus:
    """
    Thin wrapper around Redis Streams for job dispatch and event routing.
    Lazy-imports redis to avoid hard dependency in non-async environments.
    """

    def __init__(self, redis_url: str = REDIS_URL):
        self._url = redis_url
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import redis
                self._client = redis.from_url(self._url, decode_responses=True)
                self._client.ping()
                log.info("event_bus.redis_connected", url=self._url)
            except Exception as e:
                log.warning("event_bus.redis_unavailable", error=str(e))
                self._client = _NullRedis()   # no-op fallback
        return self._client

    def _ensure_groups(self):
        """Create consumer groups if they don't already exist."""
        for stream, group in CONSUMER_GROUPS.items():
            try:
                self.client.xgroup_create(stream, group, id="0", mkstream=True)
            except Exception:
                pass   # BUSYGROUP error means group already exists — ok

    # ── Dispatch ─────────────────────────────────────────────────────────────

    def _publish(self, stream: str, payload: dict) -> str:
        """Publish a message to a Redis Stream. Returns the message ID."""
        flat = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in payload.items()}
        try:
            msg_id = self.client.xadd(stream, flat)
            log.debug("event_bus.published", stream=stream, msg_id=msg_id)
            return msg_id
        except Exception as e:
            log.error("event_bus.publish_failed", stream=stream, error=str(e))
            return ""

    def dispatch_audio_job(self, video_id: str, scene_manifest: dict) -> str:
        job_id = str(uuid.uuid4())
        return self._publish(STREAM_AUDIO, {
            "job_id":        job_id,
            "video_id":      video_id,
            "scene_manifest": scene_manifest,
            "dispatched_at": datetime.now(timezone.utc).isoformat(),
        })

    def dispatch_visual_job(self, video_id: str, timing_manifest: dict, scene_manifest: dict) -> str:
        job_id = str(uuid.uuid4())
        return self._publish(STREAM_VISUAL, {
            "job_id":          job_id,
            "video_id":        video_id,
            "timing_manifest": timing_manifest,
            "scene_manifest":  scene_manifest,
            "dispatched_at":   datetime.now(timezone.utc).isoformat(),
        })

    def dispatch_render_job(self, video_id: str, timeline_manifest: dict) -> str:
        job_id = str(uuid.uuid4())
        return self._publish(STREAM_RENDER, {
            "job_id":            job_id,
            "video_id":          video_id,
            "timeline_manifest": timeline_manifest,
            "dispatched_at":     datetime.now(timezone.utc).isoformat(),
        })

    def publish_completion(
        self,
        video_id: str,
        job_type: str,
        result: dict,
    ) -> str:
        return self._publish(STREAM_DONE, {
            "video_id":    video_id,
            "job_type":    job_type,
            "result":      result,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })

    def publish_error(self, video_id: str, job_type: str, error: str) -> str:
        return self._publish(STREAM_ERRORS, {
            "video_id": video_id,
            "job_type": job_type,
            "error":    error,
            "ts":       datetime.now(timezone.utc).isoformat(),
        })

    # ── Consume ──────────────────────────────────────────────────────────────

    def consume(
        self,
        stream: str,
        group: str,
        consumer: str = "worker-1",
        count: int = 1,
        block_ms: int = 5000,
    ) -> Generator[dict, None, None]:
        """
        Blocking consumer for a Redis Stream group.
        Yields parsed message dicts with an extra 'id' key for acking.
        """
        self._ensure_groups()
        while True:
            try:
                messages = self.client.xreadgroup(
                    group, consumer, {stream: ">"}, count=count, block=block_ms
                )
                for _, msg_list in (messages or []):
                    for msg_id, data in msg_list:
                        parsed = {
                            k: _try_json(v)
                            for k, v in data.items()
                        }
                        parsed["_id"] = msg_id
                        yield parsed
            except Exception as e:
                log.error("event_bus.consume_error", stream=stream, error=str(e))
                time.sleep(2)

    def ack(self, stream: str, group: str, msg_id: str):
        try:
            self.client.xack(stream, group, msg_id)
        except Exception as e:
            log.warning("event_bus.ack_failed", error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _try_json(v: str):
    try:
        return json.loads(v)
    except (json.JSONDecodeError, TypeError):
        return v


class _NullRedis:
    """No-op Redis client used when Redis is unavailable (dev mode)."""

    def ping(self): return True
    def xadd(self, *a, **kw): return "0-0"
    def xgroup_create(self, *a, **kw): pass
    def xreadgroup(self, *a, **kw): return []
    def xack(self, *a, **kw): pass


# Singleton for import convenience
event_bus = EventBus()
