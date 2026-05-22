"""
workers/render_worker.py
─────────────────────────────────────────────────────────────────────────────
Celery Render Worker — handles final video assembly and FFmpeg rendering.

Queue: render
Concurrency: 1 (I/O + CPU intensive — serialise renders)

Startup:
    celery -A workers.render_worker worker --queues=render --concurrency=1
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os

import structlog

log = structlog.get_logger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

try:
    from celery import Celery

    app = Celery("render_worker", broker=REDIS_URL, backend=REDIS_URL)
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        task_routes={"workers.render_worker.*": {"queue": "render"}},
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        task_time_limit=3600,       # 1 hour hard timeout
        task_soft_time_limit=3300,  # 55 min soft timeout (triggers warning)
    )

    @app.task(name="workers.render_worker.render_video", bind=True, max_retries=1)
    def render_video(self, timeline_manifest_dict: dict) -> dict:
        """Celery task: compile and render a TimelineManifest to MP4."""
        from contracts.timeline_manifest import TimelineManifest
        from renderer.timeline_compiler import compile_timeline
        from renderer.ffmpeg_builder import render_timeline

        video_id = timeline_manifest_dict.get("video_id", "unknown")
        log.info("render_worker.start", video_id=video_id)

        try:
            manifest = TimelineManifest(**timeline_manifest_dict)
            plan     = compile_timeline(manifest)
            output   = render_timeline(plan)
            log.info("render_worker.done", output=output, video_id=video_id)
            return {"status": "ok", "output_path": output, "video_id": video_id}
        except Exception as exc:
            log.error("render_worker.failed", video_id=video_id, error=str(exc))
            raise self.retry(exc=exc, countdown=30)

except ImportError:
    log.warning("render_worker.celery_not_installed — worker disabled")
    app = None
