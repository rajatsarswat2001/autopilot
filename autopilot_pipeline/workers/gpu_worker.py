"""
workers/gpu_worker.py
─────────────────────────────────────────────────────────────────────────────
Celery GPU Worker — handles SDXL image generation off the main process.

Architecture:
  • LangGraph supervisor dispatches jobs via Redis Streams (event_bus.py)
  • This Celery worker consumes from autopilot:jobs:visual
  • GPU models are loaded ONCE at worker startup, not per-task
  • Results published back to autopilot:events:done

Startup:
    celery -A workers.gpu_worker worker --queues=gpu --concurrency=1 --loglevel=info

Note: --concurrency=1 is intentional — GPU is a single resource.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
from typing import Any

import structlog

log = structlog.get_logger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ─────────────────────────────────────────────────────────────────────────────
# Celery app
# ─────────────────────────────────────────────────────────────────────────────

try:
    from celery import Celery

    app = Celery("gpu_worker", broker=REDIS_URL, backend=REDIS_URL)
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_routes={"workers.gpu_worker.*": {"queue": "gpu"}},
        worker_prefetch_multiplier=1,   # process one task at a time
        task_acks_late=True,            # ack after completion, not on receive
    )

    @app.task(name="workers.gpu_worker.generate_image", bind=True, max_retries=2)
    def generate_image(self, video_id: str, scene_id: int, prompt: str, output_path: str) -> dict:
        """Celery task: SDXL image generation for one scene."""
        log.info("gpu_worker.generate_image", video_id=video_id, scene_id=scene_id)
        try:
            from tools.nim_tools import generate_image_sdxl
            path = generate_image_sdxl(prompt=prompt, output_path=output_path)
            return {"status": "ok", "path": path, "scene_id": scene_id}
        except Exception as exc:
            log.error("gpu_worker.generate_image_failed", error=str(exc))
            raise self.retry(exc=exc, countdown=10)

    @app.task(name="workers.gpu_worker.render_ken_burns", bind=True, max_retries=2)
    def render_ken_burns(
        self,
        image_path: str,
        output_path: str,
        duration_s: float,
        motion: str = "zoom_in",
    ) -> dict:
        """Celery task: Ken Burns video from still image."""
        log.info("gpu_worker.render_ken_burns", image=image_path, duration=duration_s)
        try:
            from tools.ffmpeg_tools import image_to_video
            result = image_to_video(
                image_path=image_path,
                output_path=output_path,
                duration_s=duration_s,
                fps=30,
                ken_burns=True,
                motion=motion,
            )
            return {"status": "ok", "path": result}
        except Exception as exc:
            raise self.retry(exc=exc, countdown=5)

except ImportError:
    log.warning("gpu_worker.celery_not_installed — worker disabled")
    app = None
