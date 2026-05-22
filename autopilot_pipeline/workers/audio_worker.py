"""
workers/audio_worker.py
─────────────────────────────────────────────────────────────────────────────
Celery Audio Worker — handles TTS synthesis off the main LangGraph process.

Queue: audio
Concurrency: 2 (CPU-bound, safe to run 2 in parallel)

Startup:
    celery -A workers.audio_worker worker --queues=audio --concurrency=2
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os

import structlog

log = structlog.get_logger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

try:
    from celery import Celery

    app = Celery("audio_worker", broker=REDIS_URL, backend=REDIS_URL)
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        task_routes={"workers.audio_worker.*": {"queue": "audio"}},
        worker_prefetch_multiplier=1,
        task_acks_late=True,
    )

    @app.task(name="workers.audio_worker.synthesise_scene", bind=True, max_retries=3)
    def synthesise_scene(
        self,
        scene_id: int,
        narration: str,
        output_path: str,
        emotion_exaggeration: float = 0.5,
    ) -> dict:
        """Celery task: TTS synthesis for one scene narration."""
        log.info("audio_worker.synthesise", scene_id=scene_id, chars=len(narration))
        try:
            from tools.tts_tools import TTSChain
            from tools.ffmpeg_tools import measure_audio_duration

            chain = TTSChain()
            tier = chain.synthesise(
                text=narration,
                output_path=output_path,
                emotion_exaggeration=emotion_exaggeration,
            )
            duration = measure_audio_duration(output_path)
            return {
                "status": "ok",
                "scene_id": scene_id,
                "path": output_path,
                "duration_s": duration,
                "tier": tier,
            }
        except Exception as exc:
            log.error("audio_worker.synthesise_failed", scene_id=scene_id, error=str(exc))
            raise self.retry(exc=exc, countdown=5)

    @app.task(name="workers.audio_worker.synthesise_batch")
    def synthesise_batch(scenes: list[dict], video_id: str, output_dir: str) -> list[dict]:
        """Synthesise all scenes in a manifest and return timing info."""
        results = []
        for scene in scenes:
            out_path = f"{output_dir}/{video_id}_scene_{scene['scene_id']:03d}.wav"
            result = synthesise_scene.apply(kwargs={
                "scene_id":            scene["scene_id"],
                "narration":           scene["narration"],
                "output_path":         out_path,
                "emotion_exaggeration": scene.get("emotion_exaggeration", 0.5),
            }).get()
            results.append(result)
        return results

except ImportError:
    log.warning("audio_worker.celery_not_installed — worker disabled")
    app = None
