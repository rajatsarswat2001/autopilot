"""
agents/audio_agent.py
─────────────────────────────────────────────────────────────────────────────
Audio Agent — AUDIO-FIRST TTS pipeline with 4-tier fallback chain.

TTS Tier Priority:
  Tier 1: Chatterbox TTS (MIT, Resemble AI) — best prosody, emotion control
  Tier 2: NVIDIA Magpie TTS (NIM microservice) — multilingual, 22kHz
  Tier 3: Edge TTS (Microsoft, free neural voices, no key required)
  Tier 4: pyttsx3 (offline CPU — never fails, sounds robotic)

Audio-First Principle:
  • Script is generated FIRST, audio is synthesised SECOND.
  • Each scene's actual WAV duration is measured by ffprobe.
  • The TimingManifest is built from REAL durations, not estimates.
  • No audio is ever cut or compressed to fit a pre-set time window.

Performance:
  • Scenes are synthesised in PARALLEL using ThreadPoolExecutor.
  • Edge TTS is network I/O — parallelism gives near-linear speedup.
  • Chatterbox (GPU) falls back to sequential to avoid VRAM contention.
  • max_workers capped at 4 to avoid API rate-limit exhaustion.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from contracts.timing_manifest import AudioScene, TimingManifest
from tools.tts_tools import TTSChain
from tools.ffmpeg_tools import measure_audio_duration
from workflows.pipeline_state import AgentError, PipelineState

log = structlog.get_logger(__name__)

OUTPUT_DIR   = Path(os.getenv("AUDIO_OUTPUT_DIR", "outputs/audio")).resolve()
_MAX_WORKERS = int(os.getenv("AUDIO_PARALLEL_WORKERS", "4"))
_PYTTSX3_LOCK = threading.Lock()  # pyttsx3 is not thread-safe


# ─────────────────────────────────────────────────────────────────────────────
# Per-scene synthesis helper (runs inside thread pool)
# ─────────────────────────────────────────────────────────────────────────────

def _synthesise_scene(
    scene: dict,
    video_id: str,
    output_dir: Path,
) -> tuple[AudioScene, dict]:
    """
    Synthesise TTS for a single scene. Thread-safe for all tiers.
    Returns (AudioScene, updated_scene_dict).
    """
    scene_id     = scene["scene_id"]
    narration    = scene.get("narration", "")
    emotion      = scene.get("emotion_exaggeration", 0.5)
    emotion_tone = scene.get("emotional_tone", None)   # from script_agent
    out_path     = output_dir / f"{video_id}_scene_{scene_id:03d}.wav"

    log.info("audio_agent.synthesising", scene_id=scene_id,
             chars=len(narration), tone=emotion_tone)

    tts = TTSChain()
    try:
        tier = tts.synthesise(
            text=narration,
            output_path=str(out_path),
            emotion_exaggeration=emotion,
            emotion_tone=emotion_tone,   # drives Chatterbox exaggeration per-scene
        )
    except Exception as e:
        log.error("audio_agent.tts_chain_fatal", scene_id=scene_id, error=str(e))
        _write_silence(str(out_path), duration_s=1.0)
        tier = "silence"

    duration = measure_audio_duration(str(out_path))

    audio_scene = AudioScene(
        scene_id=scene_id,
        audio_path=str(out_path),
        duration_s=duration,
        start_s=0.0,
        end_s=0.0,
        tts_tier=tier if tier in ("chatterbox", "magpie", "edge", "pyttsx3") else "pyttsx3",
        sample_rate=24000,
        channels=1,
    )
    updated_scene = {
        **scene,
        "audio_path":      str(out_path),
        "duration_s":      duration,      # used by caption_tools
        "duration_hint_s": duration,      # backward compat
    }
    return audio_scene, updated_scene


def _use_sequential_mode() -> bool:
    """Return True when Chatterbox + CUDA is active (avoid VRAM contention)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        import importlib.util
        return importlib.util.find_spec("chatterbox") is not None
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

def audio_node(state: PipelineState) -> dict[str, Any]:
    """
    Audio Agent node.

    Reads:  scene_manifest, video_id
    Writes: timing_manifest, tts_tier_used, audio_scenes, job_status, errors
    """
    manifest_dict = state.get("scene_manifest")
    video_id      = state.get("video_id", str(uuid.uuid4()))

    if not manifest_dict:
        err: AgentError = {
            "agent": "audio", "error": "No scene_manifest in state",
            "timestamp": datetime.now(timezone.utc).isoformat(), "recoverable": False,
        }
        return {"errors": [err]}

    scenes = manifest_dict.get("scenes", [])
    if not scenes:
        err = {
            "agent": "audio", "error": "Empty scenes list",
            "timestamp": datetime.now(timezone.utc).isoformat(), "recoverable": False,
        }
        return {"errors": [err]}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sequential = _use_sequential_mode()
    n_workers  = 1 if sequential else min(_MAX_WORKERS, len(scenes))

    log.info(
        "audio_agent.start",
        scenes=len(scenes),
        workers=n_workers,
        mode="sequential" if sequential else "parallel",
    )

    # ── Submit all scenes to thread pool ──────────────────────────────────────
    results: dict[int, tuple[AudioScene, dict]] = {}

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_synthesise_scene, scene, video_id, OUTPUT_DIR): scene["scene_id"]
            for scene in scenes
        }
        for future in as_completed(futures):
            scene_id = futures[future]
            try:
                audio_scene, updated_scene = future.result()
                results[scene_id] = (audio_scene, updated_scene)
            except Exception as e:
                log.error("audio_agent.scene_failed", scene_id=scene_id, error=str(e))
                silence_path = str(OUTPUT_DIR / f"{video_id}_scene_{scene_id:03d}.wav")
                _write_silence(silence_path, duration_s=1.0)
                # Find the original scene dict for this id
                orig_scene = next((s for s in scenes if s["scene_id"] == scene_id), {})
                results[scene_id] = (
                    AudioScene(
                        scene_id=scene_id,
                        audio_path=silence_path,
                        duration_s=1.0,
                        start_s=0.0, end_s=0.0,
                        tts_tier="pyttsx3",
                        sample_rate=24000, channels=1,
                    ),
                    {**orig_scene, "audio_path": silence_path, "duration_hint_s": 1.0},
                )

    # ── Reassemble in original scene order ────────────────────────────────────
    audio_scenes:   list[AudioScene] = []
    updated_scenes: list[dict]       = []
    tts_tier_used = "pyttsx3"

    for scene in sorted(scenes, key=lambda s: s["scene_id"]):
        sid = scene["scene_id"]
        if sid not in results:
            continue
        audio_scene, updated_scene = results[sid]
        audio_scenes.append(audio_scene)
        updated_scenes.append(updated_scene)
        tts_tier_used = audio_scene.tts_tier

    timing = TimingManifest.build(
        video_id=video_id,
        scenes=audio_scenes,
        tts_tier=tts_tier_used,
    )

    log.info(
        "audio_agent.done",
        scenes=len(audio_scenes),
        total_s=timing.total_duration_s,
        tier=tts_tier_used,
        parallel=(not sequential),
    )

    updated_manifest = {**manifest_dict, "scenes": updated_scenes}

    return {
        "timing_manifest": timing.model_dump(mode="json"),
        "tts_tier_used":   tts_tier_used,
        "audio_scenes":    [s.model_dump(mode="json") for s in audio_scenes],
        "scene_manifest":  updated_manifest,
        "job_status":      "visual",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _write_silence(path: str, duration_s: float = 1.0) -> None:
    """Write a silent WAV file as an absolute last resort."""
    import wave, struct
    sample_rate = 24000
    samples     = int(sample_rate * duration_s)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * samples, *([0] * samples)))
