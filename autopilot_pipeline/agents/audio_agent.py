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
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from contracts.timing_manifest import AudioScene, TimingManifest
from tools.tts_tools import TTSChain
from tools.ffmpeg_tools import measure_audio_duration
from workflows.pipeline_state import AgentError, PipelineState

log = structlog.get_logger(__name__)

OUTPUT_DIR = Path(os.getenv("AUDIO_OUTPUT_DIR", "outputs/audio")).resolve()


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
        err = {"agent": "audio", "error": "Empty scenes list",
               "timestamp": datetime.now(timezone.utc).isoformat(), "recoverable": False}
        return {"errors": [err]}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tts = TTSChain()

    audio_scenes: list[AudioScene] = []
    tts_tier_used: str = "pyttsx3"   # will be overwritten by actual tier

    for scene in scenes:
        scene_id  = scene["scene_id"]
        narration = scene.get("narration", "")
        emotion   = scene.get("emotion_exaggeration", 0.5)
        out_path  = OUTPUT_DIR / f"{video_id}_scene_{scene_id:03d}.wav"

        log.info("audio_agent.synthesising", scene_id=scene_id, chars=len(narration))

        try:
            tier = tts.synthesise(
                text=narration,
                output_path=str(out_path),
                emotion_exaggeration=emotion,
            )
            tts_tier_used = tier
        except Exception as e:
            # This should never propagate — TTSChain has its own final fallback
            log.error("audio_agent.tts_chain_fatal", scene_id=scene_id, error=str(e))
            # Create 1-second silence as absolute last resort
            _write_silence(str(out_path), duration_s=1.0)
            tier = "silence"

        duration = measure_audio_duration(str(out_path))

        audio_scenes.append(AudioScene(
            scene_id=scene_id,
            audio_path=str(out_path),
            duration_s=duration,
            start_s=0.0,   # assigned by TimingManifest.build()
            end_s=0.0,
            tts_tier=tier if tier in ("chatterbox", "magpie", "edge", "pyttsx3") else "pyttsx3",
            sample_rate=24000,
            channels=1,
        ))

        # Patch scene_manifest with audio path and duration for downstream agents
        scene["audio_path"]      = str(out_path)
        scene["duration_hint_s"] = duration

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
    )

    return {
        "timing_manifest": timing.model_dump(mode="json"),
        "tts_tier_used":   tts_tier_used,
        "audio_scenes":    [s.model_dump(mode="json") for s in audio_scenes],
        "scene_manifest":  manifest_dict,   # updated with audio paths
        "job_status":      "visual",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _write_silence(path: str, duration_s: float = 1.0):
    """Write a silent WAV file as an absolute last resort."""
    import wave
    import struct
    sample_rate = 24000
    samples = int(sample_rate * duration_s)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * samples, *([0] * samples)))
