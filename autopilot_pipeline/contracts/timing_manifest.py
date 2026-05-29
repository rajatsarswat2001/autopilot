"""
contracts/timing_manifest.py
─────────────────────────────────────────────────────────────────────────────
Data contract for timing_manifest.json produced by Audio Agent.

AUDIO-FIRST PRINCIPLE:
  Timeline is built AFTER measuring actual TTS durations via ffprobe.
  Visuals conform to audio — audio never gets cut to fit visuals.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AudioScene(BaseModel):
    """Audio information for a single scene."""

    scene_id: int = Field(..., ge=1)
    audio_path: str = Field(..., description="Absolute path to the generated WAV")
    duration_s: float = Field(..., gt=0.0)
    start_s: float = Field(0.0, ge=0.0)
    end_s: float = Field(0.0, ge=0.0)
    tts_tier: Literal["chatterbox", "kokoro", "magpie", "edge", "pyttsx3"]
    sample_rate: int = 24000
    channels: int = 1

    @field_validator("audio_path")
    @classmethod
    def must_be_absolute(cls, v: str) -> str:
        if not Path(v).is_absolute():
            raise ValueError(f"audio_path must be absolute, got: {v}")
        return v

    def frames_at(self, fps: int = 30) -> int:
        return int(self.duration_s * fps)


class TimingManifest(BaseModel):
    """
    Master timing manifest.
    Produced by Audio Agent; consumed by Visual Director + Assembly Agent.
    """

    schema_version: str = "1.0"
    video_id: str
    total_duration_s: float = Field(..., gt=0.0)
    tts_tier_used: str
    sample_rate: int = 24000
    scenes: list[AudioScene] = Field(..., min_length=1)

    @classmethod
    def build(
        cls,
        video_id: str,
        scenes: list[AudioScene],
        tts_tier: str,
    ) -> "TimingManifest":
        """
        Factory: assigns start/end timestamps then assembles the manifest.
        Call this after all scenes have been synthesised and measured.
        """
        t = 0.0
        for s in scenes:
            s.start_s = round(t, 3)
            s.end_s = round(t + s.duration_s, 3)
            t += s.duration_s

        return cls(
            video_id=video_id,
            total_duration_s=round(t, 3),
            tts_tier_used=tts_tier,
            scenes=scenes,
        )

    def get_scene(self, scene_id: int) -> AudioScene | None:
        return next((s for s in self.scenes if s.scene_id == scene_id), None)

    def duration_minutes(self) -> float:
        return self.total_duration_s / 60
