"""
contracts/scene_manifest.py
─────────────────────────────────────────────────────────────────────────────
Pydantic v2 data contract for the scene_manifest.json produced by Script Agent.

This is the master contract — Audio Agent, Visual Director, Entropy Engine,
and Compliance Agent all read from it.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Maps LLM-invented mood words to valid enum values
_MOOD_ALIASES: dict[str, str] = {
    "hopeful":      "inspiring",
    "optimistic":   "inspiring",
    "uplifting":    "inspiring",
    "positive":     "warm",
    "happy":        "warm",
    "friendly":     "warm",
    "urgent":       "tense",
    "alarming":     "tense",
    "serious":      "tense",
    "scary":        "tense",
    "sad":          "dramatic",
    "emotional":    "dramatic",
    "intense":      "dramatic",
    "informative":  "neutral",
    "educational":  "neutral",
    "analytical":   "neutral",
    "exciting":     "curious",
    "intriguing":   "curious",
    "surprising":   "shocking",
    "unexpected":   "shocking",
}


class Scene(BaseModel):
    """One scene = one TTS narration unit + paired visual asset."""

    scene_id: int = Field(..., ge=1)
    narration: str = Field(
        ..., min_length=20,
        description="Voiceover text; 1–3 natural-sounding sentences"
    )
    visual_prompt_A: str = Field(
        ..., min_length=10,
        description="Detailed image/video generation prompt for the first half of the scene"
    )
    b_roll_keyword_A: str = Field(..., description="2–4 word Pexels search query for the first half")
    visual_prompt_B: str = Field(
        default="",
        description="Detailed image/video generation prompt for the second half of the scene (A/B split)"
    )
    b_roll_keyword_B: str = Field(default="", description="2–4 word Pexels search query for the second half")
    emotional_tone: Literal["tense", "curious", "inspiring", "shocking", "warm", "neutral", "dramatic"] = "neutral"

    @field_validator("emotional_tone", mode="before")
    @classmethod
    def normalise_mood(cls, v: Any) -> str:
        """Silently map LLM-invented mood words to valid enum values."""
        if isinstance(v, str):
            low = v.lower().strip()
            if low in ("tense", "curious", "inspiring", "shocking", "warm", "neutral", "dramatic"):
                return low
            return _MOOD_ALIASES.get(low, "neutral")
        return "neutral"
    emotion_exaggeration: float = Field(0.5, ge=0.0, le=1.0)

    # Filled by downstream agents — None until populated
    duration_hint_s: Optional[float] = Field(None, description="Set by Audio Agent after TTS")
    visual_asset_path_A: Optional[str] = Field(None, description="Set by Visual Director (First half)")
    visual_asset_path_B: Optional[str] = Field(None, description="Set by Visual Director (Second half)")
    audio_path: Optional[str] = Field(None, description="Set by Audio Agent")

    @field_validator("narration")
    @classmethod
    def reject_filler_phrases(cls, v: str) -> str:
        banned = [
            "in conclusion", "as we can see", "it is worth noting",
            "don't forget to subscribe", "welcome to my channel",
            "in this video i will", "today we're going to",
        ]
        low = v.lower()
        for phrase in banned:
            if phrase in low:
                raise ValueError(f"Narration contains banned filler: '{phrase}'")
        return v

    @field_validator("narration")
    @classmethod
    def minimum_word_count(cls, v: str) -> str:
        if len(v.split()) < 10:
            raise ValueError("Narration must be ≥ 10 words")
        return v


class SceneManifest(BaseModel):
    """
    Master contract produced by Script Agent.
    All downstream agents read from this manifest.
    Schema version: 1.1
    """

    schema_version: str = "1.1"
    video_id: str
    title: str = Field(..., min_length=10, max_length=100)
    niche: str
    target_cpm_tier: int = Field(..., ge=1, le=3)
    hook: str = Field(..., min_length=20, description="Opening hook — triggers loss aversion or curiosity")
    scenes: list[Scene] = Field(..., min_length=3, max_length=25)
    call_to_action: str = Field(
        default="What money mistake have YOU made? Drop it in the comments below.",
        min_length=10,
    )
    tags: list[str] = Field(
        default=["personal finance", "money tips", "financial advice"],
        min_length=1,
        max_length=15,
    )

    # Populated after scoring phases
    uniqueness_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    entropy_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    compliance_score: Optional[dict] = None

    @field_validator("title")
    @classmethod
    def reject_spam_title(cls, v: str) -> str:
        if len(re.findall(r"\d+", v)) > 5:
            raise ValueError("Title contains too many numbers (spam signal)")
        return v

    @model_validator(mode="after")
    def scenes_are_sequential(self) -> "SceneManifest":
        ids = [s.scene_id for s in self.scenes]
        expected = list(range(1, len(ids) + 1))
        if ids != expected:
            raise ValueError(
                f"scene_id must be sequential starting at 1. Got: {ids}"
            )
        return self

    @property
    def total_narration_words(self) -> int:
        return sum(len(s.narration.split()) for s in self.scenes)

    @property
    def estimated_duration_s(self) -> float:
        """Rough estimate: 130 WPM average narration speed."""
        return self.total_narration_words / 130 * 60

    def to_pipeline_dict(self) -> dict:
        """Return plain dict for PipelineState.scene_manifest."""
        return self.model_dump(mode="json")
