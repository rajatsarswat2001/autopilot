"""
contracts/visual_manifest.py
─────────────────────────────────────────────────────────────────────────────
Data contract for visual_manifest.json produced by Visual Director.

Source priority (per scene):
  1. Pexels video clip  (free, royalty-free)
  2. SDXL via NIM      (hosted GPU, OpenAI-compat)
  3. SDXL local        (requires local GPU)
  4. Replicate API     (fallback cloud GPU)
  5. Placeholder frame (solid colour — last resort, never fails)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class VisualScene(BaseModel):
    """Visual asset information for one scene."""

    scene_id: int = Field(..., ge=1)
    asset_path: str = Field(..., description="Absolute path to video clip or image file")
    asset_type: Literal["video_clip", "image", "placeholder"]
    source: Literal["pexels", "sdxl_nim", "sdxl_local", "replicate", "placeholder"]
    width: int = 1920
    height: int = 1080
    duration_s: Optional[float] = None  # None for images; duration from timing manifest
    pexels_id: Optional[str] = None
    sdxl_prompt: Optional[str] = None

    # Ken Burns config for static images
    needs_ken_burns: bool = True
    motion_direction: Optional[Literal["zoom_in", "zoom_out", "pan_left", "pan_right"]] = None


class VisualManifest(BaseModel):
    """Master visual manifest produced by Visual Director."""

    schema_version: str = "1.0"
    video_id: str
    output_resolution: str = "1920x1080"
    fps: int = 30
    scenes: list[VisualScene] = Field(..., min_length=1)

    def get_scene(self, scene_id: int) -> VisualScene | None:
        return next((s for s in self.scenes if s.scene_id == scene_id), None)

    @property
    def pexels_scene_count(self) -> int:
        return sum(1 for s in self.scenes if s.source == "pexels")

    @property
    def generated_scene_count(self) -> int:
        return sum(1 for s in self.scenes if s.source in ("sdxl_nim", "sdxl_local", "replicate"))

    @property
    def placeholder_scene_count(self) -> int:
        return sum(1 for s in self.scenes if s.source == "placeholder")
