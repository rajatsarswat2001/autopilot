"""
contracts/visual_manifest.py
─────────────────────────────────────────────────────────────────────────────
Data contract for visual_manifest.json produced by Visual Director.

Source priority (per scene):
  1. Wan2.1 1.3B T2V   (Kaggle T4 GPU, cuda:1, bfloat16)
  2. Pexels video clip  (free, royalty-free stock — when DISABLE_STOCK != 1)
  3. Pollinations FLUX  (free AI image via API, no key required)
  4. SDXL via NIM       (hosted GPU, OpenAI-compat API)
  5. SDXL local         (requires local GPU)
  6. Replicate API      (fallback cloud GPU)
  7. Placeholder frame  (solid-colour gradient PNG — absolute last resort)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class VisualScene(BaseModel):
    """Visual asset information for one scene."""

    scene_id: int = Field(..., ge=1)
    asset_path_A: str = Field(..., description="Absolute path to first video clip or image file")
    asset_path_B: str = Field(..., description="Absolute path to second video clip or image file")
    asset_type_A: Literal["video_clip", "image", "placeholder"]
    asset_type_B: Literal["video_clip", "image", "placeholder"]
    source_A: Literal["ltx", "wan21", "pexels", "pixabay", "pollinations", "sdxl_nim", "sdxl_local", "replicate", "placeholder"]
    source_B: Literal["ltx", "wan21", "pexels", "pixabay", "pollinations", "sdxl_nim", "sdxl_local", "replicate", "placeholder"]
    width: int = 1920
    height: int = 1080
    duration_s: Optional[float] = None  # None for images; duration from timing manifest
    
    # Ken Burns config for static images
    needs_ken_burns_A: bool = True
    needs_ken_burns_B: bool = True
    motion_direction_A: Optional[Literal["zoom_in", "zoom_out", "pan_left", "pan_right"]] = None
    motion_direction_B: Optional[Literal["zoom_in", "zoom_out", "pan_left", "pan_right"]] = None


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
        return sum(1 for s in self.scenes if s.source_A == "pexels" or s.source_B == "pexels")

    @property
    def generated_scene_count(self) -> int:
        _generated = {"ltx", "wan21", "pollinations", "sdxl_nim", "sdxl_local", "replicate"}
        return sum(1 for s in self.scenes if s.source_A in _generated or s.source_B in _generated)

    @property
    def placeholder_scene_count(self) -> int:
        return sum(1 for s in self.scenes if s.source_A == "placeholder" or s.source_B == "placeholder")
