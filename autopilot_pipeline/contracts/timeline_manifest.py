"""
contracts/timeline_manifest.py
─────────────────────────────────────────────────────────────────────────────
Declarative timeline manifest — the final render specification consumed by
the renderer layer (renderer/timeline_compiler.py → renderer/ffmpeg_builder.py).

DESIGN — Shotstack-style composition:
  Represent the video as a declarative JSON timeline rather than imperative
  FFmpeg commands. Benefits:
    • Partial re-renders — only changed scenes
    • Distributed rendering — clips processed independently
    • Scene caching    — reuse unchanged renders
    • Human editable   — producers can tweak before final render
    • LLM revisable    — compliance/entropy agents can modify in-place
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from contracts.timing_manifest import TimingManifest
    from contracts.visual_manifest import VisualManifest


# ─────────────────────────────────────────────────────────────────────────────
# Sub-objects
# ─────────────────────────────────────────────────────────────────────────────

class TextOverlay(BaseModel):
    """Optional lower-third or caption overlay for a clip."""

    text: str
    font_size: int = 52
    color: str = "#FFFFFF"
    position: Literal["center", "bottom", "top"] = "bottom"
    start_offset_s: float = 0.3    # seconds after clip start
    duration_s: Optional[float] = None  # None = full clip
    fade_in_s: float = 0.25
    fade_out_s: float = 0.25
    bg_color: str = "#00000088"    # semi-transparent background


class MusicTrack(BaseModel):
    """Background music overlaid across the entire video."""

    path: str
    volume: float = 0.10           # low under voiceover
    fade_in_s: float = 2.0
    fade_out_s: float = 4.0
    loop: bool = True


class TimelineClip(BaseModel):
    """
    One composited unit: visual layer + audio layer + optional overlays.
    Maps 1:1 to a scene.
    """

    scene_id: int
    # Timeline position
    start_s: float
    end_s: float
    duration_s: float              # = end_s - start_s

    # Visual track A/B split
    visual_path_A: str
    visual_path_B: str
    visual_type_A: Literal["video_clip", "image", "placeholder"]
    visual_type_B: Literal["video_clip", "image", "placeholder"]
    visual_width: int = 1920
    visual_height: int = 1080
    ken_burns_A: bool = False
    ken_burns_B: bool = False
    motion_direction_A: Optional[Literal["zoom_in", "zoom_out", "pan_left", "pan_right"]] = None
    motion_direction_B: Optional[Literal["zoom_in", "zoom_out", "pan_left", "pan_right"]] = None

    # Audio track
    audio_path: str
    audio_volume: float = 1.0

    # Overlays
    text_overlays: list[TextOverlay] = Field(default_factory=list)

    # Transitions
    transition_in:  Literal["fade", "cut", "dissolve"] = "fade"
    transition_out: Literal["fade", "cut", "dissolve"] = "fade"
    transition_duration_s: float = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# Master manifest
# ─────────────────────────────────────────────────────────────────────────────

class TimelineManifest(BaseModel):
    """
    Complete declarative render specification.
    Assembled by Assembly Agent from TimingManifest + VisualManifest.
    Consumed by renderer/timeline_compiler.py.
    """

    schema_version: str = "1.0"
    video_id: str
    output_path: str
    total_duration_s: float
    fps: int = 30
    width: int = 1920
    height: int = 1080

    clips: list[TimelineClip] = Field(..., min_length=1)
    music_track: Optional[MusicTrack] = None
    watermark_path: Optional[str] = None
    intro_path: Optional[str] = None
    outro_path: Optional[str] = None

    @classmethod
    def from_manifests(
        cls,
        video_id: str,
        output_path: str,
        timing: "TimingManifest",
        visual: "VisualManifest",
    ) -> "TimelineManifest":
        """
        Merge TimingManifest + VisualManifest into a single TimelineManifest.
        Audio pacing drives everything; visuals fill the gaps.
        """
        clips: list[TimelineClip] = []

        for audio_scene in timing.scenes:
            vis_scene = visual.get_scene(audio_scene.scene_id)
            if vis_scene is None:
                continue  # skip missing — should not happen

            clips.append(
                TimelineClip(
                    scene_id=audio_scene.scene_id,
                    start_s=audio_scene.start_s,
                    end_s=audio_scene.end_s,
                    duration_s=audio_scene.duration_s,
                    visual_path_A=vis_scene.asset_path_A,
                    visual_path_B=vis_scene.asset_path_B,
                    visual_type_A="video_clip" if vis_scene.asset_type_A == "video_clip" else "image",
                    visual_type_B="video_clip" if vis_scene.asset_type_B == "video_clip" else "image",
                    visual_width=vis_scene.width,
                    visual_height=vis_scene.height,
                    ken_burns_A=(
                        vis_scene.needs_ken_burns_A
                        and vis_scene.asset_type_A in ("image", "placeholder")
                    ),
                    ken_burns_B=(
                        vis_scene.needs_ken_burns_B
                        and vis_scene.asset_type_B in ("image", "placeholder")
                    ),
                    motion_direction_A=vis_scene.motion_direction_A,
                    motion_direction_B=vis_scene.motion_direction_B,
                    audio_path=audio_scene.audio_path,
                )
            )

        return cls(
            video_id=video_id,
            output_path=output_path,
            total_duration_s=timing.total_duration_s,
            clips=clips,
        )
