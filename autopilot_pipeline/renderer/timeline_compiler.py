"""
renderer/timeline_compiler.py
─────────────────────────────────────────────────────────────────────────────
Converts a declarative TimelineManifest into a concrete render plan
(list of FFmpeg operations) without actually executing anything.

This separation enables:
  • Dry-run validation before committing GPU time
  • Partial re-render: only re-compute changed clips
  • Distributed rendering: distribute clips across workers
  • Cache check: skip rendering clips already in cache

Returns a RenderPlan object consumed by ffmpeg_builder.render_timeline().
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import structlog

from contracts.timeline_manifest import TimelineClip, TimelineManifest
from renderer.cache_manager import ClipCache
from tools.ffmpeg_tools import create_ken_burns_effect, loop_video_to_duration

log = structlog.get_logger(__name__)

SCRATCH_DIR = Path(os.getenv("SCRATCH_DIR", "outputs/video/scratch")).resolve()

ClipStatus = Literal["cached", "needs_render", "needs_loop", "needs_kb"]


@dataclass
class CompiledClip:
    """A single clip that has been resolved and is ready to render."""
    scene_id: int
    visual_path: str          # final video clip path (ready for FFmpeg concat)
    audio_path: str
    duration_s: float
    start_s: float
    end_s: float
    transition_in: str
    transition_out: str
    transition_duration_s: float
    status: ClipStatus = "needs_render"


@dataclass
class RenderPlan:
    """Complete render plan for the final video."""
    video_id: str
    output_path: str
    total_duration_s: float
    fps: int
    width: int
    height: int
    clips: list[CompiledClip] = field(default_factory=list)
    music_track_path: str | None = None
    music_volume: float = 0.10
    music_fade_in: float = 2.0
    music_fade_out: float = 4.0
    watermark_path: str | None = None
    intro_path: str | None = None
    outro_path: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Compiler
# ─────────────────────────────────────────────────────────────────────────────

def compile_timeline(manifest: TimelineManifest) -> RenderPlan:
    """
    Convert TimelineManifest → RenderPlan.

    For each clip:
      1. Check cache — if cached, mark as "cached" and skip render
      2. If visual is an image → plan Ken Burns conversion
      3. If visual is a video shorter than required → plan loop
      4. Otherwise → mark as ready
    """
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    cache = ClipCache()

    plan = RenderPlan(
        video_id=manifest.video_id,
        output_path=manifest.output_path,
        total_duration_s=manifest.total_duration_s,
        fps=manifest.fps,
        width=manifest.width,
        height=manifest.height,
        watermark_path=manifest.watermark_path,
        intro_path=manifest.intro_path,
        outro_path=manifest.outro_path,
    )

    if manifest.music_track:
        plan.music_track_path = manifest.music_track.path
        plan.music_volume     = manifest.music_track.volume
        plan.music_fade_in    = manifest.music_track.fade_in_s
        plan.music_fade_out   = manifest.music_track.fade_out_s

    for clip in manifest.clips:
        compiled = _compile_clip(clip, manifest.fps, cache)
        plan.clips.append(compiled)

    cached_count = sum(1 for c in plan.clips if c.status == "cached")
    log.info(
        "timeline_compiler.done",
        clips=len(plan.clips),
        cached=cached_count,
        to_render=len(plan.clips) - cached_count,
    )
    return plan


def _compile_visual(visual_path: str, visual_type: str, ken_burns: bool, width: int, height: int, duration: float, fps: int, scene_id: int, sub_label: str, motion_direction: Optional[str] = None) -> str:
    out_path = str(SCRATCH_DIR / f"clip_{scene_id:03d}_{sub_label}.mp4")
    if visual_type == "image" or ken_burns:
        try:
            create_ken_burns_effect(
                input_image_path=visual_path,
                output_video_path=out_path,
                duration_s=duration,
                resolution=f"{width}x{height}",
                frame_rate=fps,
                motion=motion_direction,
            )
            return out_path
        except Exception as e:
            log.warning("compiler.kb_failed", scene_id=scene_id, label=sub_label, error=str(e))
    elif visual_type == "video_clip":
        try:
            loop_video_to_duration(visual_path, out_path, duration_s=duration)
            return out_path
        except Exception as e:
            log.warning("compiler.loop_failed", scene_id=scene_id, label=sub_label, error=str(e))
    return visual_path

def _compile_clip(clip: TimelineClip, fps: int, cache: ClipCache) -> CompiledClip:
    """Resolve one TimelineClip into a CompiledClip ready for FFmpeg."""
    scene_id    = clip.scene_id
    duration    = clip.duration_s
    half_dur    = duration / 2.0

    # ── Cache check ───────────────────────────────────────────────────────────
    cache_key   = cache.key(f"{clip.visual_path_A}|{clip.visual_path_B}", clip.audio_path, duration)
    cached_path = cache.get(cache_key)
    if cached_path:
        log.debug("compiler.cache_hit", scene_id=scene_id)
        return CompiledClip(
            scene_id=scene_id,
            visual_path=cached_path,
            audio_path=clip.audio_path,
            duration_s=duration,
            start_s=clip.start_s,
            end_s=clip.end_s,
            transition_in=clip.transition_in,
            transition_out=clip.transition_out,
            transition_duration_s=clip.transition_duration_s,
            status="cached",
        )

    path_a = _compile_visual(clip.visual_path_A, clip.visual_type_A, clip.ken_burns_A, clip.visual_width, clip.visual_height, half_dur, fps, scene_id, "A", clip.motion_direction_A)
    path_b = _compile_visual(clip.visual_path_B, clip.visual_type_B, clip.ken_burns_B, clip.visual_width, clip.visual_height, half_dur, fps, scene_id, "B", clip.motion_direction_B)

    import subprocess
    visual_path = str(SCRATCH_DIR / f"clip_{scene_id:03d}_merged.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", path_a,
                "-i", path_b,
                "-filter_complex",
                f"[0:v]scale={clip.visual_width}:{clip.visual_height}:force_original_aspect_ratio=decrease,pad={clip.visual_width}:{clip.visual_height}:(ow-iw)/2:(oh-ih)/2,setsar=1[v0];"
                f"[1:v]scale={clip.visual_width}:{clip.visual_height}:force_original_aspect_ratio=decrease,pad={clip.visual_width}:{clip.visual_height}:(ow-iw)/2:(oh-ih)/2,setsar=1[v1];"
                f"[v0][v1]concat=n=2:v=1:a=0[v]",
                "-map", "[v]",
                "-t", str(duration),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "20",
                visual_path,
            ],
            check=True, capture_output=True, timeout=120
        )
        status = "needs_render"
    except Exception as e:
        log.error("compiler.merge_failed", scene_id=scene_id, error=str(e))
        visual_path = path_a
        status = "needs_render"

    cache.put(cache_key, visual_path)

    return CompiledClip(
        scene_id=scene_id,
        visual_path=visual_path,
        audio_path=clip.audio_path,
        duration_s=duration,
        start_s=clip.start_s,
        end_s=clip.end_s,
        transition_in=clip.transition_in,
        transition_out=clip.transition_out,
        transition_duration_s=clip.transition_duration_s,
        status=status,
    )
