"""
agents/assembly_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Assembly nodes — broken into 3 parts to isolate failures.
1. timeline_node: builds manifests and captions
2. render_node: executes FFmpeg
3. qa_thumbnail_node: runs QA and makes thumbnail
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from contracts.timeline_manifest import TimelineManifest, MusicTrack
from contracts.timing_manifest import TimingManifest
from contracts.visual_manifest import VisualManifest
from renderer.timeline_compiler import compile_timeline
from renderer.ffmpeg_builder import render_timeline
from renderer.thumbnail_generator import create_thumbnail
from tools.caption_tools import generate_captions
from tools.music_tools import generate_background_music, get_dominant_mood
from tools.ffmpeg_tools import (
    measure_video_duration,
    get_video_resolution,
    measure_loudness_lufs,
)
from workflows.pipeline_state import AgentError, PipelineState

log = structlog.get_logger(__name__)

OUTPUT_DIR   = Path(os.getenv("VIDEO_OUTPUT_DIR",  "outputs/video")).resolve()

QA_MIN_DURATION_S    = 30.0     # minimum acceptable video length
QA_MAX_DURATION_SKEW = 0.15     # 15% tolerance vs. timing manifest total
QA_MIN_LUFS          = -70.0    # safety net: catches truly silent videos only
QA_MAX_LUFS          = -6.0     # hard ceiling to avoid clipping


def _qa_check(video_path: str, expected_duration_s: float) -> tuple[bool, str]:
    notes: list[str] = []

    actual_dur = measure_video_duration(video_path)
    if actual_dur < QA_MIN_DURATION_S:
        notes.append(f"Video too short: {actual_dur:.1f}s < {QA_MIN_DURATION_S}s minimum")
    skew = abs(actual_dur - expected_duration_s) / max(expected_duration_s, 1)
    if skew > QA_MAX_DURATION_SKEW:
        notes.append(
            f"Duration skew {skew:.1%} > {QA_MAX_DURATION_SKEW:.0%} "
            f"(expected {expected_duration_s:.1f}s, got {actual_dur:.1f}s)"
        )

    resolution = get_video_resolution(video_path)
    if resolution and resolution not in ("1920x1080", "1080x1920", "1280x720"):
        log.warning("assembly_agent.resolution_mismatch", resolution=resolution)

    lufs = measure_loudness_lufs(video_path)
    if lufs is not None:
        if lufs < QA_MIN_LUFS:
            notes.append(f"Audio too quiet: {lufs:.1f} LUFS < {QA_MIN_LUFS} floor")
        elif lufs > QA_MAX_LUFS:
            notes.append(f"Audio too loud: {lufs:.1f} LUFS > {QA_MAX_LUFS} ceiling")

    passed = len(notes) == 0
    return passed, " | ".join(notes) if notes else "All QA checks passed"


def timeline_node(state: PipelineState) -> dict[str, Any]:
    timing_dict  = state.get("timing_manifest")
    visual_dict  = state.get("visual_manifest")
    manifest_dict = state.get("scene_manifest", {})
    video_id     = state.get("video_id", str(uuid.uuid4()))

    if not timing_dict or not visual_dict:
        err: AgentError = {
            "agent": "timeline",
            "error": "Missing timing_manifest or visual_manifest",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recoverable": False,
        }
        return {"errors": [err], "job_status": "failed"}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timing = TimingManifest(**timing_dict)
    visual = VisualManifest(**visual_dict)
    output_path = str(OUTPUT_DIR / f"{video_id}.mp4")

    timeline = TimelineManifest.from_manifests(
        video_id=video_id,
        output_path=output_path,
        timing=timing,
        visual=visual,
    )

    mood = get_dominant_mood(manifest_dict)
    music_path = generate_background_music(
        mood=mood,
        duration_s=timeline.total_duration_s + 5.0,
        video_id=video_id,
    )
    if music_path:
        timeline.music_track = MusicTrack(
            path=music_path,
            volume=0.10,
            fade_in_s=2.0,
            fade_out_s=4.0,
        )
        log.info("timeline_node.music_ready", path=music_path, mood=mood)

    niche = state.get("target_niche", "default")
    caption_path = generate_captions(
        scene_manifest=manifest_dict,
        timing_manifest=timing_dict,
        output_dir=str(OUTPUT_DIR),
        niche=niche,
        video_id=video_id,
    )

    return {
        "timeline_manifest": timeline.model_dump(mode="json"),
        "final_video_path":  output_path,
        "caption_path":      caption_path, # Passed down to render
        "job_status":        "render",
    }


def render_node(state: PipelineState) -> dict[str, Any]:
    timeline_dict = state.get("timeline_manifest")
    if not timeline_dict:
        return {"job_status": "failed"}

    timeline = TimelineManifest(**timeline_dict)
    output_path = state.get("final_video_path")
    caption_path = state.get("caption_path") # Requires state change

    try:
        compiled_plan = compile_timeline(timeline)
        render_timeline(compiled_plan, output_path=output_path, caption_ass_path=caption_path)
        log.info("render_node.success", path=output_path)
        return {"job_status": "qa_thumbnail"}
    except Exception as e:
        err = {
            "agent": "render",
            "error": f"Render failed: {e}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recoverable": False,
        }
        log.error("render_node.failed", error=str(e))
        return {"errors": [err], "job_status": "failed"}


def qa_thumbnail_node(state: PipelineState) -> dict[str, Any]:
    output_path = state.get("final_video_path")
    timeline_dict = state.get("timeline_manifest")
    manifest_dict = state.get("scene_manifest", {})
    niche = state.get("target_niche", "default")
    video_id = state.get("video_id", str(uuid.uuid4()))

    if not output_path or not timeline_dict:
        return {"job_status": "failed"}

    thumb_path = str(OUTPUT_DIR / f"{video_id}_thumb.jpg")
    try:
        title = manifest_dict.get("title", "")
        result = create_thumbnail(output_path, title, thumb_path, niche=niche)
        if result is None:
            thumb_path = None
    except Exception as e:
        log.warning("qa_thumbnail_node.thumbnail_failed", error=str(e))
        thumb_path = None

    timeline = TimelineManifest(**timeline_dict)
    qa_passed, qa_notes = _qa_check(output_path, timeline.total_duration_s)
    log.info("qa_thumbnail_node.qa", passed=qa_passed, notes=qa_notes)

    return {
        "thumbnail_path": thumb_path,
        "qa_passed":      qa_passed,
        "qa_notes":       qa_notes,
        "job_status":     "seo" if qa_passed else "failed",
    }
