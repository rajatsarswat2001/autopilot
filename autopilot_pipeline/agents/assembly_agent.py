"""
agents/assembly_agent.py
─────────────────────────────────────────────────────────────────────────────
Assembly Agent — merges timing + visual manifests into a TimelineManifest,
drives the renderer, and runs QA checks on the final video.

Responsibilities:
  1. Build TimelineManifest from TimingManifest + VisualManifest
  2. Optionally add background music track
  3. Invoke renderer/timeline_compiler → renderer/ffmpeg_builder
  4. Run QA: duration check, resolution check, audio level check
  5. Write final_video_path + thumbnail_path to state
─────────────────────────────────────────────────────────────────────────────
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
from tools.caption_tools import generate_captions
from tools.ffmpeg_tools import (
    measure_video_duration,
    get_video_resolution,
    measure_loudness_lufs,
    generate_thumbnail,
)
from workflows.pipeline_state import AgentError, PipelineState

log = structlog.get_logger(__name__)

OUTPUT_DIR   = Path(os.getenv("VIDEO_OUTPUT_DIR",  "outputs/video")).resolve()
MUSIC_TRACKS = list(Path("data/assets/music").glob("*.mp3")) if Path("data/assets/music").exists() else []

QA_MIN_DURATION_S    = 30.0     # minimum acceptable video length
QA_MAX_DURATION_SKEW = 0.15     # 15% tolerance vs. timing manifest total
QA_MIN_LUFS          = -70.0    # safety net: catches truly silent videos only
QA_MAX_LUFS          = -6.0     # hard ceiling to avoid clipping


# ─────────────────────────────────────────────────────────────────────────────
# QA
# ─────────────────────────────────────────────────────────────────────────────

def _qa_check(video_path: str, expected_duration_s: float) -> tuple[bool, str]:
    """Run post-render QA. Returns (passed, notes)."""
    notes: list[str] = []

    # Duration check
    actual_dur = measure_video_duration(video_path)
    if actual_dur < QA_MIN_DURATION_S:
        notes.append(f"Video too short: {actual_dur:.1f}s < {QA_MIN_DURATION_S}s minimum")
    skew = abs(actual_dur - expected_duration_s) / max(expected_duration_s, 1)
    if skew > QA_MAX_DURATION_SKEW:
        notes.append(
            f"Duration skew {skew:.1%} > {QA_MAX_DURATION_SKEW:.0%} "
            f"(expected {expected_duration_s:.1f}s, got {actual_dur:.1f}s)"
        )

    # Resolution check (warn only — Pexels clips may vary)
    resolution = get_video_resolution(video_path)
    if resolution and resolution not in ("1920x1080", "1080x1920", "1280x720"):
        log.warning("assembly_agent.resolution_mismatch", resolution=resolution)

    # Loudness check
    lufs = measure_loudness_lufs(video_path)
    if lufs is not None:
        if lufs < QA_MIN_LUFS:
            notes.append(f"Audio too quiet: {lufs:.1f} LUFS < {QA_MIN_LUFS} floor")
        elif lufs > QA_MAX_LUFS:
            notes.append(f"Audio too loud: {lufs:.1f} LUFS > {QA_MAX_LUFS} ceiling")

    passed = len(notes) == 0
    return passed, " | ".join(notes) if notes else "All QA checks passed"


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

def assembly_node(state: PipelineState) -> dict[str, Any]:
    """
    Assembly Agent node.

    Reads:  timing_manifest, visual_manifest, scene_manifest, video_id
    Writes: timeline_manifest, final_video_path, thumbnail_path,
            qa_passed, qa_notes, errors
    """
    timing_dict  = state.get("timing_manifest")
    visual_dict  = state.get("visual_manifest")
    manifest_dict = state.get("scene_manifest", {})
    video_id     = state.get("video_id", str(uuid.uuid4()))

    if not timing_dict or not visual_dict:
        err: AgentError = {
            "agent": "assembly",
            "error": "Missing timing_manifest or visual_manifest",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recoverable": False,
        }
        return {"errors": [err]}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timing = TimingManifest(**timing_dict)
    visual = VisualManifest(**visual_dict)

    output_path = str(OUTPUT_DIR / f"{video_id}.mp4")

    # ── Build TimelineManifest ────────────────────────────────────────────────
    timeline = TimelineManifest.from_manifests(
        video_id=video_id,
        output_path=output_path,
        timing=timing,
        visual=visual,
    )

    # Add background music if available
    if MUSIC_TRACKS:
        import random
        timeline.music_track = MusicTrack(
            path=str(random.choice(MUSIC_TRACKS)),
            volume=0.10,
            fade_in_s=2.0,
            fade_out_s=4.0,
        )

    log.info("assembly_agent.timeline_built",
             clips=len(timeline.clips), duration=timeline.total_duration_s)

    # ── Generate captions (ASS subtitle file) ─────────────────────────────────
    niche = state.get("niche", "default")
    caption_path = generate_captions(
        scene_manifest=manifest_dict,
        timing_manifest=timing_dict,
        output_dir=str(OUTPUT_DIR),
        niche=niche,
        video_id=video_id,
    )
    if caption_path:
        log.info("assembly_agent.captions_ready", path=caption_path)

    # ── Render ────────────────────────────────────────────────────────────────
    try:
        compiled_plan = compile_timeline(timeline)
        render_timeline(compiled_plan, output_path=output_path,
                        caption_ass_path=caption_path)
    except Exception as e:
        err = {
            "agent": "assembly",
            "error": f"Render failed: {e}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recoverable": False,
        }
        log.error("assembly_agent.render_failed", error=str(e))
        return {"errors": [err]}

    # ── Generate thumbnail ────────────────────────────────────────────────────
    thumb_path = str(OUTPUT_DIR / f"{video_id}_thumb.jpg")
    try:
        # Grab frame from first visual scene
        first_visual = visual.scenes[0].asset_path if visual.scenes else None
        title        = manifest_dict.get("title", "")
        generate_thumbnail(
            video_or_image_path=first_visual or output_path,
            output_path=thumb_path,
            title_text=title,
        )
    except Exception as e:
        log.warning("assembly_agent.thumbnail_failed", error=str(e))
        thumb_path = None

    # ── QA ────────────────────────────────────────────────────────────────────
    qa_passed, qa_notes = _qa_check(output_path, timing.total_duration_s)
    log.info("assembly_agent.qa", passed=qa_passed, notes=qa_notes)

    return {
        "timeline_manifest": timeline.model_dump(mode="json"),
        "final_video_path":  output_path,
        "thumbnail_path":    thumb_path,
        "qa_passed":         qa_passed,
        "qa_notes":          qa_notes,
        "job_status":        "upload" if qa_passed else "failed",
    }
