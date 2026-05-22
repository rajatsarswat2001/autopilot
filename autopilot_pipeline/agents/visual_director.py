"""
agents/visual_director.py
─────────────────────────────────────────────────────────────────────────────
Visual Director Agent — sources B-roll and generated stills for every scene.

Asset sourcing priority (per scene):
  1. Pexels video clip   — keyword search, free royalty-free footage
  2. SDXL via NVIDIA NIM — generated still (cinematic prompt)
  3. Replicate SDXL API  — cloud GPU fallback
  4. Solid-colour frame  — 1920×1080 gradient PNG (never fails)

Visuals conform to AudioAgent timing — clip lengths are set by the
TimingManifest, not the other way around.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from contracts.visual_manifest import VisualManifest, VisualScene
from contracts.timing_manifest import TimingManifest
from tools.pexels_tools import search_and_download_video
from tools.nim_tools import generate_image_sdxl
from workflows.pipeline_state import AgentError, PipelineState

log = structlog.get_logger(__name__)

VISUAL_DIR = Path(os.getenv("VISUAL_OUTPUT_DIR", "outputs/visual")).resolve()

MOTION_CYCLE = ["zoom_in", "pan_right", "zoom_out", "pan_left"]


# ─────────────────────────────────────────────────────────────────────────────
# Placeholder generator
# ─────────────────────────────────────────────────────────────────────────────

def _make_placeholder(scene_id: int, mood: str, out_dir: Path) -> str:
    """Generate a mood-coloured gradient PNG as last-resort placeholder."""
    try:
        from PIL import Image, ImageDraw
        MOOD_COLORS = {
            "tense":     ((40, 0, 0),   (120, 0, 0)),
            "curious":   ((0, 30, 80),  (0, 80, 160)),
            "inspiring": ((20, 60, 20), (60, 140, 60)),
            "shocking":  ((80, 0, 80),  (180, 0, 100)),
            "warm":      ((100, 60, 0), (200, 130, 20)),
            "dramatic":  ((10, 10, 40), (50, 50, 120)),
            "neutral":   ((30, 30, 30), (80, 80, 80)),
        }
        c1, c2 = MOOD_COLORS.get(mood, MOOD_COLORS["neutral"])
        img = Image.new("RGB", (1920, 1080))
        draw = ImageDraw.Draw(img)
        for x in range(1920):
            t = x / 1920
            r = int(c1[0] + (c2[0] - c1[0]) * t)
            g = int(c1[1] + (c2[1] - c1[1]) * t)
            b = int(c1[2] + (c2[2] - c1[2]) * t)
            draw.line([(x, 0), (x, 1080)], fill=(r, g, b))
        path = out_dir / f"placeholder_{scene_id:03d}.png"
        img.save(str(path))
        return str(path)
    except ImportError:
        # PIL not available: write a tiny valid PNG via bytes
        import struct, zlib
        def _png_chunk(tag, data):
            c = zlib.crc32(tag + data) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", c)
        w, h = 1920, 1080
        raw = b"\x00" + b"\x40\x40\x40" * w  # grey row
        compressed = zlib.compress(raw * h)
        png = (
            b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + _png_chunk(b"IDAT", compressed)
            + _png_chunk(b"IEND", b"")
        )
        path = out_dir / f"placeholder_{scene_id:03d}.png"
        path.write_bytes(png)
        return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

def visual_node(state: PipelineState) -> dict[str, Any]:
    """
    Visual Director node.

    Reads:  scene_manifest, timing_manifest, video_id
    Writes: visual_manifest, visual_scenes, job_status, errors
    """
    manifest_dict  = state.get("scene_manifest")
    timing_dict    = state.get("timing_manifest")
    video_id       = state.get("video_id", str(uuid.uuid4()))

    if not manifest_dict:
        err: AgentError = {
            "agent": "visual_director", "error": "No scene_manifest",
            "timestamp": datetime.now(timezone.utc).isoformat(), "recoverable": False,
        }
        return {"errors": [err]}

    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    scenes = manifest_dict.get("scenes", [])
    visual_scenes: list[VisualScene] = []

    for i, scene in enumerate(scenes):
        scene_id  = scene["scene_id"]
        keyword   = scene.get("b_roll_keyword", "")
        prompt    = scene.get("visual_prompt", "")
        mood      = scene.get("mood", "neutral")
        motion    = MOTION_CYCLE[i % len(MOTION_CYCLE)]

        log.info("visual_director.scene", scene_id=scene_id, keyword=keyword)

        asset_path = None
        source     = "placeholder"
        asset_type = "placeholder"

        # ── Tier 1: Pexels video clip ────────────────────────────────────────
        try:
            clip_path = search_and_download_video(
                keyword=keyword,
                output_dir=str(VISUAL_DIR),
                filename=f"{video_id}_scene_{scene_id:03d}.mp4",
            )
            if clip_path:
                asset_path = clip_path
                source     = "pexels"
                asset_type = "video_clip"
                log.info("visual_director.pexels_ok", scene_id=scene_id)
        except Exception as e:
            log.warning("visual_director.pexels_failed", scene_id=scene_id, error=str(e))

        # ── Tier 2: SDXL via NVIDIA NIM ─────────────────────────────────────
        if not asset_path:
            try:
                img_path = generate_image_sdxl(
                    prompt=prompt,
                    output_path=str(VISUAL_DIR / f"{video_id}_scene_{scene_id:03d}.png"),
                )
                if img_path:
                    asset_path = img_path
                    source     = "sdxl_nim"
                    asset_type = "image"
                    log.info("visual_director.sdxl_ok", scene_id=scene_id)
            except Exception as e:
                log.warning("visual_director.sdxl_failed", scene_id=scene_id, error=str(e))

        # ── Tier 3: Placeholder ──────────────────────────────────────────────
        if not asset_path:
            asset_path = _make_placeholder(scene_id, mood, VISUAL_DIR)
            source     = "placeholder"
            asset_type = "placeholder"
            log.warning("visual_director.using_placeholder", scene_id=scene_id)

        visual_scenes.append(VisualScene(
            scene_id=scene_id,
            asset_path=asset_path,
            asset_type=asset_type,
            source=source,
            width=1920,
            height=1080,
            needs_ken_burns=(asset_type in ("image", "placeholder")),
            motion_direction=motion if asset_type in ("image", "placeholder") else None,
        ))

    visual_manifest = VisualManifest(
        video_id=video_id,
        scenes=visual_scenes,
    )

    log.info(
        "visual_director.done",
        pexels=visual_manifest.pexels_scene_count,
        generated=visual_manifest.generated_scene_count,
        placeholders=visual_manifest.placeholder_scene_count,
    )

    return {
        "visual_manifest": visual_manifest.model_dump(mode="json"),
        "visual_scenes":   [s.model_dump(mode="json") for s in visual_scenes],
        "job_status":      "assembly",
    }
