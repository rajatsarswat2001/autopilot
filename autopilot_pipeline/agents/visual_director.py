"""
agents/visual_director.py
─────────────────────────────────────────────────────────────────────────────
Visual Director Agent — sources AI-generated or B-roll visuals for every scene.

Asset sourcing priority (per clip, Tier 1 through Tier 4):
  Tier 1a: Wan2.1 I2V   — Image-to-Video anchored to a master FLUX keyframe
                          (WAN21_I2V_ENABLED=1 + WAN21_ENABLED=1)
           Eliminates style drift: every clip shares the same visual identity.
  Tier 1b: Wan2.1 T2V   — Pure text-to-video (WAN21_ENABLED=1)
  Tier 2:  Pexels        — keyword-matched stock footage (DISABLE_STOCK != 1)
  Tier 3:  Pollinations  — free FLUX-powered AI still (no API key needed)
  Tier 4:  Placeholder   — mood-coloured gradient PNG (absolute last resort)

I2V Anchor Flow:
  1. generate_anchor_image() — calls Pollinations FLUX to produce a master keyframe
  2. generate_i2v_clip()     — passes the keyframe to WanImageToVideoPipeline
     This forces the diffusion transformer to retain the exact visual identity,
     color palette, and composition of the anchor across all clips in the scene.

Visuals conform to AudioAgent timing — clip lengths are set by the
TimingManifest, not the other way around.

Performance:
  • Tier 2 (Pexels) scenes run in PARALLEL via ThreadPoolExecutor (I/O-bound).
  • Tier 1 (Wan2.1) runs sequentially to maximise GPU utilisation.
  • max_workers defaults to 4; override via VISUAL_PARALLEL_WORKERS env var.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from contracts.visual_manifest import VisualManifest, VisualScene
from contracts.timing_manifest import TimingManifest
from tools.pexels_tools import search_and_download_video
from tools.wan21_tools import (
    generate_video_clip,
    generate_i2v_clip,
    generate_anchor_image,
    is_wan21_available,
)
import requests
from urllib.parse import quote as url_encode
from workflows.pipeline_state import AgentError, PipelineState

log = structlog.get_logger(__name__)

VISUAL_DIR   = Path(os.getenv("VISUAL_OUTPUT_DIR", "outputs/visual")).resolve()
MOTION_CYCLE = ["zoom_in", "pan_right", "zoom_out", "pan_left"]
_MAX_WORKERS = int(os.getenv("VISUAL_PARALLEL_WORKERS", "4"))
# Set WAN21_I2V_ENABLED=1 to enable I2V anchor-locked mode
_I2V_ENABLED = os.getenv("WAN21_I2V_ENABLED", "0").strip() == "1"


# ─────────────────────────────────────────────────────────────────────────────
# Placeholder generator
# ─────────────────────────────────────────────────────────────────────────────

def _make_placeholder(scene_id: int, mood: str, out_dir: Path, split_label: str = "") -> str:
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
        img  = Image.new("RGB", (1920, 1080))
        draw = ImageDraw.Draw(img)
        for x in range(1920):
            t = x / 1920
            r = int(c1[0] + (c2[0] - c1[0]) * t)
            g = int(c1[1] + (c2[1] - c1[1]) * t)
            b = int(c1[2] + (c2[2] - c1[2]) * t)
            draw.line([(x, 0), (x, 1080)], fill=(r, g, b))
        label_suffix = f"_{split_label}" if split_label else ""
        path = out_dir / f"placeholder_{scene_id:03d}{label_suffix}.png"
        img.save(str(path))
        return str(path)
    except ImportError:
        import struct, zlib
        def _png_chunk(tag: bytes, data: bytes) -> bytes:
            crc = zlib.crc32(tag + data) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)
        w, h = 1920, 1080
        raw        = b"\x00" + b"\x40\x40\x40" * w
        compressed = zlib.compress(raw * h)
        png = (
            b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + _png_chunk(b"IDAT", compressed)
            + _png_chunk(b"IEND", b"")
        )
        label_suffix = f"_{split_label}" if split_label else ""
        path = out_dir / f"placeholder_{scene_id:03d}{label_suffix}.png"
        path.write_bytes(png)
        return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# Per-scene sourcing helper (runs inside thread pool)
# ─────────────────────────────────────────────────────────────────────────────

def _source_asset(
    keyword: str,
    prompt: str,
    mood: str,
    out_dir: Path,
    video_id: str,
    scene_id: int,
    split_label: str,
    orientation: str,
    duration_s: float = 5.0,
    niche: str = "default",
    anchor_image_path: str | None = None,   # I2V anchor for style-locked generation
) -> tuple[str, str, str]:
    """
    Source a single visual asset for one clip (A or B) of a scene.
    Returns (path, asset_type, source_label).
    """
    # ── Tier 1a: Wan2.1 I2V — anchor-locked video (eliminates style drift) ────
    if _I2V_ENABLED and is_wan21_available() and prompt and anchor_image_path:
        clip_path = str(out_dir / f"{video_id}_scene_{scene_id:03d}_{split_label}_i2v.mp4")
        try:
            result = generate_i2v_clip(
                prompt=prompt,
                anchor_image_path=anchor_image_path,
                output_path=clip_path,
                duration_s=duration_s,
                niche=niche,
            )
            if result:
                log.info("visual_director.i2v_ok", scene_id=scene_id, label=split_label)
                return result, "video_clip", "wan21"
        except Exception as e:
            log.warning("visual_director.i2v_failed", scene_id=scene_id, error=str(e)[:120])

    # ── Tier 1b: Wan2.1 T2V — pure text-to-video ──────────────────────────────
    if is_wan21_available() and prompt:
        clip_path = str(out_dir / f"{video_id}_scene_{scene_id:03d}_{split_label}_wan.mp4")
        try:
            result = generate_video_clip(
                prompt=prompt,
                output_path=clip_path,
                duration_s=duration_s,
                niche=niche,
            )
            if result:
                log.info("visual_director.wan21_ok", scene_id=scene_id, label=split_label)
                return result, "video_clip", "wan21"
        except Exception as e:
            log.warning("visual_director.wan21_failed_falling_back",
                        scene_id=scene_id, error=str(e)[:120])

    # ── Tier 2: Pexels stock footage ───────────────────────────────────────────
    if os.environ.get("DISABLE_STOCK", "0") != "1":
        try:
            clip_path = search_and_download_video(
                keyword=keyword,
                output_dir=str(out_dir),
                filename=f"{video_id}_scene_{scene_id:03d}_{split_label}.mp4",
                orientation=orientation,
            )
            if clip_path:
                log.info("visual_director.pexels_ok", scene_id=scene_id, label=split_label)
                return clip_path, "video_clip", "pexels"
        except Exception as e:
            log.warning("visual_director.pexels_failed",
                        scene_id=scene_id, label=split_label, error=str(e)[:120])
    else:
        log.info("visual_director.stock_disabled", scene_id=scene_id, label=split_label)

    # ── Tier 3: HuggingFace Inference API — FLUX.1-schnell (free, no rate limit) ─
    # Set HF_TOKEN in .env (free HuggingFace account, no credit card needed)
    hf_token = os.getenv("HF_TOKEN", "")
    if hf_token and prompt:
        try:
            w, h = (1080, 1920) if orientation in ("portrait", "reel", "short") else (1920, 1080)
            hf_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
            hf_resp = requests.post(
                hf_url,
                headers={"Authorization": f"Bearer {hf_token}"},
                json={"inputs": prompt,
                      "parameters": {"width": w, "height": h, "num_inference_steps": 4}},
                timeout=90,
            )
            if hf_resp.status_code == 200:
                img_out = out_dir / f"{video_id}_scene_{scene_id:03d}_{split_label}_hf.png"
                img_out.write_bytes(hf_resp.content)
                if img_out.stat().st_size > 10_000:
                    log.info("visual_director.hf_flux_ok", scene_id=scene_id, label=split_label)
                    return str(img_out), "image", "pollinations"
            else:
                log.warning("visual_director.hf_flux_error",
                            status=hf_resp.status_code, scene_id=scene_id)
        except Exception as e:
            log.warning("visual_director.hf_flux_failed",
                        scene_id=scene_id, label=split_label, error=str(e)[:120])

    # ── Tier 4: Pollinations FLUX still (fallback if HF_TOKEN not set / 402) ──
    try:
        w, h = (1080, 1920) if orientation in ("portrait", "reel", "short") else (1920, 1080)
        encoded  = url_encode(prompt)
        poll_url = (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width={w}&height={h}&nologo=true&model=flux&seed={scene_id}"
        )
        resp = requests.get(poll_url, stream=True, timeout=60)
        resp.raise_for_status()
        img_out = out_dir / f"{video_id}_scene_{scene_id:03d}_{split_label}.png"
        with open(img_out, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        if img_out.stat().st_size > 10_000:
            log.info("visual_director.pollinations_ok", scene_id=scene_id, label=split_label)
            return str(img_out), "image", "pollinations"
    except Exception as e:
        log.warning("visual_director.pollinations_failed",
                    scene_id=scene_id, label=split_label, error=str(e)[:120])

    # ── Tier 5: Placeholder (absolute last resort) ─────────────────────────────
    log.warning("visual_director.using_placeholder", scene_id=scene_id, label=split_label)
    return _make_placeholder(scene_id, mood, out_dir, split_label), "placeholder", "placeholder"

def _source_scene(
    scene: dict,
    scene_index: int,
    video_id: str,
    out_dir: Path,
) -> VisualScene:
    """
    Source all visual assets for a single scene.
    When I2V mode is enabled, first generates a FLUX anchor image then
    passes it into WanImageToVideoPipeline for both A and B clips.
    """
    scene_id = scene["scene_id"]
    mood     = scene.get("emotional_tone", "neutral")
    motion_a = MOTION_CYCLE[(scene_index * 2) % len(MOTION_CYCLE)]
    motion_b = MOTION_CYCLE[(scene_index * 2 + 1) % len(MOTION_CYCLE)]

    log.info("visual_director.scene", scene_id=scene_id, i2v=_I2V_ENABLED)

    orientation        = os.environ.get("FORMAT", "youtube").lower()
    pexels_orientation = "portrait" if orientation in ("reel", "short") else "landscape"
    niche              = os.environ.get("NICHE", "default")

    # ── I2V: Generate master anchor image for this scene ───────────────────
    # The anchor image locks the visual identity across A and B clips.
    anchor_path: str | None = None
    if _I2V_ENABLED and is_wan21_available():
        pr_a = scene.get("visual_prompt_A", "")
        if pr_a:
            anchor_out = str(out_dir / f"{video_id}_scene_{scene_id:03d}_anchor.png")
            anchor_path = generate_anchor_image(
                prompt=pr_a,
                output_path=anchor_out,
                niche=niche,
            )
            if anchor_path:
                log.info("visual_director.anchor_ready",
                         scene_id=scene_id, path=anchor_path)
            else:
                log.warning("visual_director.anchor_failed_continuing",
                            scene_id=scene_id)

    # ── Source A and B clips ────────────────────────────────────────────────
    kw_a = scene.get("b_roll_keyword_A", "")
    pr_a = scene.get("visual_prompt_A", "")
    path_a, type_a, src_a = _source_asset(
        kw_a, pr_a, mood, out_dir, video_id, scene_id, "A",
        pexels_orientation, duration_s=5.0, niche=niche,
        anchor_image_path=anchor_path,
    )

    kw_b = scene.get("b_roll_keyword_B", "")
    pr_b = scene.get("visual_prompt_B", "")
    path_b, type_b, src_b = _source_asset(
        kw_b, pr_b, mood, out_dir, video_id, scene_id, "B",
        pexels_orientation, duration_s=5.0, niche=niche,
        anchor_image_path=anchor_path,
    )

    width, height = (1080, 1920) if orientation in ("reel", "short") else (1920, 1080)

    return VisualScene(
        scene_id=scene_id,
        asset_path_A=path_a, asset_path_B=path_b,
        asset_type_A=type_a, asset_type_B=type_b,
        source_A=src_a, source_B=src_b,
        width=width,
        height=height,
        needs_ken_burns_A=(type_a in ("image", "placeholder")),
        needs_ken_burns_B=(type_b in ("image", "placeholder")),
        motion_direction_A=motion_a if type_a in ("image", "placeholder") else None,
        motion_direction_B=motion_b if type_b in ("image", "placeholder") else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

def visual_node(state: PipelineState) -> dict[str, Any]:
    """
    Visual Director node.

    Reads:  scene_manifest, timing_manifest, video_id
    Writes: visual_manifest, visual_scenes, job_status, errors
    """
    manifest_dict = state.get("scene_manifest")
    video_id      = state.get("video_id", str(uuid.uuid4()))

    if not manifest_dict:
        err: AgentError = {
            "agent": "visual_director", "error": "No scene_manifest",
            "timestamp": datetime.now(timezone.utc).isoformat(), "recoverable": False,
        }
        return {"errors": [err]}

    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    scenes    = manifest_dict.get("scenes", [])

    # Force sequential processing if Wan2.1 is active to prevent VRAM spikes and thread OOMs
    wan21_active = is_wan21_available()
    n_workers = 1 if wan21_active else min(_MAX_WORKERS, max(len(scenes), 1))

    log.info(
        "visual_director.start",
        scenes=len(scenes),
        workers=n_workers,
        mode="sequential" if wan21_active else "parallel",
    )

    # ── Submit all scenes in parallel ─────────────────────────────────────────
    results: dict[int, VisualScene] = {}

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_source_scene, scene, i, video_id, VISUAL_DIR): scene["scene_id"]
            for i, scene in enumerate(scenes)
        }
        for future in as_completed(futures):
            scene_id = futures[future]
            try:
                results[scene_id] = future.result()
            except Exception as e:
                log.error("visual_director.scene_exception", scene_id=scene_id, error=str(e))
                matching = next((s for s in scenes if s["scene_id"] == scene_id), {})
                placeholder_a = _make_placeholder(scene_id, matching.get("emotional_tone", "neutral"), VISUAL_DIR, "A")
                placeholder_b = _make_placeholder(scene_id, matching.get("emotional_tone", "neutral"), VISUAL_DIR, "B")
                results[scene_id] = VisualScene(
                    scene_id=scene_id,
                    asset_path_A=placeholder_a, asset_path_B=placeholder_b,
                    asset_type_A="placeholder", asset_type_B="placeholder",
                    source_A="placeholder", source_B="placeholder",
                    width=1920, height=1080,
                    needs_ken_burns_A=True, needs_ken_burns_B=True,
                    motion_direction_A="zoom_in", motion_direction_B="zoom_out",
                )

    # ── Reassemble in scene order ─────────────────────────────────────────────
    visual_scenes: list[VisualScene] = [
        results[scene["scene_id"]]
        for scene in sorted(scenes, key=lambda s: s["scene_id"])
        if scene["scene_id"] in results
    ]

    visual_manifest = VisualManifest(video_id=video_id, scenes=visual_scenes)

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
