"""
agents/visual_director.py
─────────────────────────────────────────────────────────────────────────
Visual Director Agent — sources AI-generated visuals for every scene.

Asset sourcing priority (per clip, Tier 1 through Tier 4):
  Tier 1: Wan2.1 T2V-1.3B — primary AI video
           ~8.2 GB VRAM peak, 81 frames @ 24fps
           Internally managed by tools/video_gen_tools.py.
  Tier 2: LTX-Video I2V — fallback if anchor provided
           (auto-selected if Wan2.1 fails or as primary I2V)
  Tier 3: CogVideoX-2B — last resort fallback
  Tier 4: Pexels stock footage — keyword-matched (DISABLE_STOCK != 1)
  Tier 5: Pollinations FLUX — free AI still image with Ken Burns animation
  Tier 5: Placeholder            — mood-coloured gradient PNG (absolute last resort)

CRITICAL (T4 GPU):
  T4 (Turing arch) does NOT support bfloat16 natively — it is emulated.
  video_gen_tools always loads models in float16 for native T4 speed.

Performance:
  • Scene-level: sequential (one scene at a time). video_gen_tools handles
    intra-scene A/B clip parallelism across dual GPUs internally.
  • CogVideoX-2B: ~2-4 min per clip on T4 @ 25 steps.
  • LTX-Video fallback: ~60-90 sec per clip @ 8 steps (distilled).
─────────────────────────────────────────────────────────────────────────
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
from tools.stock_tools import search_and_download_stock_clip
from tools.video_gen_tools import (
    generate_video_clip,
    generate_video_pair,
    generate_i2v_clip,
    generate_anchor_image,
    is_video_gen_available,
)
import requests
from urllib.parse import quote as url_encode
from workflows.pipeline_state import AgentError, PipelineState

log = structlog.get_logger(__name__)

VISUAL_DIR   = Path(os.getenv("VISUAL_OUTPUT_DIR", "outputs/visual")).resolve()
MOTION_CYCLE = ["zoom_in", "pan_right", "zoom_out", "pan_left"]
_MAX_WORKERS = int(os.getenv("VISUAL_PARALLEL_WORKERS", "4"))
# I2V mode: generate a FLUX anchor image then animate it with LTX-Video I2V.
# This produces far superior quality vs pure T2V — enabled by default.
_I2V_ENABLED = os.environ.get("WAN21_I2V_ENABLED", "1").strip() == "1"


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
    # ── Tier 1: AI Video (I2V preferred, T2V fallback) ────────────────────────────
    if is_video_gen_available() and prompt:
        clip_path = str(out_dir / f"{video_id}_scene_{scene_id:03d}_{split_label}_ai.mp4")
        try:
            result = None
            src = "cogvideox"
            if anchor_image_path and _I2V_ENABLED:
                result = generate_i2v_clip(
                    prompt=prompt,
                    anchor_image_path=anchor_image_path,
                    output_path=clip_path,
                    duration_s=duration_s,
                    niche=niche,
                )
                src = "ltx"
            
            if not result:
                # Evict I2V pipeline to free VRAM before loading CogVideoX
                from tools.video_gen_tools import _evict_pipeline, _video_device
                import gc, torch
                _evict_pipeline(f"ltx_i2v_{_video_device(1).split(':')[-1]}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                result = generate_video_clip(
                    prompt=prompt,
                    output_path=clip_path,
                    duration_s=duration_s,
                    niche=niche,
                )
                src = os.environ.get("VIDEO_GEN_MODEL", "wan21") if result else "cogvideox"

            if result:
                log.info("visual_director.ai_video_ok", scene_id=scene_id, label=split_label, mode=src)
                return result, "video_clip", src
        except Exception as e:
            log.warning("visual_director.ai_video_failed",
                        scene_id=scene_id, error=str(e)[:120])

    # ── Tier 2: Pexels stock footage ───────────────────────────────────────────
    if os.environ.get("DISABLE_STOCK", "0") != "1":
        try:
            clip_path, provider = search_and_download_stock_clip(
                keyword=keyword,
                output_dir=str(out_dir),
                filename=f"{video_id}_scene_{scene_id:03d}_{split_label}.mp4",
                orientation=orientation,
            )
            if clip_path:
                log.info(f"visual_director.{provider}_ok", scene_id=scene_id, label=split_label)
                return clip_path, "video_clip", provider
        except Exception as e:
            log.warning("visual_director.stock_failed",
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
    import time
    _POLL_ATTEMPTS = 3
    for _attempt in range(1, _POLL_ATTEMPTS + 1):
        try:
            w, h = (1080, 1920) if orientation in ("portrait", "reel", "short") else (1920, 1080)
            encoded  = url_encode(prompt)
            poll_url = (
                f"https://image.pollinations.ai/prompt/{encoded}"
                f"?width={w}&height={h}&nologo=true&model=flux&seed={scene_id}"
            )
            resp = requests.get(poll_url, stream=True, timeout=90)
            resp.raise_for_status()
            img_out = out_dir / f"{video_id}_scene_{scene_id:03d}_{split_label}.png"
            with open(img_out, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            if img_out.stat().st_size > 10_000:
                log.info("visual_director.pollinations_ok", scene_id=scene_id, label=split_label, attempt=_attempt)
                return str(img_out), "image", "pollinations"
        except Exception as e:
            log.warning("visual_director.pollinations_failed_attempt",
                        scene_id=scene_id, label=split_label, attempt=_attempt, error=str(e)[:120])
            if _attempt < _POLL_ATTEMPTS:
                time.sleep(3)

    # ── Tier 5: Placeholder (absolute last resort) ─────────────────────────────
    log.warning("visual_director.using_placeholder", scene_id=scene_id, label=split_label)
    return _make_placeholder(scene_id, mood, out_dir, split_label), "placeholder", "placeholder"

def _source_scene(
    scene: dict,
    scene_index: int,
    video_id: str,
    out_dir: Path,
    timing_scenes: list[dict],
) -> VisualScene:
    scene_id = scene["scene_id"]
    mood     = scene.get("emotional_tone", "neutral")
    prompt_A = scene.get("visual_prompt_A", scene.get("narration", ""))
    prompt_B = scene.get("visual_prompt_B", prompt_A)
    keyword_A = scene.get("b_roll_keyword_A", "abstract")
    keyword_B = scene.get("b_roll_keyword_B", keyword_A)
    niche = scene.get("niche", "default")

    # Extract duration from timing_manifest
    timing = next((t for t in timing_scenes if t["scene_id"] == scene_id), {})
    duration_s = timing.get("duration_s", 5.0)

    orientation = os.environ.get("FORMAT", "youtube").lower()
    width, height = (1080, 1920) if orientation in ("reel", "short") else (1920, 1080)

    # Source A
    anchor_path_A = str(out_dir / f"anchor_{scene_id:03d}_A.png")
    anchor_A = generate_anchor_image(prompt=prompt_A, output_path=anchor_path_A, niche=niche)
    path_A, type_A, source_A = _source_asset(
        keyword=keyword_A, prompt=prompt_A, mood=mood, out_dir=out_dir,
        video_id=video_id, scene_id=scene_id, split_label="A",
        orientation=orientation, duration_s=duration_s, niche=niche, anchor_image_path=anchor_A,
    )

    # Source B
    anchor_path_B = str(out_dir / f"anchor_{scene_id:03d}_B.png")
    anchor_B = generate_anchor_image(prompt=prompt_B, output_path=anchor_path_B, niche=niche)
    path_B, type_B, source_B = _source_asset(
        keyword=keyword_B, prompt=prompt_B, mood=mood, out_dir=out_dir,
        video_id=video_id, scene_id=scene_id, split_label="B",
        orientation=orientation, duration_s=duration_s, niche=niche, anchor_image_path=anchor_B,
    )

    return VisualScene(
        scene_id=scene_id,
        asset_path_A=path_A,
        asset_path_B=path_B,
        asset_type_A=type_A,
        asset_type_B=type_B,
        source_A=source_A,
        source_B=source_B,
        width=width,
        height=height,
        needs_ken_burns_A=False,
        needs_ken_burns_B=False,
        motion_direction_A=None,
        motion_direction_B=None,
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

    import gc
    import torch
    
    # Clear VRAM after audio phase before any video generation begins
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    video_gen_active = is_video_gen_available()
    n_workers = 1

    log.info(
        "visual_director.start",
        scenes=len(scenes),
        workers=n_workers,
        mode="sequential (T2I -> I2V)",
    )

    # ── Submit all scenes in parallel ─────────────────────────────────────────
    results: dict[int, VisualScene] = {}

    timing_manifest = state.get("timing_manifest", {})
    timing_scenes = timing_manifest.get("scenes", [])

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_source_scene, scene, i, video_id, VISUAL_DIR, timing_scenes): scene["scene_id"]
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

    ai_count = sum(
        1 for s in visual_scenes
        if getattr(s, 'source_A', '') in ('cogvideox', 'ltx', 'ltx_i2v', 'wan21', 'wan22_gguf')
        or getattr(s, 'source_B', '') in ('cogvideox', 'ltx', 'ltx_i2v', 'wan21', 'wan22_gguf')
    )
    log.info(
        "visual_director.done",
        ai_generated=ai_count,
        pexels=visual_manifest.pexels_scene_count,
        generated=visual_manifest.generated_scene_count,
        placeholders=visual_manifest.placeholder_scene_count,
    )

    return {
        "visual_manifest": visual_manifest.model_dump(mode="json"),
        "visual_scenes":   [s.model_dump(mode="json") for s in visual_scenes],
        "job_status":      "assembly",
    }
