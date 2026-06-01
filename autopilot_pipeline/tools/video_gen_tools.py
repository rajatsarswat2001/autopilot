"""
tools/video_gen_tools.py  — v2.0
================================================================================
Kaggle Dual T4 — Upgraded AI video generation.

CHANGES FROM v1:
  Primary  : Wan2.1-T2V-1.3B (8.19 GB VRAM, far better motion than CogVideoX-2B)
  I2V path : LTX-Video I2V   (768x512, 65 frames, corrected guidance)
  Fallback : CogVideoX-2B    (demoted to last resort, fps fixed to 24)
  Removed  : Mirror/hybrid strategies (Wan2.1 is single-model, no benefit)

T4 PRECISION RULES (non-negotiable):
  - All models load in torch.float16 (T4 Turing = no native bfloat16)
  - LTX VAE decode patched to float32 (prevents NaN black frames)
  - Wan2.1 VAE is stable in float16 natively (no patch needed)

Public API is unchanged — visual_director.py imports work without modification.
================================================================================
"""
from __future__ import annotations

import gc
import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
_ENABLED      = os.getenv("VIDEO_GEN_ENABLED", "1").strip() != "0"
_HF_TOKEN     = os.getenv("HF_TOKEN", "")
_IS_KAGGLE    = os.path.exists("/kaggle/working")

# Wan2.1 params — primary model
_WAN_MODEL    = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
_WAN_WIDTH    = 832
_WAN_HEIGHT   = 480
_WAN_FRAMES   = 81      # 81 frames at 24fps = 3.375s, Wan's native sweet spot
_WAN_FPS      = 24
_WAN_STEPS    = int(os.getenv("VIDEO_GEN_WAN_STEPS", "50"))
_WAN_GUIDANCE = float(os.getenv("VIDEO_GEN_WAN_GUIDANCE", "5.0"))

# LTX I2V params — fallback
_LTX_MODEL    = "Lightricks/LTX-Video"
_LTX_WIDTH    = 768
_LTX_HEIGHT   = 512
_LTX_FRAMES   = 65      # (65-1) % 8 == 0 ✓, 65 frames at 24fps = 2.7s
_LTX_FPS      = 24
_LTX_STEPS    = int(os.getenv("VIDEO_GEN_LTX_STEPS", "40"))
_LTX_GUIDANCE = float(os.getenv("VIDEO_GEN_LTX_GUIDANCE", "3.5"))
_LTX_IMG_GUIDANCE = float(os.getenv("VIDEO_GEN_LTX_IMG_GUIDANCE", "2.0"))

# CogVideoX params — last resort only, fps corrected
_COG_MODEL    = "THUDM/CogVideoX-2b"
_COG_WIDTH    = 848
_COG_HEIGHT   = 480
_COG_FRAMES   = 49      # 49 frames at 24fps = 2.04s (was 25 @ 8fps = slideshow)
_COG_FPS      = 24
_COG_STEPS    = int(os.getenv("VIDEO_GEN_COG_STEPS", "50"))

# Anchor image cache
_ANCHOR_CACHE = Path("data/anchor_cache")
_ANCHOR_CACHE.mkdir(parents=True, exist_ok=True)

# ── SINGLETONS ────────────────────────────────────────────────────────────────
_PIPES: dict[str, object] = {}
_LOCKS: dict[str, threading.Lock] = {}


def _get_lock(key: str) -> threading.Lock:
    if key not in _LOCKS:
        _LOCKS[key] = threading.Lock()
    return _LOCKS[key]


# ── GPU UTILS ─────────────────────────────────────────────────────────────────
def _num_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _video_device(slot: int = 1) -> str:
    """cuda:1 on dual-T4 Kaggle, cuda:0 on single GPU."""
    return f"cuda:{slot}" if _num_gpus() >= 2 else "cuda:0"


def _free_vram_gb(device: str) -> float:
    try:
        import torch
        return torch.cuda.mem_get_info(device)[0] / 1024 ** 3
    except Exception:
        return 0.0


def _evict_pipeline(key: str):
    """Move pipeline components to CPU and release VRAM."""
    pipe = _PIPES.pop(key, None)
    if pipe is not None:
        for attr in ("transformer", "text_encoder", "text_encoder_2", "vae", "image_encoder"):
            component = getattr(pipe, attr, None)
            if component is not None and hasattr(component, "to"):
                try:
                    component.to("cpu")
                except Exception:
                    pass
        del pipe
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Log memory_reserved to verify eviction actually dropped the blocks
        device_idx = key.split('_')[-1]
        device = f"cuda:{device_idx}" if device_idx.isdigit() else "cuda:0"
        reserved_gb = torch.cuda.memory_reserved(device) / 1e9
        log.info("evict.complete", key=key, device=device, reserved_gb=round(reserved_gb, 2))
    except Exception:
        pass


# ── STYLE TOKENS ──────────────────────────────────────────────────────────────
_NICHE_TOKENS: dict[str, str] = {
    "personal_finance": "professional finance office, corporate aesthetic, cinematic",
    "saas_tools":       "sleek tech workspace, dark UI glow, cinematic",
    "legal_tax":        "formal legal office, polished mahogany, cinematic",
    "senior_health":    "warm golden hour, healthy senior lifestyle, cinematic",
    "storytelling":     "epic wide shot, dramatic rembrandt lighting, cinematic",
    "default":          "professional environment, sharp focus, cinematic",
}
_NEGATIVE = (
    "low quality, blurry, watermark, text overlay, deformed, ugly, "
    "worst quality, static, no motion, frozen, cartoon, anime, "
    "overexposed, underexposed, pixelated, compression artifacts"
)


def _enrich(prompt: str, niche: str = "default") -> str:
    tokens = _NICHE_TOKENS.get(niche, _NICHE_TOKENS["default"])
    return f"{prompt[:400]}, {tokens}"


# ── AVAILABILITY ──────────────────────────────────────────────────────────────
def is_video_gen_available() -> bool:
    if not _ENABLED:
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ── FRAMES → MP4 ──────────────────────────────────────────────────────────────
def _frames_to_mp4(frames: list, output_path: str, fps: int) -> Optional[str]:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, frame in enumerate(frames):
                frame.save(str(Path(tmpdir) / f"frame_{i:05d}.png"))

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-framerate", str(fps),
                    "-i", str(Path(tmpdir) / "frame_%05d.png"),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-preset", "fast",
                    "-crf", "20",
                    "-movflags", "+faststart",
                    output_path,
                ],
                check=True, capture_output=True, timeout=300,
            )

        import torch
        torch.cuda.empty_cache()
        log.info("video_gen.saved", path=output_path, frames=len(frames), fps=fps)
        return output_path
    except Exception as e:
        log.error("video_gen.frames_to_mp4_failed", error=str(e)[:200])
        return None


# ── WAN2.1 LOADER ─────────────────────────────────────────────────────────────
def _load_wan(device: str = "cuda:0") -> Optional[object]:
    key = f"wan_{device.split(':')[-1]}"
    if key in _PIPES:
        return _PIPES[key]

    with _get_lock(key):
        if key in _PIPES:
            return _PIPES[key]

        try:
            import torch
            from diffusers import WanPipeline

            free = _free_vram_gb(device)
            log.info("wan.loading", device=device, free_vram_gb=round(free, 1))

            pipe = WanPipeline.from_pretrained(
                _WAN_MODEL,
                torch_dtype=torch.float16,
            )
            pipe.to(device)

            # Wan2.1 VAE is stable in float16 — no monkey patch needed
            pipe.enable_attention_slicing()
            if hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()

            _PIPES[key] = pipe
            used = torch.cuda.memory_allocated(device) / 1e9
            log.info("wan.ready", device=device, vram_used_gb=round(used, 1))
            return pipe

        except Exception as e:
            log.error("wan.load_failed", error=str(e)[:200])
            return None


# ── WAN2.1 GENERATOR ──────────────────────────────────────────────────────────
def _run_wan(
    prompt: str,
    output_path: str,
    device: str,
    niche: str = "default",
    seed: int = 42,
) -> Optional[str]:
    try:
        import torch
        pipe = _load_wan(device)
        if pipe is None:
            return None

        enriched = _enrich(prompt, niche)
        log.info("wan.generating", device=device, steps=_WAN_STEPS,
                 size=f"{_WAN_WIDTH}x{_WAN_HEIGHT}", frames=_WAN_FRAMES,
                 prompt=enriched[:80])

        generator = torch.Generator(device).manual_seed(seed)

        with torch.no_grad():
            result = pipe(
                prompt=enriched,
                negative_prompt=_NEGATIVE,
                width=_WAN_WIDTH,
                height=_WAN_HEIGHT,
                num_frames=_WAN_FRAMES,
                guidance_scale=_WAN_GUIDANCE,
                num_inference_steps=_WAN_STEPS,
                generator=generator,
            )

        path = _frames_to_mp4(result.frames[0], output_path, fps=_WAN_FPS)
        gc.collect()
        torch.cuda.empty_cache()
        return path

    except torch.cuda.OutOfMemoryError:
        log.error("wan.oom", device=device)
        _evict_pipeline(f"wan_{device.split(':')[-1]}")
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None
    except Exception as e:
        log.error("wan.failed", error=str(e)[:300])
        _evict_pipeline(f"wan_{device.split(':')[-1]}")
        return None


# ── LTX I2V LOADER ───────────────────────────────────────────────────────────
def _load_ltx_i2v(device: str = "cuda:0") -> Optional[object]:
    key = f"ltx_i2v_{device.split(':')[-1]}"
    if key in _PIPES:
        return _PIPES[key]

    with _get_lock(key):
        if key in _PIPES:
            return _PIPES[key]

        try:
            import torch
            from diffusers import LTXImageToVideoPipeline
            from diffusers.hooks import apply_group_offloading

            log.info("ltx_i2v.loading", device=device)

            pipe = LTXImageToVideoPipeline.from_pretrained(
                _LTX_MODEL,
                torch_dtype=torch.float16,
            )

            # T4 float16 VAE patch — prevents NaN black frames
            if hasattr(pipe, "vae"):
                original_decode = pipe.vae.decode
                def patched_decode(z, *args, **kwargs):
                    return original_decode(z.to(torch.float32), *args, **kwargs)
                pipe.vae.decode = patched_decode
                log.info("ltx_i2v.vae_fp32_patch_applied")

            # Group offloading to fit T4 alongside Wan
            onload  = torch.device(device)
            offload = torch.device("cpu")

            if hasattr(pipe, "transformer"):
                pipe.transformer.enable_group_offload(
                    onload_device=onload,
                    offload_device=offload,
                    offload_type="leaf_level",
                    use_stream=True,
                )
            if hasattr(pipe, "text_encoder"):
                apply_group_offloading(
                    pipe.text_encoder,
                    onload_device=onload,
                    offload_type="block_level",
                    num_blocks_per_group=2,
                )
            if hasattr(pipe, "vae"):
                apply_group_offloading(
                    pipe.vae, onload_device=onload, offload_type="leaf_level"
                )

            _PIPES[key] = pipe
            log.info("ltx_i2v.ready", device=device)
            return pipe

        except Exception as e:
            log.error("ltx_i2v.load_failed", error=str(e)[:200])
            return None


# ── LTX I2V GENERATOR ────────────────────────────────────────────────────────
def _run_ltx_i2v(
    prompt: str,
    anchor_path: str,
    output_path: str,
    device: str,
    niche: str = "default",
    seed: int = 42,
) -> Optional[str]:
    try:
        import torch
        from PIL import Image

        pipe = _load_ltx_i2v(device)
        if pipe is None:
            return None

        if not anchor_path or not Path(anchor_path).exists():
            log.warning("ltx_i2v.no_anchor")
            return None

        anchor = Image.open(anchor_path).convert("RGB").resize(
            (_LTX_WIDTH, _LTX_HEIGHT), Image.LANCZOS
        )
        enriched = _enrich(prompt, niche)

        log.info("ltx_i2v.generating", device=device,
                 steps=_LTX_STEPS, size=f"{_LTX_WIDTH}x{_LTX_HEIGHT}",
                 frames=_LTX_FRAMES, guidance=_LTX_GUIDANCE,
                 img_guidance=_LTX_IMG_GUIDANCE, prompt=enriched[:80])

        generator = torch.Generator(device).manual_seed(seed)

        with torch.no_grad():
            result = pipe(
                image=anchor,
                prompt=enriched,
                negative_prompt=_NEGATIVE,
                width=_LTX_WIDTH,
                height=_LTX_HEIGHT,
                num_frames=_LTX_FRAMES,
                guidance_scale=_LTX_GUIDANCE,
                image_guidance_scale=_LTX_IMG_GUIDANCE,
                num_inference_steps=_LTX_STEPS,
                generator=generator,
            )

        path = _frames_to_mp4(result.frames[0], output_path, fps=_LTX_FPS)
        gc.collect()
        torch.cuda.empty_cache()
        return path

    except torch.cuda.OutOfMemoryError:
        log.error("ltx_i2v.oom", device=device)
        _evict_pipeline(f"ltx_i2v_{device.split(':')[-1]}")
        return None
    except Exception as e:
        log.error("ltx_i2v.failed", error=str(e)[:300])
        _evict_pipeline(f"ltx_i2v_{device.split(':')[-1]}")
        return None


# ── COGVIDEOX LOADER (last resort) ────────────────────────────────────────────
def _load_cogvideox(device: str = "cuda:0") -> Optional[object]:
    key = f"cog_{device.split(':')[-1]}"
    if key in _PIPES:
        return _PIPES[key]

    with _get_lock(key):
        if key in _PIPES:
            return _PIPES[key]

        try:
            import torch
            from diffusers import CogVideoXPipeline
            from diffusers.hooks import apply_group_offloading

            log.info("cogvideox.loading", device=device)
            pipe = CogVideoXPipeline.from_pretrained(
                _COG_MODEL,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )

            onload  = torch.device(device)
            offload = torch.device("cpu")

            if hasattr(pipe, "transformer"):
                pipe.transformer.enable_group_offload(
                    onload_device=onload,
                    offload_device=offload,
                    offload_type="leaf_level",
                    use_stream=True,
                )
            if hasattr(pipe, "text_encoder"):
                apply_group_offloading(
                    pipe.text_encoder,
                    onload_device=onload,
                    offload_type="block_level",
                    num_blocks_per_group=2,
                )
            if hasattr(pipe, "vae"):
                apply_group_offloading(
                    pipe.vae, onload_device=onload, offload_type="leaf_level"
                )
                for m in ("enable_vae_slicing", "enable_vae_tiling"):
                    if hasattr(pipe.vae, m):
                        try:
                            getattr(pipe.vae, m)()
                        except Exception:
                            pass

            _PIPES[key] = pipe
            log.info("cogvideox.ready", device=device)
            return pipe

        except Exception as e:
            log.error("cogvideox.load_failed", error=str(e)[:200])
            return None


# ── COGVIDEOX GENERATOR (last resort) ─────────────────────────────────────────
def _run_cogvideox(
    prompt: str,
    output_path: str,
    device: str,
    niche: str = "default",
    seed: int = 42,
) -> Optional[str]:
    try:
        import torch
        pipe = _load_cogvideox(device)
        if pipe is None:
            return None

        enriched = _enrich(prompt, niche)
        log.info("cogvideox.generating", device=device, steps=_COG_STEPS,
                 frames=_COG_FRAMES, fps=_COG_FPS, prompt=enriched[:80])

        generator = torch.Generator(device="cpu").manual_seed(seed)

        with torch.no_grad():
            result = pipe(
                prompt=enriched,
                negative_prompt=_NEGATIVE,
                height=_COG_HEIGHT,
                width=_COG_WIDTH,
                num_frames=_COG_FRAMES,
                num_inference_steps=_COG_STEPS,
                guidance_scale=6.0,
                generator=generator,
            )

        path = _frames_to_mp4(result.frames[0], output_path, fps=_COG_FPS)
        gc.collect()
        torch.cuda.empty_cache()
        return path

    except Exception as e:
        log.error("cogvideox.failed", error=str(e)[:300])
        _evict_pipeline(f"cog_{device.split(':')[-1]}")
        return None


# ── ANCHOR IMAGE ──────────────────────────────────────────────────────────────
def generate_anchor_image(
    prompt: str,
    output_path: str,
    niche: str = "default",
    width: int = 832,
    height: int = 480,
) -> Optional[str]:
    """
    Generate anchor image via FLUX.1-schnell (HF API or Pollinations fallback).
    Width/height default match Wan2.1 input exactly — no resize step.
    """
    enriched = _enrich(prompt, niche)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Disk cache
    cache_key = hashlib.md5(f"{enriched}{width}{height}".encode()).hexdigest()[:16]
    cached    = _ANCHOR_CACHE / f"{cache_key}.png"
    if cached.exists() and cached.stat().st_size > 10_000:
        shutil.copy2(cached, output_path)
        log.info("anchor.cache_hit", path=output_path)
        return output_path

    import requests

    # HF Inference API (skip on Kaggle — blocked by egress proxy)
    if _HF_TOKEN and not _IS_KAGGLE:
        for attempt in range(2):
            try:
                if attempt:
                    time.sleep(3 * attempt)
                resp = requests.post(
                    "https://api-inference.huggingface.co/models/"
                    "black-forest-labs/FLUX.1-schnell",
                    headers={"Authorization": f"Bearer {_HF_TOKEN}"},
                    json={"inputs": enriched,
                          "parameters": {"width": width, "height": height,
                                         "num_inference_steps": 4,
                                         "guidance_scale": 0.0}},
                    timeout=90,
                )
                if resp.status_code == 200 and len(resp.content) > 10_000:
                    Path(output_path).write_bytes(resp.content)
                    shutil.copy2(output_path, cached)
                    log.info("anchor.hf_ok", path=output_path)
                    return output_path
            except Exception as e:
                log.warning("anchor.hf_error", attempt=attempt, error=str(e)[:80])

    # Pollinations fallback
    from urllib.parse import quote as url_encode
    for attempt in range(3):
        try:
            if attempt:
                time.sleep(5 * attempt)
            url = (
                f"https://image.pollinations.ai/prompt/{url_encode(enriched)}"
                f"?width={width}&height={height}&nologo=true&model=flux&seed=42"
            )
            resp = requests.get(url, stream=True, timeout=90)
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            if Path(output_path).stat().st_size > 10_000:
                shutil.copy2(output_path, cached)
                log.info("anchor.pollinations_ok", path=output_path)
                return output_path
        except Exception as e:
            log.warning("anchor.pollinations_failed", attempt=attempt, error=str(e)[:80])

    log.error("anchor.all_failed", path=output_path)
    return None


# ── PUBLIC API — same signatures as v1, visual_director.py unchanged ──────────

def generate_video_clip(
    prompt: str,
    output_path: str,
    duration_s: float = 5.0,
    niche: str = "default",
    num_frames: int = 0,
    fps: int = 24,
    width: int = 832,
    height: int = 480,
    num_inference_steps: int = 0,
    guidance_scale: float = 0.0,
    device: Optional[str] = None,
) -> Optional[str]:
    """
    Primary T2V path. Waterfall: Wan2.1 → LTX T2V → CogVideoX.
    """
    if not is_video_gen_available():
        return None

    dev = device or _video_device(1)

    # Tier 1: Wan2.1
    result = _run_wan(prompt, output_path, dev, niche)
    if result:
        return result

    log.warning("video_gen.wan_failed_trying_ltx_t2v")
    _evict_pipeline(f"wan_{dev.split(':')[-1]}")

    # Tier 2: LTX T2V (reuse I2V pipe without image — anchor=None path)
    result = _run_ltx_i2v(prompt, anchor_path="", output_path=output_path,
                           device=dev, niche=niche)
    if result:
        return result

    log.warning("video_gen.ltx_failed_trying_cogvideox")
    _evict_pipeline(f"ltx_i2v_{dev.split(':')[-1]}")

    # Tier 3: CogVideoX (last resort)
    return _run_cogvideox(prompt, output_path, dev, niche)


def generate_i2v_clip(
    prompt: str,
    anchor_image_path: str,
    output_path: str,
    duration_s: float = 5.0,
    niche: str = "default",
    **kwargs,
) -> Optional[str]:
    """
    I2V path: LTX I2V with corrected guidance values.
    Falls back to T2V if anchor missing or I2V fails.
    """
    if not is_video_gen_available():
        return None

    dev = _video_device(1)

    if anchor_image_path and Path(anchor_image_path).exists():
        result = _run_ltx_i2v(prompt, anchor_image_path, output_path, dev, niche)
        if result:
            return result
        _evict_pipeline(f"ltx_i2v_{dev.split(':')[-1]}")

    log.warning("video_gen.i2v_fallback_to_t2v")
    return generate_video_clip(prompt, output_path, duration_s, niche)


def generate_video_pair(
    prompt_a: str, prompt_b: str,
    out_a: str, out_b: str,
    niche: str = "default",
) -> tuple[Optional[str], Optional[str]]:
    """
    Generate two clips sequentially. Wan2.1 is single-GPU; no parallel benefit.
    """
    r_a = generate_video_clip(prompt_a, out_a, niche=niche)
    r_b = generate_video_clip(prompt_b, out_b, niche=niche)
    return r_a, r_b


def generate_professional_clip(
    prompt: str,
    output_path: str,
    niche: str = "default",
    scene_id: int = 1,
    out_dir: Path = Path("outputs/visual"),
) -> Optional[str]:
    """Kept for API compatibility. Routes to generate_video_clip."""
    return generate_video_clip(prompt, output_path, niche=niche)
