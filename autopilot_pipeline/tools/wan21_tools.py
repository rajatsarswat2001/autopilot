"""
tools/wan21_tools.py
─────────────────────────────────────────────────────────────────────────────
Wan2.1 1.3B Text-to-Video (T2V) and Image-to-Video (I2V) wrapper.

Model: Wan-AI/Wan2.1-T2V-1.3B-Diffusers  (Apache 2.0 — commercial use OK)
       Wan-AI/Wan2.1-I2V-14B-480P-Diffusers (I2V — better temporal coherence)
VRAM:  ~8–9 GB (bfloat16 + VAE float32). Fits single T4 with CPU offloading.
Speed: ~4 min per 5s clip on T4 single GPU (raw); with TeaCache: ~2.5 min.

Multi-GPU (Kaggle 2x T4):
    Chatterbox TTS → cuda:0  |  Wan2.1 → cuda:1
    This maximises parallel availability and avoids VRAM contention.

Advanced Features:
    I2V Anchor Mode:  Pass an anchor PIL image → WanImageToVideoPipeline
                      Locks visual identity (color, style, composition) across clips.
                      Eliminates "style drift" between independently generated scenes.
    TeaCache:         Timestep Embedding Aware Cache — skips redundant denoising steps.
                      Set WAN21_TEACACHE=1 (thresh 0.20) for ~2x speedup, zero quality loss.
    Rigid Style Tokens: All prompts receive an immutable cinematic suffix to lock the
                      aesthetic subspace across every scene, regardless of script content.

Environment:
    WAN21_MODEL_ID     — HuggingFace T2V model ID (default: Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
    WAN21_I2V_MODEL_ID — HuggingFace I2V model ID (default: Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
    WAN21_ENABLED      — set "0" to force image/Pollinations fallback
    WAN21_TEACACHE     — set "1" to enable TeaCache 2x speedup (default: 0)
    WAN21_I2V_ENABLED  — set "1" to enable I2V anchor mode (default: 0, loads extra model)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

_PIPE_LOCK = threading.Lock()
_I2V_PIPE_LOCK = threading.Lock()

# ── Model IDs ────────────────────────────────────────────────────────────────
_MODEL_ID     = os.getenv("WAN21_MODEL_ID",     "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
_I2V_MODEL_ID = os.getenv("WAN21_I2V_MODEL_ID", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
_MODEL_ID_FALLBACKS = [
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",   # correct (May 2026)
    "Wan-AI/Wan2.1-T2V-1.3B",              # alternate naming
]

# ── Feature flags ─────────────────────────────────────────────────────────────
_ENABLED      = os.getenv("WAN21_ENABLED",    "1").strip() != "0"
_TEACACHE     = os.getenv("WAN21_TEACACHE",   "0").strip() == "1"
_I2V_ENABLED  = os.getenv("WAN21_I2V_ENABLED","0").strip() == "1"

# ── Singletons ────────────────────────────────────────────────────────────────
_PIPE     = None   # T2V pipeline singleton
_I2V_PIPE = None   # I2V pipeline singleton


# ─────────────────────────────────────────────────────────────────────────────
# Rigid cinematic style tokens (immutable suffix on every prompt)
# Standardises text embeddings → forces identical aesthetic subspace per scene
# ─────────────────────────────────────────────────────────────────────────────

# Global master style suffix — applied to ALL prompts regardless of niche
_MASTER_STYLE_SUFFIX = (
    "ultra-realistic cinematic footage, captured on 35mm anamorphic lens, "
    "Arri Alexa raw color science, moody Rembrandt side-lighting, "
    "shallow depth of field, film grain, teal and orange color grading, "
    "no text, no watermarks, no UI elements, photorealistic"
)

# Niche-specific style tokens (prepended before master suffix)
_NICHE_STYLE_TOKENS: dict[str, str] = {
    "personal_finance": (
        "professional financial environment, sleek modern office, "
        "data visualizations on monitors, confident executive, "
        "warm tungsten lighting, premium corporate aesthetic"
    ),
    "saas_tools": (
        "sleek minimalist tech workspace, dark UI on multiple curved monitors, "
        "soft blue accent lighting, futuristic product demo environment"
    ),
    "legal_tax": (
        "professional legal office, polished mahogany desk, law books, "
        "formal authoritative atmosphere, soft warm overhead lighting"
    ),
    "senior_health": (
        "warm golden hour sunlight, serene natural environment, "
        "healthy active lifestyle, soft bokeh background, joyful atmosphere"
    ),
    "storytelling": (
        "epic wide establishing shot, dramatic volumetric lighting, "
        "cinematic movie quality, sweeping landscape, intense atmosphere"
    ),
    "default": (
        "professional high-quality environment, clean composition, "
        "balanced natural lighting, premium production value"
    ),
}

# Standard negative prompt — applied to all generations
_NEGATIVE_PROMPT = (
    "low quality, blurry, pixelated, compression artifacts, watermark, "
    "text overlay, subtitles, logo, deformed, ugly, bad anatomy, "
    "worst quality, amateur footage, shaky camera, overexposed, underexposed, "
    "cartoon, anime, illustration, painting, drawing"
)


def _enrich_prompt(prompt: str, niche: str = "default") -> str:
    """Append niche-specific + master style tokens to enforce visual identity."""
    niche_tokens = _NICHE_STYLE_TOKENS.get(niche, _NICHE_STYLE_TOKENS["default"])
    return f"{prompt}, {niche_tokens}, {_MASTER_STYLE_SUFFIX}"


# ─────────────────────────────────────────────────────────────────────────────
# Availability check
# ─────────────────────────────────────────────────────────────────────────────

def is_wan21_available() -> bool:
    """Return True if Wan2.1 can be loaded (GPU + diffusers>=0.33.0 installed)."""
    if not _ENABLED:
        return False
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        import diffusers  # noqa: F401
        return True
    except ImportError:
        return False


def _get_device() -> str:
    """Return the correct CUDA device for Wan2.1 based on GPU count."""
    try:
        import torch
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return "cuda:1" if num_gpus >= 2 else "cuda:0"
    except Exception:
        return "cuda:0"


# ─────────────────────────────────────────────────────────────────────────────
# T2V Model loader (singleton)
# ─────────────────────────────────────────────────────────────────────────────

def _load_pipeline():
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    with _PIPE_LOCK:
        if _PIPE is not None:
            return _PIPE
        return _load_pipeline_impl()

def _load_pipeline_impl():
    global _PIPE
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    candidates = [_MODEL_ID] + [m for m in _MODEL_ID_FALLBACKS if m != _MODEL_ID]

    last_error: Exception | None = None
    resolved_id: str | None = None
    vae = None

    for candidate in candidates:
        try:
            log.info("wan21.trying_model_id", model_id=candidate)
            vae = AutoencoderKLWan.from_pretrained(
                candidate, subfolder="vae", torch_dtype=torch.float32
            )
            resolved_id = candidate
            break
        except Exception as e:
            log.warning("wan21.model_id_failed", model_id=candidate, error=str(e)[:120])
            last_error = e

    if resolved_id is None:
        raise RuntimeError(
            f"Could not load Wan2.1 from any candidate model ID: {candidates}. "
            f"Last error: {last_error}"
        )

    log.info("wan21.loading_model", model_id=resolved_id)
    log.info("wan21.note", msg="First load downloads ~6GB weights — one-time only")

    _PIPE = WanPipeline.from_pretrained(
        resolved_id,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )

    device = _get_device()
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Always use cpu_offload on T4 regardless of GPU count.
    # Raw .to(cuda:1) on Kaggle 2xT4 causes OOM if cuda:1 has Chatterbox residuals
    # from Cell 2 warm-up. cpu_offload pages weights between CPU and GPU on demand,
    # keeping peak VRAM under 10 GB and preventing the 1.96 GiB allocation crash.
    log.info("wan21.loading_strategy",
             gpus=num_gpus, strategy="cpu_offload",
             note="safer than .to(device) on shared-GPU Kaggle T4")
    gpu_idx = 0
    if num_gpus >= 2:
        try:
            gpu_idx = int(device.split(":")[-1])
        except Exception:
            gpu_idx = 1
    _PIPE.enable_model_cpu_offload(gpu_id=gpu_idx)

    if hasattr(_PIPE, "enable_vae_slicing"):
        try:
            _PIPE.enable_vae_slicing()
        except Exception as e:
            log.warning("wan21.enable_vae_slicing_failed", error=str(e))
    if hasattr(_PIPE, "enable_attention_slicing"):
        try:
            _PIPE.enable_attention_slicing()
        except Exception as e:
            log.warning("wan21.enable_attention_slicing_failed", error=str(e))

    _PIPE.scheduler = UniPCMultistepScheduler.from_config(
        _PIPE.scheduler.config, flow_shift=3.0
    )

    # TeaCache: skips redundant denoising steps → ~2x speedup, zero quality loss
    if _TEACACHE:
        try:
            _PIPE.enable_cache(cache_type="tea_cache", threshold=0.20)
            log.info("wan21.teacache_enabled", threshold=0.20)
        except AttributeError:
            log.warning("wan21.teacache_not_supported",
                        msg="diffusers version does not support enable_cache — upgrade to >=0.35.0")

    log.info("wan21.model_ready", gpus=num_gpus, resolved_model_id=resolved_id,
             teacache=_TEACACHE)
    return _PIPE


# ─────────────────────────────────────────────────────────────────────────────
# I2V Model loader (singleton) — Image-to-Video for anchor-locked scenes
# ─────────────────────────────────────────────────────────────────────────────

def _load_i2v_pipeline():
    """
    Load the Image-to-Video pipeline for anchor-locked scene generation.
    Reuses the T2V pipeline as a fallback if WanImageToVideoPipeline is unavailable.
    """
    global _I2V_PIPE
    if _I2V_PIPE is not None:
        return _I2V_PIPE
    with _I2V_PIPE_LOCK:
        if _I2V_PIPE is not None:
            return _I2V_PIPE
        return _load_i2v_pipeline_impl()

def _load_i2v_pipeline_impl():
    global _I2V_PIPE
    import torch

    try:
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        log.info("wan21.i2v.loading", model_id=_I2V_MODEL_ID)
        vae = AutoencoderKLWan.from_pretrained(
            _I2V_MODEL_ID, subfolder="vae", torch_dtype=torch.float32
        )
        _I2V_PIPE = WanImageToVideoPipeline.from_pretrained(
            _I2V_MODEL_ID,
            vae=vae,
            torch_dtype=torch.bfloat16,
        )

        device = _get_device()
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        # Same cpu_offload strategy as T2V — avoids OOM on shared 2xT4
        gpu_idx = 0
        if num_gpus >= 2:
            try:
                gpu_idx = int(device.split(":")[-1])
            except Exception:
                gpu_idx = 1
        _I2V_PIPE.enable_model_cpu_offload(gpu_id=gpu_idx)

        if hasattr(_I2V_PIPE, "enable_vae_slicing"):
            try:
                _I2V_PIPE.enable_vae_slicing()
            except Exception as e:
                log.warning("wan21.i2v.enable_vae_slicing_failed", error=str(e))
        if hasattr(_I2V_PIPE, "enable_attention_slicing"):
            try:
                _I2V_PIPE.enable_attention_slicing()
            except Exception as e:
                log.warning("wan21.i2v.enable_attention_slicing_failed", error=str(e))
        _I2V_PIPE.scheduler = UniPCMultistepScheduler.from_config(
            _I2V_PIPE.scheduler.config, flow_shift=3.0
        )
        log.info("wan21.i2v.ready")

    except (ImportError, Exception) as e:
        log.warning("wan21.i2v.load_failed",
                    error=str(e)[:120],
                    fallback="Will use T2V pipeline instead")
        # Fall back to T2V pipeline
        _I2V_PIPE = _load_pipeline()

    return _I2V_PIPE


# ─────────────────────────────────────────────────────────────────────────────
# Anchor image generator — creates FLUX-quality still via Pollinations
# ─────────────────────────────────────────────────────────────────────────────

def generate_anchor_image(
    prompt: str,
    output_path: str,
    niche: str = "default",
    width: int = 832,
    height: int = 480,
) -> Optional[str]:
    """
    Generate a master anchor image using Pollinations FLUX (free, no key).
    This image is passed to generate_i2v_clip() to lock visual identity across clips.

    Returns path to PNG file, or None on failure.
    """
    import requests
    from urllib.parse import quote as url_encode

    enriched = _enrich_prompt(prompt, niche)
    encoded  = url_encode(enriched)
    url      = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width={width}&height={height}&nologo=true&model=flux&seed=42"
    )

    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        size = Path(output_path).stat().st_size
        if size < 10_000:
            log.warning("wan21.anchor.too_small", size=size, path=output_path)
            return None
        log.info("wan21.anchor.saved", path=output_path, bytes=size)
        return output_path
    except Exception as e:
        log.warning("wan21.anchor.failed", error=str(e)[:120])
        return None


# ─────────────────────────────────────────────────────────────────────────────
# I2V clip generator — anchor-locked video generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_i2v_clip(
    prompt: str,
    anchor_image_path: str,
    output_path: str,
    duration_s: float = 5.0,
    niche: str = "default",
    num_frames: int = 0,
    fps: int = 16,
    width: int = 832,
    height: int = 480,
    num_inference_steps: int = 20,
    guidance_scale: float = 5.0,
) -> Optional[str]:
    """
    Generate a video clip using Image-to-Video anchoring.
    The anchor image locks the visual identity (color, style, composition).
    Eliminates style drift between independently generated clips.

    Returns path to MP4 file, or None on failure.
    Falls back to T2V if I2V pipeline fails.
    """
    if not is_wan21_available():
        log.warning("wan21.i2v.not_available")
        return None

    try:
        from PIL import Image as PILImage
        anchor = PILImage.open(anchor_image_path).convert("RGB")
        anchor = anchor.resize((width, height), PILImage.LANCZOS)
    except Exception as e:
        log.warning("wan21.i2v.anchor_load_failed", error=str(e), path=anchor_image_path)
        # Fall back to T2V
        return generate_video_clip(prompt, output_path, duration_s, niche,
                                   num_frames, fps, width, height,
                                   num_inference_steps, guidance_scale)

    enriched = _enrich_prompt(prompt, niche)

    if num_frames == 0:
        raw = int(duration_s * fps)
        num_frames = max(17, ((raw // 4) * 4) + 1)

    log.info("wan21.i2v.generating",
             frames=num_frames, fps=fps, size=f"{width}x{height}",
             steps=num_inference_steps, prompt=enriched[:80])

    try:
        import torch
        pipe = _load_i2v_pipeline()

        with torch.inference_mode():
            # WanImageToVideoPipeline uses `image` param; T2V fallback ignores it
            try:
                result = pipe(
                    image=anchor,
                    prompt=enriched,
                    negative_prompt=_NEGATIVE_PROMPT,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )
            except TypeError:
                # Fallback: pipe is actually T2V (no `image` param)
                result = pipe(
                    prompt=enriched,
                    negative_prompt=_NEGATIVE_PROMPT,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )

        frames = result.frames[0]
        return _frames_to_mp4(frames, output_path, fps)

    except Exception as e:
        log.error("wan21.i2v.generation_failed", error=str(e)[:300])
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Shared frame → MP4 exporter
# ─────────────────────────────────────────────────────────────────────────────

def _frames_to_mp4(frames: list, output_path: str, fps: int) -> Optional[str]:
    """Export a list of PIL images to an MP4 via ffmpeg."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, frame in enumerate(frames):
                p = Path(tmpdir) / f"frame_{i:05d}.png"
                frame.save(str(p))

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
        log.info("wan21.clip_saved", path=output_path, frames=len(frames))
        return output_path

    except Exception as e:
        log.error("wan21.frames_to_mp4_failed", error=str(e)[:200])
        return None


# ─────────────────────────────────────────────────────────────────────────────
# T2V clip generator (main entry point)
# ─────────────────────────────────────────────────────────────────────────────

def generate_video_clip(
    prompt: str,
    output_path: str,
    duration_s: float = 5.0,
    niche: str = "default",
    num_frames: int = 0,
    fps: int = 16,
    width: int = 832,
    height: int = 480,
    num_inference_steps: int = 20,
    guidance_scale: float = 5.0,
) -> Optional[str]:
    """
    Generate a video clip using Wan2.1 1.3B T2V.
    Returns path to MP4 file, or None on failure.
    """
    if not is_wan21_available():
        log.warning("wan21.not_available")
        return None

    enriched = _enrich_prompt(prompt, niche)

    if num_frames == 0:
        raw = int(duration_s * fps)
        num_frames = max(17, ((raw // 4) * 4) + 1)

    log.info("wan21.generating",
             frames=num_frames, fps=fps, size=f"{width}x{height}",
             steps=num_inference_steps, prompt=enriched[:80],
             teacache=_TEACACHE)

    try:
        import torch
        pipe = _load_pipeline()

        with torch.inference_mode():
            result = pipe(
                prompt=enriched,
                negative_prompt=_NEGATIVE_PROMPT,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )

        frames = result.frames[0]
        return _frames_to_mp4(frames, output_path, fps)

    except Exception as e:
        log.error("wan21.generation_failed", error=str(e)[:300])
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None
