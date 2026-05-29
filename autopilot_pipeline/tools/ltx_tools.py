"""
tools/ltx_tools.py
─────────────────────────────────────────────────────────────────────────────
LTX-Video Text-to-Video wrapper.

Model: Lightricks/LTX-Video  (Apache 2.0 — commercial use OK)
VRAM:  ~9 GB bfloat16 with CPU offloading. Fits a single T4 (15.6 GB).
Speed: ~15-20 seconds per 5s clip on T4  — 16× faster than Wan2.1 1.3B.

Multi-GPU (Kaggle 2× T4):
    Chatterbox TTS → cuda:0  |  LTX-Video → cuda:1
    Full 15.6 GB on cuda:1 = headroom for 704×480 @ 81 frames with no OOM.

Frame count rule:  num_frames must satisfy (num_frames - 1) % 8 == 0
    Valid: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97 …
    Default: 65 frames at 24 fps ≈ 2.7 s clip  (fast + reliable on T4)

Environment:
    LTX_MODEL_ID   — HuggingFace model ID (default: Lightricks/LTX-Video)
    LTX_ENABLED    — set "0" to force Pollinations/placeholder fallback
    LTX_STEPS      — inference steps (default: 25; increase to 50 for quality)
    LTX_GUIDANCE   — guidance scale (default: 3.0; LTX-Video uses low values)
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

# ── Config ────────────────────────────────────────────────────────────────────
_MODEL_ID   = os.getenv("LTX_MODEL_ID", "Lightricks/LTX-Video")
_ENABLED    = os.getenv("LTX_ENABLED", "1").strip() != "0"
_STEPS      = int(os.getenv("LTX_STEPS",    "25"))    # 25=fast, 50=quality
_GUIDANCE   = float(os.getenv("LTX_GUIDANCE", "3.0")) # LTX uses ~3.0 (lower than SD)

# ── Singleton ─────────────────────────────────────────────────────────────────
_PIPE      = None
_PIPE_LOCK = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Cinematic style tokens — applied to every prompt for visual consistency
# ─────────────────────────────────────────────────────────────────────────────

_NICHE_STYLE: dict[str, str] = {
    "personal_finance": (
        "professional financial environment, sleek modern office, "
        "data visualizations on monitors, cinematic lighting"
    ),
    "saas_tools": (
        "sleek tech workspace, dark UI on monitors, soft blue accent lighting, "
        "futuristic product demo environment"
    ),
    "legal_tax": (
        "professional legal office, polished desk, law books, "
        "warm authoritative atmosphere"
    ),
    "senior_health": (
        "warm golden hour sunlight, serene natural environment, "
        "healthy active lifestyle, soft bokeh background"
    ),
    "storytelling": (
        "epic wide establishing shot, dramatic volumetric lighting, "
        "cinematic movie quality, sweeping landscape"
    ),
    "default": (
        "professional high-quality environment, clean composition, "
        "balanced natural lighting, premium production value"
    ),
}

_MASTER_SUFFIX = (
    "ultra-realistic cinematic footage, 35mm anamorphic lens, "
    "moody side-lighting, shallow depth of field, film grain, "
    "teal and orange color grading, no text, no watermarks, photorealistic"
)

_NEGATIVE_PROMPT = (
    "low quality, blurry, pixelated, watermark, text overlay, logo, "
    "deformed, worst quality, shaky camera, overexposed, underexposed, "
    "cartoon, anime, illustration, painting"
)


def _enrich_prompt(prompt: str, niche: str = "default") -> str:
    """Append niche-specific + master style tokens to every prompt."""
    style = _NICHE_STYLE.get(niche, _NICHE_STYLE["default"])
    return f"{prompt}, {style}, {_MASTER_SUFFIX}"


# ─────────────────────────────────────────────────────────────────────────────
# Availability check
# ─────────────────────────────────────────────────────────────────────────────

def is_ltx_available() -> bool:
    """Return True if LTX-Video can run (GPU + diffusers>=0.32 with LTXPipeline)."""
    if not _ENABLED:
        return False
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        from diffusers import LTXPipeline  # noqa: F401
        return True
    except ImportError:
        return False


def _get_device() -> str:
    """Return cuda:1 on 2-GPU Kaggle (reserved for video), cuda:0 on single GPU."""
    try:
        import torch
        return "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"
    except Exception:
        return "cuda:0"


def _get_gpu_id() -> int:
    try:
        import torch
        return 1 if torch.cuda.device_count() >= 2 else 0
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Model loader (singleton)
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
    from diffusers import LTXPipeline

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_id   = 1 if num_gpus >= 2 else 0

    log.info("ltx.loading_model", model_id=_MODEL_ID,
             note="First load downloads ~9 GB weights — one-time only",
             num_gpus=num_gpus)

    _PIPE = LTXPipeline.from_pretrained(
        _MODEL_ID,
        torch_dtype=torch.bfloat16,
    )

    # CPU offload: keeps weights in system RAM, pages to GPU only during compute.
    # Peak active VRAM stays ~5-7 GB — very comfortable on a 15.6 GB T4.
    log.info("ltx.loading_strategy",
             gpus=num_gpus, strategy="cpu_offload", gpu_id=gpu_id,
             note="Weights paged to system RAM; peak active VRAM ~5-7 GB")
    _PIPE.enable_model_cpu_offload(gpu_id=gpu_id)

    # Enable VAE slicing for large batch safety
    if hasattr(_PIPE, "enable_vae_slicing"):
        try:
            _PIPE.enable_vae_slicing()
            log.info("ltx.vae_slicing_enabled")
        except Exception as e:
            log.warning("ltx.vae_slicing_failed", error=str(e))

    log.info("ltx.model_ready", model_id=_MODEL_ID, steps=_STEPS, guidance=_GUIDANCE)
    return _PIPE


# ─────────────────────────────────────────────────────────────────────────────
# Frame → MP4 exporter
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
        log.info("ltx.clip_saved", path=output_path, frames=len(frames))
        return output_path

    except Exception as e:
        log.error("ltx.frames_to_mp4_failed", error=str(e)[:200])
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main T2V clip generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_ltx_clip(
    prompt: str,
    output_path: str,
    duration_s: float = 5.0,
    niche: str = "default",
    num_frames: int = 0,
    fps: int = 24,
    width: int = 704,    # 704×480 = 16:9-ish, fits comfortably in T4 VRAM
    height: int = 480,
    num_inference_steps: int = 0,  # 0 = use LTX_STEPS env var
    guidance_scale: float = 0.0,   # 0.0 = use LTX_GUIDANCE env var
) -> Optional[str]:
    """
    Generate a video clip using LTX-Video.
    Frame count must satisfy (num_frames - 1) % 8 == 0.
    Returns path to MP4 file, or None on failure.
    """
    if not is_ltx_available():
        log.warning("ltx.not_available")
        return None

    steps    = num_inference_steps or _STEPS
    guidance = guidance_scale or _GUIDANCE
    enriched = _enrich_prompt(prompt, niche)

    # LTX-Video frame count rule: (num_frames - 1) % 8 == 0
    if num_frames == 0:
        raw        = int(duration_s * fps)
        # Round UP to nearest valid LTX frame count (8n + 1)
        n_blocks   = max(1, (raw + 6) // 8)
        num_frames = n_blocks * 8 + 1          # e.g. 5s@24fps → 120 → 121
        num_frames = min(num_frames, 97)       # cap at 97 for T4 safety

    log.info("ltx.generating",
             frames=num_frames, fps=fps, size=f"{width}x{height}",
             steps=steps, guidance=guidance, prompt=enriched[:80])

    try:
        import torch
        pipe = _load_pipeline()

        with torch.inference_mode():
            result = pipe(
                prompt=enriched,
                negative_prompt=_NEGATIVE_PROMPT,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=guidance,
            )

        frames = result.frames[0]
        return _frames_to_mp4(frames, output_path, fps)

    except Exception as e:
        log.error("ltx.generation_failed", error=str(e)[:300])
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None
