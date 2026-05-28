"""
tools/wan21_tools.py
─────────────────────────────────────────────────────────────────────────────
Wan2.1 1.3B Text-to-Video wrapper for the AutoPilot pipeline.

Model: Wan-AI/Wan2.1-T2V-1.3B-Diffusers  (Apache 2.0 — commercial use OK)
VRAM:  ~8–9 GB (bfloat16 + VAE float32). Fits single T4 with CPU offloading.
Speed: ~4 min per 5s clip on T4 single GPU; faster on 2x T4.

Multi-GPU (Kaggle 2x T4):
    Chatterbox TTS loads on cuda:0, Wan2.1 loads on cuda:1.
    This maximises parallel availability and avoids VRAM contention.

Usage:
    from tools.wan21_tools import generate_video_clip, is_wan21_available

    if is_wan21_available():
        path = generate_video_clip(
            prompt="cinematic aerial view of a city, golden hour, 4K",
            duration_s=5.0,
            output_path="/tmp/clip.mp4",
        )

Environment:
    WAN21_MODEL_ID  — HuggingFace model ID
                      default: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    WAN21_ENABLED   — set to "0" to disable and force image/Pollinations fallback
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# Research (May 2026): correct HF model ID is Wan-AI org, not Wan-Video
# Fallback list: tries IDs in order until one succeeds
_MODEL_ID  = os.getenv("WAN21_MODEL_ID", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
_MODEL_ID_FALLBACKS = [
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",   # correct (May 2026)
    "Wan-AI/Wan2.1-T2V-1.3B",              # alternate naming
    # Note: Wan-Video/... is the old wrong org — excluded intentionally
]
_ENABLED   = os.getenv("WAN21_ENABLED", "1").strip() != "0"
_PIPE      = None   # singleton model instance (loaded once per process)


# ─────────────────────────────────────────────────────────────────────────────
# Availability check
# ─────────────────────────────────────────────────────────────────────────────

def is_wan21_available() -> bool:
    """Return True if Wan2.1 can be loaded (GPU + diffusers installed)."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Model loader (singleton to avoid reloading weights each scene)
# ─────────────────────────────────────────────────────────────────────────────

def _load_pipeline():
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    import torch
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    # Build candidate list: user-specified ID first, then known-good fallbacks
    candidates = [_MODEL_ID] + [m for m in _MODEL_ID_FALLBACKS if m != _MODEL_ID]

    last_error: Exception | None = None
    resolved_id: str | None = None
    vae = None

    for candidate in candidates:
        try:
            log.info("wan21.trying_model_id", model_id=candidate)
            # VAE must be float32 for best decoding quality (per HF Wan2.1 docs)
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

    # Use bfloat16 for the main pipeline (research recommendation for Wan2.1)
    _PIPE = WanPipeline.from_pretrained(
        resolved_id,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )

    # Multi-GPU strategy: on 2x T4, use cuda:1 for Wan2.1
    # This leaves cuda:0 free for Chatterbox TTS to run in parallel
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus >= 2:
        log.info("wan21.multi_gpu", gpus=num_gpus, wan21_device="cuda:1")
        _PIPE = _PIPE.to("cuda:1")
    elif num_gpus == 1:
        # Single T4: use CPU offloading to keep within 15.6GB budget
        log.info("wan21.single_gpu", device="cuda:0", mode="cpu_offload")
        _PIPE.enable_model_cpu_offload()
    # else: no GPU — pipeline will run on CPU (slow but functional)

    # Memory optimisations (effective on both single and multi-GPU)
    _PIPE.enable_vae_slicing()
    _PIPE.enable_attention_slicing()

    # UniPC scheduler tuned for 480P (flow_shift=3.0 per HF recommendation)
    # Use flow_shift=5.0 for 720P if running on 14B model
    _PIPE.scheduler = UniPCMultistepScheduler.from_config(
        _PIPE.scheduler.config, flow_shift=3.0
    )

    log.info("wan21.model_ready", gpus=num_gpus, resolved_model_id=resolved_id)
    return _PIPE


# ─────────────────────────────────────────────────────────────────────────────
# Niche-specific style token appender
# ─────────────────────────────────────────────────────────────────────────────

_NICHE_STYLE_TOKENS: dict[str, str] = {
    "personal_finance": "financial charts, professional, cinematic lighting, 4K",
    "saas_tools":       "sleek UI, tech aesthetic, dark mode, professional, cinematic",
    "legal_tax":        "professional office, courtroom, formal lighting, cinematic",
    "senior_health":    "warm sunlight, nature, healthy lifestyle, cinematic",
    "storytelling":     "epic cinematic, dramatic lighting, movie quality, 4K",
    "default":          "cinematic, professional, high quality, 4K, smooth motion",
}


def _enrich_prompt(prompt: str, niche: str = "default") -> str:
    """Append niche-specific style tokens to a visual prompt."""
    style = _NICHE_STYLE_TOKENS.get(niche, _NICHE_STYLE_TOKENS["default"])
    return f"{prompt}, {style}"


# ─────────────────────────────────────────────────────────────────────────────
# Main generation function
# ─────────────────────────────────────────────────────────────────────────────

def generate_video_clip(
    prompt: str,
    output_path: str,
    duration_s: float = 5.0,
    niche: str = "default",
    num_frames: int = 0,    # 0 = auto-compute from duration
    fps: int = 16,          # Wan2.1 1.3B default
    width: int = 832,
    height: int = 480,      # 480P for 1.3B model (use 720P for 14B)
    num_inference_steps: int = 20,  # 20 = good quality/speed balance on T4
    guidance_scale: float = 5.0,
) -> Optional[str]:
    """
    Generate a video clip using Wan2.1 1.3B.

    Returns path to MP4 file, or None on failure.
    """
    if not is_wan21_available():
        log.warning("wan21.not_available")
        return None

    enriched = _enrich_prompt(prompt, niche)

    if num_frames == 0:
        # Wan2.1 requires frames = 4n+1 (e.g. 17, 33, 49, 81)
        raw = int(duration_s * fps)
        num_frames = max(17, ((raw // 4) * 4) + 1)

    log.info("wan21.generating",
             frames=num_frames, fps=fps, size=f"{width}x{height}",
             steps=num_inference_steps, prompt=enriched[:80])

    try:
        import torch
        pipe = _load_pipeline()

        with torch.inference_mode():
            result = pipe(
                prompt=enriched,
                negative_prompt=(
                    "low quality, blurry, artifacts, watermark, text overlay, "
                    "deformed, ugly, bad anatomy, worst quality"
                ),
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )

        frames = result.frames[0]   # list of PIL Images

        # Export frames → MP4 via ffmpeg
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_paths = []
            for i, frame in enumerate(frames):
                p = Path(tmpdir) / f"frame_{i:05d}.png"
                frame.save(str(p))
                frame_paths.append(str(p))

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

        log.info("wan21.clip_saved", path=output_path, frames=len(frames))

        # Free GPU memory after generation
        torch.cuda.empty_cache()
        return output_path

    except Exception as e:
        log.error("wan21.generation_failed", error=str(e)[:300])
        # Clear VRAM on failure too
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None
