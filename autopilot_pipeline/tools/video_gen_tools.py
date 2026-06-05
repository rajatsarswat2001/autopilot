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
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import structlog

# ── PATCH FOR DIFFUSERS 0.33 WITH TRANSFORMERS 4.46.3 ───────────────────────
try:
    import transformers.utils
    if not hasattr(transformers.utils, "FLAX_WEIGHTS_NAME"):
        transformers.utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

log = structlog.get_logger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
_ENABLED      = os.getenv("VIDEO_GEN_ENABLED", "1").strip() != "0"
_HF_TOKEN     = os.getenv("HF_TOKEN", "")
_IS_KAGGLE    = os.path.exists("/kaggle/working")
_FORCED_MODEL = os.getenv("VIDEO_GEN_MODEL", "").strip().lower()

# Wan2.1 params — primary model
_WAN_MODEL    = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
_WAN_WIDTH    = 832
_WAN_HEIGHT   = 480
_WAN_FPS      = 24
_WAN_STEPS    = int(os.getenv("VIDEO_GEN_WAN_STEPS", "20"))
_WAN_GUIDANCE = float(os.getenv("VIDEO_GEN_WAN_GUIDANCE", "5.0"))

# ── TESTING MODE: hard cap on clip duration ──────────────────────────────────
# Set MAX_VIDEO_DURATION_S=20 in .env to cap all clips to ≤20 seconds.
# Wan2.1 at 24fps: frames must be odd (81=3.4s, 121=5s, 241=10s, 481=20s)
_MAX_DURATION_S = float(os.getenv("MAX_VIDEO_DURATION_S", "20.0"))  # cap: 15-20s
_MAX_FRAMES     = int(_MAX_DURATION_S * _WAN_FPS)  # e.g. 20s*24fps = 480 frames
_WAN_FRAMES     = 81   # default clip length (3.4s) — overridden per-call by duration_s



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
    pipe = _PIPES.pop(key, None)
    if pipe is None:
        return
    
    for attr in ("transformer", "text_encoder", "text_encoder_2", 
                 "vae", "image_encoder", "scheduler"):
        component = getattr(pipe, attr, None)
        if component is not None and hasattr(component, "to"):
            try:
                component.to("cpu")
            except Exception:
                pass
    
    # Remove CPU offload hooks — this is what releases the pinned memory
    if hasattr(pipe, "_all_hooks"):
        for hook in pipe._all_hooks:
            hook.remove()
        pipe._all_hooks = []
    
    del pipe
    gc.collect()
    
    try:
        import torch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
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
        from PIL import Image
        import numpy as np
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, frame in enumerate(frames):
                if hasattr(frame, 'cpu'):
                    frame = frame.cpu().numpy()
                if isinstance(frame, np.ndarray):
                    if frame.ndim == 4:
                        frame = frame[0]
                    # channel-first → channel-last
                    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
                        frame = np.transpose(frame, (1, 2, 0))
                    # any float (including float16) → uint8
                    if frame.dtype != np.uint8:
                        frame = (np.clip(frame.astype(np.float32), 0.0, 1.0) * 255).astype(np.uint8)
                    frame = Image.fromarray(frame)
                
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

            # FIX 1: Running natively on GPU for max speed on A100 (40GB VRAM)
            # No CPU offloading needed.
            pipe.to(device)

            # FIX 2: slice_size=1 is mandatory — the default "auto" is a no-op
            pipe.enable_attention_slicing(slice_size=1)

            if hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()

            _PIPES[key] = pipe
            free_after = _free_vram_gb(device)
            log.info("wan.ready", device=device, free_vram_gb_after=round(free_after, 1))
            return pipe

        except torch.cuda.OutOfMemoryError as e:
            # FIX 3: catch OOM explicitly so eviction fires correctly
            log.error("wan.load_oom", device=device, error=str(e)[:200])
            _evict_pipeline(f"wan_{device.split(':')[-1]}")
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            return None
        except Exception as e:
            log.error("wan.load_failed", error=str(e)[:200])
            _evict_pipeline(f"wan_{device.split(':')[-1]}")
            return None


# ── WAN2.1 GENERATOR ──────────────────────────────────────────────────────────
def _run_wan(
    prompt: str,
    output_path: str,
    device: str,
    niche: str = "default",
    seed: int = 42,
    duration_s: float = 0.0,  # 0 = use default _WAN_FRAMES
) -> Optional[str]:
    try:
        import torch
        pipe = _load_wan(device)
        if pipe is None:
            return None

        enriched = _enrich(prompt, niche)

        # ── Compute frames from requested duration, capped at MAX_VIDEO_DURATION_S ──
        if duration_s > 0:
            raw_frames = int(duration_s * _WAN_FPS)
        else:
            raw_frames = _WAN_FRAMES
        # Wan2.1 requires odd frame count; cap at _MAX_FRAMES
        capped = min(raw_frames, _MAX_FRAMES)
        frames = capped if capped % 2 == 1 else max(1, capped - 1)
        actual_duration_s = frames / _WAN_FPS

        log.info("wan.generating", device=device, steps=_WAN_STEPS,
                 size=f"{_WAN_WIDTH}x{_WAN_HEIGHT}", frames=frames,
                 duration_s=round(actual_duration_s, 1),
                 max_cap_s=_MAX_DURATION_S,
                 prompt=enriched[:80])

        generator = torch.Generator(device).manual_seed(seed)

        with torch.no_grad():
            result = pipe(
                prompt=enriched,
                negative_prompt=_NEGATIVE,
                width=_WAN_WIDTH,
                height=_WAN_HEIGHT,
                num_frames=frames,
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

    log.error("anchor.all_failed", path=output_path)
    return None


def _generate_wangp_clip(prompt: str, output_path: str, niche: str) -> Optional[str]:
    """
    Generates a video using the standalone WanGP application in headless mode.
    Writes a JSON queue file and executes wgp.py via subprocess.
    """
    wangp_dir = os.environ.get("WANGP_DIR", "/kaggle/working/Wan2GP")
    if not os.path.exists(wangp_dir):
        log.error("wangp.missing_dir", dir=wangp_dir)
        return None

    job_file = os.path.join(wangp_dir, f"job_{int(time.time())}.json")
    
    # WanGP expected JSON structure (approximated for headless batch)
    job_data = {
        "prompt": prompt,
        "model": "wan22_14b_gguf",
        "output_path": os.path.abspath(output_path),
        "resolution": "832x480",
        "frames": 81
    }
    
    try:
        with open(job_file, "w") as f:
            json.dump([job_data], f, indent=2)
            
        log.info("wangp.starting_job", job_file=job_file)
        
        result = subprocess.run(
            ["python", "wgp.py", "--process", job_file],
            cwd=wangp_dir,
            capture_output=True,
            text=True,
            timeout=3600 # 1 hour timeout
        )
        
        if result.returncode == 0 and os.path.exists(output_path):
            log.info("wangp.success", output=output_path)
            return output_path
        else:
            log.error("wangp.failed", stdout=result.stdout[-500:], stderr=result.stderr[-500:])
            return None
            
    except Exception as e:
        log.error("wangp.exception", error=str(e))
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

    if _FORCED_MODEL == "wan21":
        return _run_wan(prompt, output_path, dev, niche, duration_s=duration_s)
    elif _FORCED_MODEL == "wan22_gguf":
        return _generate_wangp_clip(prompt, output_path, niche)
    elif _FORCED_MODEL == "wan22":
        # Route to ComfyUI dual-GPU workers launched in Cell 5 (Kaggle notebook).
        # Falls back to wan21 diffusers if KAGGLE_NGROK_URL is not set.
        api_url = os.getenv("KAGGLE_NGROK_URL", "").strip()
        if api_url:
            try:
                import requests as _req
                fps    = _WAN_FPS
                frames = int(min(duration_s or _MAX_DURATION_S, _MAX_DURATION_S) * fps)
                frames = frames if frames % 2 == 1 else max(1, frames - 1)
                enriched = _enrich(prompt, niche)
                log.info("wan22.comfyui_request", url=api_url,
                         frames=frames, duration_s=round(frames / fps, 1))
                resp = _req.post(
                    f"{api_url.rstrip('/')}/generate_video",
                    data={
                        "prompt": enriched,
                        "seed":   str(abs(hash(prompt)) % (2 ** 31)),
                        "steps":  str(_WAN_STEPS),
                        "width":  str(_WAN_WIDTH),
                        "height": str(_WAN_HEIGHT),
                        "frames": str(frames),
                    },
                    timeout=1800,
                )
                if resp.status_code == 200:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "wb") as _f:
                        _f.write(resp.content)
                    log.info("wan22.comfyui_success", path=output_path)
                    return output_path
                else:
                    log.error("wan22.comfyui_failed", status=resp.status_code)
            except Exception as _e:
                log.error("wan22.comfyui_error", error=str(_e)[:200])
        log.warning("wan22.fallback_to_wan21",
                    reason="KAGGLE_NGROK_URL not set or ComfyUI unreachable")
        return _run_wan(prompt, output_path, dev, niche, duration_s=duration_s)

    # Default: Wan2.1 diffusers
    result = _run_wan(prompt, output_path, dev, niche, duration_s=duration_s)
    if result:
        return result

    return None


def generate_i2v_clip(
    prompt: str,
    anchor_image_path: str,
    output_path: str,
    duration_s: float = 5.0,
    niche: str = "default",
    **kwargs,
) -> Optional[str]:
    """
    I2V path: Headless Kaggle ComfyUI API (Wan 2.1 14B Q6_K).
    """
    if not _ENABLED:
        return None

    if not anchor_image_path or not Path(anchor_image_path).exists():
        anchor_image_path = None
        log.warning("video_gen.missing_anchor_image_falling_back_to_t2v")

    api_url = os.getenv("KAGGLE_NGROK_URL", "").strip()
    if not api_url:
        log.warning("video_gen.missing_kaggle_url")
        # Fall back to legacy if no URL is provided
        return generate_video_clip(prompt, output_path, duration_s, niche)

    log.info("video_gen.kaggle_api_starting", prompt=prompt)
    
    try:
        import requests
        
        files = {}
        if anchor_image_path:
            with open(anchor_image_path, "rb") as f:
                # We must read it entirely into memory since we're passing dict of bytes or file objects
                img_data = f.read()
            files = {"image": (Path(anchor_image_path).name, img_data, "image/png")}
            
        data = {"prompt": _enrich(prompt, niche)}
        
        # This is a long-running request (10-15 mins), so timeout must be high
        response = requests.post(
            f"{api_url.rstrip('/')}/generate_video",
            files=files if files else None,
            data=data,
            timeout=1800  # 30 minutes
        )
            
        if response.status_code == 200:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as out_f:
                out_f.write(response.content)
            log.info("video_gen.kaggle_api_success", path=output_path)
            return output_path
        else:
            log.error("video_gen.kaggle_api_failed", status=response.status_code, text=response.text)
            return generate_video_clip(prompt, output_path, duration_s, niche)
            
    except Exception as e:
        log.error("video_gen.kaggle_api_error", error=str(e))
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
