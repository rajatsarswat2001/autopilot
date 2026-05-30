"""
tools/video_gen_tools.py
────────────────────────────────────────────────────────────────────────────────
Kaggle T4 x2 — Unified AI video generation.
Replaces wan21_tools.py entirely.

CRITICAL T4 INSIGHT: T4 (Turing) does NOT support bfloat16 natively.
Using bfloat16 on T4 triggers slow emulated execution. Always use float16.

Tier 1: CogVideoX-2B INT8  — primary, ~6-7 GB VRAM, 49 frames @ 8fps, Apache 2.0
Tier 2: LTX-Video 0.9      — fast fallback, ~9 GB VRAM, 97 frames @ 24fps
Tier 3: (caller falls through to Pollinations / placeholder)

Strategy selection via VIDEO_GEN_STRATEGY env var:
    mirror   (default) — CogVideoX on both GPUs, A+B clips in parallel (2x speed)
    hybrid             — LTX-Video(cuda:0) + CogVideoX(cuda:1) simultaneously
    sequential         — single GPU, one clip at a time (safe fallback)

GPU Assignment (dual T4):
    cuda:0  → Chatterbox TTS during audio phase (~2 GB, cleared before visual)
              Then → CogVideoX (mirror) OR LTX-Video (hybrid)
    cuda:1  → CogVideoX-2B always

Environment variables:
    VIDEO_GEN_ENABLED      1/0  — master switch (default 1)
    VIDEO_GEN_INT8         1/0  — CogVideoX INT8 quantization (default 1)
    VIDEO_GEN_STRATEGY     mirror|hybrid|sequential (default mirror)
    VIDEO_GEN_COG_STEPS    inference steps for CogVideoX (default 25)
    VIDEO_GEN_LTX_STEPS    inference steps for LTX-Video (default 8)
    WAN21_ENABLED          set to 0 to disable old wan21_tools loading
────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_ENABLED   = os.getenv("VIDEO_GEN_ENABLED", "1").strip() != "0"
_STRATEGY  = os.getenv("VIDEO_GEN_STRATEGY", "mirror").lower()  # mirror|hybrid|sequential
_USE_INT8  = os.getenv("VIDEO_GEN_INT8", "1").strip() != "0"
_HF_TOKEN  = os.getenv("HF_TOKEN", "")
_STEPS_COG = int(os.getenv("VIDEO_GEN_COG_STEPS", "25"))
_STEPS_LTX = int(os.getenv("VIDEO_GEN_LTX_STEPS", "8"))

# ── GPU assignment ─────────────────────────────────────────────────────────────
def _num_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0

def _video_device(slot: int = 1) -> str:
    """Return cuda:1 for video on 2-GPU Kaggle, cuda:0 on single GPU."""
    n = _num_gpus()
    if n >= 2:
        return f"cuda:{slot}"   # slot 0 = TTS, slot 1 = video
    return "cuda:0"

# ── Singletons ─────────────────────────────────────────────────────────────────
_PIPES: dict[str, object] = {}
_LOCKS: dict[str, threading.Lock] = {
    "cog_0": threading.Lock(),
    "cog_1": threading.Lock(),
    "ltx_0": threading.Lock(),
}

# ── Style tokens ───────────────────────────────────────────────────────────────
_STYLE = (
    "ultra-realistic cinematic footage, 35mm anamorphic lens, "
    "Arri Alexa color science, moody side-lighting, shallow depth of field, "
    "film grain, teal and orange color grading, no text, no watermarks, photorealistic"
)
_NICHE_TOKENS: dict[str, str] = {
    "personal_finance": (
        "professional financial environment, sleek modern office, "
        "warm tungsten lighting, premium corporate aesthetic"
    ),
    "saas_tools":    "sleek tech workspace, dark UI glow, blue accent lighting",
    "legal_tax":     "professional legal office, polished desk, formal atmosphere",
    "senior_health": "warm golden hour, serene natural, healthy active lifestyle",
    "storytelling":  "epic wide shot, dramatic lighting, cinematic movie quality",
    "default":       "professional environment, balanced natural lighting",
}
_NEGATIVE = (
    "low quality, blurry, watermark, text overlay, deformed, "
    "worst quality, amateur footage, shaky camera, overexposed"
)

def _enrich(prompt: str, niche: str = "default") -> str:
    tokens = _NICHE_TOKENS.get(niche, _NICHE_TOKENS["default"])
    return f"{prompt}, {tokens}, {_STYLE}"


# ── Availability ───────────────────────────────────────────────────────────────
def is_video_gen_available() -> bool:
    if not _ENABLED:
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

# Backward-compat alias used by visual_director.py
is_wan21_available = is_video_gen_available


# ── CogVideoX-2B loader ────────────────────────────────────────────────────────
def _load_cogvideox(device: str = "cuda:1") -> object:
    slot = device.split(":")[-1]
    key  = f"cog_{slot}"
    if _LOCKS.get(key) is None:
        _LOCKS[key] = threading.Lock()
    if key in _PIPES:
        return _PIPES[key]

    with _LOCKS[key]:
        if key in _PIPES:
            return _PIPES[key]

        import torch
        from diffusers import CogVideoXPipeline

        # ⚠️  T4 = Turing. bfloat16 is EMULATED → SLOW. Always use float16.
        gpu_id = int(device.split(":")[-1])

        # ── Step 1: Load T5 with bitsandbytes NF4 ──────────────────────────
        # Root-cause fix for persistent VRAM OOM on 15.6 GB T4:
        #   T5 in float16 = 9.5 GB  →  heavily fragments VRAM post-offload
        #                              Transformer activations can't find contiguous blocks
        #   T5 in NF4 (bitsandbytes) = ~2.1 GB  → 7.4 GB saved, fragmentation gone
        #
        # WHY bitsandbytes and NOT torchao:
        #   torchao wraps tensors as AffineQuantizedTensor (tensor-level).
        #   accelerate's cpu_offload hooks call .to(device) on tensors → TypeError.
        #
        #   bitsandbytes replaces nn.Linear → bnb.Linear4bit (module-level).
        #   accelerate's hooks move modules to/from GPU normally → full compatibility.
        text_encoder = None
        try:
            from transformers import T5EncoderModel, BitsAndBytesConfig
            log.info("cogvideox.t5_nf4_loading", device=device)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            text_encoder = T5EncoderModel.from_pretrained(
                "THUDM/CogVideoX-2b",
                subfolder="text_encoder",
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            log.info("cogvideox.t5_nf4_ready", device=device)
        except Exception as e:
            log.warning("cogvideox.t5_nf4_failed_using_float16", error=str(e)[:120])
            text_encoder = None  # fall through to full float16 pipeline

        # ── Step 2: Load full pipeline ──────────────────────────────────────
        log.info("cogvideox.loading", device=device, dtype="float16",
                 t5_nf4=(text_encoder is not None))

        load_kwargs: dict = dict(torch_dtype=torch.float16, use_safetensors=True)
        if text_encoder is not None:
            load_kwargs["text_encoder"] = text_encoder

        pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", **load_kwargs)

        # ── Step 3: CPU offload (accelerate component-level hooks) ──────────
        # Works correctly with bitsandbytes bnb.Linear4bit modules.
        # Peak VRAM: T5 NF4 ~2.1 GB + Transformer ~4 GB + activations ~3 GB = ~9 GB
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload(gpu_id=gpu_id)
        else:
            pipe.to(device)

        # ── Step 4: Memory-saving inference hooks ───────────────────────────
        # attention_slicing(1) = one head at a time → max activation reduction
        pipe.enable_attention_slicing(slice_size=1)
        for method in ("enable_vae_slicing", "enable_vae_tiling"):
            if hasattr(pipe, method):
                try:
                    getattr(pipe, method)()
                except Exception:
                    pass

        _PIPES[key] = pipe
        used = torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0
        log.info("cogvideox.ready", device=device, vram_gb=round(used, 1),
                 t5_mode="nf4" if text_encoder is not None else "float16")
        return pipe



# ── LTX-Video loader ───────────────────────────────────────────────────────────
def _load_ltx(device: str = "cuda:0") -> object:
    slot = device.split(":")[-1]
    key  = f"ltx_{slot}"
    if _LOCKS.get(key) is None:
        _LOCKS[key] = threading.Lock()
    if key in _PIPES:
        return _PIPES[key]

    with _LOCKS[key]:
        if key in _PIPES:
            return _PIPES[key]

        import torch
        from diffusers import LTXPipeline

        log.info("ltx.loading", device=device)

        # Try float16 first (faster on T4); fall back to bfloat16 if weights require it
        try:
            pipe = LTXPipeline.from_pretrained(
                "Lightricks/LTX-Video",
                torch_dtype=torch.float16,
            )
        except Exception:
            log.warning("ltx.fp16_load_failed_trying_bf16")
            pipe = LTXPipeline.from_pretrained(
                "Lightricks/LTX-Video",
                torch_dtype=torch.bfloat16,
            )

        gpu_id = int(device.split(":")[-1])
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload(gpu_id=gpu_id)
        else:
            pipe.to(device)

        for method in ("enable_vae_slicing", "enable_attention_slicing"):
            if hasattr(pipe, method):
                try:
                    getattr(pipe, method)()
                except Exception:
                    pass

        _PIPES[key] = pipe
        log.info("ltx.ready", device=device)
        return pipe


# ── Frame list → MP4 ──────────────────────────────────────────────────────────
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
        log.info("video_gen.saved", path=output_path, frames=len(frames))
        return output_path
    except Exception as e:
        log.error("video_gen.frames_to_mp4_failed", error=str(e)[:200])
        return None


# ── CogVideoX single-clip generator ───────────────────────────────────────────
def _run_cogvideox(
    prompt: str,
    output_path: str,
    device: str,
    niche: str = "default",
) -> Optional[str]:
    try:
        import torch
        pipe     = _load_cogvideox(device)
        enriched = _enrich(prompt, niche)

        log.info("cogvideox.generating", device=device,
                 steps=_STEPS_COG, prompt=enriched[:80])

        with torch.inference_mode():
            result = pipe(
                prompt=enriched,
                negative_prompt=_NEGATIVE,
                height=480,
                width=720,
                num_frames=25,        # reduced from 49 to 25 to fit in 16GB VRAM in FP16
                num_inference_steps=_STEPS_COG,
                guidance_scale=6.0,
            )

        return _frames_to_mp4(result.frames[0], output_path, fps=8)

    except Exception as e:
        log.error("cogvideox.failed", device=device, error=str(e)[:300])
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None


# ── LTX-Video single-clip generator ───────────────────────────────────────────
def _run_ltx(
    prompt: str,
    output_path: str,
    device: str,
    niche: str = "default",
) -> Optional[str]:
    try:
        import torch
        pipe     = _load_ltx(device)
        enriched = _enrich(prompt, niche)

        log.info("ltx.generating", device=device,
                 steps=_STEPS_LTX, prompt=enriched[:80])

        with torch.inference_mode():
            result = pipe(
                prompt=enriched,
                negative_prompt=_NEGATIVE,
                height=480,
                width=704,
                num_frames=97,        # 97 frames @ 24fps ≈ 4.0s
                num_inference_steps=_STEPS_LTX,
                guidance_scale=3.0,
            )

        return _frames_to_mp4(result.frames[0], output_path, fps=24)

    except Exception as e:
        log.error("ltx.failed", device=device, error=str(e)[:300])
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None


# ════════════════════════════════════════════════════════════════════════════════
# Strategy implementations
# ════════════════════════════════════════════════════════════════════════════════

def _generate_pair(
    prompt_a: str, prompt_b: str,
    out_a: str, out_b: str,
    niche: str,
) -> tuple[Optional[str], Optional[str]]:
    """
    Generate clip A and clip B according to the selected strategy.
    Called from generate_video_clip when the caller provides two prompts/paths.
    """
    n = _num_gpus()

    if _STRATEGY == "mirror" and n >= 2:
        # ── Mirror: CogVideoX on both GPUs simultaneously ────────────────────
        log.info("strategy.mirror")
        results: dict[str, Optional[str]] = {}

        def _gen_cog(prompt, out, device, label):
            results[label] = _run_cogvideox(prompt, out, device, niche)

        with ThreadPoolExecutor(max_workers=2) as pool:
            fa = pool.submit(_gen_cog, prompt_a, out_a, "cuda:0", "A")
            fb = pool.submit(_gen_cog, prompt_b, out_b, "cuda:1", "B")
            fa.result(); fb.result()

        return results.get("A"), results.get("B")

    elif _STRATEGY == "hybrid" and n >= 2:
        # ── Hybrid: LTX(cuda:0) + CogVideoX(cuda:1) simultaneously ──────────
        log.info("strategy.hybrid")
        results: dict[str, Optional[str]] = {}

        def _gen_ltx(prompt, out, device, label):
            results[label] = _run_ltx(prompt, out, device, niche)

        def _gen_cog(prompt, out, device, label):
            results[label] = _run_cogvideox(prompt, out, device, niche)

        with ThreadPoolExecutor(max_workers=2) as pool:
            fa = pool.submit(_gen_ltx, prompt_a, out_a, "cuda:0", "A")
            fb = pool.submit(_gen_cog, prompt_b, out_b, "cuda:1", "B")
            fa.result(); fb.result()

        return results.get("A"), results.get("B")

    else:
        # ── Sequential: single GPU, safe ─────────────────────────────────────
        log.info("strategy.sequential")
        dev = _video_device(1)
        r_a = _run_cogvideox(prompt_a, out_a, dev, niche)
        if r_a is None:
            r_a = _run_ltx(prompt_a, out_a, dev, niche)
        r_b = _run_cogvideox(prompt_b, out_b, dev, niche)
        if r_b is None:
            r_b = _run_ltx(prompt_b, out_b, dev, niche)
        return r_a, r_b


# ════════════════════════════════════════════════════════════════════════════════
# Public API — same signatures as wan21_tools.py
# ════════════════════════════════════════════════════════════════════════════════

def generate_video_clip(
    prompt: str,
    output_path: str,
    duration_s: float = 5.0,  # kept for interface compat; models use fixed frames
    niche: str = "default",
    num_frames: int = 0,       # kept for interface compat
    fps: int = 8,              # kept for interface compat
    width: int = 720,
    height: int = 480,
    num_inference_steps: int = 0,  # 0 = use env var default
    guidance_scale: float = 0.0,   # 0.0 = use model default
) -> Optional[str]:
    """
    Generate a single video clip. Primary: CogVideoX-2B. Fallback: LTX-Video.
    Returns path to MP4 or None.
    """
    if not is_video_gen_available():
        log.warning("video_gen.not_available")
        return None

    dev = _video_device(1)

    # Try CogVideoX first
    result = _run_cogvideox(prompt, output_path, device=dev, niche=niche)
    if result:
        return result

    log.warning("video_gen.cogvideox_failed_trying_ltx")
    return _run_ltx(prompt, output_path, device=dev, niche=niche)


def generate_anchor_image(
    prompt: str,
    output_path: str,
    niche: str = "default",
    width: int = 720,
    height: int = 480,
) -> Optional[str]:
    """
    Generate a high-quality anchor still image.
    Uses HuggingFace FLUX.1-schnell API (zero local VRAM) if HF_TOKEN is set.
    Falls back to Pollinations FLUX (free, no key needed).
    """
    enriched = _enrich(prompt, niche)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Try HF Inference API (FLUX.1-schnell) ─────────────────────────────────
    if _HF_TOKEN:
        for attempt in range(2):
            try:
                import requests
                time.sleep(1.0 + attempt * 2)
                resp = requests.post(
                    "https://api-inference.huggingface.co/models/"
                    "black-forest-labs/FLUX.1-schnell",
                    headers={"Authorization": f"Bearer {_HF_TOKEN}"},
                    json={
                        "inputs": enriched,
                        "parameters": {
                            "width": width, "height": height,
                            "num_inference_steps": 4,
                            "guidance_scale": 0.0,
                        },
                    },
                    timeout=90,
                )
                if resp.status_code == 200 and len(resp.content) > 10_000:
                    Path(output_path).write_bytes(resp.content)
                    log.info("anchor.hf_flux_ok", path=output_path)
                    return output_path
                log.warning("anchor.hf_flux_fail",
                            status=resp.status_code, attempt=attempt)
            except Exception as e:
                log.warning("anchor.hf_flux_error",
                            attempt=attempt, error=str(e)[:100])

    # ── Fallback: Pollinations FLUX (no key, free) ────────────────────────────
    from urllib.parse import quote as url_encode
    import requests
    for attempt in range(3):
        try:
            time.sleep(2.0 + attempt * 3)
            encoded = url_encode(enriched)
            url = (
                f"https://image.pollinations.ai/prompt/{encoded}"
                f"?width={width}&height={height}&nologo=true&model=flux&seed=42"
            )
            resp = requests.get(url, stream=True, timeout=90)
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            if Path(output_path).stat().st_size > 10_000:
                log.info("anchor.pollinations_ok", path=output_path,
                         attempt=attempt)
                return output_path
        except Exception as e:
            log.warning("anchor.pollinations_failed",
                        attempt=attempt, error=str(e)[:100])

    log.error("anchor.all_sources_failed", path=output_path)
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
    Image-to-Video — falls back to T2V in this module.
    (LTX-Video 0.9 natively supports I2V via LTXImageToVideoPipeline;
     add that upgrade here when ready.)
    """
    log.info("video_gen.i2v_fallback_to_t2v")
    return generate_video_clip(prompt, output_path, duration_s, niche)
