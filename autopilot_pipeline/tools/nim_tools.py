"""
tools/nim_tools.py
─────────────────────────────────────────────────────────────────────────────
NVIDIA NIM API wrappers — LLM inference + image generation.

LLM client:
  • OpenAI-compatible, targets integrate.api.nvidia.com
  • Exponential back-off retry (3 attempts)
  • Rate-limit: NIM free-tier is 40 RPM → automatic sleep on 429

Image generation (SDXL):
  • Uses NVIDIA Picasso NIM (or compatible SDXL endpoint)
  • Falls back to a local SDXL call via diffusers if NIM unavailable
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import requests
import structlog

log = structlog.get_logger(__name__)

NIM_BASE_URL     = "https://integrate.api.nvidia.com/v1"
NIM_IMAGE_URL    = "https://ai.api.nvidia.com/v1/genai/stabilityai/sdxl-turbo"
DEFAULT_LLM      = "meta/llama-3.3-70b-instruct"
DEFAULT_IMG_SIZE = "1024x1024"


# ─────────────────────────────────────────────────────────────────────────────
# LLM helpers
# ─────────────────────────────────────────────────────────────────────────────

def nim_chat_completion(
    messages: list[dict],
    model: str = DEFAULT_LLM,
    temperature: float = 0.8,
    max_tokens: int = 4096,
) -> str:
    """
    Call NVIDIA NIM chat completion endpoint.
    Returns the assistant message content string.
    Raises RuntimeError if NVIDIA_API_KEY is not set.
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set")

    from openai import OpenAI, RateLimitError, APIError

    client = OpenAI(base_url=NIM_BASE_URL, api_key=api_key)

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except RateLimitError:
            wait = 5 * (2 ** attempt)
            log.warning("nim.rate_limit", wait=wait, attempt=attempt)
            time.sleep(wait)
        except APIError as e:
            log.error("nim.api_error", error=str(e), attempt=attempt)
            if attempt == 2:
                raise
            time.sleep(3)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Image generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_image_sdxl(prompt: str, output_path: str) -> str | None:
    """
    Generate an image via NVIDIA NIM SDXL-Turbo endpoint.
    Saves PNG to output_path and returns the path.
    Falls back to local diffusers if NIM is unavailable.
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    if api_key:
        try:
            return _nim_sdxl(prompt, output_path, api_key)
        except Exception as e:
            log.warning("nim.sdxl_failed", error=str(e))

    # Fallback: local SDXL via diffusers
    return _local_sdxl(prompt, output_path)


def _nim_sdxl(prompt: str, output_path: str, api_key: str) -> str:
    """Call NVIDIA NIM SDXL-Turbo image generation."""
    payload = {
        "text_prompts": [
            {"text": f"cinematic, 4K, high quality, {prompt}", "weight": 1.0},
            {"text": "blurry, ugly, watermark, text, low quality", "weight": -1.0},
        ],
        "sampler":           "DDIM",
        "steps":             4,               # Turbo is fast at 4 steps
        "cfg_scale":         1.5,
        "seed":              0,
        "style_preset":      "cinematic",
        "width":             1024,
        "height":            1024,
    }

    resp = requests.post(
        NIM_IMAGE_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept":        "application/json",
        },
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()

    data   = resp.json()
    b64    = data["artifacts"][0]["base64"]
    img_bytes = base64.b64decode(b64)

    Path(output_path).write_bytes(img_bytes)
    log.debug("nim.sdxl_ok", path=output_path)
    return output_path


def _local_sdxl(prompt: str, output_path: str) -> str | None:
    """Local SDXL via diffusers (requires GPU + diffusers installed)."""
    try:
        import torch
        from diffusers import AutoPipelineForText2Image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)

        image = pipe(
            prompt=f"cinematic, 4K, {prompt}",
            negative_prompt="blurry, watermark, low quality",
            num_inference_steps=4,
            guidance_scale=0.0,
        ).images[0]

        image = image.resize((1920, 1080))
        image.save(output_path)
        log.debug("nim.local_sdxl_ok", path=output_path)
        return output_path

    except Exception as e:
        log.warning("nim.local_sdxl_failed", error=str(e))
        return None
