"""
tools/tts_tools.py
─────────────────────────────────────────────────────────────────────────────
TTSChain — 5-tier Text-to-Speech waterfall system.

Tier 1: Chatterbox TTS  (GPU, MIT, Resemble AI)
         Best prosody + emotion control. Runs on cuda:0.
         On 2× T4: cuda:0 reserved for TTS, cuda:1 for Wan2.1 video.
         Requires transformers==4.46.3 (install via kaggle_setup.py).

Tier 2: Kokoro-82M  (GPU/CPU, Apache 2.0, free)
         82M params, 210× real-time on GPU, <2GB VRAM.
         Best free TTS fallback. No API key. 50+ English voices.
         Requires: pip install kokoro soundfile  +  apt install espeak-ng

Tier 3: NVIDIA Magpie NIM  (cloud TTS microservice)
         Requires: NVIDIA_API_KEY env var.

Tier 4: Edge TTS  (Microsoft neural voices, FREE)
         Requires: pip install edge-tts

Tier 5: pyttsx3  (offline CPU, monotone, always available)
         Requires: pip install pyttsx3

Each tier raises on failure; TTSChain advances to the next tier.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import asyncio
import tempfile
from pathlib import Path
from typing import Literal

import structlog

log = structlog.get_logger(__name__)

TtsTier = Literal["chatterbox", "kokoro", "magpie", "edge", "pyttsx3"]

EDGE_VOICE    = os.getenv("EDGE_TTS_VOICE",    "en-US-GuyNeural")
PYTTSX3_RATE  = int(os.getenv("PYTTSX3_RATE",  "165"))
MAGPIE_VOICE  = os.getenv("MAGPIE_VOICE",      "English-US.Female-1")

# Chatterbox emotion → exaggeration mapping
# Chatterbox's exaggeration range: 0.0 (monotone) → 1.0 (very dramatic)
_TONE_EXAGGERATION: dict[str, float] = {
    "urgent":       0.85,
    "dramatic":     0.90,
    "shocking":     0.80,
    "curious":      0.55,
    "inspiring":    0.65,
    "hopeful":      0.60,
    "melancholic":  0.45,
    "warm":         0.50,
    "authoritative":0.55,
    "neutral":      0.50,
    "default":      0.50,
}

def _resolve_exaggeration(emotion_tone: str | None) -> float:
    """Return exaggeration value for a given emotional tone string."""
    # Global override from env var takes priority
    env_val = os.getenv("CHATTERBOX_EXAGGERATION", "")
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            pass
    if not emotion_tone:
        return 0.5
    return _TONE_EXAGGERATION.get(emotion_tone.lower(), 0.5)


# Global model caches for singleton performance
_CHATTERBOX_MODEL = None
_KOKORO_PIPELINE = None


def release_chatterbox() -> None:
    """Release Chatterbox from cuda:0 after audio synthesis. Call before visual gen."""
    global _CHATTERBOX_MODEL
    if _CHATTERBOX_MODEL is None:
        return
    try:
        del _CHATTERBOX_MODEL
        _CHATTERBOX_MODEL = None
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device="cuda:0")
            torch.cuda.empty_cache()
            free_gb = torch.cuda.mem_get_info("cuda:0")[0] / 1024 ** 3
            log.info("tts.chatterbox_released",
                     device="cuda:0", free_gb=round(free_gb, 2))
    except Exception as e:
        log.warning("tts.chatterbox_release_failed", error=str(e))


def _tts_chatterbox(text: str, output_path: str, emotion_exaggeration: float = 0.5) -> None:
    global _CHATTERBOX_MODEL
    import torch
    try:
        import os
        os.environ['TRANSFORMERS_ATTN_IMPLEMENTATION'] = 'eager'
        from chatterbox.tts import ChatterboxTTS
    except ImportError:
        raise ImportError("chatterbox-tts not installed. Run: pip install chatterbox-tts")

    # Pin to cuda:0 — on 2x T4, cuda:1 is reserved for Wan2.1 video generation
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    if _CHATTERBOX_MODEL is None:
        _CHATTERBOX_MODEL = ChatterboxTTS.from_pretrained(device=device)
    model = _CHATTERBOX_MODEL

    import re
    # Add breath pauses after sentences
    spaced_text = re.sub(r'([.!?])\s+', r'\1   ', text)

    wav = model.generate(
        text=spaced_text,
        exaggeration=emotion_exaggeration,
        cfg_weight=0.3,
    )

    import torchaudio
    import tempfile
    import subprocess
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    torchaudio.save(tmp_path, wav, model.sr)

    # Slow down voice slightly — 0.92 keeps natural prosody without adding excess length
    subprocess.run([
        "ffmpeg", "-y", "-i", tmp_path, "-filter:a", "atempo=0.92", output_path
    ], check=True, capture_output=True)
    Path(tmp_path).unlink(missing_ok=True)

    log.debug("tts.chatterbox_ok", chars=len(text), path=output_path,
              exaggeration=emotion_exaggeration, slowed=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2: Kokoro-82M (Apache 2.0, 82M params, 210x RT on GPU)
# Best free TTS fallback — cleaner than Edge TTS, no API key, <2GB VRAM
# 50+ English voices: af_heart, af_bella, am_adam, bf_emma, bm_george ...
# ─────────────────────────────────────────────────────────────────────────────

KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")  # American English female


def _tts_kokoro(text: str, output_path: str) -> None:
    global _KOKORO_PIPELINE
    try:
        from kokoro import KPipeline
    except ImportError:
        raise ImportError(
            "kokoro not installed. Run: pip install kokoro soundfile "
            "and: apt install espeak-ng"
        )

    import soundfile as sf
    import numpy as np

    if _KOKORO_PIPELINE is None:
        _KOKORO_PIPELINE = KPipeline(lang_code="a")  # 'a' = American English
    pipeline = _KOKORO_PIPELINE
    generator = pipeline(text, voice=KOKORO_VOICE)

    chunks = []
    for _, _, audio in generator:
        chunks.append(audio)

    if not chunks:
        raise RuntimeError("Kokoro produced no audio chunks")

    import tempfile
    import subprocess
    from pathlib import Path
    
    combined = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, combined, 24000)
    
    # Apply pace normalization for Kokoro — 0.92 matches Chatterbox setting
    subprocess.run([
        "ffmpeg", "-y", "-i", tmp_path, "-filter:a", "atempo=0.92", output_path
    ], check=True, capture_output=True)
    Path(tmp_path).unlink(missing_ok=True)
    
    log.debug("tts.kokoro_ok", voice=KOKORO_VOICE, chars=len(text), path=output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2: NVIDIA Magpie NIM
# ─────────────────────────────────────────────────────────────────────────────

def _tts_magpie(text: str, output_path: str) -> None:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set — cannot use Magpie NIM")

    import requests

    resp = requests.post(
        "https://ai.api.nvidia.com/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "Accept":        "audio/wav",
        },
        json={
            "model":  "magpie-tts",
            "input":  text,
            "voice":  MAGPIE_VOICE,
            "format": "wav",
        },
        timeout=60,
    )
    resp.raise_for_status()
    Path(output_path).write_bytes(resp.content)
    log.debug("tts.magpie_ok", chars=len(text), path=output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3: Edge TTS
# ─────────────────────────────────────────────────────────────────────────────

def _tts_edge(text: str, output_path: str) -> None:
    try:
        import edge_tts
    except ImportError:
        raise ImportError("edge-tts not installed. Run: pip install edge-tts")

    async def _run():
        communicate = edge_tts.Communicate(text, EDGE_VOICE)
        # Edge TTS outputs MP3; we save to a temp file then convert to WAV
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        await communicate.save(tmp_path)
        # Convert MP3 → WAV via ffmpeg and apply pace normalization
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-ar", "24000", "-ac", "1", "-filter:a", "atempo=0.92", output_path],
            check=True, capture_output=True,
        )
        Path(tmp_path).unlink(missing_ok=True)

    # asyncio.run() raises RuntimeError inside Jupyter/ipykernel because an
    # event loop is already running. Use the running loop directly instead.
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside Jupyter / ipykernel — schedule as a task and wait
            import concurrent.futures
            future = concurrent.futures.Future()

            async def _run_and_resolve():
                try:
                    await _run()
                    future.set_result(None)
                except Exception as exc:
                    future.set_exception(exc)

            loop.create_task(_run_and_resolve())
            # Block the current thread until the coroutine finishes.
            # concurrent.futures.Future.result() is thread-safe.
            future.result(timeout=60)
        else:
            loop.run_until_complete(_run())
    except RuntimeError:
        # Last resort: plain asyncio.run() — works in scripts, not Jupyter
        asyncio.run(_run())
    log.debug("tts.edge_ok", voice=EDGE_VOICE, chars=len(text))


# ─────────────────────────────────────────────────────────────────────────────
# Tier 4: pyttsx3 (CPU, offline, always available)
# ─────────────────────────────────────────────────────────────────────────────

def _tts_pyttsx3(text: str, output_path: str) -> None:
    try:
        import pyttsx3
    except ImportError:
        raise ImportError("pyttsx3 not installed. Run: pip install pyttsx3")

    engine = pyttsx3.init()
    engine.setProperty("rate", PYTTSX3_RATE)
    engine.setProperty("volume", 1.0)

    # pyttsx3 can only save via save_to_file + runAndWait
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    log.debug("tts.pyttsx3_ok", chars=len(text))


# ─────────────────────────────────────────────────────────────────────────────
# Chain
# ─────────────────────────────────────────────────────────────────────────────

class TTSChain:
    """
    Waterfall TTS chain. Tries each tier in order; advances on failure.
    Returns the tier name that succeeded.
    """

    def synthesise(
        self,
        text: str,
        output_path: str,
        emotion_exaggeration: float = 0.5,
        emotion_tone: str | None = None,  # if set, overrides emotion_exaggeration
    ) -> TtsTier:
        # Resolve final exaggeration from tone (takes priority over raw float)
        if emotion_tone:
            emotion_exaggeration = _resolve_exaggeration(emotion_tone)

        tiers: list[tuple[TtsTier, callable]] = [
            ("chatterbox", lambda: _tts_chatterbox(text, output_path, emotion_exaggeration)),
            ("kokoro",     lambda: _tts_kokoro(text, output_path)),
            ("magpie",     lambda: _tts_magpie(text, output_path)),
            ("edge",       lambda: _tts_edge(text, output_path)),
            ("pyttsx3",    lambda: _tts_pyttsx3(text, output_path)),
        ]

        for tier_name, fn in tiers:
            try:
                fn()
                log.info("tts.tier_succeeded", tier=tier_name)
                return tier_name
            except Exception as e:
                log.warning("tts.tier_failed", tier=tier_name, reason=str(e)[:120])

        # Absolute fallback — write silence (handled by audio_agent)
        raise RuntimeError("All TTS tiers exhausted — caller must handle silence fallback")
