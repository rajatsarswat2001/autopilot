"""
tools/tts_tools.py
─────────────────────────────────────────────────────────────────────────────
TTSChain — 4-tier Text-to-Speech fallback system.

Tier 1: Chatterbox TTS (MIT licence, Resemble AI)
         Best prosody, supports emotion_exaggeration parameter.
         Requires: pip install chatterbox-tts  (or chatterbox-audio)
         Model: ~1.8 GB, auto-downloads on first run.

Tier 2: NVIDIA Magpie NIM (cloud TTS microservice)
         Requires: NVIDIA_API_KEY

Tier 3: Edge TTS (Microsoft neural voices, FREE, no API key)
         Requires: pip install edge-tts
         Async under the hood, wrapped synchronously here.

Tier 4: pyttsx3 (offline CPU, monotone but never fails)
         Requires: pip install pyttsx3

Each tier returns the tier name string on success and raises on failure,
allowing TTSChain to advance to the next tier.
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

TtsTier = Literal["chatterbox", "magpie", "edge", "pyttsx3"]

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
    "authoritative":0.40,
    "neutral":      0.35,
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


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1: Chatterbox
# ─────────────────────────────────────────────────────────────────────────────

def _tts_chatterbox(text: str, output_path: str, emotion_exaggeration: float = 0.5) -> None:
    import torch
    try:
        import os
        os.environ['TRANSFORMERS_ATTN_IMPLEMENTATION'] = 'eager'
        from chatterbox.tts import ChatterboxTTS
    except ImportError:
        raise ImportError("chatterbox-tts not installed. Run: pip install chatterbox-tts")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTTS.from_pretrained(device=device)

    wav = model.generate(
        text=text,
        exaggeration=emotion_exaggeration,
        cfg_weight=0.5,
    )

    import torchaudio
    torchaudio.save(output_path, wav, model.sr)

    # Clear GPU cache after synthesis to free space for Wan2.1 / ACE-Step
    if torch.cuda.is_available():
        del model
        torch.cuda.empty_cache()

    log.debug("tts.chatterbox_ok", chars=len(text), path=output_path,
              exaggeration=emotion_exaggeration)


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
        # Convert MP3 → WAV via ffmpeg
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-ar", "24000", "-ac", "1", output_path],
            check=True, capture_output=True,
        )
        Path(tmp_path).unlink(missing_ok=True)

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
