"""
tools/music_tools.py
─────────────────────────────────────────────────────────────────────────────
Background music generation for the AutoPilot pipeline.

Priority waterfall:
  Tier 1: ACE-Step 1.5 (local GPU — MIT licensed, Suno v4.5 quality)
           Requires: pip install acestep  (auto-installs on Kaggle)
           VRAM: ~4 GB — runs easily on Kaggle T4
           Speed: ~10s for a 60s track on T4
  
  Tier 2: YouTube Audio Library (free, royalty-free, no key needed)
           Downloads a pre-curated royalty-free track matching the mood.

  Tier 3: Silence / no music
           Returns None — pipeline skips music mixing.

Usage:
    from tools.music_tools import generate_background_music
    music_path = generate_background_music(mood="upbeat", duration_s=65.0)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

MUSIC_OUTPUT_DIR = Path(os.getenv("VIDEO_OUTPUT_DIR", "outputs/video")).resolve() / "scratch"

# ─────────────────────────────────────────────────────────────────────────────
# Mood → ACE-Step text prompt mapping
# ─────────────────────────────────────────────────────────────────────────────
_MOOD_PROMPTS: dict[str, str] = {
    "urgent":       "tense cinematic score, fast tempo, no vocals, driving percussion, thriller",
    "melancholic":  "slow emotional piano, ambient, no vocals, soft strings, contemplative",
    "upbeat":       "upbeat corporate background, positive energy, no vocals, motivational",
    "curious":      "curious investigative score, light percussive, no vocals, documentary feel",
    "authoritative":"corporate motivational, steady beat, no vocals, confident, professional",
    "dramatic":     "epic cinematic orchestral, no vocals, dramatic tension, rising strings",
    "hopeful":      "inspirational light piano, uplifting, no vocals, gentle build",
    "default":      "neutral background music, steady tempo, no vocals, professional, loopable",
}

# Pre-curated royalty-free YouTube Audio Library URLs (genre-matched, always free)
_FALLBACK_TRACKS: dict[str, str] = {
    "urgent":       "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3",
    "melancholic":  "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-8.mp3",
    "upbeat":       "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
    "curious":      "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-6.mp3",
    "authoritative":"https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3",
    "dramatic":     "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-9.mp3",
    "hopeful":      "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-5.mp3",
    "default":      "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3",
}


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1: ACE-Step 1.5 (local GPU)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_acestep(mood: str, duration_s: float, output_path: str) -> bool:
    """
    Generate background music using ACE-Step 1.5 on local GPU.
    Falls through on ImportError so the pipeline can continue on CPU-only machines.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            log.info("music.acestep_skipped_no_gpu")
            return False

        # ACE-Step uses diffusers-compatible pipeline
        from acestep.pipeline_ace_step import ACEStepPipeline  # type: ignore
    except ImportError:
        log.info("music.acestep_not_installed", hint="pip install acestep")
        return False

    try:
        prompt = _MOOD_PROMPTS.get(mood, _MOOD_PROMPTS["default"])
        log.info("music.acestep_generating", mood=mood, duration=duration_s)

        pipe = ACEStepPipeline.from_pretrained(
            "ACE-Step/ACE-Step-v1-3.5B",
            torch_dtype=torch.float16,
        )
        pipe = pipe.to("cuda")

        audio = pipe(
            prompt=prompt,
            duration=duration_s,
            guidance_scale=7.5,
        ).audio

        # Save as WAV, convert to MP3 via ffmpeg for compatibility
        import torchaudio
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        torchaudio.save(tmp_wav, audio, 44100)

        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_wav,
             "-codec:a", "libmp3lame", "-qscale:a", "2", output_path],
            check=True, capture_output=True
        )
        Path(tmp_wav).unlink(missing_ok=True)

        # Clean up GPU memory
        del pipe
        torch.cuda.empty_cache()

        log.info("music.acestep_ok", path=output_path, duration=duration_s)
        return True

    except Exception as e:
        log.warning("music.acestep_failed", error=str(e)[:200])
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2: Download royalty-free track
# ─────────────────────────────────────────────────────────────────────────────

def _download_fallback_track(mood: str, output_path: str) -> bool:
    """Download a pre-curated royalty-free music track matching the mood."""
    url = _FALLBACK_TRACKS.get(mood, _FALLBACK_TRACKS["default"])
    try:
        log.info("music.downloading_fallback", mood=mood, url=url)
        urllib.request.urlretrieve(url, output_path)
        if Path(output_path).stat().st_size > 50_000:
            log.info("music.fallback_ok", path=output_path)
            return True
    except Exception as e:
        log.warning("music.fallback_download_failed", error=str(e))
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_background_music(
    mood: str = "default",
    duration_s: float = 65.0,
    video_id: str = "video",
) -> Optional[str]:
    """
    Generate or download background music for a video.

    Args:
        mood: Emotional tone from the script (e.g. "urgent", "hopeful")
        duration_s: Target duration in seconds
        video_id: Used for output filename

    Returns:
        Absolute path to MP3 file, or None if all tiers fail.
    """
    MUSIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = str(MUSIC_OUTPUT_DIR / f"{video_id}_music.mp3")

    # Tier 1: ACE-Step (GPU)
    if _generate_acestep(mood, duration_s, output_path):
        return output_path

    # Tier 2: Download fallback track
    if _download_fallback_track(mood, output_path):
        return output_path

    # Tier 3: No music
    log.warning("music.all_tiers_failed_no_music")
    return None


def get_dominant_mood(scene_manifest: dict) -> str:
    """
    Extract the dominant emotional tone from the scene manifest.
    Returns the most common emotional_tone across all scenes.
    """
    from collections import Counter
    tones = [
        s.get("emotional_tone", "default")
        for s in scene_manifest.get("scenes", [])
        if s.get("emotional_tone")
    ]
    if not tones:
        return "default"
    return Counter(tones).most_common(1)[0][0]
