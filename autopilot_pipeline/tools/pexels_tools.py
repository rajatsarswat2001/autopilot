"""
tools/pexels_tools.py
─────────────────────────────────────────────────────────────────────────────
Pexels API wrappers for searching and downloading royalty-free B-roll.

Requires: PEXELS_API_KEY (free at pexels.com/api)
Caches downloaded clips under data/assets/pexels/ to avoid re-downloading.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Optional

import requests
import structlog

log = structlog.get_logger(__name__)

PEXELS_BASE   = "https://api.pexels.com/videos"
CACHE_DIR     = Path(os.getenv("PEXELS_CACHE_DIR", "data/assets/pexels")).resolve()
MIN_WIDTH     = 1280   # refuse clips below HD
PREFERRED_W   = 1920
PREFERRED_H   = 1080
MAX_DURATION  = 30     # seconds — don't download huge clips
MIN_DURATION  = 4      # skip very short clips


# ─────────────────────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────────────────────

def search_videos(keyword: str, per_page: int = 10) -> list[dict]:
    """
    Search Pexels for videos matching keyword.
    Returns list of video metadata dicts (already filtered for quality).
    """
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        log.warning("pexels.no_api_key")
        return []

    try:
        resp = requests.get(
            f"{PEXELS_BASE}/search",
            headers={"Authorization": api_key},
            params={
                "query":    keyword,
                "per_page": per_page,
                "size":     "large",
                "orientation": "landscape",
            },
            timeout=15,
        )
        resp.raise_for_status()
        videos = resp.json().get("videos", [])
        return [v for v in videos if _passes_quality(v)]
    except Exception as e:
        log.warning("pexels.search_error", keyword=keyword, error=str(e))
        return []


def _passes_quality(video: dict) -> bool:
    """Return True if the video meets minimum quality requirements."""
    duration = video.get("duration", 0)
    if duration < MIN_DURATION or duration > MAX_DURATION:
        return False
    files = video.get("video_files", [])
    return any(
        f.get("width", 0) >= MIN_WIDTH
        for f in files
        if f.get("file_type") == "video/mp4"
    )


def _best_file_url(video: dict) -> Optional[str]:
    """
    Pick the best (closest to 1920×1080) MP4 file URL from a Pexels video.
    """
    files = [
        f for f in video.get("video_files", [])
        if f.get("file_type") == "video/mp4" and f.get("width", 0) >= MIN_WIDTH
    ]
    if not files:
        return None

    # Sort by proximity to preferred resolution
    files.sort(
        key=lambda f: abs(f.get("width", 0) - PREFERRED_W) + abs(f.get("height", 0) - PREFERRED_H)
    )
    return files[0].get("link")


# ─────────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(url: str) -> Path:
    slug = hashlib.md5(url.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{slug}.mp4"


def download_video(url: str, output_path: str) -> str:
    """Download a video URL to output_path. Uses local cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = _cache_path(url)

    if cached.exists():
        log.debug("pexels.cache_hit", url=url[:60])
        shutil.copy2(str(cached), output_path)
        return output_path

    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        # Save to cache
        shutil.copy2(output_path, str(cached))
        log.debug("pexels.downloaded", url=url[:60], path=output_path)
        return output_path
    except Exception as e:
        log.warning("pexels.download_failed", url=url[:60], error=str(e))
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Combined search + download
# ─────────────────────────────────────────────────────────────────────────────

def search_and_download_video(
    keyword: str,
    output_dir: str,
    filename: str,
) -> Optional[str]:
    """
    Search Pexels for keyword, download best matching clip.
    Returns absolute path to downloaded file, or None on failure.
    """
    videos = search_videos(keyword, per_page=15)

    # Try fallback keyword if no results (strip adjectives from keyword)
    if not videos and " " in keyword:
        short_kw = keyword.split()[-1]  # last word is usually the noun
        log.info("pexels.retrying_short_keyword", short=short_kw)
        videos = search_videos(short_kw, per_page=10)

    if not videos:
        return None

    output_path = str(Path(output_dir) / filename)
    for video in videos[:3]:
        url = _best_file_url(video)
        if not url:
            continue
        try:
            return download_video(url, output_path)
        except Exception:
            continue

    return None
