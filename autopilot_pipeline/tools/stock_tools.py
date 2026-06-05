"""
tools/stock_tools.py
─────────────────────────────────────────────────────────────────────────────
Unified Pixabay & Pexels API wrappers for searching and downloading 
royalty-free B-roll.

Requires: PEXELS_API_KEY (pexels.com/api) and PIXABAY_API_KEY (pixabay.com/api)
Caches downloaded clips under data/assets/stock/ to avoid re-downloading.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import hashlib
import os
import shutil
import urllib.parse
from pathlib import Path
from typing import Optional, List, Dict, Any

import requests
import structlog

log = structlog.get_logger(__name__)

CACHE_DIR     = Path(os.getenv("STOCK_CACHE_DIR", "data/assets/stock")).resolve()
MIN_WIDTH     = 1080
MIN_DURATION  = 4      # skip very short clips
MAX_DURATION  = 60     # skip very long clips

# ─────────────────────────────────────────────────────────────────────────────
# Pixabay API
# ─────────────────────────────────────────────────────────────────────────────
def search_pixabay(keyword: str, min_duration: int = MIN_DURATION) -> List[Dict[str, Any]]:
    api_key = os.getenv("PIXABAY_API_KEY")
    if not api_key:
        log.warning("pixabay.no_api_key")
        return []

    try:
        params = {
            "q": keyword,
            "video_type": "all",
            "per_page": 50,
            "key": api_key,
        }
        url = f"https://pixabay.com/api/videos/?{urllib.parse.urlencode(params)}"
        resp = requests.get(url, timeout=(10, 30))
        resp.raise_for_status()
        
        results = []
        hits = resp.json().get("hits", [])
        for hit in hits:
            duration = hit.get("duration", 0)
            if duration < min_duration or duration > MAX_DURATION:
                continue
            
            videos = hit.get("videos", {})
            for size, data in videos.items():
                if isinstance(data, dict):
                    w = data.get("width", 0)
                    h = data.get("height", 0)
                    if w >= MIN_WIDTH or h >= MIN_WIDTH:
                        results.append({
                            "provider": "pixabay",
                            "url": data.get("url"),
                            "duration": duration,
                            "width": w,
                            "height": h
                        })
                        break # Got the highest quality for this hit
        return results
    except Exception as e:
        log.warning("pixabay.search_error", keyword=keyword, error=str(e))
        return []

# ─────────────────────────────────────────────────────────────────────────────
# Pexels API
# ─────────────────────────────────────────────────────────────────────────────
def search_pexels(keyword: str, orientation: str = "landscape", min_duration: int = MIN_DURATION) -> List[Dict[str, Any]]:
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        log.warning("pexels.no_api_key")
        return []

    try:
        url = "https://api.pexels.com/videos/search"
        headers = {"Authorization": api_key}
        params = {
            "query": keyword,
            "per_page": 20,
            "orientation": orientation
        }
        resp = requests.get(url, headers=headers, params=params, timeout=(10, 30))
        resp.raise_for_status()
        
        results = []
        videos = resp.json().get("videos", [])
        for v in videos:
            duration = v.get("duration", 0)
            if duration < min_duration or duration > MAX_DURATION:
                continue
            
            files = v.get("video_files", [])
            valid_files = [f for f in files if f.get("file_type") == "video/mp4" and (f.get("width", 0) >= MIN_WIDTH or f.get("height", 0) >= MIN_WIDTH)]
            if not valid_files:
                continue
            
            # Sort to get highest quality
            valid_files.sort(key=lambda f: f.get("width", 0) * f.get("height", 0), reverse=True)
            best_file = valid_files[0]
            
            results.append({
                "provider": "pexels",
                "url": best_file.get("link"),
                "duration": duration,
                "width": best_file.get("width", 0),
                "height": best_file.get("height", 0)
            })
        return results
    except Exception as e:
        log.warning("pexels.search_error", keyword=keyword, error=str(e))
        return []

# ─────────────────────────────────────────────────────────────────────────────
# Download & Unified Logic
# ─────────────────────────────────────────────────────────────────────────────
def _cache_path(url: str) -> Path:
    # URL might have query strings; strip them to keep cache consistent
    base_url = url.split("?")[0]
    slug = hashlib.md5(base_url.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{slug}.mp4"

def download_video(url: str, output_path: str) -> str:
    """Download a video URL to output_path. Uses local cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = _cache_path(url)

    if cached.exists() and cached.stat().st_size > 0:
        log.debug("stock.cache_hit", url=url[:60])
        shutil.copy2(str(cached), output_path)
        return output_path

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        with requests.get(url, stream=True, headers=headers, timeout=(15, 60)) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    
        if os.path.getsize(output_path) > 0:
            shutil.copy2(output_path, str(cached))
            log.debug("stock.downloaded", url=url[:60], path=output_path)
            return output_path
        else:
            raise Exception("Downloaded file is empty")
    except Exception as e:
        log.warning("stock.download_failed", url=url[:60], error=str(e))
        if os.path.exists(output_path):
            os.remove(output_path)
        raise

def search_and_download_stock_clip(
    keyword: str,
    output_dir: str,
    filename: str,
    orientation: str = "portrait",
    preferred_width: int = 1080,
    preferred_height: int = 1920
) -> tuple[Optional[str], str]:
    """
    Search Pixabay and Pexels for keyword, score results based on match to 
    target resolution, and download the best clip.
    Returns (absolute_path, provider_name) or (None, "") on failure.
    """
    all_clips = []
    
    # 1. Fetch from both platforms
    pix_clips = search_pixabay(keyword)
    pex_clips = search_pexels(keyword, orientation=orientation)
    
    # Fallback to shorter keyword if nothing found
    if not pix_clips and not pex_clips and " " in keyword:
        short_kw = keyword.split()[-1]
        log.info("stock.retrying_short_keyword", short=short_kw)
        pix_clips = search_pixabay(short_kw)
        pex_clips = search_pexels(short_kw, orientation=orientation)
        
    all_clips.extend(pix_clips)
    all_clips.extend(pex_clips)
    
    if not all_clips:
        return None, ""
        
    # 2. Rank clips based on how close they are to the desired aspect ratio/resolution
    # Lower score is better.
    def score_clip(c):
        w = c.get("width", 0)
        h = c.get("height", 0)
        if w == 0 or h == 0:
            return 999999
        # Perfect aspect ratio match is huge bonus
        ratio_diff = abs((w/h) - (preferred_width/preferred_height))
        res_diff = abs(w - preferred_width) + abs(h - preferred_height)
        return (ratio_diff * 10000) + res_diff
        
    all_clips.sort(key=score_clip)
    
    # 3. Download best clip
    output_path = str(Path(output_dir) / filename)
    for clip in all_clips[:5]: # Try top 5
        url = clip.get("url")
        if not url:
            continue
        try:
            res_path = download_video(url, output_path)
            return res_path, clip.get("provider", "unknown")
        except Exception:
            continue
            
    return None, ""
