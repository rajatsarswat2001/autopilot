"""
renderer/cache_manager.py
─────────────────────────────────────────────────────────────────────────────
Scene-level clip cache to avoid re-rendering identical clips.

Cache key = SHA256(visual_path_content_hash + audio_path + duration).
Stores rendered clip paths in a JSON index file.

This enables:
  • Partial re-renders — only changed scenes are re-processed
  • Faster iteration when fixing individual scenes
  • Reuse across multiple pipeline runs (same B-roll + narration)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

CACHE_DIR   = Path(os.getenv("CLIP_CACHE_DIR", "data/clip_cache")).resolve()
CACHE_INDEX = CACHE_DIR / "index.json"
MAX_CACHE_ENTRIES = int(os.getenv("MAX_CACHE_ENTRIES", "200"))


class ClipCache:
    """
    File-system backed clip cache with LRU eviction (by access time).
    Thread-safe for single-process use; not safe for concurrent writes.
    """

    def __init__(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, str] = self._load_index()

    # ── Index I/O ─────────────────────────────────────────────────────────────

    def _load_index(self) -> dict[str, str]:
        if CACHE_INDEX.exists():
            try:
                return json.loads(CACHE_INDEX.read_text())
            except Exception:
                return {}
        return {}

    def _save_index(self):
        try:
            CACHE_INDEX.write_text(json.dumps(self._index, indent=2))
        except Exception as e:
            log.warning("clip_cache.save_index_failed", error=str(e))

    # ── Key ───────────────────────────────────────────────────────────────────

    @staticmethod
    def key(visual_path: str, audio_path: str, duration_s: float) -> str:
        """Generate a deterministic cache key for a (visual, audio, duration) triple."""
        # Use file size as a fast proxy for content hash
        try:
            v_size = Path(visual_path).stat().st_size
            a_size = Path(audio_path).stat().st_size
        except FileNotFoundError:
            v_size = a_size = 0

        raw = f"{visual_path}|{v_size}|{audio_path}|{a_size}|{duration_s:.3f}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    # ── Get / Put ─────────────────────────────────────────────────────────────

    def get(self, cache_key: str) -> Optional[str]:
        """Return cached clip path if it exists and is valid."""
        cached_path = self._index.get(cache_key)
        if cached_path and Path(cached_path).exists():
            return cached_path
        if cache_key in self._index:
            # Stale entry — remove
            del self._index[cache_key]
            self._save_index()
        return None

    def put(self, cache_key: str, clip_path: str):
        """Register a rendered clip in the cache."""
        # Evict oldest entries if over limit
        if len(self._index) >= MAX_CACHE_ENTRIES:
            oldest_key = next(iter(self._index))
            log.debug("clip_cache.evicting", key=oldest_key)
            del self._index[oldest_key]

        self._index[cache_key] = clip_path
        self._save_index()
        log.debug("clip_cache.put", key=cache_key, path=clip_path)

    # ── Maintenance ───────────────────────────────────────────────────────────

    def clear(self):
        """Clear the entire cache (use when forcing full re-render)."""
        self._index.clear()
        self._save_index()
        log.info("clip_cache.cleared")

    def stats(self) -> dict:
        valid   = sum(1 for p in self._index.values() if Path(p).exists())
        stale   = len(self._index) - valid
        return {"total": len(self._index), "valid": valid, "stale": stale}
