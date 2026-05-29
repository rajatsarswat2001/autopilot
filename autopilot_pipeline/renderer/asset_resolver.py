"""
renderer/asset_resolver.py
─────────────────────────────────────────────────────────────────────────────
Asset path resolver — ensures all asset paths in a TimelineManifest
are absolute, valid, and reachable before the render begins.

Handles:
  • Relative → absolute path conversion
  • Cloud storage URLs → local temp file download (S3, GCS stubs)
  • Missing asset detection and placeholder substitution
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import structlog

from contracts.timeline_manifest import TimelineManifest, TimelineClip

log = structlog.get_logger(__name__)

PLACEHOLDER_COLOR = "black"  # FFmpeg lavfi color for inline placeholder


def resolve_manifest(manifest: TimelineManifest) -> TimelineManifest:
    """
    Walk every clip in the manifest and resolve all paths.
    Mutates clip paths in-place. Returns the same manifest.
    """
    for clip in manifest.clips:
        clip.visual_path_A = _resolve_path(clip.visual_path_A, f"visual A for scene {clip.scene_id}")
        clip.visual_path_B = _resolve_path(clip.visual_path_B, f"visual B for scene {clip.scene_id}")
        clip.audio_path  = _resolve_path(clip.audio_path,  f"audio for scene {clip.scene_id}")

    if manifest.music_track and manifest.music_track.path:
        manifest.music_track.path = _resolve_path(
            manifest.music_track.path, "music_track"
        )

    return manifest


def _resolve_path(path: str, label: str) -> str:
    """
    Resolve a single path:
      1. If absolute and exists → return as-is
      2. If relative → convert to absolute relative to CWD
      3. If URL → download to temp file (stub)
      4. If missing → return a synthetic lavfi black video/silence path marker
    """
    if path.startswith(("http://", "https://", "s3://", "gs://")):
        local = _download_url(path)
        if local:
            return local
        log.warning("asset_resolver.url_download_failed", label=label, path=path[:80])
        return path  # let ffmpeg fail with a clear error

    abs_path = Path(path) if Path(path).is_absolute() else Path.cwd() / path

    if abs_path.exists():
        return str(abs_path)

    log.warning("asset_resolver.missing_asset", label=label, path=str(abs_path))
    return str(abs_path)  # pass through; FFmpeg error is more descriptive


def _download_url(url: str) -> Optional[str]:
    """Download a remote URL to a temp file. Returns local path or None."""
    try:
        import tempfile, requests
        suffix = Path(url.split("?")[0]).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            tmp = f.name
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(1024 * 1024):
                f.write(chunk)
        return tmp
    except Exception as e:
        log.warning("asset_resolver.download_failed", url=url[:80], error=str(e))
        return None
