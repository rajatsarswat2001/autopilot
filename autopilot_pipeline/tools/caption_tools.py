"""
tools/caption_tools.py
─────────────────────────────────────────────────────────────────────────────
Modern styled caption / subtitle generator for the AutoPilot pipeline.

Produces an ASS (Advanced SubStation Alpha) subtitle file with:
  - Word-by-word timing derived from narration text + scene durations
  - TikTok/Reels-style large bold captions at bottom of frame
  - Per-niche color themes (yellow/cyan/gold highlights)
  - Semi-transparent background box for readability
  - No Whisper transcription needed — uses exact script narration

Usage:
    from tools.caption_tools import generate_captions
    ass_path = generate_captions(scene_manifest, timing_manifest, output_dir, niche)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Style presets per niche
# ─────────────────────────────────────────────────────────────────────────────

# ASS colours are in &HAABBGGRR format (alpha, blue, green, red)
_STYLES: dict[str, dict] = {
    "personal_finance": {
        "primary":    "&H00FFFFFF",   # white text
        "highlight":  "&H0000FFFF",   # yellow highlight (active word)
        "shadow":     "&H00000000",   # black shadow
        "back_colour":"&H99000000",   # semi-transparent black box
        "fontname":   "Arial",
        "fontsize":   "68",
        "bold":       "1",
        "margin_v":   "80",           # pixels from bottom
    },
    "saas_tools": {
        "primary":    "&H00FFFFFF",
        "highlight":  "&H00FFFF00",   # cyan highlight
        "shadow":     "&H00000000",
        "back_colour":"&H99000000",
        "fontname":   "Arial",
        "fontsize":   "66",
        "bold":       "1",
        "margin_v":   "80",
    },
    "legal_tax": {
        "primary":    "&H00FFFFFF",
        "highlight":  "&H0000D7FF",   # gold highlight
        "shadow":     "&H00000000",
        "back_colour":"&H99000000",
        "fontname":   "Arial",
        "fontsize":   "64",
        "bold":       "1",
        "margin_v":   "80",
    },
    "senior_health": {
        "primary":    "&H00FFFFFF",
        "highlight":  "&H0000FF7F",   # green highlight
        "shadow":     "&H00000000",
        "back_colour":"&H99000000",
        "fontname":   "Arial",
        "fontsize":   "72",           # larger for senior audience
        "bold":       "1",
        "margin_v":   "100",
    },
    "storytelling": {
        "primary":    "&H00FFFFFF",
        "highlight":  "&H000080FF",   # orange highlight
        "shadow":     "&H00000000",
        "back_colour":"&H99000000",
        "fontname":   "Arial",
        "fontsize":   "68",
        "bold":       "1",
        "margin_v":   "80",
    },
    "default": {
        "primary":    "&H00FFFFFF",
        "highlight":  "&H0000FFFF",   # yellow
        "shadow":     "&H00000000",
        "back_colour":"&H99000000",
        "fontname":   "Arial",
        "fontsize":   "68",
        "bold":       "1",
        "margin_v":   "80",
    },
}

# Average speaking rate (words per second) — Edge TTS is about 3.0 WPS
_DEFAULT_WPS = 3.0
# Min/max word hold time in seconds
_MIN_WORD_S  = 0.18
_MAX_WORD_S  = 0.80
# Words per caption line (controls how many words appear together)
_WORDS_PER_LINE = 4


# ─────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _seconds_to_ass(secs: float) -> str:
    """Convert seconds to ASS timestamp format H:MM:SS.cc"""
    secs = max(0.0, secs)
    h    = int(secs // 3600)
    m    = int((secs % 3600) // 60)
    s    = int(secs % 60)
    cs   = int(round((secs - int(secs)) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _clean_text(text: str) -> str:
    """Strip markdown, extra whitespace, and special chars from narration."""
    text = re.sub(r"\*+", "", text)          # remove bold/italic markers
    text = re.sub(r"#+\s*", "", text)        # remove headings
    text = re.sub(r"\s+", " ", text)         # normalise spaces
    return text.strip()


def _split_into_groups(words: list[str], per_line: int) -> list[list[str]]:
    """Split word list into groups of N for caption lines."""
    return [words[i:i + per_line] for i in range(0, len(words), per_line)]


# ─────────────────────────────────────────────────────────────────────────────
# Core: build word timings from scene data
# ─────────────────────────────────────────────────────────────────────────────

def build_word_timings(
    scenes: list[dict],          # list of scene dicts with narration + duration_hint_s
    global_offset: float = 0.0,  # seconds offset (e.g. for intro silence)
) -> list[dict]:
    """
    For each word in each scene's narration, compute (start, end, word, scene_id).
    Distributes words proportionally across each scene's audio duration.
    Returns list of dicts: {start, end, word, scene_id, is_last_in_group}
    """
    timings: list[dict] = []
    cursor = global_offset

    for scene in scenes:
        scene_id  = scene.get("scene_id", 0)
        narration = _clean_text(scene.get("narration", ""))
        duration  = float(scene.get("duration_s") or scene.get("duration_hint_s") or 0.0)

        if not narration:
            cursor += duration
            continue

        words = narration.split()
        if not words:
            cursor += duration
            continue

        # If duration is unknown/0, estimate from word count
        if duration <= 0:
            duration = len(words) / _DEFAULT_WPS

        # Time per word (clamped)
        time_per_word = min(_MAX_WORD_S, max(_MIN_WORD_S, duration / len(words)))

        # Ensure words fill exactly the scene duration
        actual_total = time_per_word * len(words)
        scale = duration / actual_total if actual_total > 0 else 1.0
        time_per_word *= scale

        groups = _split_into_groups(words, _WORDS_PER_LINE)
        word_idx = 0

        for group in groups:
            group_start = cursor + word_idx * time_per_word
            for j, word in enumerate(group):
                w_start = cursor + word_idx * time_per_word
                w_end   = w_start + time_per_word
                is_last = (j == len(group) - 1)
                timings.append({
                    "start":           w_start,
                    "end":             w_end,
                    "word":            word,
                    "scene_id":        scene_id,
                    "group":           group,
                    "group_start":     group_start,
                    "group_end":       group_start + len(group) * time_per_word,
                    "word_idx_in_group": j,
                    "is_last_in_group": is_last,
                })
                word_idx += 1

        cursor += duration

    return timings


# ─────────────────────────────────────────────────────────────────────────────
# ASS file generation
# ─────────────────────────────────────────────────────────────────────────────

def _ass_header(style: dict, resolution: tuple[int, int] = (1920, 1080)) -> str:
    """Generate the ASS file header with style definitions."""
    w, h = resolution
    fontsize  = style["fontsize"]
    fontname  = style["fontname"]
    bold      = style["bold"]
    primary   = style["primary"]
    shadow    = style["shadow"]
    back      = style["back_colour"]
    margin_v  = style["margin_v"]

    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
ScaledBorderAndShadow: yes
YCbCr Matrix: None

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{fontname},{fontsize},{primary},&H000000FF,&H00000000,{back},{bold},0,0,0,100,100,0,0,3,4,0,2,40,40,{margin_v},1
Style: Highlight,{fontname},{fontsize},{style["highlight"]},&H000000FF,&H00000000,{back},{bold},0,0,0,110,110,0,0,3,4,0,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def generate_ass_file(
    word_timings: list[dict],
    output_path:  str,
    niche:        str = "default",
    resolution:   tuple[int, int] = (1920, 1080),
) -> str:
    """
    Generate a complete ASS subtitle file with word-level highlight captions.

    Strategy: For each word group (N words per line), emit one Dialogue line
    per word where the active word is highlighted and all others are normal.
    This creates the word-pop highlight effect.
    """
    style = _STYLES.get(niche, _STYLES["default"])
    lines = [_ass_header(style, resolution)]

    # Group timings by their group identity (group_start is unique per group)
    from itertools import groupby
    groups: dict[tuple, list[dict]] = {}
    for t in word_timings:
        key = (t["scene_id"], t["group_start"])
        if key not in groups:
            groups[key] = []
        groups[key].append(t)

    highlight_col = style["highlight"]  # e.g. &H0000FFFF (yellow)
    normal_col    = style["primary"]    # white

    for (scene_id, group_start), group_words in sorted(groups.items()):
        # For each word in the group, emit a subtitle line where that word is highlighted
        for active_idx, active_word in enumerate(group_words):
            t_start = _seconds_to_ass(active_word["start"])
            t_end   = _seconds_to_ass(active_word["end"])

            # Build the text with color tags
            parts = []
            for i, w in enumerate(group_words):
                word_text = w["word"]
                if i == active_idx:
                    # Highlighted active word — slightly larger + colored
                    parts.append(
                        f"{{\\c{highlight_col}\\fscx110\\fscy110}}{word_text.upper()}{{\\r}}"
                    )
                else:
                    parts.append(
                        f"{{\\c{normal_col}}}{word_text}{{\\r}}"
                    )

            text = "  ".join(parts)  # double-space for readability

            lines.append(
                f"Dialogue: 0,{t_start},{t_end},Default,,0,0,0,,{text}"
            )

    content = "\n".join(lines)
    Path(output_path).write_text(content, encoding="utf-8-sig")
    log.info("captions.ass_written", path=output_path, events=len(groups))
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# High-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_captions(
    scene_manifest:  dict,
    timing_manifest: dict,
    output_dir:      str,
    niche:           str = "default",
    video_id:        str = "video",
) -> Optional[str]:
    """
    Full caption generation pipeline:
      1. Extract scenes with timing info
      2. Build word timings
      3. Write ASS file to output_dir

    Returns path to .ass file, or None on error.
    """
    try:
        scenes_raw = scene_manifest.get("scenes", [])
        if not scenes_raw:
            log.warning("captions.no_scenes")
            return None

        # Merge duration_hint_s from timing_manifest into scenes
        timing_scenes = {
            s["scene_id"]: s
            for s in timing_manifest.get("scenes", [])
        } if timing_manifest else {}

        scenes = []
        for s in scenes_raw:
            sid = s.get("scene_id", 0)
            merged = dict(s)
            if sid in timing_scenes:
                ts = timing_scenes[sid]
                # AudioScene uses duration_s; fall back to duration_hint_s for compat
                dur = ts.get("duration_s") or ts.get("duration_hint_s") or 0
                merged["duration_s"] = float(dur)
            scenes.append(merged)

        word_timings = build_word_timings(scenes)
        if not word_timings:
            log.warning("captions.no_timings")
            return None

        ass_path = str(Path(output_dir) / f"{video_id}_captions.ass")
        generate_ass_file(word_timings, ass_path, niche=niche)

        log.info("captions.done", path=ass_path, words=len(word_timings), niche=niche)
        return ass_path

    except Exception as e:
        log.warning("captions.failed", error=str(e))
        return None
