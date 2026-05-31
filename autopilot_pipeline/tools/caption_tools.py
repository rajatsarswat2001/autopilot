"""
tools/caption_tools.py
─────────────────────────────────────────────────────────────────────────────
Modern styled caption / subtitle generator for the AutoPilot pipeline.

Produces an ASS (Advanced SubStation Alpha) subtitle file with:
  - Word-by-word timing (WhisperX frame-perfect if available, proportional fallback)
  - TikTok/Reels-style large bold karaoke word-pop captions
  - Per-niche color themes (yellow/cyan/gold highlights)
  - Parametric subtitle randomization (font, size jitter) for demonetization resistance
  - Semi-transparent background box for readability

Caption timing tiers (best → fastest):
  Tier 1: WhisperX    — frame-perfect word-boundary alignment from actual audio WAV
  Tier 2: Proportional — distributes words evenly across measured audio duration (fallback)

Usage:
    from tools.caption_tools import generate_captions
    ass_path = generate_captions(scene_manifest, timing_manifest, output_dir, niche)

    # With WhisperX word-boundary alignment from actual audio:
    ass_path = generate_captions(..., audio_path="/path/to/narration.wav")
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Style presets per niche
# ASS colours: &HAABBGGRR (alpha, blue, green, red)
# ─────────────────────────────────────────────────────────────────────────────

_BASE_STYLES: dict[str, dict] = {
    "personal_finance": {
        "primary":    "&H00FFFFFF",   # white text
        "highlight":  "&H0000FFFF",   # yellow highlight (active word)
        "shadow":     "&H00000000",   # black shadow
        "back_colour":"&H99000000",   # semi-transparent black box
        "fontnames":  ["LiberationSans-Bold", "Liberation Sans", "DejaVuSans-Bold", "DejaVu Sans"],
        "fontsize":   68,
        "bold":       "1",
        "margin_v":   80,
    },
    "saas_tools": {
        "primary":    "&H00FFFFFF",
        "highlight":  "&H00FFFF00",   # cyan highlight
        "shadow":     "&H00000000",
        "back_colour":"&H99000000",
        "fontnames":  ["LiberationSans-Bold", "Liberation Sans", "DejaVuSans-Bold", "DejaVu Sans"],
        "fontsize":   66,
        "bold":       "1",
        "margin_v":   80,
    },
    "legal_tax": {
        "primary":    "&H00FFFFFF",
        "highlight":  "&H0000D7FF",   # gold highlight
        "shadow":     "&H00000000",
        "back_colour":"&H99000000",
        "fontnames":  ["LiberationSans-Bold", "Liberation Sans", "DejaVuSans-Bold", "DejaVu Sans"],
        "fontsize":   64,
        "bold":       "1",
        "margin_v":   80,
    },
    "senior_health": {
        "primary":    "&H00FFFFFF",
        "highlight":  "&H0000FF7F",   # green highlight
        "shadow":     "&H00000000",
        "back_colour":"&H99000000",
        "fontnames":  ["LiberationSans-Bold", "Liberation Sans", "DejaVuSans-Bold", "DejaVu Sans"],
        "fontsize":   72,           # larger for senior audience
        "bold":       "1",
        "margin_v":   100,
    },
    "storytelling": {
        "primary":    "&H00FFFFFF",
        "highlight":  "&H000080FF",   # orange highlight
        "shadow":     "&H00000000",
        "back_colour":"&H99000000",
        "fontnames":  ["LiberationSans-Bold", "Liberation Sans", "DejaVuSans-Bold", "DejaVu Sans"],
        "fontsize":   68,
        "bold":       "1",
        "margin_v":   80,
    },
    "default": {
        "primary":    "&H00FFFFFF",
        "highlight":  "&H0000FFFF",   # yellow
        "shadow":     "&H00000000",
        "back_colour":"&H99000000",
        "fontnames":  ["LiberationSans-Bold", "Liberation Sans", "DejaVuSans-Bold", "DejaVu Sans"],
        "fontsize":   68,
        "bold":       "1",
        "margin_v":   80,
    },
}

# Speaking rate / timing constants
_DEFAULT_WPS    = 2.8   # words per second (realistic Chatterbox rate; Edge=3.0)
_MIN_WORD_S     = 0.18
_MAX_WORD_S     = 0.80
_WORDS_PER_LINE = 4     # words per caption chunk


# ─────────────────────────────────────────────────────────────────────────────
# Parametric style randomiser
# Generates a per-video fingerprint to bypass YouTube "reused content" detection
# ─────────────────────────────────────────────────────────────────────────────

def _get_parametric_style(niche: str, video_id: str = "default") -> dict:
    """
    Return a niche style dict with small per-video randomisation applied.
    Uses video_id as seed so the same video always produces the same style.

    Randomises:
      - Font selection (from niche font pool)
      - Font size ±2 px
      - MarginV ±4 px
    """
    base = dict(_BASE_STYLES.get(niche, _BASE_STYLES["default"]))

    # Deterministic seed from video_id for reproducibility
    seed = int(hashlib.md5(video_id.encode()).hexdigest()[:8], 16)
    rng  = random.Random(seed)

    base["fontname"] = rng.choice(base["fontnames"])
    base["fontsize"] = str(base["fontsize"] + rng.randint(-2, 2))
    base["margin_v"] = str(base["margin_v"] + rng.randint(-4, 4))

    return base


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
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_into_groups(words: list[str], per_line: int) -> list[list[str]]:
    """Split word list into groups of N for caption lines."""
    return [words[i:i + per_line] for i in range(0, len(words), per_line)]


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1: WhisperX word-boundary alignment
# ─────────────────────────────────────────────────────────────────────────────

def _try_whisperx_align(audio_path: str, scenes: list[dict]) -> Optional[list[dict]]:
    """
    Use WhisperX to extract frame-perfect word-level timestamps from the audio.
    Returns a flat list of {start, end, word} dicts, or None if WhisperX unavailable.

    WhisperX produces karaoke-grade alignment — words snap to actual phoneme boundaries.
    Falls back silently to proportional timing if not installed.
    """
    try:
        import whisperx  # type: ignore
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        log.info("captions.whisperx.loading", device=device)
        model = whisperx.load_model("base.en", device=device, compute_type=compute_type)
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=8)

        # Alignment (maps to word-level timestamps)
        align_model, metadata = whisperx.load_align_model(
            language_code="en", device=device
        )
        aligned = whisperx.align(
            result["segments"], align_model, metadata, audio, device
        )

        word_timings: list[dict] = []
        for segment in aligned.get("word_segments", []):
            word = segment.get("word", "").strip()
            start = segment.get("start", 0.0)
            end   = segment.get("end", 0.0)
            if word:
                word_timings.append({"word": word, "start": start, "end": end})

        log.info("captions.whisperx.aligned", words=len(word_timings))
        return word_timings if word_timings else None

    except ImportError:
        log.debug("captions.whisperx.not_installed",
                  msg="Install whisperx for frame-perfect captions: pip install whisperx")
        return None
    except Exception as e:
        log.warning("captions.whisperx.failed", error=str(e)[:120])
        return None


def _enrich_whisperx_timings(flat_timings: list[dict], scenes: list[dict]) -> list[dict]:
    """
    Convert flat WhisperX {word, start, end} list into the grouped format
    expected by generate_ass_file (adds scene_id, group, group_start etc).
    Assigns scene IDs by matching word timestamps to scene time windows.
    """
    # Build scene time windows from timing info
    scene_windows: list[tuple[float, float, int]] = []
    cursor = 0.0
    for s in scenes:
        dur = float(s.get("duration_s") or s.get("duration_hint_s") or 0.0)
        scene_id = s.get("scene_id", 0)
        scene_windows.append((cursor, cursor + dur, scene_id))
        cursor += dur

    def _get_scene_id(t: float) -> int:
        for start, end, sid in scene_windows:
            if start <= t < end:
                return sid
        return scene_windows[-1][2] if scene_windows else 0

    # Group words into caption chunks of _WORDS_PER_LINE
    grouped: list[dict] = []
    words = flat_timings
    for i in range(0, len(words), _WORDS_PER_LINE):
        group_words = words[i:i + _WORDS_PER_LINE]
        group_texts = [w["word"] for w in group_words]
        group_start = group_words[0]["start"]

        for j, w in enumerate(group_words):
            grouped.append({
                "start":            w["start"],
                "end":              w["end"],
                "word":             w["word"],
                "scene_id":         _get_scene_id(w["start"]),
                "group":            group_texts,
                "group_start":      group_start,
                "group_end":        group_words[-1]["end"],
                "word_idx_in_group": j,
                "is_last_in_group": j == len(group_words) - 1,
            })

    return grouped


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2: Proportional timing (fallback)
# ─────────────────────────────────────────────────────────────────────────────

def build_word_timings(
    scenes: list[dict],
    global_offset: float = 0.0,
) -> list[dict]:
    """
    Proportional word timing fallback (no audio required).
    For each word in each scene's narration, compute (start, end, word, scene_id).
    Returns list of dicts: {start, end, word, scene_id, is_last_in_group, ...}
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

        if duration <= 0:
            duration = len(words) / _DEFAULT_WPS

        total_chars = sum(len(w) for w in words)
        char_duration = duration / total_chars if total_chars > 0 else 0

        word_durations = []
        for w in words:
            # Proportional time based on word length
            w_dur = char_duration * len(w)
            w_dur = min(_MAX_WORD_S, max(_MIN_WORD_S, w_dur))
            word_durations.append(w_dur)

        actual_total = sum(word_durations)
        scale = duration / actual_total if actual_total > 0 else 1.0
        word_durations = [d * scale for d in word_durations]

        groups = _split_into_groups(words, _WORDS_PER_LINE)
        word_idx = 0
        scene_cursor = cursor

        for group in groups:
            group_start = scene_cursor
            
            # First pass to find group end
            group_dur = sum(word_durations[word_idx + k] for k in range(len(group)))
            group_end = group_start + group_dur
            
            for j, word in enumerate(group):
                w_dur = word_durations[word_idx]
                w_start = scene_cursor
                w_end = w_start + w_dur
                is_last = (j == len(group) - 1)
                
                timings.append({
                    "start":            w_start,
                    "end":              w_end,
                    "word":             word,
                    "scene_id":         scene_id,
                    "group":            group,
                    "group_start":      group_start,
                    "group_end":        group_end,
                    "word_idx_in_group": j,
                    "is_last_in_group": is_last,
                })
                scene_cursor += w_dur
                word_idx += 1

        cursor += duration

    return timings


# ─────────────────────────────────────────────────────────────────────────────
# ASS file generation
# ─────────────────────────────────────────────────────────────────────────────

def _ass_header(style: dict, resolution: tuple[int, int] = (1920, 1080)) -> str:
    """Generate the ASS file header with style definitions."""
    w, h      = resolution
    fontsize  = style["fontsize"]
    fontname  = style.get("fontname", "Arial")
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
    video_id:     str = "default",
) -> str:
    """
    Generate a complete ASS subtitle file with karaoke word-pop highlight captions.
    Applies parametric randomisation for demonetization resistance.
    """
    style = _get_parametric_style(niche, video_id)
    lines = [_ass_header(style, resolution)]

    groups: dict[tuple, list[dict]] = {}
    for t in word_timings:
        key = (t["scene_id"], t["group_start"])
        if key not in groups:
            groups[key] = []
        groups[key].append(t)

    highlight_col = style["highlight"]
    normal_col    = style["primary"]

    for (scene_id, group_start), group_words in sorted(groups.items()):
        for active_idx, active_word in enumerate(group_words):
            t_start = _seconds_to_ass(active_word["start"])
            t_end   = _seconds_to_ass(active_word["end"])

            parts = []
            for i, w in enumerate(group_words):
                word_text = w["word"]
                if i == active_idx:
                    # Active word: highlighted, uppercase, slightly larger
                    parts.append(
                        f"{{\\c{highlight_col}\\fscx110\\fscy110}}{word_text.upper()}{{\\r}}"
                    )
                else:
                    parts.append(f"{{\\c{normal_col}}}{word_text}{{\\r}}")

            text = "  ".join(parts)
            lines.append(f"Dialogue: 0,{t_start},{t_end},Default,,0,0,0,,{text}")

    content = "\n".join(lines)
    Path(output_path).write_text(content, encoding="utf-8-sig")
    log.info("captions.ass_written", path=output_path, events=len(groups),
             font=style.get("fontname"), fontsize=style.get("fontsize"))
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
    audio_path:      Optional[str] = None,    # if set, try WhisperX alignment
) -> Optional[str]:
    """
    Full caption generation pipeline:
      Tier 1 (if audio_path set): WhisperX frame-perfect word-boundary alignment
      Tier 2: Proportional word timing from narration text + scene durations

    Returns path to .ass file, or None on error.
    """
    try:
        scenes_raw = scene_manifest.get("scenes", [])
        if not scenes_raw:
            log.warning("captions.no_scenes")
            return None

        # Merge duration_s from timing_manifest into scenes
        timing_scenes = {
            s["scene_id"]: s
            for s in timing_manifest.get("scenes", [])
        } if timing_manifest else {}

        scenes: list[dict] = []
        for s in scenes_raw:
            sid    = s.get("scene_id", 0)
            merged = dict(s)
            if sid in timing_scenes:
                ts  = timing_scenes[sid]
                dur = ts.get("duration_s") or ts.get("duration_hint_s") or 0
                merged["duration_s"] = float(dur)
            scenes.append(merged)

        # ── Tier 1: WhisperX (frame-perfect from actual audio) ────────────────
        word_timings: Optional[list[dict]] = None
        if audio_path and Path(audio_path).exists():
            log.info("captions.trying_whisperx", audio=audio_path)
            flat_timings = _try_whisperx_align(audio_path, scenes)
            if flat_timings:
                word_timings = _enrich_whisperx_timings(flat_timings, scenes)
                log.info("captions.tier1_whisperx", words=len(word_timings))

        # ── Tier 2: Proportional timing (fallback) ────────────────────────────
        if not word_timings:
            word_timings = build_word_timings(scenes)
            if word_timings:
                log.info("captions.tier2_proportional", words=len(word_timings))

        if not word_timings:
            log.warning("captions.no_timings")
            return None

        ass_path = str(Path(output_dir) / f"{video_id}_captions.ass")
        generate_ass_file(word_timings, ass_path,
                          niche=niche, video_id=video_id)

        log.info("captions.done", path=ass_path, words=len(word_timings), niche=niche)
        return ass_path

    except Exception as e:
        log.warning("captions.failed", error=str(e))
        return None
