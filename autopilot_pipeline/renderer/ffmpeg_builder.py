"""
renderer/ffmpeg_builder.py
─────────────────────────────────────────────────────────────────────────────
FFmpeg command builder and executor.

Takes a RenderPlan from timeline_compiler.py and produces the final MP4
using a concat-demuxer approach:

  1. Write a concat list file (inputs.txt) referencing each processed clip
  2. Mux audio + video per clip (silent video + voiceover WAV)
  3. Concatenate all clips with fade transitions
  4. Mix background music at low volume
  5. Loudnorm final audio to -16 LUFS
  6. Output to final_video_path

This is the ONLY place in the codebase that calls subprocess ffmpeg for
the final assembly — keeps complexity contained and auditable.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import structlog

from renderer.timeline_compiler import RenderPlan

log = structlog.get_logger(__name__)

FFMPEG_PRESET   = os.getenv("FFMPEG_PRESET",   "fast")    # ultrafast/fast/medium/slow
FFMPEG_CRF      = int(os.getenv("FFMPEG_CRF",   "20"))    # 18=near-lossless, 23=default
FFMPEG_AUDIO_BR = os.getenv("FFMPEG_AUDIO_BR",  "192k")
TARGET_LUFS     = float(os.getenv("TARGET_LUFS", "-16.0"))
SCRATCH_DIR     = Path(os.getenv("SCRATCH_DIR",  "outputs/video/scratch")).resolve()


# ─────────────────────────────────────────────────────────────────────────────
# Per-clip mux (video + audio)
# ─────────────────────────────────────────────────────────────────────────────

def _mux_clip(visual_path: str, audio_path: str, output_path: str, duration_s: float, width: int = 1920, height: int = 1080) -> str:
    """Combine one video clip with its voiceover WAV, scaling and padding to target resolution."""
    vf_filter = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i",  visual_path,
            "-i",  audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-vf",  vf_filter,
            "-t",   str(duration_s),
            "-c:v", "libx264",
            "-preset", FFMPEG_PRESET,
            "-crf",    str(FFMPEG_CRF),
            "-c:a",    "aac",
            "-b:a",    FFMPEG_AUDIO_BR,
            "-shortest",
            output_path,
        ],
        check=True, capture_output=True, timeout=300,
    )
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Fade transition filter
# ─────────────────────────────────────────────────────────────────────────────

def _build_fade_filter(clip_paths: list[str], durations: list[float], fade_s: float = 0.25) -> str:
    """
    Build an FFmpeg filter_complex string that:
      - Applies fade-out to end of each clip
      - Applies fade-in to start of each clip
      - Concatenates all clips
    """
    n = len(clip_paths)
    parts = []

    for i in range(n):
        dur   = durations[i]
        fade_out_t = max(0, dur - fade_s)
        parts.append(
            f"[{i}:v]fade=t=in:st=0:d={fade_s},"
            f"fade=t=out:st={fade_out_t:.3f}:d={fade_s}[v{i}];"
        )
        parts.append(
            f"[{i}:a]afade=t=in:st=0:d={fade_s},"
            f"afade=t=out:st={fade_out_t:.3f}:d={fade_s}[a{i}];"
        )

    vstreams = "".join(f"[v{i}]" for i in range(n))
    astreams = "".join(f"[a{i}]" for i in range(n))
    parts.append(f"{vstreams}concat=n={n}:v=1:a=0[outv];")
    parts.append(f"{astreams}concat=n={n}:v=0:a=1[outa]")

    return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────

def render_timeline(plan: RenderPlan, output_path: str | None = None,
                    caption_ass_path: str | None = None) -> str:
    """
    Execute the full render pipeline for a RenderPlan.
    Returns the path to the finished MP4.
    """
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    final_path = output_path or plan.output_path

    # ── Step 1: Mux each clip (video + voiceover) ─────────────────────────────
    muxed_clips: list[str] = []
    for i, clip in enumerate(plan.clips):
        mux_out = str(SCRATCH_DIR / f"muxed_{clip.scene_id:03d}.mp4")
        log.info("render.muxing", scene_id=clip.scene_id)
        try:
            _mux_clip(clip.visual_path, clip.audio_path, mux_out, clip.duration_s, plan.width, plan.height)
            muxed_clips.append(mux_out)
        except subprocess.CalledProcessError as e:
            log.error("render.mux_failed", scene_id=clip.scene_id,
                      stderr=e.stderr.decode()[:300] if e.stderr else "")
            raise

    # ── Step 2: Concat via filter_complex with fades ─────────────────────────
    log.info("render.concatenating", clips=len(muxed_clips))
    durations = [c.duration_s for c in plan.clips]

    concat_out = str(SCRATCH_DIR / "concat_no_music.mp4")

    inputs = []
    for p in muxed_clips:
        inputs += ["-i", p]

    filter_str = _build_fade_filter(muxed_clips, durations, fade_s=0.25)

    subprocess.run(
        [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_str,
            "-map", "[outv]",
            "-map", "[outa]",
            "-c:v", "libx264",
            "-preset", FFMPEG_PRESET,
            "-crf",    str(FFMPEG_CRF),
            "-c:a",    "aac",
            "-b:a",    FFMPEG_AUDIO_BR,
            concat_out,
        ],
        check=True, capture_output=True, timeout=1800,
    )

    # ── Step 3: Mix background music ─────────────────────────────────────────
    if plan.music_track_path and Path(plan.music_track_path).exists():
        log.info("render.mixing_music", track=plan.music_track_path)
        music_mixed = str(SCRATCH_DIR / "with_music.mp4")
        total_dur   = plan.total_duration_s
        vol         = plan.music_volume
        fade_in     = plan.music_fade_in
        fade_out    = plan.music_fade_out
        fade_out_st = max(0, total_dur - fade_out)

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", concat_out,
                "-stream_loop", "-1", "-i", plan.music_track_path,
                "-filter_complex",
                f"[1:a]volume={vol},"
                f"afade=t=in:st=0:d={fade_in},"
                f"afade=t=out:st={fade_out_st:.1f}:d={fade_out}[music];"
                f"[music][0:a]sidechaincompress=threshold=0.03:ratio=4:attack=5:release=50[ducked_music];"
                f"[0:a][ducked_music]amix=inputs=2:duration=first[aout]",
                "-map", "0:v",
                "-map", "[aout]",
                "-t", str(total_dur),
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", FFMPEG_AUDIO_BR,
                music_mixed,
            ],
            check=True, capture_output=True, timeout=600,
        )
        concat_out = music_mixed

    # ── Step 4: Loudnorm to target LUFS ────────────────────────────────────
    log.info("render.loudnorm", target_lufs=TARGET_LUFS)
    from tools.ffmpeg_tools import normalise_audio

    # If we have captions, loudnorm into a temp file first
    if caption_ass_path and Path(caption_ass_path).exists():
        loudnorm_out = str(SCRATCH_DIR / "loudnorm_no_captions.mp4")
        normalise_audio(concat_out, loudnorm_out, target_lufs=TARGET_LUFS)

        # ── Step 5: Burn captions (ASS) into video ───────────────────────────
        log.info("render.burning_captions", ass=caption_ass_path)
        # Escape Windows backslashes for FFmpeg filter path
        ass_escaped = caption_ass_path.replace("\\", "/").replace(":", "\\:")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i",  loudnorm_out,
                    "-vf", f"ass='{ass_escaped}'",
                    "-c:v", "libx264",
                    "-preset", FFMPEG_PRESET,
                    "-crf",    str(FFMPEG_CRF),
                    "-c:a",    "copy",
                    final_path,
                ],
                check=True, capture_output=True, timeout=1800,
            )
        except subprocess.CalledProcessError as e:
            log.warning("render.captions_failed",
                        stderr=e.stderr.decode()[:400] if e.stderr else "",
                        fallback="using video without captions")
            # Fallback: rename loudnorm output as final (no captions)
            import shutil
            shutil.copy2(loudnorm_out, final_path)
    else:
        normalise_audio(concat_out, final_path, target_lufs=TARGET_LUFS)

    log.info("render.done", output=final_path)
    return final_path
