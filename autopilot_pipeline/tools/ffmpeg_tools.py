"""
tools/ffmpeg_tools.py
─────────────────────────────────────────────────────────────────────────────
FFmpeg utility wrappers used across the pipeline.

All functions require ffmpeg and ffprobe to be on PATH.
Install: https://ffmpeg.org/download.html or via `winget install ffmpeg`
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Probing
# ─────────────────────────────────────────────────────────────────────────────

def _ffprobe(path: str, select_streams: str = "", show_entries: str = "") -> dict:
    """Run ffprobe and return parsed JSON output."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json"]
    if select_streams:
        cmd += ["-select_streams", select_streams]
    if show_entries:
        cmd += ["-show_entries", show_entries]
    cmd.append(path)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr[:200]}")
    return json.loads(result.stdout)


def measure_audio_duration(path: str) -> float:
    """Return audio duration in seconds via ffprobe. Falls back to 1.0 on error."""
    try:
        info = _ffprobe(path, select_streams="a:0", show_entries="stream=duration")
        streams = info.get("streams", [])
        if streams and "duration" in streams[0]:
            return float(streams[0]["duration"])
        # Try format-level duration
        info2 = _ffprobe(path, show_entries="format=duration")
        return float(info2.get("format", {}).get("duration", 1.0))
    except Exception as e:
        log.warning("ffprobe.audio_duration_failed", path=path, error=str(e))
        return 1.0


def measure_video_duration(path: str) -> float:
    """Return video duration in seconds."""
    try:
        info = _ffprobe(path, show_entries="format=duration")
        return float(info.get("format", {}).get("duration", 0.0))
    except Exception as e:
        log.warning("ffprobe.video_duration_failed", path=path, error=str(e))
        return 0.0


def get_video_resolution(path: str) -> Optional[str]:
    """Return 'WIDTHxHEIGHT' string or None on error."""
    try:
        info = _ffprobe(path, select_streams="v:0", show_entries="stream=width,height")
        streams = info.get("streams", [])
        if streams:
            w = streams[0].get("width", 0)
            h = streams[0].get("height", 0)
            return f"{w}x{h}"
    except Exception as e:
        log.warning("ffprobe.resolution_failed", error=str(e))
    return None


def measure_loudness_lufs(path: str) -> Optional[float]:
    """
    Measure integrated loudness (LUFS) using FFmpeg's ebur128 filter.
    Returns float or None on error.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"],
            capture_output=True, text=True, timeout=120,
        )
        # Parse from stderr
        for line in result.stderr.split("\n"):
            if "I:" in line and "LUFS" in line:
                parts = line.split()
                idx = parts.index("I:")
                return float(parts[idx + 1])
    except Exception as e:
        log.warning("ffmpeg.lufs_failed", error=str(e))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_audio(input_path: str, output_path: str, sample_rate: int = 24000, channels: int = 1) -> str:
    """Convert any audio format to mono WAV at given sample rate."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         "-ar", str(sample_rate), "-ac", str(channels), output_path],
        check=True, capture_output=True,
    )
    return output_path


def loop_video_to_duration(input_path: str, output_path: str, duration_s: float) -> str:
    """Loop a video clip to fill the required duration.

    Fast path: if the input clip is already >= requested duration, trim
    using stream-copy (`-c copy`) which is much faster and avoids
    re-encoding. Otherwise fall back to concatenating multiple copies
    and re-encoding to guarantee exact duration and target pixel layout.
    """
    video_duration = measure_video_duration(input_path)

    # Fast path: simply trim with stream copy when possible
    if video_duration >= duration_s and video_duration > 0:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-t", str(duration_s),
            "-c", "copy",
            output_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError as e:
            log.warning("ffmpeg.trim_copy_failed", input=input_path, error=str(e))

    # Slow path: build a concat list and re-encode to ensure correct length
    num_loops = 1
    if video_duration > 0:
        num_loops = int(duration_s / video_duration) + 1

    loop_file_path = Path(output_path).with_suffix(".looplist.txt")
    try:
        with open(loop_file_path, "w", encoding="utf-8") as f:
            for _ in range(num_loops):
                f.write(f"file '{Path(input_path).resolve()}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(loop_file_path),
            "-t", str(duration_s),
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        return output_path
    except subprocess.CalledProcessError as e:
        log.warning("ffmpeg.loop_failed", input=input_path, error=str(e))
        raise
    finally:
        try:
            if loop_file_path.exists():
                loop_file_path.unlink()
        except Exception:
            pass


def image_to_video(
    image_path: str,
    output_path: str,
    duration_s: float,
    fps: int = 30,
    ken_burns: bool = True,
    motion: str = "zoom_in",
) -> str:
    """
    Convert a still image to a video clip with optional Ken Burns effect.
    """
    scale_filter = "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080"

    if ken_burns:
        MOTIONS = {
            "zoom_in":   "scale=8000:-1,zoompan=z='min(zoom+0.0015,1.5)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={d}:s=1920x1080",
            "zoom_out":  "scale=8000:-1,zoompan=z='if(lte(zoom,1.0),1.5,max(1.001,zoom-0.0015))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={d}:s=1920x1080",
            "pan_left":  "scale=8000:-1,zoompan=z=1.2:x='min(x+1,iw-iw/zoom)':y='ih/2-(ih/zoom/2)':d={d}:s=1920x1080",
            "pan_right": "scale=8000:-1,zoompan=z=1.2:x='max(0,iw-iw/zoom-x)':y='ih/2-(ih/zoom/2)':d={d}:s=1920x1080",
        }
        frames = int(duration_s * fps)
        vf = MOTIONS.get(motion, MOTIONS["zoom_in"]).format(d=frames)
    else:
        vf = scale_filter

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-loop", "1",
            "-framerate", str(fps),
            "-i", image_path,
            "-vf", vf,
            "-t", str(duration_s),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast",
            output_path,
        ],
        check=True, capture_output=True, timeout=300,
    )
    return output_path


def create_ken_burns_effect(
    input_image_path: str,
    output_video_path: str,
    duration_s: float,
    resolution: str = "1920x1080",
    frame_rate: int = 30,
) -> str:
    """
    Generate a video from an image using a randomized Ken Burns effect.

    The function randomly picks one of several zoom/pan motion patterns.
    """
    total_frames = int(duration_s * frame_rate)

    effects = [
        {
            "z": "min(zoom+0.001,1.5)",
            "x": "iw/2-(iw/zoom/2)",
            "y": "ih/2-(ih/zoom/2)",
        },
        {
            "z": "1.2",
            "x": "min(on*iw/200, iw-iw/zoom)",
            "y": "min(on*ih/200, ih-ih/zoom)",
        },
        {
            "z": "1.2",
            "x": "max(iw-iw/zoom-on*iw/200, 0)",
            "y": "max(ih-ih/zoom-on*ih/200, 0)",
        },
        {
            "z": "max(1.0,1.5-0.001*on)",
            "x": "iw/2-(iw/zoom/2)",
            "y": "ih/2-(ih/zoom/2)",
        },
    ]

    effect = random.choice(effects)
    zoom_pan_filter = (
        f"zoompan=z='{effect['z']}':x='{effect['x']}':y='{effect['y']}':"
        f"d={total_frames}:s={resolution}:fps={frame_rate}"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        input_image_path,
        "-vf",
        zoom_pan_filter,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-t",
        str(duration_s),
        output_video_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    except subprocess.CalledProcessError as e:
        log.warning(
            "ffmpeg.ken_burns_failed",
            input=input_image_path,
            error=e.stderr.decode(errors="ignore")[:300],
        )
        raise

    return output_video_path


# ─────────────────────────────────────────────────────────────────────────────
# Thumbnail
# ─────────────────────────────────────────────────────────────────────────────

def generate_thumbnail(
    video_or_image_path: str,
    output_path: str,
    title_text: str = "",
) -> str:
    """
    Generate a YouTube thumbnail (1280×720 JPEG).
    If input is a video: grab first frame.
    If input is an image: scale and crop.
    Optionally overlays title text via PIL.
    """
    path = Path(video_or_image_path)
    suffix = path.suffix.lower()

    if suffix in (".mp4", ".mov", ".avi", ".mkv"):
        # Extract first frame
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_or_image_path,
             "-vframes", "1", "-vf", "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720",
             output_path],
            check=True, capture_output=True,
        )
    else:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_or_image_path,
             "-vf", "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720",
             output_path],
            check=True, capture_output=True,
        )

    # Overlay title text via PIL if available
    if title_text:
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.open(output_path).convert("RGB")
            draw = ImageDraw.Draw(img)

            # Dark gradient bar at bottom
            bar = Image.new("RGBA", (1280, 120), (0, 0, 0, 180))
            img.paste(bar, (0, 600), bar)

            try:
                font = ImageFont.truetype("arial.ttf", 52)
            except Exception:
                font = ImageFont.load_default()

            draw.text((40, 610), title_text[:60], font=font, fill=(255, 255, 255))
            img.save(output_path, "JPEG", quality=95)
        except ImportError:
            pass  # PIL not available — raw frame thumbnail is fine

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalise_audio(input_path: str, output_path: str, target_lufs: float = -16.0) -> str:
    """Loudnorm audio to target LUFS (EBU R128 two-pass)."""
    # Pass 1: measure
    result = subprocess.run(
        ["ffmpeg", "-i", input_path,
         "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    # Parse JSON from stderr
    stderr = result.stderr
    start  = stderr.rfind("{")
    end    = stderr.rfind("}") + 1
    if start >= 0 and end > start:
        stats = json.loads(stderr[start:end])
        il    = stats.get("input_i",  "-16.0")
        lra   = stats.get("input_lra", "11.0")
        tp    = stats.get("input_tp",  "-1.5")
        off   = stats.get("input_thresh", "-26.0")
        # Pass 2: apply
        af = (
            f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11"
            f":measured_I={il}:measured_LRA={lra}"
            f":measured_TP={tp}:measured_thresh={off}:linear=true"
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-af", af, output_path],
            check=True, capture_output=True,
        )
    else:
        # Fallback: single-pass (less accurate)
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-af", f"loudnorm=I={target_lufs}",
             output_path],
            check=True, capture_output=True,
        )
    return output_path
