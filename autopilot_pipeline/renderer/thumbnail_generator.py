"""
renderer/thumbnail_generator.py
─────────────────────────────────────────────────────────────────────────────
Create YouTube thumbnails by extracting a frame and overlaying title text.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import subprocess
import textwrap
import uuid
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def extract_frame(video_path: str, frame_output_path: str, timestamp: str = "00:00:03") -> bool:
    """
    Extract a single frame from a video at a specific timestamp using FFmpeg.
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", timestamp,
        "-i", video_path,
        "-vframes", "1",
        frame_output_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="ignore")
        print(f"Error extracting frame from {video_path}: {err}")
        return False


def create_thumbnail(video_path: str, title_text: str, output_path: str, font_path: str = "font.ttf") -> str | None:
    """
    Create a thumbnail by overlaying text on a frame extracted from the video.
    """
    temp_frame_path = str(Path(output_path).with_name(f"thumb_frame_{uuid.uuid4().hex}.jpg"))

    if not extract_frame(video_path, temp_frame_path):
        print("Aborting thumbnail creation due to frame extraction failure.")
        return None

    try:
        image = Image.open(temp_frame_path).convert("RGBA")
        width, height = image.size

        draw = ImageDraw.Draw(image)
        overlay_color = (0, 0, 0, 128)
        draw.rectangle([(0, 0), (width, height)], fill=overlay_color)

        font_size = max(24, int(height / 8))
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            print(f"Warning: Font file not found at {font_path}. Using default font.")
            font = ImageFont.load_default()

        avg_char_width = max(10, font_size // 2)
        wrap_width = max(10, int(width * 0.9 / avg_char_width))
        wrapped_text = textwrap.fill((title_text or "").upper(), width=wrap_width)

        text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center")
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (width - text_width) / 2
        text_y = (height - text_height) / 2

        draw.multiline_text(
            (text_x, text_y),
            wrapped_text,
            font=font,
            fill=(255, 255, 255, 255),
            align="center",
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(output_path, "JPEG", quality=90)
        print(f"Thumbnail saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"An error occurred during thumbnail creation: {e}")
        return None
    finally:
        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)


if __name__ == "__main__":
    if os.path.exists("final_video_youtube.mp4"):
        create_thumbnail(
            "final_video_youtube.mp4",
            "The Surprising And Secret History Of Coffee",
            "thumbnail.jpg",
        )
    else:
        print("Run renderer first to generate a video for thumbnail creation.")
