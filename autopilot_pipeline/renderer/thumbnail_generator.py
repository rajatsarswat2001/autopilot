"""
renderer/thumbnail_generator.py
─────────────────────────────────────────────────────────────────────────────
Create YouTube thumbnails:
  1. Generate a cinematic background via FLUX.1 through Pollinations AI (free)
  2. Overlay bold title text using a downloaded Google Font (Oswald-Bold)
  3. Fallback: extract a video frame if FLUX call fails

Font resolution order (4-tier):
  1. THUMBNAIL_FONT_PATH env var (custom font)
  2. ~/.cache/autopilot/Oswald-Bold.ttf  (auto-downloaded from Google Fonts)
  3. /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf  (Kaggle/Ubuntu system)
  4. PIL default bitmap (8px — absolute last resort, never fails)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import subprocess
import textwrap
import urllib.request
import uuid
from pathlib import Path

import structlog
from PIL import Image, ImageDraw, ImageFont

log = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Font resolution
# ─────────────────────────────────────────────────────────────────────────────
_FONT_CACHE_DIR = Path.home() / ".cache" / "autopilot"
_FONT_CACHE_PATH = _FONT_CACHE_DIR / "Oswald-Bold.ttf"
# Correct direct download URL (the google/fonts repo restructured in 2024)
_FONT_URL = (
    "https://raw.githubusercontent.com/google/fonts/main/ofl/oswald/static/Oswald-Bold.ttf"
)


def _resolve_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return the best available font at the requested size."""
    # 1. User-specified env var
    custom = os.getenv("THUMBNAIL_FONT_PATH", "")
    if custom and Path(custom).exists():
        try:
            return ImageFont.truetype(custom, size)
        except Exception:
            pass

    # 2. Cached Google Font (Oswald-Bold — downloads once)
    if not _FONT_CACHE_PATH.exists():
        try:
            _FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            log.info("thumbnail.downloading_font", url=_FONT_URL)
            urllib.request.urlretrieve(_FONT_URL, _FONT_CACHE_PATH)
        except Exception as e:
            log.warning("thumbnail.font_download_failed", error=str(e))

    if _FONT_CACHE_PATH.exists():
        try:
            return ImageFont.truetype(str(_FONT_CACHE_PATH), size)
        except Exception:
            pass

    # 3. System fonts — DejaVuSans-Bold (always present on Kaggle/Ubuntu)
    _SYSTEM_FONTS = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",   # Kaggle / Ubuntu
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",                     # macOS
        "C:/Windows/Fonts/arialbd.ttf",                            # Windows dev
    ]
    for sys_font in _SYSTEM_FONTS:
        if Path(sys_font).exists():
            try:
                font = ImageFont.truetype(sys_font, size)
                log.info("thumbnail.using_system_font", path=sys_font)
                return font
            except Exception:
                pass

    # 4. PIL default bitmap (absolute last resort — tiny but never fails)
    log.warning("thumbnail.using_default_font")
    return ImageFont.load_default()


# ─────────────────────────────────────────────────────────────────────────────
# FLUX background via Pollinations AI (free, no key)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_flux_background(prompt: str, output_path: str, width: int = 1280, height: int = 720) -> bool:
    """
    Generate a cinematic thumbnail background using FLUX.1 via Pollinations AI.
    Completely free — no API key required.
    """
    import urllib.parse
    safe_prompt = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{safe_prompt}?width={width}&height={height}&model=flux&nologo=true&enhance=true"
    try:
        urllib.request.urlretrieve(url, output_path)
        if Path(output_path).stat().st_size > 10_000:
            log.info("thumbnail.flux_bg_ok", path=output_path)
            return True
    except Exception as e:
        log.warning("thumbnail.flux_bg_failed", error=str(e))
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Frame extraction fallback
# ─────────────────────────────────────────────────────────────────────────────

def extract_frame(video_path: str, frame_output_path: str, timestamp: str = "00:00:03") -> bool:
    """Extract a single frame from a video at a specific timestamp using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", timestamp,
        "-i", video_path,
        "-vframes", "1",
        frame_output_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return Path(frame_output_path).exists() and Path(frame_output_path).stat().st_size > 1000
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="ignore")
        log.warning("thumbnail.frame_extract_failed", error=err[:200])
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def create_thumbnail(
    video_path: str,
    title_text: str,
    output_path: str,
    niche: str = "default",
    font_path: str | None = None,   # kept for backward compat (ignored; use env var)
) -> str | None:
    """
    Create a cinematic YouTube thumbnail:
      1. Try FLUX.1 via Pollinations for an AI-generated background
      2. Fall back to extracting a frame from the rendered video
      3. Overlay bold title text with drop shadow
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_bg = str(Path(output_path).with_name(f"thumb_bg_{uuid.uuid4().hex}.jpg"))

    # ── Background ──────────────────────────────────────────────────────────
    # Build a cinematic FLUX prompt from the title, explicitly forbidding text
    # so the Python PIL text overlay remains legible.
    niche_style_map = {
        "personal_finance": "dramatic financial charts, city skyline at dusk, cinematic lighting",
        "saas_tools":       "futuristic tech interface, dark UI glow, cyberpunk aesthetic",
        "legal_tax":        "scales of justice, professional office, golden lighting",
        "senior_health":    "healthy elderly lifestyle, warm sunlight, nature background",
        "storytelling":     "epic cinematic scene, dramatic lighting, movie poster style",
        "default":          "dramatic cinematic background, dark gradient, professional",
    }
    style_hint = niche_style_map.get(niche, niche_style_map["default"])
    flux_prompt = f"Background image for '{title_text}', {style_hint}, textless, no text, empty space in center, 4K, ultra high quality, clean composition"

    bg_ready = _generate_flux_background(flux_prompt, tmp_bg)

    if not bg_ready:
        log.info("thumbnail.falling_back_to_frame")
        bg_ready = extract_frame(video_path, tmp_bg)

    if not bg_ready or not Path(tmp_bg).exists() or Path(tmp_bg).stat().st_size < 1000:
        log.warning("thumbnail.no_background_available_or_invalid")
        return None

    try:
        image = Image.open(tmp_bg).convert("RGBA").resize((1280, 720), Image.LANCZOS)
        width, height = image.size
        draw = ImageDraw.Draw(image)

        # Semi-transparent dark gradient overlay for text legibility
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        ov_draw = ImageDraw.Draw(overlay)
        for y in range(height // 2, height):
            alpha = int(180 * (y - height // 2) / (height // 2))
            ov_draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))
        image = Image.alpha_composite(image, overlay)
        draw = ImageDraw.Draw(image)

        # ── Text ────────────────────────────────────────────────────────────
        font_size = max(48, int(height / 7))
        font = _resolve_font(font_size)

        avg_char_width = max(10, font_size // 2)
        wrap_width = max(10, int(width * 0.85 / avg_char_width))
        wrapped = textwrap.fill((title_text or "").upper(), width=wrap_width)

        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, align="center")
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = (width - tw) / 2
        ty = height - th - 60  # bottom area

        # Drop shadow
        draw.multiline_text((tx + 4, ty + 4), wrapped, font=font,
                            fill=(0, 0, 0, 200), align="center")
        # Main text
        draw.multiline_text((tx, ty), wrapped, font=font,
                            fill=(255, 255, 255, 255), align="center")

        image.convert("RGB").save(output_path, "JPEG", quality=92)
        log.info("thumbnail.saved", path=output_path)
        return output_path

    except Exception as e:
        log.error("thumbnail.creation_failed", error=str(e))
        return None
    finally:
        try:
            Path(tmp_bg).unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    if os.path.exists("final_video_youtube.mp4"):
        create_thumbnail(
            "final_video_youtube.mp4",
            "The Surprising And Secret History Of Coffee",
            "thumbnail.jpg",
        )
    else:
        print("Run renderer first to generate a video for thumbnail creation.")
