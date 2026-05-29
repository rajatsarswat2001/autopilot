"""
agents/upload_agent.py
─────────────────────────────────────────────────────────────────────────────
Upload Agent — YouTube Data API v3 upload with metadata from scene_manifest.

Prerequisites:
  • GOOGLE_CLIENT_SECRET_FILE env var pointing to OAuth2 client_secret.json
    OR YOUTUBE_ACCESS_TOKEN for service-account / pre-authorised flow.
  • google-api-python-client, google-auth, google-auth-oauthlib installed.

Privacy defaults to "private" — safe for review before publishing.
Set YOUTUBE_PRIVACY=public|unlisted|private via .env to override.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from workflows.pipeline_state import AgentError, PipelineState

log = structlog.get_logger(__name__)

PRIVACY_STATUS = os.getenv("YOUTUBE_PRIVACY", "private")
UPLOAD_CHUNK   = 8 * 1024 * 1024   # 8 MB resumable upload chunks


# ─────────────────────────────────────────────────────────────────────────────
# YouTube client
# ─────────────────────────────────────────────────────────────────────────────

def _get_youtube_service():
    """
    Build YouTube Data API v3 client.
    Priority:
      1. YOUTUBE_ACCESS_TOKEN env var (pre-authorised token)
      2. GOOGLE_CLIENT_SECRET_FILE + stored token cache
    """
    try:
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials

        access_token = os.getenv("YOUTUBE_ACCESS_TOKEN")
        if access_token:
            creds = Credentials(token=access_token)
            return build("youtube", "v3", credentials=creds)

        secret_file = os.getenv("GOOGLE_CLIENT_SECRET_FILE")
        if secret_file and Path(secret_file).exists():
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            import pickle

            token_path = Path("data/yt_token.pickle")
            creds = None
            if token_path.exists():
                with open(token_path, "rb") as f:
                    creds = pickle.load(f)
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        secret_file,
                        scopes=["https://www.googleapis.com/auth/youtube.upload"],
                    )
                    creds = flow.run_local_server(port=0)
                with open(token_path, "wb") as f:
                    pickle.dump(creds, f)
            return build("youtube", "v3", credentials=creds)

        log.warning("upload_agent.no_youtube_credentials")
        return None

    except ImportError as e:
        log.warning("upload_agent.google_lib_missing", error=str(e))
        return None


def _upload_video(
    youtube,
    video_path: str,
    title: str,
    description: str,
    tags: list[str],
    category_id: str = "27",  # Education
    thumbnail_path: str | None = None,
) -> dict:
    """Resumable upload with retry. Returns insert response dict."""
    from googleapiclient.http import MediaFileUpload

    # Truncate tags to keep total combined comma-separated length under 490 characters (YouTube strict limit is 500 chars)
    filtered_tags = []
    current_len = 0
    for t in tags:
        added_len = len(t) + (1 if filtered_tags else 0)
        if current_len + added_len > 490:
            break
        filtered_tags.append(t)
        current_len += added_len

    body = {
        "snippet": {
            "title":      title[:100],
            "description": description,
            "tags":        filtered_tags,
            "categoryId":  category_id,
        },
        "status": {
            "privacyStatus":          PRIVACY_STATUS,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        video_path,
        mimetype="video/mp4",
        chunksize=UPLOAD_CHUNK,
        resumable=True,
    )

    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    for attempt in range(5):
        try:
            status, response = request.next_chunk()
            if response:
                break
        except Exception as e:
            log.warning("upload_agent.chunk_error", attempt=attempt, error=str(e))
            time.sleep(2 ** attempt)

    if thumbnail_path and response and Path(thumbnail_path).exists():
        try:
            youtube.thumbnails().set(
                videoId=response["id"],
                media_body=MediaFileUpload(thumbnail_path, mimetype="image/jpeg"),
            ).execute()
        except Exception as e:
            log.warning("upload_agent.thumbnail_upload_failed", error=str(e))

    return response or {}


# ─────────────────────────────────────────────────────────────────────────────
# Description builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_description(manifest_dict: dict, source_urls: list[str]) -> str:
    hook      = manifest_dict.get("hook", "")
    scenes    = manifest_dict.get("scenes", [])
    cta       = manifest_dict.get("call_to_action", "")
    narration = " ".join(s.get("narration", "") for s in scenes[:3])

    desc_parts = [hook, "", narration[:500], "", cta, ""]
    if source_urls:
        desc_parts += ["─── Sources ───"] + [f"• {u}" for u in source_urls[:5]]

    return "\n".join(desc_parts)[:4900]   # YouTube 5000 char limit


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

def upload_node(state: PipelineState) -> dict[str, Any]:
    """
    Upload Agent node.

    Reads:  final_video_path, thumbnail_path, scene_manifest,
            source_urls, video_id
    Writes: youtube_video_id, youtube_url, job_status, errors
    """
    video_path     = state.get("final_video_path")
    thumb_path     = state.get("thumbnail_path")
    manifest_dict  = state.get("scene_manifest", {})
    source_urls    = state.get("source_urls", [])

    if not video_path or not Path(video_path).exists():
        err: AgentError = {
            "agent": "upload",
            "error": f"Video file not found: {video_path}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recoverable": False,
        }
        return {"errors": [err]}

    title       = manifest_dict.get("title", "AutoPilot Video")
    tags        = manifest_dict.get("tags", [])
    description = _build_description(manifest_dict, source_urls)

    youtube = _get_youtube_service()

    if not youtube:
        log.warning("upload_agent.skipping_no_credentials")
        return {
            "youtube_video_id": None,
            "youtube_url":      f"file://{video_path}",
            "job_status":       "done",
        }

    log.info("upload_agent.uploading", title=title, privacy=PRIVACY_STATUS)

    try:
        response = _upload_video(
            youtube=youtube,
            video_path=video_path,
            title=title,
            description=description,
            tags=tags,
            thumbnail_path=thumb_path,
        )
        vid_id = response.get("id", "")
        yt_url = f"https://www.youtube.com/watch?v={vid_id}" if vid_id else ""

        log.info("upload_agent.done", video_id=vid_id, url=yt_url)

        return {
            "youtube_video_id": vid_id,
            "youtube_url":      yt_url,
            "job_status":       "done",
        }

    except Exception as e:
        err = {
            "agent": "upload",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recoverable": True,
        }
        log.error("upload_agent.upload_failed", error=str(e))
        return {"errors": [err], "youtube_url": f"file://{video_path}"}
