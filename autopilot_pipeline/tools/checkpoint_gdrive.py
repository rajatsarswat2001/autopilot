"""
tools/checkpoint_gdrive.py
─────────────────────────────────────────────────────────────────────────────
Session checkpoint + Google Drive auto-upload for Kaggle environments.

Solves two Kaggle constraints:
  1. 9-hour session timeout: state.json checkpoint lets the pipeline resume
     exactly where it left off after a session restart.
  2. Ephemeral /kaggle/working/ storage: all generated assets are uploaded
     to Google Drive automatically so nothing is lost when the session ends.

Usage in kaggle_run.ipynb:
    from tools.checkpoint_gdrive import CheckpointManager, GDriveUploader

    # Checkpoint manager — call .mark_done(scene_id) after each scene
    cp = CheckpointManager("/kaggle/working/autopilot/state.json")
    if cp.is_done(scene_id):
        print(f"Scene {scene_id} already done — skipping")
        continue
    # ... generate scene ...
    cp.mark_done(scene_id, metadata={"path": clip_path, "source": "wan21"})

    # Google Drive uploader — uploads each output as it's generated
    uploader = GDriveUploader(
        service_account_json=os.getenv("GDRIVE_SA_JSON"),  # JSON string from Kaggle Secret
        folder_id=os.getenv("GDRIVE_FOLDER_ID"),           # target Drive folder
    )
    uploader.upload("/kaggle/working/.../video.mp4", "video.mp4")

Environment (Kaggle Secrets):
    GDRIVE_SA_JSON    — JSON string of Google Cloud Service Account credentials
    GDRIVE_FOLDER_ID  — ID of the target Google Drive folder (from the URL)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import structlog

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Session Checkpoint Manager
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    """
    Persists pipeline progress to state.json after every scene.
    On restart, reads the file and skips already-completed scenes.

    state.json format:
    {
      "video_id": "abc123",
      "niche": "personal_finance",
      "topic": "Five money mistakes...",
      "completed_scenes": [1, 2, 3],
      "scene_metadata": {
        "1": {"path": "/kaggle/working/.../clip_001.mp4", "source": "wan21"},
        ...
      },
      "pipeline_done": false,
      "started_at": "2026-05-29T02:00:00Z",
      "last_updated": "2026-05-29T02:15:00Z"
    }
    """

    def __init__(self, state_path: str = "/kaggle/working/autopilot/state.json"):
        self.state_path = Path(state_path)
        self._state: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                log.info("checkpoint.resumed",
                         scenes_done=len(data.get("completed_scenes", [])),
                         video_id=data.get("video_id"))
                return data
            except Exception as e:
                log.warning("checkpoint.load_failed", error=str(e))
        return {
            "video_id": None,
            "niche": None,
            "topic": None,
            "completed_scenes": [],
            "scene_metadata": {},
            "pipeline_done": False,
            "started_at": _now_iso(),
            "last_updated": _now_iso(),
        }

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state["last_updated"] = _now_iso()
        self.state_path.write_text(json.dumps(self._state, indent=2))

    def init_session(self, video_id: str, niche: str, topic: str) -> None:
        """Call at the start of a new pipeline run to record session metadata."""
        # Only reset if this is a genuinely new video (different video_id)
        if self._state.get("video_id") != video_id:
            self._state.update({
                "video_id": video_id,
                "niche": niche,
                "topic": topic,
                "completed_scenes": [],
                "scene_metadata": {},
                "pipeline_done": False,
                "started_at": _now_iso(),
            })
            self._save()
            log.info("checkpoint.new_session", video_id=video_id, niche=niche)
        else:
            log.info("checkpoint.resuming_session",
                     video_id=video_id,
                     scenes_already_done=len(self._state.get("completed_scenes", [])))

    def is_done(self, scene_id: int) -> bool:
        """Return True if this scene was already successfully completed."""
        return scene_id in self._state.get("completed_scenes", [])

    def mark_done(self, scene_id: int, metadata: Optional[dict] = None) -> None:
        """Mark a scene as done and persist state immediately."""
        if scene_id not in self._state["completed_scenes"]:
            self._state["completed_scenes"].append(scene_id)
        if metadata:
            self._state["scene_metadata"][str(scene_id)] = metadata
        self._save()
        log.info("checkpoint.scene_done", scene_id=scene_id,
                 total_done=len(self._state["completed_scenes"]))

    def mark_pipeline_done(self, final_video_path: str, youtube_url: str = "") -> None:
        """Mark the entire pipeline as complete."""
        self._state["pipeline_done"] = True
        self._state["final_video_path"] = final_video_path
        self._state["youtube_url"] = youtube_url
        self._save()
        log.info("checkpoint.pipeline_done", video=final_video_path)

    def get_completed_scenes(self) -> list[int]:
        return list(self._state.get("completed_scenes", []))

    def get_scene_metadata(self, scene_id: int) -> dict:
        return self._state.get("scene_metadata", {}).get(str(scene_id), {})

    @property
    def video_id(self) -> Optional[str]:
        return self._state.get("video_id")

    @property
    def is_pipeline_done(self) -> bool:
        return bool(self._state.get("pipeline_done"))


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Google Drive Auto-Uploader
# ─────────────────────────────────────────────────────────────────────────────

class GDriveUploader:
    """
    Uploads generated files to Google Drive using a Service Account.
    Safe to use from Kaggle — no OAuth browser flow required.

    Setup:
      1. Create a Google Cloud project → enable Drive API
      2. Create a Service Account → download JSON key
      3. Share your target Drive folder with the service account email
      4. Store the JSON key string in Kaggle Secrets as GDRIVE_SA_JSON
      5. Store the folder ID in Kaggle Secrets as GDRIVE_FOLDER_ID
    """

    def __init__(
        self,
        service_account_json: Optional[str] = None,
        folder_id: Optional[str] = None,
    ):
        """
        service_account_json: raw JSON string of the service account credentials
                              (from Kaggle Secret GDRIVE_SA_JSON)
        folder_id: Google Drive folder ID where uploads will go
        """
        self._folder_id = folder_id or os.getenv("GDRIVE_FOLDER_ID")
        sa_json = service_account_json or os.getenv("GDRIVE_SA_JSON")
        self._service = None
        self._enabled = bool(sa_json and self._folder_id)

        if not self._enabled:
            log.warning("gdrive.disabled",
                        msg="GDRIVE_SA_JSON and GDRIVE_FOLDER_ID not set — uploads skipped")
            return

        try:
            import json as _json
            from google.oauth2.service_account import Credentials
            from googleapiclient.discovery import build

            creds_dict = _json.loads(sa_json) if isinstance(sa_json, str) else sa_json
            creds = Credentials.from_service_account_info(
                creds_dict,
                scopes=["https://www.googleapis.com/auth/drive.file"],
            )
            self._service = build("drive", "v3", credentials=creds)
            log.info("gdrive.authenticated", folder_id=self._folder_id)

        except ImportError:
            log.warning("gdrive.missing_deps",
                        msg="Install: pip install google-api-python-client google-auth")
            self._enabled = False
        except Exception as e:
            log.warning("gdrive.auth_failed", error=str(e)[:120])
            self._enabled = False

    def upload(
        self,
        local_path: str,
        remote_name: Optional[str] = None,
        mimetype: Optional[str] = None,
        retries: int = 3,
    ) -> Optional[str]:
        """
        Upload a file to the configured Google Drive folder.
        Returns the Drive file ID on success, None on failure.
        """
        if not self._enabled or not self._service:
            log.debug("gdrive.upload_skipped", path=local_path)
            return None

        local = Path(local_path)
        if not local.exists():
            log.warning("gdrive.file_not_found", path=local_path)
            return None

        remote_name = remote_name or local.name
        if mimetype is None:
            mimetype = _guess_mimetype(local)

        from googleapiclient.http import MediaFileUpload

        for attempt in range(retries):
            try:
                media   = MediaFileUpload(str(local), mimetype=mimetype, resumable=True)
                meta    = {"name": remote_name, "parents": [self._folder_id]}
                request = self._service.files().create(
                    body=meta, media_body=media, fields="id,webViewLink"
                )

                response = None
                while response is None:
                    status, response = request.next_chunk()
                    if status:
                        pct = int(status.progress() * 100)
                        log.debug("gdrive.upload_progress", file=remote_name, pct=pct)

                file_id   = response.get("id", "")
                view_link = response.get("webViewLink", "")
                size_mb   = local.stat().st_size / 1_048_576
                log.info("gdrive.uploaded",
                         file=remote_name, size_mb=round(size_mb, 1),
                         file_id=file_id, link=view_link)
                return file_id

            except Exception as e:
                log.warning("gdrive.upload_error",
                            attempt=attempt + 1, error=str(e)[:120])
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)

        log.error("gdrive.upload_failed_all_retries", file=remote_name)
        return None

    def upload_batch(self, files: list[str]) -> dict[str, Optional[str]]:
        """Upload multiple files. Returns {local_path: drive_file_id}."""
        return {f: self.upload(f) for f in files}

    @property
    def is_enabled(self) -> bool:
        return self._enabled


def _guess_mimetype(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".mp4":  "video/mp4",
        ".mov":  "video/quicktime",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".ass":  "text/plain",
        ".srt":  "text/plain",
        ".wav":  "audio/wav",
        ".json": "application/json",
    }.get(ext, "application/octet-stream")
