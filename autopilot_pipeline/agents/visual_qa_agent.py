"""
agents/visual_qa_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Visual QA Agent — Runs sanity checks on visual clips before
passing them to the Assembly agent.

Why this exists:
  Prevents the pipeline from assembling a video with broken,
  missing, or pure placeholder clips.

Checks:
  - File exists and is >0 bytes
  - If type == "video_clip", duration is > 0s
  - Source is not "placeholder" unless absolutely intended
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import structlog

from tools.ffmpeg_tools import measure_video_duration
from workflows.pipeline_state import AgentError, PipelineState

log = structlog.get_logger(__name__)


def visual_qa_node(state: PipelineState) -> dict[str, Any]:
    """
    Visual QA node.
    Reads: visual_manifest
    Writes: visual_qa_passed, visual_qa_notes, job_status
    """
    visual_dict = state.get("visual_manifest")
    
    if not visual_dict:
        err: AgentError = {
            "agent": "visual_qa",
            "error": "Missing visual_manifest",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recoverable": False,
        }
        return {"errors": [err], "job_status": "failed"}

    scenes = visual_dict.get("scenes", [])
    if not scenes:
        return {
            "visual_qa_passed": False,
            "visual_qa_notes": "No scenes found in visual manifest.",
            "job_status": "failed",
        }

    failed_clips = []
    warning_notes = []

    for scene in scenes:
        sid = scene.get("scene_id")
        
        for asset_key, type_key, source_key in [
            ("asset_path_A", "asset_type_A", "source_A"),
            ("asset_path_B", "asset_type_B", "source_B"),
        ]:
            path = scene.get(asset_key)
            atype = scene.get(type_key)
            source = scene.get(source_key)
            
            # Missing file check
            if not path or not os.path.exists(path):
                failed_clips.append(f"Scene {sid} {asset_key} missing: {path}")
                continue
                
            # Zero byte check
            if os.path.getsize(path) == 0:
                failed_clips.append(f"Scene {sid} {asset_key} is 0 bytes.")
                continue

            # Placeholder check - soft fail (warning)
            if source == "placeholder":
                warning_notes.append(f"Scene {sid} {asset_key} is a placeholder.")
                continue
                
            # Duration check for videos
            if atype == "video_clip":
                dur = measure_video_duration(path)
                if dur <= 0.1:
                    failed_clips.append(f"Scene {sid} {asset_key} duration {dur}s is invalid.")

    if failed_clips:
        notes = " | ".join(failed_clips)
        log.warning("visual_qa.failed", notes=notes)
        return {
            "visual_qa_passed": False,
            "visual_qa_notes": notes,
            "job_status": "failed",
        }

    final_notes = "All clips passed QA."
    if warning_notes:
        final_notes = "Warnings: " + " | ".join(warning_notes)

    log.info("visual_qa.passed", scenes=len(scenes), warnings=len(warning_notes))
    return {
        "visual_qa_passed": True,
        "visual_qa_notes": final_notes,
        "job_status": "assembly",
    }
