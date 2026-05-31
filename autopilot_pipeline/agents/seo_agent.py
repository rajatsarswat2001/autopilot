"""
agents/seo_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEO Agent — generates high-quality YouTube metadata packages
(descriptions, hashtags, chapter markers) before upload.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

import structlog

from tools.llm_client import call_llm
from workflows.pipeline_state import AgentError, PipelineState

log = structlog.get_logger(__name__)

_SYSTEM = """
You are an expert YouTube SEO specialist.
Your job is to generate highly optimized metadata for a YouTube video.

RULES:
1. Write a compelling description (3-4 paragraphs) incorporating the topic naturally.
2. The first 2 lines must be extremely keyword-rich and hook the viewer.
3. Include exactly 15 highly relevant hashtags at the bottom of the description.
4. Generate chapter markers based on the scenes provided, starting with "00:00 Intro".
5. Generate a short, punchy caption for cross-posting to Pinterest/Instagram.
6. Return ONLY a valid JSON object matching the provided structure. No markdown, no commentary.
"""

_USER = """
Topic: {topic}
Title: {title}

Scenes:
{scenes_json}

Return ONLY a JSON object with this exact structure:
{{
  "description": "<full youtube description including paragraphs, chapters, and hashtags at the bottom>",
  "chapters": [
    {{"time": "00:00", "title": "Intro"}},
    ...
  ],
  "hashtags": ["#tag1", "#tag2", ...],
  "social_caption": "<short punchy caption for IG/Pinterest>"
}}
"""


def seo_node(state: PipelineState) -> dict[str, Any]:
    """
    SEO Agent node.
    Reads:  scene_manifest, selected_topic
    Writes: seo_metadata, job_status
    """
    manifest_dict = state.get("scene_manifest", {})
    topic = state.get("selected_topic", "")
    title = manifest_dict.get("title", "Untitled Video")
    scenes = manifest_dict.get("scenes", [])

    if not scenes:
        log.warning("seo_agent.no_scenes_skipping")
        return {"job_status": "upload"}

    # Pass minimal data to LLM
    slim_scenes = [
        {
            "scene_id": s.get("scene_id"),
            "narration": s.get("narration"),
        }
        for s in scenes
    ]

    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": _USER.format(
                topic=topic,
                title=title,
                scenes_json=json.dumps(slim_scenes, indent=2),
            )
        }
    ]

    try:
        raw = call_llm(messages, temperature=0.7, max_tokens=1500)
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("` ")
        metadata = json.loads(raw)
        
        log.info("seo_agent.success")
        return {
            "seo_metadata": metadata,
            "job_status": "upload"
        }
    except Exception as e:
        log.warning("seo_agent.failed", error=str(e))
        return {"job_status": "upload"}
