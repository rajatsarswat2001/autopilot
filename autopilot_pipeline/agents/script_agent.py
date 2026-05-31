"""
agents/script_agent.py
─────────────────────────────────────────────────────────────────────────────
Script Writer Agent — LLM + self-critic loop.

Pipeline:
  1. Generate script from research notes + topic via NIM LLM
  2. Parse into validated SceneManifest (Pydantic)
  3. Score uniqueness (anti-spam heuristic)
  4. If score < 0.70 or parse fails → self-critic loop rewrites
  5. Max 3 revisions before halting with fatal error

LLM Fallback Chain:
  • NVIDIA NIM  → meta/llama-3.3-70b-instruct
  • OpenAI      → gpt-4o-mini
  • Local Ollama → llama3:8b-instruct-q4_K_M
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from openai import APIError, OpenAI, RateLimitError
from pydantic import ValidationError

from contracts.scene_manifest import Scene, SceneManifest
from tools.llm_client import call_llm, get_llm_client
from workflows.pipeline_state import AgentError, PipelineState

log = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# LLM Client — waterfall fallback
# ─────────────────────────────────────────────────────────────────────────────

# _get_llm_client is now handled by tools.llm_client (shared, with key rotation)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are an expert YouTube scriptwriter specialising in high-CPM educational content.

RULES (non-negotiable):
1. Open with a BOLD hook that exploits loss aversion or curiosity.
2. Structure the narrative strictly: HOOK -> CONTEXT -> MECHANISM -> TWIST -> CTA.
3. Inject at least ONE surprising fact or counter-intuitive claim.
4. Add ONE personal-sounding anecdote or first-person observation.
5. NEVER use: "in conclusion", "as we can see", "it is worth noting",
   "welcome to", "don't forget to subscribe", "in this video".
6. Each scene narration: 1–3 sentences, ≥ 10 words, conversational tone.
7. Return ONLY valid JSON. No markdown fences, no preamble, no comments.
8. Visual prompts MUST be highly detailed, photorealistic, and cinematic. You MUST include explicit ACTION VERBS and motion descriptors (e.g., "walking rapidly", "gesturing with hands", "camera pans left", "water flowing"). Do not describe static scenes.
"""

_SCHEMA = """\
{
  "video_id": "<uuid>",
    "title": "<compelling YouTube title, 10-70 chars; top-level key for thumbnail text>",
  "niche": "<niche key>",
  "target_cpm_tier": <1|2|3>,
  "hook": "<opening sentence that creates strong curiosity or urgency>",
  "scenes": [
    {
      "scene_id": <int, start at 1, sequential>,
      "narration": "<1-3 sentences of voiceover, conversational tone>",
      "visual_prompt_A": "<highly detailed, photorealistic, cinematic image generation prompt including subject, action, environment, lighting, and camera angle>",
      "b_roll_keyword_A": "<2-4 word Pexels search query>",
      "visual_prompt_B": "<(Optional) secondary detailed visual prompt>",
      "b_roll_keyword_B": "<(Optional) secondary Pexels search query>",
      "emotional_tone": "<tense|curious|inspiring|shocking|warm|neutral|dramatic>",
      "emotion_exaggeration": <0.0-1.0>
    }
  ],
  "call_to_action": "<end-of-video engagement line>",
  "tags": ["<tag1>", "<tag2>", ...]
}
"""

_CRITIC = """\
You reviewed the script below and found issues. Fix ALL of them.

ISSUES:
{issues}

ORIGINAL JSON:
{original}

Requirements:
- Fix every listed issue.
- Make the hook MORE specific and emotionally provocative.
- Add a surprising fact not in the original.
- Return the complete corrected JSON. No markdown, no commentary.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Uniqueness heuristic scorer
# ─────────────────────────────────────────────────────────────────────────────

_SIGNALS: list[tuple[str, float]] = [
    # Positive
    (r"\b(actually|surprisingly|contrary|most people|nobody talks about|I found|I used to)\b", +0.15),
    (r"\b(I|my|I've|I was|imagine)\b", +0.10),
    (r"\b(study|research|data|percent|statistic|scientists|according to)\b", +0.10),
    (r"\?", +0.05),
    # Negative
    (r"\b(welcome to|in this video|don't forget to subscribe|hit the bell)\b", -0.25),
    (r"\b(amazing|incredible|unbelievable|mind-blowing|life-changing)\b", -0.10),
    (r"(.)\1{3,}", -0.15),
    (r"\b(tip \d+|number \d+|step \d+)\b", -0.05),
]


def _score_uniqueness(manifest: SceneManifest) -> float:
    text = manifest.hook + " " + " ".join(s.narration for s in manifest.scenes)
    score = 0.50
    for pattern, weight in _SIGNALS:
        hits = len(re.findall(pattern, text, re.IGNORECASE))
        score += weight * min(hits, 3)
    return max(0.0, min(1.0, round(score, 3)))


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(messages: list[dict], temperature: float) -> str:
    """Call LLM with full key rotation across all providers."""
    return call_llm(messages, temperature=temperature)


def _parse_manifest(raw: str, video_id: str, niche: str) -> tuple[SceneManifest | None, list[str]]:
    issues: list[str] = []
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("` ")

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        issues.append(f"JSON parse error at char {e.pos}: {e.msg}. Snippet: {cleaned[:200]}")
        return None, issues

    data["video_id"] = video_id
    data.setdefault("niche", niche)

    try:
        return SceneManifest(**data), []
    except ValidationError as e:
        for err in e.errors():
            issues.append(f"[{'.'.join(str(x) for x in err['loc'])}] {err['msg']}")
        return None, issues


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

def script_writer_node(state: PipelineState) -> dict[str, Any]:
    """
    Script Writer Agent node.

    Reads:  selected_topic, target_niche, research_notes,
            script_revisions, script_draft (critic loop)
    Writes: script_draft, scene_manifest, uniqueness_score,
            script_revisions, errors
    """
    topic     = state.get("selected_topic", "high-value financial topic")
    niche     = state.get("target_niche", "personal_finance")
    research  = state.get("research_notes", "")
    video_id  = state.get("video_id", str(uuid.uuid4()))
    revisions = state.get("script_revisions", 0)
    prev_draft = state.get("script_draft")
    prev_score = state.get("uniqueness_score", 0.0)

    log.info("script_agent.start", topic=topic, revision=revisions)

    # ── Build prompt ─────────────────────────────────────────────────────────
    if revisions == 0 or not prev_draft:
        user_msg = (
            f"Topic: {topic}\nNiche: {niche}\n"
            f"Research:\n{research or 'Use general knowledge.'}\n\n"
            f"Schema:\n{_SCHEMA}\n\n"
            "Generate exactly a 4 scene script. Make the hook arresting and specific. "
            "Ensure 'title' is present as a top-level JSON field and is under 70 characters."
        )
        messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user_msg}]
        temperature = 0.80
    else:
        # Self-critic mode
        issues = []
        if not state.get("scene_manifest"):
            issues.append("Previous output was not valid JSON / failed schema validation.")
        if prev_score < 0.70:
            issues.append(
                f"Uniqueness score {prev_score:.2f} < 0.70. "
                "Add a surprising fact, a personal anecdote, and a counter-intuitive angle."
            )
        issues.append(f"Revision #{revisions}: make it significantly more original.")

        user_msg = _CRITIC.format(
            issues="\n".join(f"  - {i}" for i in issues),
            original=prev_draft[:3000],
        )
        messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user_msg}]
        temperature = 0.92

    # ── Call LLM (auto-rotates through all keys/providers) ───────────────────
    try:
        raw = _call_llm(messages, temperature)
    except Exception as e:
        err: AgentError = {
            "agent": "script_writer",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recoverable": False,
        }
        return {"errors": [err], "script_revisions": revisions + 1}

    # ── Parse + validate ─────────────────────────────────────────────────────
    manifest, parse_issues = _parse_manifest(raw, video_id, niche)

    if manifest is None:
        log.warning("script_agent.parse_failed", issues=parse_issues)
        return {
            "script_draft":     raw,
            "scene_manifest":   None,
            "uniqueness_score": 0.0,
            "script_revisions": revisions + 1,
        }

    score = _score_uniqueness(manifest)
    log.info("script_agent.done", uniqueness=score, scenes=len(manifest.scenes))

    return {
        "script_draft":     raw,
        "scene_manifest":   manifest.to_pipeline_dict(),
        "uniqueness_score": score,
        "script_revisions": revisions + 1,
        "job_status":       "scripting",
    }
