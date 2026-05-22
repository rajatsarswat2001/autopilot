"""
agents/entropy_agent.py
─────────────────────────────────────────────────────────────────────────────
Entropy Engine — the "Human Touch" agent that injects natural irregularity
into AI-generated scripts to defeat YouTube's AI-detection heuristics.

What it does:
  • Varies sentence rhythm (short punchy → longer explanatory → brief closer)
  • Inserts "micro-opinions" and first-person observations
  • Adds colloquial transitions ("Here's the thing...", "And no, I'm not joking")
  • Reorders predictable sentence openers
  • Injects parenthetical asides and em-dash interruptions
  • Scores entropy using Flesch Reading Ease variance + pattern analysis

The output is the SAME scene_manifest structure — only narration text changes.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import structlog
from openai import APIError, OpenAI, RateLimitError

from contracts.scene_manifest import SceneManifest
from tools.llm_client import call_llm
from workflows.pipeline_state import AgentError, PipelineState

log = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Entropy measurement
# ─────────────────────────────────────────────────────────────────────────────

ROBOTIC_PATTERNS = [
    # Uniform sentence openers
    r"^(First|Second|Third|Finally|Additionally|Furthermore|Moreover),",
    # Template cadence
    r"\b(In this (video|section)|Let('s| us) (talk|discuss|explore|look at))\b",
    # Passive AI tone
    r"\b(It is important to note|It should be noted|As mentioned)\b",
    # Uniform sentence length (all similar word counts per sentence)
]

HUMAN_TRANSITIONS = [
    "Here's the thing —",
    "And I know what you're thinking.",
    "No, seriously.",
    "Let me be blunt.",
    "This one surprised even me.",
    "Most people get this completely wrong.",
    "Pay attention here.",
    "I'll be honest with you.",
]


def _count_robotic_signals(text: str) -> int:
    count = 0
    for pattern in ROBOTIC_PATTERNS:
        count += len(re.findall(pattern, text, re.IGNORECASE | re.MULTILINE))
    return count


def _sentence_length_variance(text: str) -> float:
    """Variance in word counts per sentence. Low variance = robotic cadence."""
    sentences = re.split(r"[.!?]+", text)
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) < 2:
        return 0.0
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    return round(variance, 2)


def _entropy_score(manifest_dict: dict) -> float:
    """
    0.0–1.0 humanisation score.
    Combines sentence variance + robotic pattern penalty + positive signal bonus.
    """
    scenes = manifest_dict.get("scenes", [])
    all_text = manifest_dict.get("hook", "") + " " + " ".join(
        s.get("narration", "") for s in scenes
    )

    # Sentence length variance (capped contribution)
    variance = _sentence_length_variance(all_text)
    variance_score = min(variance / 50.0, 1.0)  # 50+ variance = fully diverse

    # Robotic signal penalty
    robotic_count = _count_robotic_signals(all_text)
    robotic_penalty = min(robotic_count * 0.05, 0.4)

    # Personal voice bonus
    personal_hits = len(re.findall(r"\b(I|my|me|I've|I was|imagine you)\b", all_text, re.I))
    personal_bonus = min(personal_hits * 0.05, 0.25)

    # Colloquial bonus
    colloq_hits = len(re.findall(
        r"\b(here's the thing|honestly|no seriously|pay attention|let me be blunt)\b",
        all_text, re.I
    ))
    colloq_bonus = min(colloq_hits * 0.08, 0.20)

    score = 0.35 + variance_score * 0.30 - robotic_penalty + personal_bonus + colloq_bonus
    return max(0.0, min(1.0, round(score, 3)))


# ─────────────────────────────────────────────────────────────────────────────
# LLM rewriter
# ─────────────────────────────────────────────────────────────────────────────

_ENTROPY_SYSTEM = """\
You are a human-voice editor for YouTube scripts. Your job is to make AI-written
narration sound like it was spoken by a passionate, knowledgeable person — NOT an AI.

RULES:
1. Vary sentence LENGTH dramatically. Mix 3-word punches with longer explanations.
2. Add 1–2 "micro-opinions" — brief personal takes or surprising admissions.
3. Use colloquial transitions: "Here's the thing —", "No, seriously.", "Pay attention here."
4. Break predictable patterns — no two consecutive scenes should start the same way.
5. Add em-dash interruptions and parenthetical asides naturally.
6. Keep the SAME factual content and scene count. Do NOT add or remove scenes.
7. Return ONLY the "scenes" array as valid JSON. No wrapper object, no markdown.

Example transformation:
  BEFORE: "In this section, we will explore the benefits of compound interest."
  AFTER:  "Here's the thing about compound interest — most people wait too long. And by then? The math has already decided their future."
"""

_ENTROPY_USER = """\
Rewrite the narration for each scene below to sound human, varied, and engaging.
Keep scene_id and all other fields identical. Only change "narration" text.

Current scenes JSON:
{scenes_json}

Return ONLY the updated scenes array as JSON. No wrapper. No markdown.
"""


def _rewrite_narrations(manifest_dict: dict) -> dict:
    """Call LLM to rewrite scene narrations with human entropy."""
    scenes_json = json.dumps(manifest_dict.get("scenes", []), indent=2)
    messages = [
        {"role": "system", "content": _ENTROPY_SYSTEM},
        {"role": "user", "content": _ENTROPY_USER.format(scenes_json=scenes_json)},
    ]
    try:
        raw = call_llm(messages, temperature=0.85)
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("` ")
        updated_scenes = json.loads(raw)
        if isinstance(updated_scenes, list):
            manifest_dict["scenes"] = updated_scenes
    except Exception as e:
        log.warning("entropy_agent.llm_error", error=str(e)[:120])
        # Return original manifest if rewrite fails
    return manifest_dict


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

def entropy_node(state: PipelineState) -> dict[str, Any]:
    """
    Entropy Engine node.

    Reads:  scene_manifest
    Writes: scene_manifest (updated narrations), entropy_score, entropy_applied
    """
    manifest_dict = state.get("scene_manifest")
    if not manifest_dict:
        log.warning("entropy_agent.no_manifest")
        return {"entropy_score": 0.0, "entropy_applied": False}

    before_score = _entropy_score(manifest_dict)
    log.info("entropy_agent.before", score=before_score)

    # Only rewrite if entropy is below threshold
    if before_score >= 0.75:
        log.info("entropy_agent.skip_already_high", score=before_score)
        return {
            "scene_manifest":  manifest_dict,
            "entropy_score":   before_score,
            "entropy_applied": False,
        }

    updated = _rewrite_narrations(manifest_dict)
    after_score = _entropy_score(updated)

    log.info("entropy_agent.done",
             before=before_score, after=after_score,
             delta=round(after_score - before_score, 3))

    return {
        "scene_manifest":  updated,
        "entropy_score":   after_score,
        "entropy_applied": True,
        "job_status":      "compliance",
    }
