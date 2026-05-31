"""
agents/title_ab_agent.py
─────────────────────────────────────────────────────────────────────────────
Title A/B Agent — generates 3 title+hook variants and picks the highest-
scoring one using CTR (click-through rate) heuristics.

Why this matters:
  YouTube CTR is the #1 early ranking signal for new uploads.
  A single percentage-point improvement in CTR can double channel growth.

Variant angles:
  A — Loss aversion  : what the viewer LOSES by not watching
  B — Curiosity gap  : surprising fact or number
  C — Direct benefit : concrete transformation promise

Scoring dimensions (heuristic, 0.0–1.0):
  • curiosity_gap   — information gap without becoming clickbait
  • specificity     — numbers, years, dollar amounts build trust
  • power_words     — words correlated with high-CTR thumbnails
  • length_score    — YouTube truncates at ~60 chars on mobile
  • question_bonus  — question hooks increase engagement
  • caps_penalty    — excessive caps signal spam
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import re
from typing import Any

import structlog

from tools.llm_client import call_llm
from tools.retry_utils import with_retry
from workflows.pipeline_state import PipelineState

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CTR heuristic scorer
# ─────────────────────────────────────────────────────────────────────────────

_POWER_WORDS: list[str] = [
    "secret", "truth", "exposed", "mistake", "warning", "actually",
    "hidden", "nobody", "surprising", "shocking", "real", "finally",
    "stopped", "why", "how", "never", "always", "broke", "rich",
    "free", "fast", "proven", "simple", "worst", "best",
]

_SPECIFICITY_PATTERNS: list[str] = [
    r"\b\d+\s*%\b",
    r"\$[\d,]+",
    r"\b\d{4}\b",
    r"\b\d+\s+(ways|tips|steps|reasons|mistakes|things)\b",
    r"\b(study|research|data|survey|scientists)\b",
]

_CURIOSITY_PHRASES: list[str] = [
    "why", "what nobody", "the truth", "most people", "you're doing",
    "you don't know", "changes everything", "here's why", "this is why",
    "nobody tells you", "they won't tell",
]


def score_title(title: str) -> float:
    """Heuristic CTR score 0.0–1.0. Public so it can be used externally."""
    t = title.lower()
    score = 0.40

    # Curiosity gap
    curiosity_hits = sum(1 for p in _CURIOSITY_PHRASES if p in t)
    score += min(curiosity_hits * 0.06, 0.18)

    # Specificity (numbers, stats)
    for pat in _SPECIFICITY_PATTERNS:
        if re.search(pat, title, re.IGNORECASE):
            score += 0.06
    score = min(score, 0.99)

    # Power words
    power_hits = sum(1 for w in _POWER_WORDS if re.search(rf"\b{w}\b", t))
    score += min(power_hits * 0.04, 0.12)

    # Length — 40–65 chars is mobile sweet spot
    length = len(title)
    if length < 20:
        score -= 0.20
    elif 40 <= length <= 65:
        score += 0.08
    elif 30 <= length < 40:
        score += 0.03
    elif length > 80:
        score -= 0.10

    # Question hook bonus
    if title.strip().endswith("?"):
        score += 0.05

    # ALL CAPS penalty
    if len(title) > 3:
        caps_ratio = sum(1 for c in title if c.isupper()) / len(title)
        if caps_ratio > 0.5:
            score -= 0.15

    # Clutter penalty
    if title.count("...") > 1 or title.count("…") > 1:
        score -= 0.05

    return max(0.0, min(1.0, round(score, 3)))


# ─────────────────────────────────────────────────────────────────────────────
# LLM variant generator
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a YouTube title and hook copywriter who consistently achieves 8–12% CTR.

RULES:
1. Generate EXACTLY 3 title+hook variants using different psychological angles:
   Variant A — Loss aversion  : what the viewer LOSES by not watching
   Variant B — Curiosity gap  : a surprising fact, number, or counter-intuitive claim
   Variant C — Direct benefit : the concrete transformation the viewer gets
2. Title length: 35–70 characters. No ALL CAPS. No emoji. No excessive punctuation.
3. Hook: 1 punchy sentence (10–25 words) that pairs visually with the title.
4. Return ONLY a valid JSON array — no markdown fences, no commentary.

JSON schema (return exactly this structure):
[
  {"variant": "A", "angle": "loss_aversion",  "title": "...", "hook": "..."},
  {"variant": "B", "angle": "curiosity_gap",  "title": "...", "hook": "..."},
  {"variant": "C", "angle": "direct_benefit", "title": "...", "hook": "..."}
]
"""

_USER = """\
Topic         : {topic}
Niche         : {niche}
Current title : {current_title}
Current hook  : {current_hook}
Scene preview : {scenes_preview}

Generate 3 distinct title+hook variants. Return JSON only.
"""


@with_retry(max_attempts=3, base_delay_s=4.0, exceptions=(Exception,))
def _generate_variants(topic: str, niche: str, manifest_dict: dict) -> list[dict]:
    scenes = manifest_dict.get("scenes", [])
    preview = " | ".join(s.get("narration", "")[:60] for s in scenes[:3])

    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": _USER.format(
                topic=topic,
                niche=niche,
                current_title=manifest_dict.get("title", ""),
                current_hook=manifest_dict.get("hook", ""),
                scenes_preview=preview,
            ),
        },
    ]

    raw = call_llm(messages, temperature=0.90, max_tokens=600)
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("` ")
    data = json.loads(raw)
    return data if isinstance(data, list) else []


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

def title_ab_node(state: PipelineState) -> dict[str, Any]:
    """
    Title A/B Agent node.

    Reads:  scene_manifest, selected_topic, target_niche
    Writes: scene_manifest (title+hook updated to winner),
            title_variants (all scored variants for records)
    """
    manifest_dict = state.get("scene_manifest")
    topic         = state.get("selected_topic", "")
    niche         = state.get("target_niche", "personal_finance")

    if not manifest_dict:
        log.warning("title_ab_agent.no_manifest_skipping")
        return {}

    try:
        variants = _generate_variants(topic, niche, manifest_dict)
    except Exception as e:
        log.warning("title_ab_agent.generation_failed_keeping_original", error=str(e))
        return {"title_variants": []}

    if not variants:
        log.warning("title_ab_agent.empty_response_keeping_original")
        return {"title_variants": []}

    # Score all variants
    scored: list[dict] = []
    for v in variants:
        title = v.get("title", "").strip()
        if not title:
            continue
        scored.append({**v, "ctr_score": score_title(title)})

    # Also score original so it can win if LLM variants are worse
    orig_title = manifest_dict.get("title", "")
    if orig_title:
        scored.append({
            "variant":   "original",
            "angle":     "original",
            "title":     orig_title,
            "hook":      manifest_dict.get("hook", ""),
            "ctr_score": score_title(orig_title),
        })

    scored.sort(key=lambda x: (x["ctr_score"], x["variant"] == "original"), reverse=True)
    winner = scored[0] if scored else None

    if winner and winner.get("variant") != "original":
        manifest_dict = {**manifest_dict,
                         "title": winner["title"],
                         "hook":  winner["hook"]}
        log.info(
            "title_ab_agent.winner",
            variant=winner.get("variant"),
            angle=winner.get("angle"),
            title=winner["title"],
            ctr_score=winner["ctr_score"],
        )
    else:
        log.info("title_ab_agent.original_wins",
                 ctr_score=winner["ctr_score"] if winner else 0)

    return {
        "scene_manifest": manifest_dict,
        "title_variants":  scored,
        "job_status":      "human_review",
    }
