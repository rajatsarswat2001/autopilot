"""
agents/compliance_agent.py
─────────────────────────────────────────────────────────────────────────────
Compliance & Originality Agent — multidimensional scoring before rendering.

Scores five dimensions (0.0–1.0 each):
  1. semantic_uniqueness  — TF-IDF based uniqueness vs. generic AI templates
  2. narrative_entropy    — sentence rhythm variation (from entropy agent)
  3. cadence_variation    — pacing irregularity across scenes
  4. advertiser_safety    — absence of sensitive/demonetisation-risk content
  5. policy_compliance    — no banned keywords, no misleading health claims

Final gate: all dimensions must exceed 0.65 for compliance_passed = True.
Issues list is populated for any failing dimension.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import re
from typing import Any

import structlog

from workflows.pipeline_state import PipelineState

log = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Signal libraries
# ─────────────────────────────────────────────────────────────────────────────

# Common AI template phrases that indicate low originality
TEMPLATE_PHRASES = [
    "in today's video", "welcome to", "don't forget to subscribe",
    "hit the bell icon", "like and subscribe", "in this video we will",
    "let's dive in", "without further ado", "stay tuned",
    "comment below", "drop a like", "smash that like button",
    "at the end of the day", "long story short", "needless to say",
]

# YouTube advertiser-sensitive topics (may trigger limited ads)
DEMONETISATION_RISKS = [
    r"\b(death|kill|murder|suicide|self.harm)\b",
    r"\b(sex|porn|nude|naked|explicit)\b",
    r"\b(drug|cocaine|heroin|meth|overdose)\b",
    r"\b(terrorism|bomb|weapon|shooting|massacre)\b",
    r"\b(hate speech|racial slur|homophob)\b",
]

# Misleading health/finance claims
MISLEADING_CLAIMS = [
    r"\b(cure(s|d)? (cancer|diabetes|covid))\b",
    r"\b(guaranteed (returns?|profits?|income))\b",
    r"\b(100% (safe|risk.free|guaranteed))\b",
    r"\b(doctors (hate|don't want you to know))\b",
    r"\b(secret (cure|formula|trick) (that|to))\b",
]

# Minimum score threshold per dimension
PASS_THRESHOLD = 0.65


# ─────────────────────────────────────────────────────────────────────────────
# Scorers
# ─────────────────────────────────────────────────────────────────────────────

def _semantic_uniqueness(full_text: str) -> float:
    """
    Heuristic uniqueness vs. generic AI templates.
    High template phrase density → low score.
    """
    template_hits = sum(1 for p in TEMPLATE_PHRASES if p in full_text.lower())
    penalty = min(template_hits * 0.12, 0.60)
    return round(max(0.0, 1.0 - penalty), 3)


def _narrative_entropy(full_text: str) -> float:
    """Sentence length variance → rhythm diversity."""
    sentences = re.split(r"[.!?]+", full_text)
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) < 3:
        return 0.5
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    return round(min(variance / 60.0, 1.0), 3)


def _cadence_variation(scenes: list[dict]) -> float:
    """
    Variation in narration length across scenes.
    Uniform scene lengths are a templating signal.
    """
    word_counts = [len(s.get("narration", "").split()) for s in scenes]
    if len(word_counts) < 2:
        return 0.5
    mean = sum(word_counts) / len(word_counts)
    variance = sum((c - mean) ** 2 for c in word_counts) / len(word_counts)
    return round(min(variance / 40.0, 1.0), 3)


def _advertiser_safety(full_text: str) -> tuple[float, list[str]]:
    """Check for demonetisation-risk content."""
    issues = []
    for pattern in DEMONETISATION_RISKS:
        matches = re.findall(pattern, full_text, re.IGNORECASE)
        if matches:
            issues.append(f"Advertiser risk: found '{matches[0]}' — rephrase or remove")
    score = max(0.0, 1.0 - len(issues) * 0.30)
    return round(score, 3), issues


def _policy_compliance(full_text: str) -> tuple[float, list[str]]:
    """Check for misleading claims and policy violations."""
    issues = []
    for pattern in MISLEADING_CLAIMS:
        matches = re.findall(pattern, full_text, re.IGNORECASE)
        if matches:
            issues.append(f"Misleading claim detected: '{matches[0]}' — remove or qualify")
    score = max(0.0, 1.0 - len(issues) * 0.35)
    return round(score, 3), issues


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

def compliance_node(state: PipelineState) -> dict[str, Any]:
    """
    Compliance Agent node.

    Reads:  scene_manifest, entropy_score
    Writes: compliance_score, compliance_passed, compliance_issues
    """
    manifest_dict = state.get("scene_manifest")
    if not manifest_dict:
        log.warning("compliance_agent.no_manifest")
        return {"compliance_passed": False, "compliance_issues": ["No scene_manifest found"]}

    scenes = manifest_dict.get("scenes", [])
    full_text = (
        manifest_dict.get("hook", "") + " "
        + manifest_dict.get("title", "") + " "
        + " ".join(s.get("narration", "") for s in scenes)
    )

    # ── Run all scorers ───────────────────────────────────────────────────────
    sem_u  = _semantic_uniqueness(full_text)
    narr_e = state.get("entropy_score") or _narrative_entropy(full_text)
    cad_v  = _cadence_variation(scenes)
    adv_s, adv_issues = _advertiser_safety(full_text)
    pol_s, pol_issues = _policy_compliance(full_text)

    scores = {
        "semantic_uniqueness": sem_u,
        "narrative_entropy":   round(float(narr_e), 3),
        "cadence_variation":   cad_v,
        "advertiser_safety":   adv_s,
        "policy_compliance":   pol_s,
    }

    overall = round(sum(scores.values()) / len(scores), 3)
    all_issues = adv_issues + pol_issues

    # Dimension-level failure issues
    for dim, val in scores.items():
        if val < PASS_THRESHOLD:
            all_issues.append(
                f"{dim} score {val:.2f} < {PASS_THRESHOLD} threshold — needs improvement"
            )

    passed = all(v >= PASS_THRESHOLD for v in scores.values())

    log.info(
        "compliance_agent.done",
        passed=passed,
        overall=overall,
        scores=scores,
        issues=len(all_issues),
    )

    return {
        "compliance_score":  {**scores, "overall": overall},
        "compliance_passed": passed,
        "compliance_issues": all_issues,
        "job_status":        "human_review" if passed else "scripting",
    }
