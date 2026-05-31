"""
agents/motion_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Motion Prompt Agent — rewrites raw visual prompts from the
Script Agent into CogVideoX-optimised motion prompts.

Why this exists:
  Script Agent optimises for NARRATION quality.
  This agent optimises for VIDEO GENERATION quality.
  These are completely different skills.

What it produces per scene:
  - Strong subject + action verb combination
  - Explicit camera movement (dolly, track, pan, tilt)
  - Environmental secondary motion (wind, water, crowd)
  - Temporal continuity language (no frozen frames)
  - Lighting and depth cues for cinematic quality

One LLM call handles all scenes in batch — fast and cheap.
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CogVideoX prompt structure reference
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_COGVIDEOX_STRUCTURE = """
COGVIDEOX PROMPT FORMULA:
[PRIMARY SUBJECT + STRONG ACTION VERB], [SECONDARY MOTION ELEMENT],
[EXPLICIT CAMERA MOVEMENT], [ENVIRONMENT + DEPTH], [LIGHTING QUALITY],
continuous motion throughout, no static frames, photorealistic

CAMERA MOVEMENT OPTIONS (pick one per prompt):
- camera dollies slowly forward
- camera tracks alongside subject
- camera pans left/right revealing scene  
- camera tilts up from ground to sky
- slow zoom out revealing wider environment
- handheld camera follows subject naturally
- camera circles subject slowly

SECONDARY MOTION (always include one):
- background pedestrians walking, traffic flowing
- leaves rustling in wind, trees swaying
- water rippling, flowing, splashing
- crowd moving in background
- papers shuffling, hands gesturing
- city traffic flowing past
- clouds drifting across sky

STRONG ACTION VERBS (use these not "standing/sitting"):
walks rapidly, strides confidently, gestures expressively,
counts money, types quickly, flips through documents,
looks up suddenly, turns to face camera, reaches forward,
opens envelope, points at screen, leans in, stands up
"""

_SYSTEM = f"""
You are a professional video director specialising in AI video generation.
Your job is to rewrite basic scene descriptions into high-quality motion prompts
optimised specifically for CogVideoX-2B, a text-to-video AI model.

{_COGVIDEOX_STRUCTURE}

RULES:
1. Every prompt MUST have a strong action verb — the subject must be DOING something.
2. Every prompt MUST have explicit camera movement described.
3. Every prompt MUST have at least one secondary motion element in the background.
4. Prompts must describe CONTINUOUS action, never a frozen moment.
5. Keep prompts between 60-120 words. Dense, specific, visual.
6. Stay true to the scene's EMOTIONAL TONE and NARRATION INTENT.
7. Finance niche: prefer office environments, city streets, banks, charts, documents.
8. Return ONLY valid JSON array. No markdown. No commentary.

QUALITY EXAMPLES:

WEAK (do not write like this):
"A person standing near a bank looking worried about money"

STRONG (write like this):
"A young professional walks rapidly through a modern bank lobby, clutching 
documents tightly, other customers moving past in background, camera tracks 
alongside at shoulder height, fluorescent office lighting creating sharp shadows, 
subject glances nervously at paper in hand while walking, continuous forward motion, 
no static frames, photorealistic cinematic footage"
"""

_USER = """
Niche: {niche}
Topic: {topic}

Rewrite the visual prompts for each scene below.
Keep scene_id and emotional_tone identical.
Replace visual_prompt_A and visual_prompt_B with motion-optimised versions.
If visual_prompt_B is empty, generate a complementary motion prompt for the same scene.

Scenes to rewrite:
{scenes_json}

Return ONLY a JSON array with this structure per scene:
[
  {{
    "scene_id": 1,
    "visual_prompt_A": "...",
    "visual_prompt_B": "...",
    "camera_move": "...",
    "motion_confidence": 0.0-1.0
  }}
]
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Motion quality scorer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_STRONG_VERBS = [
    "walks", "strides", "runs", "turns", "gestures", "reaches",
    "opens", "closes", "flips", "counts", "types", "points",
    "leans", "stands up", "sits down", "looks up", "moves",
    "rushes", "scrolls", "taps", "writes", "signs",
]

_CAMERA_WORDS = [
    "camera", "dolly", "pan", "tilt", "zoom", "track",
    "handheld", "follows", "circles", "reveals", "sweeps",
]

_MOTION_WORDS = [
    "flowing", "moving", "walking", "rushing", "swaying",
    "rippling", "drifting", "fluttering", "passing", "shuffling",
    "continuous", "throughout", "motion",
]


def score_motion_prompt(prompt: str) -> float:
    """
    Score a prompt 0.0-1.0 for motion quality.
    Used to decide whether LLM rewrite was worth it.
    """
    p = prompt.lower()
    score = 0.0

    verb_hits = sum(1 for v in _STRONG_VERBS if v in p)
    score += min(verb_hits * 0.15, 0.35)

    camera_hits = sum(1 for c in _CAMERA_WORDS if c in p)
    score += min(camera_hits * 0.20, 0.35)

    motion_hits = sum(1 for m in _MOTION_WORDS if m in p)
    score += min(motion_hits * 0.10, 0.30)

    # Penalise static language
    static_hits = sum(1 for s in ["standing", "sitting", "posed", "frozen",
                                   "static", "still"] if s in p)
    score -= static_hits * 0.15

    return max(0.0, min(1.0, round(score, 3)))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LangGraph node
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def motion_node(state: PipelineState) -> dict[str, Any]:
    """
    Motion Prompt Agent node.

    Reads:  scene_manifest, selected_topic, target_niche
    Writes: scene_manifest (visual_prompt_A/B enhanced),
            motion_scores, job_status
    """
    manifest_dict = state.get("scene_manifest")
    topic         = state.get("selected_topic", "")
    niche         = state.get("target_niche", "personal_finance")

    if not manifest_dict:
        log.warning("motion_agent.no_manifest_skipping")
        return {"job_status": "audio"}

    scenes = manifest_dict.get("scenes", [])
    if not scenes:
        log.warning("motion_agent.no_scenes_skipping")
        return {"job_status": "audio"}

    # Score existing prompts before rewrite
    before_scores = [
        score_motion_prompt(s.get("visual_prompt_A", ""))
        for s in scenes
    ]
    avg_before = round(sum(before_scores) / len(before_scores), 3)
    log.info("motion_agent.before", avg_score=avg_before, scenes=len(scenes))

    # Only send the fields the LLM needs — keeps token cost low
    slim_scenes = [
        {
            "scene_id":      s["scene_id"],
            "narration":     s.get("narration", "")[:120],
            "visual_prompt_A": s.get("visual_prompt_A", ""),
            "visual_prompt_B": s.get("visual_prompt_B", ""),
            "emotional_tone":  s.get("emotional_tone", "neutral"),
        }
        for s in scenes
    ]

    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": _USER.format(
                niche=niche,
                topic=topic,
                scenes_json=json.dumps(slim_scenes, indent=2),
            ),
        },
    ]

    try:
        raw = call_llm(messages, temperature=0.75, max_tokens=2000)
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("` ")
        rewritten = json.loads(raw)

        if not isinstance(rewritten, list):
            raise ValueError("LLM did not return a list")

        # Build lookup by scene_id
        rewrites = {r["scene_id"]: r for r in rewritten}

        # Merge rewritten prompts back into scenes
        updated_scenes = []
        motion_scores  = []

        for scene in scenes:
            sid = scene["scene_id"]
            if sid in rewrites:
                r = rewrites[sid]
                new_a = r.get("visual_prompt_A", scene.get("visual_prompt_A", ""))
                new_b = r.get("visual_prompt_B", scene.get("visual_prompt_B", ""))
                score = score_motion_prompt(new_a)
                motion_scores.append(score)
                updated_scenes.append({
                    **scene,
                    "visual_prompt_A": new_a,
                    "visual_prompt_B": new_b,
                    "camera_move":     r.get("camera_move", ""),
                })
                log.info(
                    "motion_agent.scene_rewritten",
                    scene_id=sid,
                    score_before=before_scores[sid - 1],
                    score_after=score,
                    camera=r.get("camera_move", "")[:40],
                )
            else:
                # LLM missed this scene — keep original
                updated_scenes.append(scene)
                motion_scores.append(before_scores[sid - 1])
                log.warning("motion_agent.scene_missing_from_llm", scene_id=sid)

        avg_after = round(sum(motion_scores) / len(motion_scores), 3)
        log.info(
            "motion_agent.done",
            avg_before=avg_before,
            avg_after=avg_after,
            delta=round(avg_after - avg_before, 3),
        )

        updated_manifest = {**manifest_dict, "scenes": updated_scenes}

        return {
            "scene_manifest": updated_manifest,
            "motion_scores":  motion_scores,
            "job_status":     "audio",
        }

    except Exception as e:
        log.warning(
            "motion_agent.failed_keeping_original",
            error=str(e)[:150],
        )
        # Non-fatal — pipeline continues with original prompts
        return {
            "scene_manifest": manifest_dict,
            "motion_scores":  before_scores,
            "job_status":     "audio",
        }
