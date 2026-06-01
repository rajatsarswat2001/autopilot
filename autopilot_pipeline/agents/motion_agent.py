"""
agents/motion_agent.py
────────────────────────────────────────────────────────────────
Motion Prompt Agent — rewrites raw visual prompts from the
Script Agent into Wan2.1-optimised cinematic scene prompts.

Why Wan2.1 needs different prompts than CogVideoX:
  CogVideoX-2B was weak at inferring motion — needed explicit
  "camera dollies forward", "background pedestrians walking" etc.

  Wan2.1 1.3B has far stronger motion priors — it infers natural
  movement from scene context. Over-specifying camera moves causes
  it to average across instructions and produce mediocre output.

  Wan2.1 responds best to dense, immersive, cinematographic
  scene descriptions — what a DP would see through the lens —
  rather than a checklist of motion requirements.

What this agent produces per scene:
  - Rich scene atmosphere and environment
  - Subject with specific, concrete action
  - Implied camera feel (lens type, depth, angle)
  - Lighting mood
  - Niche-specific visual language
  - One clean negative anchor

One LLM call handles all scenes in batch — fast and cheap.
────────────────────────────────────────────────────────────────
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


# ── WAN2.1 PROMPT REFERENCE ───────────────────────────────────────────────────

_WAN_STRUCTURE = """
WAN2.1 PROMPT FORMULA:
[Atmosphere/mood opening], [specific subject doing concrete action],
[environment with texture and depth], [implied lens/camera feel],
[lighting quality], cinematic

WHAT WAN2.1 RESPONDS TO:
- Dense atmospheric language ("rain-slicked streets", "golden dust motes")
- Specific concrete actions ("counting worn banknotes", "signing a thick contract")
- Implied motion through scene description (not "camera dollies" instructions)
- Lens feel language ("tight 85mm", "wide establishing", "shallow focus")
- Lighting that implies mood ("harsh fluorescent", "warm practicals", "rim lit")

WHAT HURTS WAN2.1 QUALITY:
- Mechanical checklists ("camera tracks alongside, secondary motion element")
- Contradictory instructions that average into mush
- Generic placeholder language ("professional environment", "cinematic lighting")
- Telling it "no static frames, continuous motion" — it already knows

NICHE VISUAL LANGUAGE:

personal_finance:
  Environments: glass-walled boardrooms, marble bank lobbies, cluttered home
  offices at 2am, trading floors with flickering tickers
  Subjects: exhausted accountant, nervous loan applicant, confident investor
  Textures: worn leather wallet, crisp paper statements, glowing spreadsheet

saas_tools:
  Environments: dark open-plan offices with monitor glow, startup lofts,
  minimalist desks with mechanical keyboards
  Subjects: developer hunched over terminal, designer dragging wireframes,
  founder on a video call
  Textures: screen reflections, cable tangles, sticky-note covered monitors

legal_tax:
  Environments: wood-panelled law libraries, fluorescent IRS offices,
  glass conference rooms at sunset
  Subjects: attorney annotating briefs, paralegal cross-referencing binders,
  nervous client across mahogany desk
  Textures: stamped official seals, highlighted clauses, heavy embossed folders

senior_health:
  Environments: sun-drenched kitchen, neighbourhood walking path at dawn,
  doctor's consultation room with afternoon light
  Subjects: active retiree, attentive physician, couple cooking together
  Textures: fresh produce, worn but loved sneakers, warm mugs

storytelling:
  Environments: rain on windows, empty diners at midnight, overgrown
  suburban houses, airport departure halls
  Subjects: figure with back to camera, hands exchanging something, a door
  closing or opening
  Textures: water on glass, cigarette smoke, crumpled letters
"""

_SYSTEM = f"""
You are a cinematographer and visual director writing prompts for Wan2.1,
a state-of-the-art text-to-video AI model.

Your prompts must read like a DP's shot description — immersive, atmospheric,
specific — not like a checklist of requirements.

{_WAN_STRUCTURE}

RULES:
1. Open with atmosphere or lighting — set the mood immediately.
2. The subject must be doing something SPECIFIC and CONCRETE, not generic.
3. Include at least one texture or material detail (leather, glass, paper, etc).
4. Use lens/depth language to imply camera feel without giving instructions.
5. Keep prompts 40-70 words. Dense and specific beats long and vague.
6. Never use: "camera dollies", "camera tracks", "no static frames",
   "continuous motion throughout", "secondary motion element",
   "professional environment", "cinematic lighting" as standalone phrase.
7. End every prompt with ", cinematic" — single word, no more.
8. Stay true to the scene's emotional tone and narration content.
9. Return ONLY valid JSON array. No markdown. No commentary.

QUALITY EXAMPLES:

WEAK (CogVideoX-style checklist — do not write like this):
"A person walks rapidly through a bank lobby, other customers moving in
background, camera tracks alongside at shoulder height, fluorescent lighting,
continuous motion throughout, no static frames, photorealistic"

STRONG (Wan2.1 atmospheric style — write like this):
"Harsh fluorescent light strips a marble bank lobby bare. A young man in a
slightly-too-large suit grips a manila folder to his chest, threading past
suited executives who don't notice him. Shallow focus on his white knuckles.
The numbers on the teller boards flicker. cinematic"

WEAK:
"An accountant sitting at a desk looking at papers, professional office,
camera pans left revealing scene, background motion"

STRONG:
"3am. A cluttered home office lit only by monitor glow. An accountant pulls
her glasses off and presses her palms against her eyes — stacks of tax returns
threatening to slide off the desk. The cursor blinks in an empty cell.
Tight 50mm. cinematic"
"""

_USER = """
Niche: {niche}
Topic: {topic}

Rewrite the visual prompts for each scene below.
Keep scene_id and emotional_tone identical.
Replace visual_prompt_A and visual_prompt_B with Wan2.1-optimised
atmospheric scene descriptions.
If visual_prompt_B is empty, write a complementary shot — different angle
or moment from the same scene.

Scenes to rewrite:
{scenes_json}

Return ONLY a JSON array with this exact structure per scene:
[
  {{
    "scene_id": 1,
    "visual_prompt_A": "...",
    "visual_prompt_B": "...",
    "shot_feel": "tight/wide/medium/aerial",
    "motion_confidence": 0.0-1.0
  }}
]
"""


# ── MOTION QUALITY SCORER — tuned for Wan2.1 prompts ─────────────────────────

# Words that indicate rich atmospheric description
_ATMOSPHERE_WORDS = [
    "light", "shadow", "glow", "dusk", "dawn", "fluorescent", "golden",
    "harsh", "soft", "rim", "flicker", "beam", "haze", "mist",
]

# Specific materials and textures — strong Wan2.1 signal
_TEXTURE_WORDS = [
    "leather", "glass", "paper", "marble", "wood", "steel", "concrete",
    "fabric", "worn", "crumpled", "polished", "frosted", "neon", "chrome",
]

# Concrete actions — subject doing something specific
_ACTION_WORDS = [
    "grips", "clutches", "flips", "counts", "signs", "tears", "slides",
    "presses", "pulls", "opens", "closes", "leans", "turns", "reaches",
    "reads", "types", "writes", "marks", "stamps", "hands",
]

# Lens/depth language — implies camera without instructing it
_LENS_WORDS = [
    "shallow", "focus", "50mm", "85mm", "24mm", "tight", "wide", "depth",
    "bokeh", "foreground", "background blurs", "rack",
]

# Penalise mechanical checklist language
_CHECKLIST_PENALTIES = [
    "camera dollies", "camera tracks", "camera pans", "camera tilts",
    "camera circles", "no static frames", "continuous motion throughout",
    "secondary motion element", "background pedestrians",
    "handheld camera follows",
]


def score_motion_prompt(prompt: str) -> float:
    p = prompt.lower()
    score = 0.30  # baseline

    atmosphere = sum(1 for w in _ATMOSPHERE_WORDS if w in p)
    score += min(atmosphere * 0.08, 0.20)

    texture = sum(1 for w in _TEXTURE_WORDS if w in p)
    score += min(texture * 0.10, 0.20)

    action = sum(1 for w in _ACTION_WORDS if w in p)
    score += min(action * 0.08, 0.16)

    lens = sum(1 for w in _LENS_WORDS if w in p)
    score += min(lens * 0.06, 0.12)

    # Penalise CogVideoX-style checklist language
    checklist = sum(1 for phrase in _CHECKLIST_PENALTIES if phrase in p)
    score -= checklist * 0.12

    # Reward appropriate prompt length (40-70 words is Wan2.1 sweet spot)
    words = len(p.split())
    if 40 <= words <= 70:
        score += 0.05
    elif words < 20:
        score -= 0.15
    elif words > 100:
        score -= 0.08

    # Reward ending with ", cinematic"
    if p.rstrip().endswith("cinematic"):
        score += 0.05

    return max(0.0, min(1.0, round(score, 3)))


# ── LANGGRAPH NODE ────────────────────────────────────────────────────────────

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

    before_scores = [
        score_motion_prompt(s.get("visual_prompt_A", ""))
        for s in scenes
    ]
    avg_before = round(sum(before_scores) / len(before_scores), 3)
    log.info("motion_agent.before", avg_score=avg_before, scenes=len(scenes))

    slim_scenes = [
        {
            "scene_id":        s["scene_id"],
            "narration":       s.get("narration", "")[:150],
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
        raw = call_llm(messages, temperature=0.80, max_tokens=3000)
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("` ")
        rewritten = json.loads(raw)

        if not isinstance(rewritten, list):
            raise ValueError("LLM did not return a list")

        rewrites = {r["scene_id"]: r for r in rewritten}

        updated_scenes = []
        motion_scores  = []

        for i, scene in enumerate(scenes):
            sid = scene["scene_id"]
            if sid in rewrites:
                r     = rewrites[sid]
                new_a = r.get("visual_prompt_A", scene.get("visual_prompt_A", ""))
                new_b = r.get("visual_prompt_B", scene.get("visual_prompt_B", ""))
                score = score_motion_prompt(new_a)
                motion_scores.append(score)
                updated_scenes.append({
                    **scene,
                    "visual_prompt_A": new_a,
                    "visual_prompt_B": new_b,
                    "shot_feel":       r.get("shot_feel", "medium"),
                })
                log.info(
                    "motion_agent.scene_rewritten",
                    scene_id=sid,
                    score_before=before_scores[i],
                    score_after=score,
                    shot=r.get("shot_feel", ""),
                    prompt_words=len(new_a.split()),
                )
            else:
                updated_scenes.append(scene)
                motion_scores.append(before_scores[i])
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
        log.warning("motion_agent.failed_keeping_original", error=str(e)[:150])
        return {
            "scene_manifest": manifest_dict,
            "motion_scores":  before_scores,
            "job_status":     "audio",
        }
