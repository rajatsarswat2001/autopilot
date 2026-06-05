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
8. Visual prompts must be atmospheric and cinematographic \u2014 describe what a DP sees through a lens, not a checklist of requirements. Include scene mood, subject doing something specific, environment texture, and lighting feel. End with ', cinematic'.
9. Subjects must be doing concrete specific actions (counting banknotes, signing a contract, gripping a phone) not generic ones (standing, looking, walking). The scene should feel like a film still that implies motion, not a stage direction.
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


# ─────────────────────────────────────────────────────────────────────────────
# PRE-WRITTEN TEST SCRIPT — bypasses ALL LLM calls when TEST_SCRIPT_ENABLED=1
# Set TEST_SCRIPT_ENABLED=1 in .env / Kaggle cell to use this during testing.
# Saves Groq / Gemini quota and removes rate-limit risk entirely.
# ─────────────────────────────────────────────────────────────────────────────

_TEST_SCRIPT_JSON = """{
  "video_id": "__PLACEHOLDER__",
  "title": "The Savings Trick Banks Hide From You",
  "niche": "personal_finance",
  "target_cpm_tier": 2,
  "hook": "Most people lose $400 a year to a mistake their bank is counting on.",
  "scenes": [
    {
      "scene_id": 1,
      "narration": "Your bank earns interest on YOUR money and shares almost none of it. The average savings account pays 0.4% while inflation runs at 3%. You're losing ground every month.",
      "visual_prompt_A": "Extreme close-up of weathered male hands methodically counting a thick stack of worn $100 bills one by one on a dark polished mahogany desk, each bill slightly dog-eared and creased from heavy use, a single antique brass banker's lamp casts a warm amber cone of light from the upper right creating deep directional shadows across every ridge and fold of the bills, the scene fades into rich chocolate darkness behind with very shallow depth of field at f/1.8, subtle motion blur on each bill flip implying urgency and repetition, visible film grain, desaturated teal-and-orange color grade, cinematic",
      "b_roll_keyword_A": "counting cash money desk",
      "visual_prompt_B": "Medium wide shot of a single clear glass mason jar sitting alone on a weathered white-painted wooden kitchen windowsill, the jar is nearly empty with only a handful of scattered pennies and dimes at the bottom visible through the glass, soft diffused gray morning light pours through sheer linen curtains behind the jar creating a gentle rim highlight along the jar's curve, outside the window a blurred suburban street shows a few parked cars in cool bokeh, the counter surface shows subtle wood grain texture, muted cool blue-gray color palette with warm amber highlights on the jar rim, slow gentle push-in camera movement over 5 seconds, photorealistic, cinematic",
      "b_roll_keyword_B": "empty savings jar window light",
      "emotional_tone": "tense",
      "emotion_exaggeration": 0.7
    },
    {
      "scene_id": 2,
      "narration": "High-yield savings accounts from online banks pay 10 to 20 times more — same FDIC protection, no fees. I switched and earned $600 extra last year without doing anything else.",
      "visual_prompt_A": "Medium shot of a confident mid-30s woman with natural makeup and relaxed professional attire leaning forward at a minimalist white oak standing desk in a bright contemporary home office, she traces a sharply rising green line graph on a large ultrawide monitor with her index finger while grinning, the graph UI shows a steep upward curve labeled 4.85 percent APY in bold green text against a dark dashboard background, large floor-to-ceiling windows behind her flood the room with soft diffused blue-white morning daylight creating a clean soft box key light on her face and hair, healthy indoor plants and a steaming ceramic mug are visible in shallow background bokeh, lens at 50mm equivalent with shallow focus on her face, warm skin tones contrasting against cool ambient environment, subtle slow dolly-in camera movement, cinematic",
      "b_roll_keyword_A": "woman laptop banking smile office",
      "visual_prompt_B": "Tight macro shot looking straight at a large 4K monitor displaying a dark-mode fintech banking dashboard, the interface shows an account balance counter ticking upward in real time with bright green percentage numbers climbing, bold white sans-serif text reads HIGH-YIELD SAVINGS 4.85% APY, glowing teal and emerald accent UI elements illuminate the dark screen surface, the glossy monitor bezel reflects a softly blurred office environment in cool blues, camera slowly racks focus from the sharp screen text to the reflection revealing bokeh office shapes, deep cool cinematic color grade with electric teal highlights and deep shadow blacks, photorealistic screen render, cinematic",
      "b_roll_keyword_B": "bank dashboard screen teal glow",
      "emotional_tone": "inspiring",
      "emotion_exaggeration": 0.6
    },
    {
      "scene_id": 3,
      "narration": "The switch takes 10 minutes. Open the account, link your old bank, move your emergency fund. Your money starts working harder tonight.",
      "visual_prompt_A": "Close-up overhead bird's-eye shot angled at 80 degrees downward looking at a pair of hands holding a sleek matte-black iPhone 15, the screen displays a clean modern fintech app transfer interface mid-completion with a circular progress animation and bold green text reading Transfer Complete with a checkmark, the phone rests just above a dark aged walnut coffee table surface showing visible wood grain and a small white ceramic espresso cup and a tiny succulent plant softly out of focus at the edges of frame, natural window light streams in from camera left casting a gentle directional shadow of the hands across the warm wood surface, 35mm equivalent lens perspective with tight vignette, warm coffeehouse color temperature, slow orbital clockwise camera rotation at low angle, cinematic 4K",
      "b_roll_keyword_A": "phone banking app transfer wood table",
      "visual_prompt_B": "Ultra slow motion macro close-up at 240fps of gold and silver coins being dropped one by one onto a growing neat cylindrical stack on a pure white seamless surface, each coin tumbles through the air in perfect detail catching a single overhead hard studio strobe that creates a crisp specular highlight along the coin edge and a sharp dramatic shadow beneath the stack, the falling coin spins in slow motion mid-air revealing engraved texture detail, the stack visibly grows taller with each addition, background is pure white with a subtle soft gray radial vignette at corners, razor-sharp focus on the topmost coin with lower stack descending into gentle bokeh at f/2.0, cold clinical white light with warm gold coin reflections, macro lens distortion, cinematic high-speed photography aesthetic",
      "b_roll_keyword_B": "coins falling stack macro slow motion",
      "emotional_tone": "curious",
      "emotion_exaggeration": 0.5
    }
  ],
  "call_to_action": "Which bank are you using? Drop it in the comments — I'll tell you if you're leaving money on the table.",
  "tags": ["personal finance", "savings account", "high yield savings", "money tips", "financial advice"]
}"""


def script_writer_node(state: PipelineState) -> dict[str, Any]:
    """
    Script Writer Agent node.

    Reads:  selected_topic, target_niche, research_notes,
            script_revisions, script_draft (critic loop)
    Writes: script_draft, scene_manifest, uniqueness_score,
            script_revisions, errors

    TESTING SHORTCUT: set TEST_SCRIPT_ENABLED=1 in env to skip all LLM calls
    and use _TEST_SCRIPT_JSON (pre-written 3-scene, ~15s, personal finance).
    """
    topic     = state.get("selected_topic", "high-value financial topic")
    niche     = state.get("target_niche", "personal_finance")
    research  = state.get("research_notes", "")
    video_id  = state.get("video_id", str(uuid.uuid4()))
    revisions = state.get("script_revisions", 0)
    prev_draft = state.get("script_draft")
    prev_score = state.get("uniqueness_score", 0.0)

    log.info("script_agent.start", topic=topic, revision=revisions)

    # ── TESTING SHORT-CIRCUIT — zero LLM calls ───────────────────────────────
    if os.getenv("TEST_SCRIPT_ENABLED", "0").strip() == "1":
        log.info("script_agent.using_prewritten_script",
                 reason="TEST_SCRIPT_ENABLED=1, skipping all LLM calls")
        raw = _TEST_SCRIPT_JSON.replace("__PLACEHOLDER__", video_id)
        manifest, parse_issues = _parse_manifest(raw, video_id, niche)
        if manifest is None:
            log.error("script_agent.prewritten_parse_failed", issues=parse_issues)
            from datetime import datetime, timezone
            return {"errors": [{"agent": "script_writer",
                                "error": f"Pre-written script parse failed: {parse_issues}",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "recoverable": False}],
                    "script_revisions": 1}
        score = _score_uniqueness(manifest)
        log.info("script_agent.done", uniqueness=score, scenes=len(manifest.scenes),
                 mode="pre_written")
        return {
            "script_draft":     raw,
            "scene_manifest":   manifest.to_pipeline_dict(),
            "uniqueness_score": score,
            "script_revisions": 1,
            "job_status":       "scripting",
        }
    # ── END TESTING SHORT-CIRCUIT ─────────────────────────────────────────────

    # ── Build prompt ─────────────────────────────────────────────────────────
    # Read scene count + duration cap from env (testing defaults: 3 scenes, 20s)
    scene_count  = int(os.getenv("SCRIPT_SCENE_COUNT", "3"))
    max_dur_s    = float(os.getenv("MAX_VIDEO_DURATION_S", "20.0"))
    # Each scene narration should be short enough to fit within the budget.
    # At ~130 WPM: max_dur_s * 130 / 60 words total across all scenes.
    words_budget = int(max_dur_s * 130 / 60)
    words_per_scene = max(10, words_budget // max(1, scene_count))

    if revisions == 0 or not prev_draft:
        user_msg = (
            f"Topic: {topic}\nNiche: {niche}\n"
            f"Research:\n{research or 'Use general knowledge.'}\n\n"
            f"Schema:\n{_SCHEMA}\n\n"
            f"Generate exactly a {scene_count} scene script. Make the hook arresting and specific. "
            f"CRITICAL: Total video must be ≤{max_dur_s:.0f} seconds. "
            f"Each scene narration MUST be ≤{words_per_scene} words (short, punchy sentences). "
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
