"""
orchestration/supervisor.py
─────────────────────────────────────────────────────────────────────────────
LangGraph Supervisor — Stateful router, checkpointing, and error bus.

Architecture:
  • LangGraph v0.3+ StateGraph with Postgres checkpointer for durability.
  • DETERMINISTIC routing (Python code, no LLM reasoning) for speed.
  • Human-in-the-loop checkpoint fires after script approval, before GPU work.
  • Error bus: any agent writes to state.errors; supervisor decides
    whether to retry, degrade gracefully, or halt.

Usage:
    python -m orchestration.supervisor --topic "savings account fees" --niche personal_finance
    python -m orchestration.supervisor --no-db   # in-memory dev mode
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import os
import uuid
from datetime import datetime, timezone
from typing import Literal

import structlog
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from workflows.pipeline_state import AgentError, PipelineState
from agents.trend_scout import trend_scout_node
from agents.research_agent import research_node
from agents.script_agent import script_writer_node
from agents.entropy_agent import entropy_node
from agents.compliance_agent import compliance_node
from agents.title_ab_agent import title_ab_node
from agents.motion_agent import motion_node
from agents.audio_agent import audio_node
from agents.visual_director import visual_node
from agents.visual_qa_agent import visual_qa_node
from agents.assembly_agent import timeline_node, render_node, qa_thumbnail_node
from agents.seo_agent import seo_node
from agents.upload_agent import upload_node

load_dotenv()
log = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MAX_SCRIPT_REVISIONS = 3
MIN_UNIQUENESS_SCORE = 0.70
MIN_ENTROPY_SCORE = 0.65
MIN_COMPLIANCE_SCORE = 0.70
POSTGRES_URI = os.getenv(
    "POSTGRES_URI", "postgresql://autopilot:autopilot@localhost:5432/autopilot"
)

# ─────────────────────────────────────────────────────────────────────────────
# Router functions — deterministic Python, no LLM calls
# ─────────────────────────────────────────────────────────────────────────────

def route_after_scout(state: PipelineState) -> Literal["research", "failed"]:
    if _has_fatal_error(state, "trend_scout"):
        return "failed"
    if not state.get("selected_topic"):
        log.warning("route.no_topic")
        return "failed"
    return "research"


def route_after_research(state: PipelineState) -> Literal["script", "failed"]:
    if _has_fatal_error(state, "research"):
        return "failed"
    return "script"   # research is best-effort; always proceed


def route_after_script(
    state: PipelineState,
) -> Literal["script", "entropy", "failed"]:
    if _has_fatal_error(state, "script_writer"):
        return "failed"

    revisions = state.get("script_revisions", 0)
    score     = state.get("uniqueness_score", 0.0)
    manifest  = state.get("scene_manifest")

    if not manifest or score < MIN_UNIQUENESS_SCORE:
        # revisions starts at 0 before first attempt; script_writer increments it.
        # So revisions==1 after first run, meaning we have MAX_SCRIPT_REVISIONS-1 retries.
        if revisions < MAX_SCRIPT_REVISIONS:
            log.info("route.script_retry", revision=revisions, score=score)
            return "script"
        log.error("route.script_max_revisions", score=score)
        return "failed"

    log.info("route.script_passed", score=score, scenes=len(manifest.get("scenes", [])))
    return "entropy"


def route_after_entropy(state: PipelineState) -> Literal["compliance", "failed"]:
    if _has_fatal_error(state, "entropy"):
        return "failed"
    return "compliance"


def route_after_compliance(
    state: PipelineState,
) -> Literal["title_ab", "script", "failed"]:
    if _has_fatal_error(state, "compliance"):
        return "failed"
    passed = state.get("compliance_passed", True)
    if not passed:
        revisions = state.get("script_revisions", 0)
        if revisions < MAX_SCRIPT_REVISIONS:
            log.warning("route.compliance_failed_rewriting")
            return "script"
        return "failed"
    return "title_ab"


def route_after_human(
    state: PipelineState,
) -> Literal["audio", "script", "failed"]:
    if state.get("human_approved") is False:
        log.info("route.human_rejected", notes=state.get("human_notes"))
        return "script"   # human feedback → fresh rewrite
    if state.get("human_approved") is True:
        return "audio"
    return "failed"


def route_after_audio(state: PipelineState) -> Literal["visual", "failed"]:
    if _has_fatal_error(state, "audio"):
        return "failed"
    if not state.get("timing_manifest"):
        return "failed"
    return "visual"


def route_after_visual(state: PipelineState) -> Literal["visual_qa", "failed"]:
    if _has_fatal_error(state, "visual_director"):
        return "failed"
    return "visual_qa"


def route_after_visual_qa(state: PipelineState) -> Literal["timeline", "failed"]:
    if not state.get("visual_qa_passed"):
        return "failed"
    return "timeline"


def route_after_timeline(state: PipelineState) -> Literal["render", "failed"]:
    if _has_fatal_error(state, "timeline"):
        return "failed"
    return "render"


def route_after_render(state: PipelineState) -> Literal["qa_thumbnail", "failed"]:
    if _has_fatal_error(state, "render"):
        return "failed"
    return "qa_thumbnail"


def route_after_qa_thumbnail(state: PipelineState) -> Literal["seo", "failed"]:
    if not state.get("qa_passed"):
        log.error("route.qa_failed", notes=state.get("qa_notes"))
        return "failed"
    return "seo"


def route_after_seo(state: PipelineState) -> Literal["upload", "failed"]:
    if _has_fatal_error(state, "seo"):
        return "failed"
    return "upload"

# ─────────────────────────────────────────────────────────────────────────────
# Special nodes
# ─────────────────────────────────────────────────────────────────────────────

def human_review_node(state: PipelineState) -> dict:
    """
    Human-in-the-loop checkpoint.
    Set AUTOPILOT_AUTO_APPROVE=1 for fully automated runs.
    Otherwise, LangGraph interrupt() pauses here; resume by reinvoking
    the graph with human_approved=True/False.
    """
    if os.getenv("AUTOPILOT_AUTO_APPROVE", "0") == "1":
        log.info("human_review.auto_approved")
        return {"human_approved": True, "job_status": "audio"}

    manifest = state.get("scene_manifest", {})
    scenes = manifest.get("scenes", [])
    preview = "\n".join(
        f"  Scene {s['scene_id']}: {s['narration'][:80]}…"
        for s in scenes[:4]
    )
    prompt = (
        f"\n{'='*64}\n"
        f"VIDEO: {manifest.get('title', 'Untitled')}\n"
        f"NICHE: {state.get('target_niche')}  |  CPM TIER: {state.get('target_cpm_tier')}\n"
        f"UNIQUENESS: {state.get('uniqueness_score', 0):.2f}  |  "
        f"ENTROPY: {state.get('entropy_score', 0):.2f}\n"
        f"SCENES ({len(scenes)} total):\n{preview}\n"
        f"{'='*64}\n"
        f"Approve and start GPU rendering? (y/n/edit): "
    )

    human_response = interrupt(prompt)
    approved = str(human_response).strip().lower().startswith("y")
    return {
        "human_approved": approved,
        "human_notes": str(human_response),
        "job_status": "audio" if approved else "scripting",
    }


def terminal_success_node(state: PipelineState) -> dict:
    log.info(
        "pipeline.complete",
        youtube_url=state.get("youtube_url"),
        video_id=state.get("youtube_video_id"),
    )
    return {"job_status": "done", "updated_at": datetime.now(timezone.utc).isoformat()}


def terminal_failure_node(state: PipelineState) -> dict:
    errors = state.get("errors", [])
    log.error("pipeline.failed", error_count=len(errors), last_errors=errors[-3:])
    return {"job_status": "failed", "updated_at": datetime.now(timezone.utc).isoformat()}

# ─────────────────────────────────────────────────────────────────────────────
# Graph assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(checkpointer=None):
    """
    Construct and compile the full StateGraph pipeline.

    Args:
        checkpointer: LangGraph checkpointer. None = in-memory (dev mode).
                      PostgresSaver for production durability.
    """
    g = StateGraph(PipelineState)

    # ── Register nodes ──────────────────────────────────────────────────────
    g.add_node("trend_scout",   trend_scout_node)
    g.add_node("research",      research_node)
    g.add_node("script",        script_writer_node)
    g.add_node("entropy",       entropy_node)
    g.add_node("compliance",    compliance_node)
    g.add_node("title_ab",      title_ab_node)
    g.add_node("motion",        motion_node)
    g.add_node("human_review",  human_review_node)
    g.add_node("audio",         audio_node)
    g.add_node("visual",        visual_node)
    g.add_node("visual_qa",     visual_qa_node)
    g.add_node("timeline",      timeline_node)
    g.add_node("render",        render_node)
    g.add_node("qa_thumbnail",  qa_thumbnail_node)
    g.add_node("seo",           seo_node)
    g.add_node("upload",        upload_node)
    g.add_node("success",       terminal_success_node)
    g.add_node("failed",        terminal_failure_node)

    # ── Entry ────────────────────────────────────────────────────────────────
    g.add_edge(START, "trend_scout")

    # ── Routing ─────────────────────────────────────────────────────────────
    g.add_conditional_edges("trend_scout",  route_after_scout,
                            {"research": "research", "failed": "failed"})
    g.add_conditional_edges("research",     route_after_research,
                            {"script": "script", "failed": "failed"})
    g.add_conditional_edges("script",       route_after_script,
                            {"script": "script", "entropy": "entropy", "failed": "failed"})
    g.add_conditional_edges("entropy",      route_after_entropy,
                            {"compliance": "compliance", "failed": "failed"})
    g.add_conditional_edges("compliance",   route_after_compliance,
                            {"title_ab": "title_ab", "script": "script", "failed": "failed"})
    g.add_edge("title_ab", "motion")
    g.add_edge("motion", "human_review")
    g.add_conditional_edges("human_review", route_after_human,
                            {"audio": "audio", "script": "script", "failed": "failed"})
    g.add_conditional_edges("audio",        route_after_audio,
                            {"visual": "visual", "failed": "failed"})
    g.add_conditional_edges("visual",       route_after_visual,
                            {"visual_qa": "visual_qa", "failed": "failed"})
    g.add_conditional_edges("visual_qa",    route_after_visual_qa,
                            {"timeline": "timeline", "failed": "failed"})
    g.add_conditional_edges("timeline",     route_after_timeline,
                            {"render": "render", "failed": "failed"})
    g.add_conditional_edges("render",       route_after_render,
                            {"qa_thumbnail": "qa_thumbnail", "failed": "failed"})
    g.add_conditional_edges("qa_thumbnail", route_after_qa_thumbnail,
                            {"seo": "seo", "failed": "failed"})
    g.add_conditional_edges("seo",          route_after_seo,
                            {"upload": "upload", "failed": "failed"})

    # ── Terminal edges ───────────────────────────────────────────────────────
    g.add_edge("upload",  "success")
    g.add_edge("success", END)
    g.add_edge("failed",  END)

    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],
    )

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _has_fatal_error(state: PipelineState, agent_name: str) -> bool:
    return any(
        e["agent"] == agent_name and not e["recoverable"]
        for e in state.get("errors", [])
    )


def make_initial_state(
    topic: str | None = None,
    niche: str = "personal_finance",
) -> PipelineState:
    now = datetime.now(timezone.utc).isoformat()
    return PipelineState(
        video_id=str(uuid.uuid4()),
        job_status="init",
        raw_trends=[],
        selected_topic=topic,
        target_niche=niche,
        target_cpm_tier=None,
        research_notes=None,
        source_urls=[],
        script_draft=None,
        scene_manifest=None,
        uniqueness_score=None,
        script_revisions=0,
        entropy_score=None,
        entropy_applied=None,
        compliance_score=None,
        compliance_passed=None,
        compliance_issues=[],
        audio_scenes=[],
        timing_manifest=None,
        tts_tier_used=None,
        visual_scenes=[],
        visual_manifest=None,
        visual_qa_passed=None,
        visual_qa_notes=None,
        timeline_manifest=None,
        final_video_path=None,
        caption_path=None,
        thumbnail_path=None,
        qa_passed=None,
        qa_notes=None,
        seo_metadata=None,
        youtube_video_id=None,
        youtube_url=None,
        errors=[],
        human_approved=None,
        human_notes=None,
        messages=[],
        title_variants=[],
        motion_scores=[],
        created_at=now,
        updated_at=now,
    )

# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────

# NOTE: The CLI entrypoint lives in main.py (project root).
# Run the pipeline with:  python main.py --niche personal_finance --approve
# supervisor.py is import-only — it only exports build_pipeline() and make_initial_state().
