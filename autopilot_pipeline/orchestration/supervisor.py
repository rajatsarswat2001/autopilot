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
from agents.audio_agent import audio_node
from agents.visual_director import visual_node
from agents.assembly_agent import assembly_node
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
    score = state.get("uniqueness_score", 0.0)
    manifest = state.get("scene_manifest")

    if not manifest or score < MIN_UNIQUENESS_SCORE:
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
) -> Literal["human_review", "script", "failed"]:
    if _has_fatal_error(state, "compliance"):
        return "failed"
    passed = state.get("compliance_passed", True)
    if not passed:
        revisions = state.get("script_revisions", 0)
        if revisions < MAX_SCRIPT_REVISIONS:
            log.warning("route.compliance_failed_rewriting")
            return "script"
        return "failed"
    return "human_review"


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


def route_after_visual(state: PipelineState) -> Literal["assembly", "failed"]:
    if _has_fatal_error(state, "visual_director"):
        return "failed"
    return "assembly"


def route_after_assembly(state: PipelineState) -> Literal["upload", "failed"]:
    if not state.get("qa_passed"):
        log.error("route.qa_failed", notes=state.get("qa_notes"))
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
    g.add_node("human_review",  human_review_node)
    g.add_node("audio",         audio_node)
    g.add_node("visual",        visual_node)
    g.add_node("assembly",      assembly_node)
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
                            {"human_review": "human_review", "script": "script", "failed": "failed"})
    g.add_conditional_edges("human_review", route_after_human,
                            {"audio": "audio", "script": "script", "failed": "failed"})
    g.add_conditional_edges("audio",        route_after_audio,
                            {"visual": "visual", "failed": "failed"})
    g.add_conditional_edges("visual",       route_after_visual,
                            {"assembly": "assembly", "failed": "failed"})
    g.add_conditional_edges("assembly",     route_after_assembly,
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
        timeline_manifest=None,
        final_video_path=None,
        thumbnail_path=None,
        qa_passed=None,
        qa_notes=None,
        youtube_video_id=None,
        youtube_url=None,
        errors=[],
        human_approved=None,
        human_notes=None,
        messages=[],
        created_at=now,
        updated_at=now,
    )

# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import structlog

    parser = argparse.ArgumentParser(description="AutoPilot Video Pipeline")
    parser.add_argument("--topic",   default=None,              help="Seed topic (Trend Scout auto-discovers if omitted)")
    parser.add_argument("--niche",   default="personal_finance", help="Target niche key")
    parser.add_argument("--no-db",   action="store_true",        help="Use in-memory checkpointer (dev mode)")
    parser.add_argument("--thread",  default=None,               help="Resume existing thread ID")
    parser.add_argument("--approve", action="store_true",        help="Auto-approve human review")
    args = parser.parse_args()

    if args.approve:
        os.environ["AUTOPILOT_AUTO_APPROVE"] = "1"

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )

    if args.no_db:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        log.info("checkpointer.memory")
    else:
        from langgraph.checkpoint.postgres import PostgresSaver
        checkpointer = PostgresSaver.from_conn_string(POSTGRES_URI)
        checkpointer.setup()
        log.info("checkpointer.postgres")

    pipeline = build_pipeline(checkpointer=checkpointer)
    thread_id = args.thread or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    log.info("pipeline.starting", thread_id=thread_id, topic=args.topic, niche=args.niche)
    initial_state = make_initial_state(topic=args.topic, niche=args.niche)

    for step in pipeline.stream(initial_state, config=config, stream_mode="updates"):
        node_name = list(step.keys())[0]
        status = step[node_name].get("job_status", "…")
        log.info("step", node=node_name, status=status)

        if node_name == "__interrupt__":
            prompt = step["__interrupt__"][0].value
            print(prompt, end="", flush=True)
            user_input = input()
            pipeline.invoke(
                {
                    "human_approved": user_input.strip().lower().startswith("y"),
                    "human_notes": user_input,
                },
                config=config,
            )
            break

    final = pipeline.get_state(config).values
    print(f"\n{'='*64}")
    print(f"Status   : {final.get('job_status')}")
    print(f"Video    : {final.get('youtube_url', 'N/A')}")
    print(f"File     : {final.get('final_video_path', 'N/A')}")
    print(f"Errors   : {len(final.get('errors', []))}")
    print(f"Thread   : {thread_id}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
