"""
workflows/pipeline_state.py
─────────────────────────────────────────────────────────────────────────────
Shared state schema for the AutoPilot pipeline.

Design choices:
  • TypedDict (not dataclass) — LangGraph requires dict-compatible state.
  • Annotated[list, operator.add] — append semantics so concurrent agents
    can safely write to list fields without overwriting each other.
  • total=False — all fields are optional at construction; routers use
    state.get() with defaults rather than attribute access.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict


class AgentError(TypedDict):
    """Structured error entry appended to state.errors by any agent."""

    agent: str         # which agent raised the error
    error: str         # human-readable description
    timestamp: str     # ISO-8601 UTC
    recoverable: bool  # True → supervisor may retry; False → halt pipeline


class PipelineState(TypedDict, total=False):
    """
    Single source of truth for one video pipeline run.
    Serialised to Postgres via LangGraph PostgresSaver checkpointer.

    Reducer convention:
      list fields  → Annotated[list, operator.add]  (safe concurrent append)
      scalar fields → plain Optional[T]             (last write wins)
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    video_id:   str   # UUID4, set at pipeline init
    job_status: str   # init | scouting | researching | scripting | …

    # ── Phase: Trend Scout ────────────────────────────────────────────────────
    raw_trends:      Annotated[list[dict], operator.add]  # all candidates scored
    selected_topic:  Optional[str]                         # winning topic string
    target_niche:    str                                   # e.g. "personal_finance"
    target_cpm_tier: Optional[int]                        # 1=peak, 2=strong, 3=moderate

    # ── Phase: Research ───────────────────────────────────────────────────────
    research_notes: Optional[str]                        # aggregated facts + quotes
    source_urls:    Annotated[list[str], operator.add]   # scraped URLs

    # ── Phase: Script Writer ──────────────────────────────────────────────────
    script_draft:     Optional[str]    # raw LLM output (for critic loop)
    scene_manifest:   Optional[dict]   # validated SceneManifest.model_dump()
    uniqueness_score: Optional[float]  # 0.0–1.0 from heuristic scorer
    script_revisions: int              # number of rewrite attempts so far

    # ── Phase: Entropy Engine ─────────────────────────────────────────────────
    entropy_score:   Optional[float]  # 0.0–1.0 humanisation index
    entropy_applied: Optional[bool]

    # ── Phase: Compliance ─────────────────────────────────────────────────────
    compliance_score:  Optional[dict]              # multidimensional scores dict
    compliance_passed: Optional[bool]
    compliance_issues: Annotated[list[str], operator.add]

    # ── Phase: Audio ─────────────────────────────────────────────────────────
    audio_scenes:    Annotated[list[dict], operator.add]  # per-scene audio info
    timing_manifest: Optional[dict]                       # TimingManifest.model_dump()
    tts_tier_used:   Optional[str]                        # "chatterbox"|"magpie"|"edge"|"pyttsx3"

    # ── Phase: Visual Director ────────────────────────────────────────────────
    visual_scenes:   Annotated[list[dict], operator.add]  # per-scene visual info
    visual_manifest: Optional[dict]                        # VisualManifest.model_dump()

    # ── Phase: Title A/B ─────────────────────────────────────────────────────
    title_variants: Annotated[list[dict], operator.add]   # all scored title variants

    # ── Phase: Assembly ───────────────────────────────────────────────────────
    timeline_manifest: Optional[dict]   # TimelineManifest — full declarative render spec
    final_video_path:  Optional[str]    # absolute path to final MP4
    thumbnail_path:    Optional[str]    # absolute path to thumbnail PNG
    qa_passed:         Optional[bool]
    qa_notes:          Optional[str]

    # ── Phase: Upload ─────────────────────────────────────────────────────────
    youtube_video_id: Optional[str]
    youtube_url:      Optional[str]

    # ── System / Meta ─────────────────────────────────────────────────────────
    errors:         Annotated[list[AgentError], operator.add]
    human_approved: Optional[bool]
    human_notes:    Optional[str]
    messages:       Annotated[list[Any], operator.add]  # LangChain message history
    created_at:     Optional[str]   # ISO-8601 UTC
    updated_at:     Optional[str]   # ISO-8601 UTC
