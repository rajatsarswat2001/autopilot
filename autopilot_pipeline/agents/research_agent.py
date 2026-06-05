"""
agents/research_agent.py
─────────────────────────────────────────────────────────────────────────────
Research Agent — web + RAG retrieval to enrich the script with facts.

Pipeline:
  1. Query Tavily for 5–8 articles about the selected topic
  2. Extract key facts, statistics, and quotes
  3. (Optional) Store to Qdrant for future RAG recall
  4. Return structured research_notes + source_urls

Research is best-effort: failures do not halt the pipeline.
The Script Agent will fall back to general LLM knowledge if notes are empty.

TESTING SHORTCUT: set TEST_SCRIPT_ENABLED=1 to skip all Tavily calls and
return hardcoded research notes matching the pre-written test script.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import textwrap
from datetime import datetime, timezone
from typing import Any

import requests
import structlog

from workflows.pipeline_state import AgentError, PipelineState

log = structlog.get_logger(__name__)

MAX_ARTICLES = 8
MAX_CHARS_PER_ARTICLE = 800


# ─────────────────────────────────────────────────────────────────────────────
# Tavily deep search
# ─────────────────────────────────────────────────────────────────────────────

def _tavily_deep(query: str, n: int = 8) -> list[dict]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": n,
                "search_depth": "advanced",   # deeper crawl
                "include_answer": True,
                "include_raw_content": False,
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception as e:
        log.warning("research_agent.tavily_error", error=str(e))
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Note builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_notes(topic: str, results: list[dict]) -> str:
    """
    Format Tavily results into structured research notes for the Script Agent.
    Emphasises numbers, statistics, and surprising facts.
    """
    lines = [
        f"RESEARCH NOTES: {topic}",
        f"Sources: {len(results)} articles | Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "=" * 60,
        "",
    ]

    for i, r in enumerate(results[:MAX_ARTICLES], 1):
        title   = r.get("title", "Untitled")
        content = r.get("content", r.get("snippet", ""))[:MAX_CHARS_PER_ARTICLE]
        url     = r.get("url", "")
        lines += [
            f"[{i}] {title}",
            f"    URL: {url}",
            f"    {textwrap.fill(content, width=90, subsequent_indent='    ')}",
            "",
        ]

    # Add Tavily's synthesised answer if available
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Optional Qdrant storage
# ─────────────────────────────────────────────────────────────────────────────

def _store_to_qdrant(video_id: str, topic: str, notes: str, source_urls: list[str]):
    """Best-effort: store research notes in Qdrant for future RAG recall."""
    try:
        from tools.rag_tools import upsert_document
        upsert_document(
            collection="research_notes",
            doc_id=video_id,
            text=notes,
            metadata={"topic": topic, "source_urls": source_urls},
        )
        log.info("research_agent.qdrant_stored", video_id=video_id)
    except Exception as e:
        log.warning("research_agent.qdrant_skip", reason=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Pre-written test research notes (used when TEST_SCRIPT_ENABLED=1)
# ─────────────────────────────────────────────────────────────────────────────

_TEST_RESEARCH_NOTES = """RESEARCH NOTES: The Savings Trick Banks Hide From You
Sources: Pre-written test data | TEST MODE (zero Tavily API calls)
============================================================

[1] Average savings account APY vs High-Yield savings 2025
    The national average savings account APY is 0.41% (FDIC, 2025).
    High-yield savings accounts from online banks offer 4.5-5.1% APY.
    That is 10-12x higher with identical FDIC insurance protection up to $250,000.

[2] Inflation erosion of savings
    US CPI inflation averaged 3.2% in 2024. A 0.4% APY savings account loses
    ~2.8% real purchasing power per year. On a $15,000 emergency fund that is
    ~$420 lost annually vs a high-yield savings account earning 4.85%.

[3] Switching time and friction
    Opening a high-yield savings account takes 5-10 minutes online.
    Funds transfer via ACH in 1-3 business days. No fees at most online banks.
    Popular options: Marcus by Goldman Sachs, Ally Bank, SoFi, American Express HYSA.

[4] Why big banks pay low rates
    Traditional banks have expensive branch networks and rely on customer inertia.
    Online banks have lower overhead and pass savings to customers as higher APY.
    FDIC insurance coverage is identical regardless of bank size or type.
"""

_TEST_SOURCE_URLS = [
    "https://www.fdic.gov/bank/statistical/guide/2025/",
    "https://www.bankrate.com/banking/savings/best-high-yield-interests-savings-accounts/",
]


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

def research_node(state: PipelineState) -> dict[str, Any]:
    """
    Research Agent node.

    Reads:  selected_topic, video_id
    Writes: research_notes, source_urls, job_status

    TESTING SHORTCUT: set TEST_SCRIPT_ENABLED=1 to skip all Tavily calls and
    return hardcoded research notes matching the pre-written test script.
    Zero API quota usage.
    """
    topic    = state.get("selected_topic", "")
    video_id = state.get("video_id", "")

    if not topic:
        log.warning("research_agent.no_topic")
        return {"research_notes": "", "source_urls": [], "job_status": "scripting"}

    # ── TESTING SHORT-CIRCUIT — zero API calls ────────────────────────────────
    if os.getenv("TEST_SCRIPT_ENABLED", "0").strip() == "1":
        log.info("research_agent.test_mode",
                 reason="TEST_SCRIPT_ENABLED=1, returning hardcoded notes, skipping Tavily")
        return {
            "research_notes": _TEST_RESEARCH_NOTES,
            "source_urls":    _TEST_SOURCE_URLS,
            "job_status":     "scripting",
        }
    # ── END TESTING SHORT-CIRCUIT ─────────────────────────────────────────────

    log.info("research_agent.start", topic=topic[:80])

    # Build richer search queries from the topic
    queries = [
        topic,
        f"{topic} statistics data 2025",
        f"{topic} expert advice surprising facts",
    ]

    all_results: list[dict] = []
    seen_urls: set[str] = set()

    for q in queries:
        for r in _tavily_deep(q, n=4):
            url = r.get("url", "")
            if url not in seen_urls:
                all_results.append(r)
                seen_urls.add(url)

    if not all_results:
        log.warning("research_agent.no_results")
        return {"research_notes": "", "source_urls": [], "job_status": "scripting"}

    notes       = _build_notes(topic, all_results)
    source_urls = [r.get("url", "") for r in all_results if r.get("url")]

    _store_to_qdrant(video_id, topic, notes, source_urls)

    log.info("research_agent.done", articles=len(all_results), chars=len(notes))

    return {
        "research_notes": notes,
        "source_urls":    source_urls,
        "job_status":     "scripting",
    }
