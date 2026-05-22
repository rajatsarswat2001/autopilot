"""
agents/trend_scout.py
─────────────────────────────────────────────────────────────────────────────
Trend Scout Agent — discovers high-CPM YouTube topics.

Strategy:
  1. If state.selected_topic is pre-seeded → skip scouting
  2. Query Tavily API for trending queries per niche seed
  3. Score each candidate: CPM tier × competition × keyword density
  4. Check YouTube Data API v3 for competition density
  5. Return the highest-scoring topic under the competition threshold

CPM Tiers:
  Tier 1 ($18–50): personal_finance, saas_tools, legal_tax
  Tier 2 ($10–20): senior_health, real_estate, storytelling
  Tier 3  ($6–12): diy, how_to, nature
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

import requests
import structlog

from workflows.pipeline_state import AgentError, PipelineState

log = structlog.get_logger(__name__)

MAX_COMPETING_VIDEOS = 500

# ─────────────────────────────────────────────────────────────────────────────
# Niche configuration
# ─────────────────────────────────────────────────────────────────────────────

NICHE_CONFIG: dict[str, dict] = {
    "personal_finance": {
        "tier": 1,
        "keywords": ["savings", "credit card", "interest", "budget", "investment", "401k", "debt", "loan"],
        "anti_keywords": ["get rich quick", "hack", "loophole"],
        "seed_queries": [
            "personal finance mistakes most people make 2025",
            "savings account secrets banks hide from you",
            "credit score myths debunked financial experts",
        ],
    },
    "saas_tools": {
        "tier": 1,
        "keywords": ["AI tool", "software", "productivity", "automation", "workflow", "SaaS"],
        "anti_keywords": [],
        "seed_queries": [
            "best AI tools small business productivity 2025",
            "Claude vs ChatGPT entrepreneurs comparison",
            "automation tools that replace expensive software",
        ],
    },
    "legal_tax": {
        "tier": 1,
        "keywords": ["tax deduction", "legal", "freelancer", "LLC", "write-off", "IRS", "audit"],
        "anti_keywords": ["tax evasion", "illegal"],
        "seed_queries": [
            "tax deductions freelancers miss every year",
            "LLC benefits explained simply small business",
            "IRS audit triggers to avoid freelancer",
        ],
    },
    "senior_health": {
        "tier": 2,
        "keywords": ["senior", "over 60", "aging", "longevity", "retirement", "elderly", "50+"],
        "anti_keywords": [],
        "seed_queries": [
            "morning routine science for people over 60",
            "supplements seniors actually need science",
            "exercises that slow aging research 2025",
        ],
    },
    "real_estate": {
        "tier": 2,
        "keywords": ["real estate", "rental", "mortgage", "housing market", "property"],
        "anti_keywords": [],
        "seed_queries": [
            "analyze rental property for beginners 2025",
            "housing market predictions explained simply",
            "real estate investing small budget strategy",
        ],
    },
    "storytelling": {
        "tier": 2,
        "keywords": ["betrayal", "revenge", "true story", "shocking story"],
        "anti_keywords": ["graphic", "explicit", "nsfw"],
        "seed_queries": [
            "viral betrayal revenge story 2025",
            "shocking true story YouTube trending",
            "inspirational comeback story viral 2025",
        ],
    },
    "diy": {
        "tier": 3,
        "keywords": ["DIY", "how to", "tutorial", "build", "make", "beginner"],
        "anti_keywords": [],
        "seed_queries": [
            "DIY home improvement beginner 2025",
            "rooftop garden tutorial budget",
            "homesteading beginners guide 2025",
        ],
    },
}

CPM_SIGNALS: list[tuple[list[str], int]] = [
    (["credit card", "savings", "tax", "IRS", "401k", "LLC", "deduction", "mortgage", "debt"],     1),
    (["AI tool", "SaaS", "ChatGPT", "Claude", "automation", "productivity software"],               1),
    (["legal", "attorney", "lawsuit", "copyright", "trademark"],                                    1),
    (["real estate", "rental", "investment property", "REI"],                                       2),
    (["senior", "over 60", "aging", "longevity", "retirement health"],                              2),
    (["story", "betrayal", "revenge", "true story"],                                                2),
    (["DIY", "how to", "tutorial", "garden", "homestead"],                                         3),
]

NICHE_FALLBACKS: dict[str, str] = {
    "personal_finance": "Why 90% of people lose money on their savings accounts",
    "saas_tools":       "Top AI tools replacing expensive software in 2025",
    "legal_tax":        "5 tax deductions freelancers always miss",
    "senior_health":    "Morning routine science for people over 60",
    "real_estate":      "How to analyze a rental property in 7 minutes",
    "storytelling":     "The betrayal that changed everything — a true story",
    "diy":              "How I built a rooftop garden for under $200",
}


# ─────────────────────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tavily_search(query: str, max_results: int = 5) -> list[dict]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        log.warning("trend_scout.tavily_no_key")
        return []
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query,
                  "max_results": max_results, "search_depth": "basic"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception as e:
        log.warning("trend_scout.tavily_error", error=str(e))
        return []


def _youtube_competition(query: str) -> int:
    """Return estimated video count for query (0 = unknown / API key missing)."""
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return 0
    try:
        resp = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet", "q": query, "type": "video",
                "maxResults": 1, "order": "relevance",
                "publishedAfter": "2024-01-01T00:00:00Z", "key": api_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return int(resp.json().get("pageInfo", {}).get("totalResults", 0))
    except Exception as e:
        log.warning("trend_scout.youtube_api_error", error=str(e))
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _classify_cpm_tier(text: str) -> int:
    low = text.lower()
    for keywords, tier in CPM_SIGNALS:
        if any(kw.lower() in low for kw in keywords):
            return tier
    return 3


def _score(title: str, snippet: str, cfg: dict, competition: int) -> float:
    combined = (title + " " + snippet).lower()
    tier      = _classify_cpm_tier(combined)
    cpm_score = 4.0 - tier   # tier 1 → 3.0, tier 2 → 2.0, tier 3 → 1.0

    comp_score = {0: 1.0}.get(competition, (
        2.0 if competition < 100 else
        1.0 if competition < MAX_COMPETING_VIDEOS else
        0.0
    ))

    kw_hits   = sum(1 for kw in cfg.get("keywords", []) if kw.lower() in combined)
    anti_hits = sum(1 for kw in cfg.get("anti_keywords", []) if kw.lower() in combined)

    return max(0.0, round(cpm_score + comp_score + min(kw_hits * 0.3, 1.5) - anti_hits, 3))


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

def trend_scout_node(state: PipelineState) -> dict[str, Any]:
    """
    Trend Scout Agent node.

    Reads:  selected_topic (pre-seed), target_niche
    Writes: selected_topic, raw_trends, target_cpm_tier, job_status, errors
    """
    niche       = state.get("target_niche", "personal_finance")
    pre_seeded  = state.get("selected_topic")

    if pre_seeded:
        tier = _classify_cpm_tier(pre_seeded)
        log.info("trend_scout.preseeded", topic=pre_seeded, tier=tier)
        return {"selected_topic": pre_seeded, "target_cpm_tier": tier, "job_status": "researching"}

    cfg = NICHE_CONFIG.get(niche, NICHE_CONFIG["personal_finance"])
    log.info("trend_scout.start", niche=niche, default_tier=cfg["tier"])

    candidates: list[dict] = []
    for query in cfg["seed_queries"]:
        results = _tavily_search(query, max_results=5)
        for r in results:
            title   = r.get("title", "").strip()
            snippet = r.get("content", r.get("snippet", ""))[:300]
            if not title:
                continue
            competition = _youtube_competition(title)
            score       = _score(title, snippet, cfg, competition)
            tier        = _classify_cpm_tier(title + " " + snippet)
            candidates.append({
                "title": title, "snippet": snippet, "url": r.get("url", ""),
                "cpm_tier": tier, "competition": competition, "score": score,
            })
        time.sleep(0.5)  # gentle rate limit

    if not candidates:
        topic = NICHE_FALLBACKS.get(niche, "Personal finance tips that actually work")
        log.warning("trend_scout.no_candidates_fallback", topic=topic)
        return {"selected_topic": topic, "raw_trends": [], "target_cpm_tier": cfg["tier"], "job_status": "researching"}

    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]
    log.info("trend_scout.selected", topic=best["title"], score=best["score"], tier=best["cpm_tier"])

    return {
        "selected_topic":  best["title"],
        "raw_trends":      candidates,
        "target_cpm_tier": best["cpm_tier"],
        "job_status":      "researching",
    }
