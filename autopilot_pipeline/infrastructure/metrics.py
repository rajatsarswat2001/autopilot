"""
infrastructure/metrics.py
─────────────────────────────────────────────────────────────────────────────
Lightweight metrics collection — Prometheus-compatible counters and histograms.

Falls back to a no-op implementation if prometheus_client is not installed.
Metrics are also logged via structlog for Grafana Loki ingestion.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import time
from functools import wraps
from typing import Callable

import structlog

log = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prometheus integration (optional)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from prometheus_client import Counter, Histogram, start_http_server

    PIPELINE_RUNS = Counter(
        "autopilot_pipeline_runs_total",
        "Total pipeline runs",
        ["niche", "status"],
    )
    LLM_CALLS = Counter(
        "autopilot_llm_calls_total",
        "LLM API calls",
        ["model", "agent"],
    )
    TTS_CALLS = Counter(
        "autopilot_tts_calls_total",
        "TTS synthesis calls",
        ["tier"],
    )
    RENDER_DURATION = Histogram(
        "autopilot_render_duration_seconds",
        "Video render duration",
        buckets=[30, 60, 120, 300, 600, 1200],
    )
    UNIQUENESS_SCORE = Histogram(
        "autopilot_uniqueness_score",
        "Script uniqueness scores",
        buckets=[0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


def start_metrics_server(port: int = 8000):
    if _PROMETHEUS_AVAILABLE:
        try:
            start_http_server(port)
            log.info("metrics.prometheus_server_started", port=port)
        except Exception as e:
            log.warning("metrics.prometheus_start_failed", error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Simple structured-log counters (always available)
# ─────────────────────────────────────────────────────────────────────────────

def record_pipeline_run(niche: str, status: str, duration_s: float):
    log.info("metric.pipeline_run", niche=niche, status=status, duration_s=round(duration_s, 1))
    if _PROMETHEUS_AVAILABLE:
        PIPELINE_RUNS.labels(niche=niche, status=status).inc()
        RENDER_DURATION.observe(duration_s)


def record_llm_call(model: str, agent: str):
    log.debug("metric.llm_call", model=model, agent=agent)
    if _PROMETHEUS_AVAILABLE:
        LLM_CALLS.labels(model=model, agent=agent).inc()


def record_tts_call(tier: str):
    log.debug("metric.tts_call", tier=tier)
    if _PROMETHEUS_AVAILABLE:
        TTS_CALLS.labels(tier=tier).inc()


def record_uniqueness(score: float):
    log.debug("metric.uniqueness", score=score)
    if _PROMETHEUS_AVAILABLE:
        UNIQUENESS_SCORE.observe(score)


# ─────────────────────────────────────────────────────────────────────────────
# Timing decorator
# ─────────────────────────────────────────────────────────────────────────────

def timed(label: str) -> Callable:
    """Decorator that logs execution time of a function."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.monotonic()
            result = fn(*args, **kwargs)
            elapsed = round(time.monotonic() - t0, 2)
            log.info("metric.timing", label=label, elapsed_s=elapsed)
            return result
        return wrapper
    return decorator
