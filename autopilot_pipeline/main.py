"""
main.py
─────────────────────────────────────────────────────────────────────────────
AutoPilot Video Pipeline — primary entrypoint.

Usage examples:

  # Auto-discover a trending topic and run fully automated
  python main.py --niche personal_finance --approve

  # Seed a specific topic (skip Trend Scout)
  python main.py --topic "Why savings accounts are losing you money" --niche personal_finance

  # Dev mode (in-memory checkpointer, no Postgres/Redis required)
  python main.py --niche saas_tools --no-db --approve

  # Resume a paused run (after human review)
  python main.py --resume --thread <thread_id> --approve

  # Run with a specific niche
  python main.py --niche legal_tax --approve

Available niches:
  personal_finance | saas_tools | legal_tax | senior_health |
  real_estate      | storytelling | diy
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ── Compatibility patch ───────────────────────────────────────────────────────
# langchain-core internally reads langchain.debug for backwards compatibility.
# langchain 1.x removed this attribute. Patch ensures it exists regardless of
# which langchain version is installed (0.3.x local or 1.2.x on Kaggle).
try:
    import langchain as _lc
    if not hasattr(_lc, 'debug'):
        _lc.debug = False
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

import structlog
from dotenv import load_dotenv

# ── Load .env before anything else ──────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")

# ── Inject FFmpeg bin directory into PATH so subprocesses can find ffmpeg ────
_ffmpeg_bin = os.getenv("FFMPEG_BIN", "")
if _ffmpeg_bin and _ffmpeg_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")


def _configure_logging(level: str = "INFO", fmt: str = "json"):
    log_level = getattr(logging, level.upper(), logging.INFO)

    if fmt == "console":
        processors = [
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        processors = [
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    logging.basicConfig(stream=sys.stdout, level=log_level)


def run(
    topic: str | None = None,
    niche: str = "personal_finance",
    no_db: bool = False,
    approve: bool = False,
    thread_id: str | None = None,
    log_level: str = "INFO",
    log_format: str = "json",
):
    """Run the full pipeline programmatically (importable by tests/notebooks)."""
    _configure_logging(log_level, log_format)
    log = structlog.get_logger("main")

    if approve:
        os.environ["AUTOPILOT_AUTO_APPROVE"] = "1"

    from orchestration.supervisor import build_pipeline, make_initial_state

    # ── Checkpointer ─────────────────────────────────────────────────────────
    if no_db:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        log.info("main.checkpointer", type="memory")
    else:
        from infrastructure.postgres import get_checkpointer
        checkpointer = get_checkpointer()
        log.info("main.checkpointer", type="postgres")

    pipeline = build_pipeline(checkpointer=checkpointer)

    tid    = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}

    log.info("main.start", thread_id=tid, topic=topic, niche=niche)

    initial_state = make_initial_state(topic=topic, niche=niche)
    start_time    = datetime.now(timezone.utc)

    # ── Stream execution ─────────────────────────────────────────────────────
    for step in pipeline.stream(initial_state, config=config, stream_mode="updates"):
        node_name = list(step.keys())[0]

        if node_name == "__interrupt__":
            # Auto-approve fast-path (AUTOPILOT_AUTO_APPROVE=1)
            if os.getenv("AUTOPILOT_AUTO_APPROVE", "0") == "1":
                log.info("main.interrupt.auto_approved")
                approved_state = {"human_approved": True, "human_notes": "auto-approved"}
            else:
                # Human-in-the-loop pause
                interrupt_data = step["__interrupt__"]
                prompt = (
                    interrupt_data[0].value
                    if interrupt_data
                    else "Approve and start rendering? (y/n): "
                )
                print(prompt, end="", flush=True)
                user_input = input()
                approved   = user_input.strip().lower().startswith("y")
                approved_state = {"human_approved": approved, "human_notes": user_input}

            # CORRECT LangGraph resume pattern:
            # update_state() writes the human decision into the checkpointed state,
            # then stream(None) resumes from that checkpoint — never restarts.
            pipeline.update_state(config, approved_state, as_node="human_review")

            for resume_step in pipeline.stream(None, config=config, stream_mode="updates"):
                resume_node = list(resume_step.keys())[0]
                resume_data = resume_step[resume_node]
                # Guard: LangGraph may yield dicts or tuples in edge cases
                if isinstance(resume_data, dict):
                    resume_status = resume_data.get("job_status", "")
                    resume_errors = resume_data.get("errors", [])
                    if resume_errors:
                        for e in resume_errors:
                            log.warning("main.agent_error",
                                        agent=e.get("agent"), error=e.get("error"),
                                        recoverable=e.get("recoverable"))
                    log.info("main.step", node=resume_node, status=resume_status)
                else:
                    log.info("main.step", node=resume_node, status="ok")
            break

        update = step[node_name]
        status = update.get("job_status", "")
        errors = update.get("errors", [])

        if errors:
            for e in errors:
                log.warning("main.agent_error",
                            agent=e.get("agent"), error=e.get("error"),
                            recoverable=e.get("recoverable"))

        log.info("main.step", node=node_name, status=status)

    # ── Final state ───────────────────────────────────────────────────────────
    final   = pipeline.get_state(config).values
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

    from infrastructure.metrics import record_pipeline_run
    record_pipeline_run(
        niche=niche,
        status=final.get("job_status", "unknown"),
        duration_s=elapsed,
    )

    _print_summary(final, tid, elapsed)
    return final


def _print_summary(state: dict, thread_id: str, elapsed_s: float):
    bar = "=" * 68
    print(f"\n{bar}")
    print(f"  AutoPilot Pipeline — Completed")
    print(bar)
    print(f"  Status    : {state.get('job_status', 'unknown').upper()}")
    print(f"  Duration  : {elapsed_s:.0f}s  ({elapsed_s/60:.1f} min)")
    print(f"  Thread    : {thread_id}")
    print(f"  Niche     : {state.get('target_niche', 'N/A')} (CPM tier {state.get('target_cpm_tier', '?')})")
    print(f"  Topic     : {str(state.get('selected_topic', 'N/A'))[:70]}")
    print(f"  Uniqueness: {float(state.get('uniqueness_score') or 0):.2f}   "
          f"Entropy: {float(state.get('entropy_score') or 0):.2f}")

    compliance = state.get("compliance_score") or {}
    if compliance:
        overall = compliance.get("overall", 0) or 0
        print(f"  Compliance: {float(overall):.2f} overall")

    print(f"  TTS Tier  : {state.get('tts_tier_used', 'N/A')}")
    print(f"  Video     : {state.get('final_video_path', 'N/A')}")
    print(f"  YouTube   : {state.get('youtube_url', 'Not uploaded')}")

    errors = state.get("errors", [])
    if errors:
        print(f"  Errors    : {len(errors)}")
        for e in errors[-3:]:
            print(f"    [{e.get('agent')}] {e.get('error', '')[:80]}")

    print(bar)



# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AutoPilot Video Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--topic",      default=None,              help="Seed topic (auto-discover if omitted)")
    parser.add_argument("--niche",      default="personal_finance", help="Target niche")
    parser.add_argument("--no-db",      action="store_true",        help="In-memory mode (no Postgres)")
    parser.add_argument("--approve",    action="store_true",        help="Auto-approve human review")
    parser.add_argument("--resume",     action="store_true",        help="Resume existing thread")
    parser.add_argument("--thread",     default=None,               help="Thread ID to resume")
    parser.add_argument("--log-level",  default="INFO",             help="Log level (DEBUG/INFO/WARNING)")
    parser.add_argument("--log-format", default="console",          choices=["json", "console"],
                        help="Log format")
    args = parser.parse_args()

    run(
        topic=args.topic,
        niche=args.niche,
        no_db=args.no_db,
        approve=args.approve,
        thread_id=args.thread if args.resume else None,
        log_level=args.log_level,
        log_format=args.log_format,
    )


if __name__ == "__main__":
    main()
