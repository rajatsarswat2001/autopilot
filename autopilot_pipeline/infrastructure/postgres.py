"""
infrastructure/postgres.py
─────────────────────────────────────────────────────────────────────────────
Postgres connection helpers and LangGraph checkpointer setup.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

POSTGRES_URI = os.getenv(
    "POSTGRES_URI",
    "postgresql://autopilot:autopilot@localhost:5432/autopilot",
)


def get_checkpointer():
    """
    Return a LangGraph PostgresSaver checkpointer.
    Falls back to MemorySaver if psycopg2 / Postgres is unavailable.
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        saver = PostgresSaver.from_conn_string(POSTGRES_URI)
        saver.setup()   # creates tables if not present
        log.info("postgres.checkpointer_ready")
        return saver
    except Exception as e:
        log.warning("postgres.unavailable_using_memory", error=str(e))
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()


def get_connection():
    """Return a raw psycopg2 connection (for custom queries)."""
    try:
        import psycopg2
        return psycopg2.connect(POSTGRES_URI)
    except Exception as e:
        log.error("postgres.connection_failed", error=str(e))
        raise


def execute(sql: str, params: tuple = (), fetch: bool = False) -> Optional[list]:
    """Execute a SQL statement. Returns rows if fetch=True."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            if fetch:
                return cur.fetchall()
            conn.commit()
    finally:
        conn.close()
    return None
