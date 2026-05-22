"""
tools/retry_utils.py
─────────────────────────────────────────────────────────────────────────────
Shared retry utilities — exponential back-off with jitter for all
API-calling agents.

Usage:
    from tools.retry_utils import with_retry

    @with_retry(max_attempts=3, exceptions=(RateLimitError,))
    def call_api(...):
        ...
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import functools
import random
import time
from typing import Any, Callable, Tuple, Type, TypeVar

import structlog

log = structlog.get_logger(__name__)
F = TypeVar("F", bound=Callable[..., Any])


def with_retry(
    max_attempts: int = 3,
    base_delay_s: float = 2.0,
    max_delay_s: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    jitter: bool = True,
) -> Callable[[F], F]:
    """
    Decorator: retry a function on specified exceptions with exponential back-off.

    Args:
        max_attempts:  Total call attempts (1 = no retry).
        base_delay_s:  Initial sleep before the first retry.
        max_delay_s:   Cap on sleep duration.
        exceptions:    Tuple of exception types to catch and retry on.
        jitter:        Add ±25% random jitter to avoid thundering-herd.
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        log.error(
                            "retry.exhausted",
                            fn=fn.__qualname__,
                            attempts=attempt,
                            error=str(exc),
                        )
                        raise
                    delay = min(base_delay_s * (2 ** (attempt - 1)), max_delay_s)
                    if jitter:
                        delay *= 0.75 + random.random() * 0.50
                    log.warning(
                        "retry.attempt",
                        fn=fn.__qualname__,
                        attempt=attempt,
                        of=max_attempts,
                        next_in_s=round(delay, 1),
                        error=str(exc),
                    )
                    time.sleep(delay)
            raise last_exc  # type: ignore[misc]
        return wrapper  # type: ignore[return-value]
    return decorator


def retry_call(
    fn: Callable[..., Any],
    args: tuple = (),
    kwargs: dict | None = None,
    max_attempts: int = 3,
    base_delay_s: float = 2.0,
    max_delay_s: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Any:
    """One-shot retry wrapper for callables you can't decorate directly."""
    decorated = with_retry(
        max_attempts=max_attempts,
        base_delay_s=base_delay_s,
        max_delay_s=max_delay_s,
        exceptions=exceptions,
    )(fn)
    return decorated(*args, **(kwargs or {}))
