"""Retry helper for transient OpenAI errors (rate limits, timeouts)."""

from __future__ import annotations

import time
from typing import Callable, TypeVar

import openai

T = TypeVar("T")

# Errors worth retrying: rate limits, transient connection/timeouts, 5xx.
_TRANSIENT = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def _retry_delay(exc: Exception, attempt: int, base: float, cap: float) -> float:
    """Seconds to wait before the next attempt, honoring Retry-After if present."""
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                return min(float(retry_after) + 0.5, cap)
        except (TypeError, ValueError, AttributeError):
            pass
    return min(base * (2 ** (attempt - 1)), cap)


def with_retries(
    func: Callable[[], T],
    *,
    max_attempts: int = 6,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
) -> T:
    """Call ``func`` and retry on transient OpenAI errors with backoff.

    Blocks (time.sleep) between attempts. Re-raises the last error once
    ``max_attempts`` is exhausted.
    """
    attempt = 0
    while True:
        try:
            return func()
        except _TRANSIENT as exc:
            attempt += 1
            if attempt >= max_attempts:
                raise
            time.sleep(_retry_delay(exc, attempt, base_delay, max_delay))
