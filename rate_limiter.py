"""
rate_limiter.py — Sliding-window token-per-minute rate limiter for Anthropic API.

Enforces a global TPM budget across all concurrent calls (orchestrator + subagents).
Uses a sliding 60-second window with actual token counts from API responses.
When the budget is exhausted, callers wait (not error) until capacity frees up.
"""

import asyncio
import time
import logging
from collections import deque

log = logging.getLogger(__name__)


class TokenBudget:
    """Sliding 60-second window tracking total tokens used."""

    def __init__(self, max_tokens_per_minute: int = 30_000):
        self.max_tpm = max_tokens_per_minute
        self._records: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()

    def _expire_old(self) -> None:
        cutoff = time.monotonic() - 60.0
        while self._records and self._records[0][0] < cutoff:
            self._records.popleft()

    def _current_usage(self) -> int:
        self._expire_old()
        return sum(count for _, count in self._records)

    async def wait_for_capacity(self) -> None:
        """Block until there is room in the budget."""
        while True:
            async with self._lock:
                usage = self._current_usage()
                if usage < self.max_tpm:
                    return
                oldest_ts = self._records[0][0] if self._records else time.monotonic()
                wait_seconds = (oldest_ts + 60.0) - time.monotonic() + 0.1

            log.info(
                "Rate limiter: %d/%d tokens used, waiting %.1fs",
                usage, self.max_tpm, wait_seconds,
            )
            await asyncio.sleep(max(wait_seconds, 0.5))

    async def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        total = input_tokens + output_tokens
        async with self._lock:
            self._records.append((time.monotonic(), total))


class RateLimitedClient:
    """
    Drop-in proxy for AsyncAnthropic that enforces token-per-minute limits.

    All existing code keeps calling client.messages.create(...) unchanged.
    The proxy intercepts it, waits for budget, calls the real API,
    records usage, and returns the response.
    """

    def __init__(self, real_client, max_tpm: int = 30_000):
        self._real_client = real_client
        self._budget = TokenBudget(max_tpm)
        self.messages = self._MessagesProxy(real_client.messages, self._budget)

    def __getattr__(self, name):
        return getattr(self._real_client, name)

    class _MessagesProxy:
        def __init__(self, real_messages, budget: "TokenBudget"):
            self._real = real_messages
            self._budget = budget

        async def create(self, **kwargs):
            await self._budget.wait_for_capacity()
            response = await self._real.create(**kwargs)
            if hasattr(response, "usage") and response.usage:
                await self._budget.record_usage(
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                )
            return response

        def __getattr__(self, name):
            return getattr(self._real, name)
