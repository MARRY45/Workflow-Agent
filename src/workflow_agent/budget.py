"""Run-wide token/cost accounting and budget enforcement."""

from __future__ import annotations

from workflow_agent.errors import BudgetExceededError
from workflow_agent.messages import Usage


class UsageTracker:
    """Aggregates usage over every LLM call of a run and enforces optional budgets.

    One tracker is shared by the planner and all step agents of a workflow. Budgets are
    checked *before* each LLM call, so a run can overshoot by at most one call.
    """

    def __init__(
        self,
        *,
        max_total_tokens: int | None = None,
        max_cost_usd: float | None = None,
    ) -> None:
        if max_total_tokens is not None and max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be positive")
        if max_cost_usd is not None and max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be positive")
        self.max_total_tokens = max_total_tokens
        self.max_cost_usd = max_cost_usd
        self._usage = Usage()

    @property
    def usage(self) -> Usage:
        return self._usage

    def record(self, usage: Usage) -> None:
        self._usage = self._usage + usage

    def exceeded(self) -> str | None:
        """A human-readable reason if a budget is exhausted, else ``None``."""
        if self.max_total_tokens is not None and self._usage.total_tokens >= self.max_total_tokens:
            return f"token budget exhausted ({self._usage.total_tokens} >= {self.max_total_tokens})"
        if self.max_cost_usd is not None and self._usage.cost_usd >= self.max_cost_usd:
            return f"cost budget exhausted (${self._usage.cost_usd:.4f} >= ${self.max_cost_usd})"
        return None

    def check(self) -> None:
        reason = self.exceeded()
        if reason is not None:
            raise BudgetExceededError(reason)
