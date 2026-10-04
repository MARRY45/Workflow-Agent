"""Exception hierarchy.

Every exception raised deliberately by the library derives from :class:`WorkflowAgentError`
so callers can catch library failures with a single ``except`` clause.
"""

from __future__ import annotations


class WorkflowAgentError(Exception):
    """Base class for all library errors."""


class ConfigurationError(WorkflowAgentError):
    """Invalid or inconsistent configuration."""


class LLMError(WorkflowAgentError):
    """An LLM request failed (after retries, if the failure was transient)."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ToolDefinitionError(WorkflowAgentError):
    """A tool was declared incorrectly (bad signature, duplicate name, ...)."""


class ToolError(WorkflowAgentError):
    """Raised *inside* a tool to report an expected, user-facing failure.

    The registry converts it into an error result that is shown to the model, so the
    message should be actionable ("file not found: x.py"), not a stack trace.
    """


class WorkspaceError(ToolError):
    """A path resolved outside of the agent workspace or is otherwise unusable."""


class BudgetExceededError(WorkflowAgentError):
    """A token or cost budget was exhausted; the run must stop."""


class PlanningError(WorkflowAgentError):
    """The planner could not produce a valid plan."""
