"""Structured run events.

Agents, the planner and the workflow engine report progress as typed events on an
:class:`EventBus`. Sinks (console reporter, JSONL trace writer, metrics, a UI ...) subscribe
to the bus; a failing sink is logged and never breaks the run.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from workflow_agent.messages import Usage

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(UTC)


class Event(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    source: str = Field(description="Emitter: an agent name, 'planner' or 'workflow'.")
    timestamp: datetime = Field(default_factory=_now)


# ------------------------------------------------------------------------------ agent


class AgentStarted(Event):
    type: Literal["agent_started"] = "agent_started"
    task: str


class LLMCallCompleted(Event):
    type: Literal["llm_call"] = "llm_call"
    turn: int
    finish_reason: str | None
    tool_calls: list[str]
    content: str | None
    usage: Usage


class ToolCallStarted(Event):
    type: Literal["tool_started"] = "tool_started"
    turn: int
    call_id: str
    tool: str
    arguments: str


class ToolCallFinished(Event):
    type: Literal["tool_finished"] = "tool_finished"
    turn: int
    call_id: str
    tool: str
    is_error: bool
    duration_s: float
    output: str


class AgentFinished(Event):
    type: Literal["agent_finished"] = "agent_finished"
    stop_reason: str
    turns: int
    output: str
    usage: Usage


# ---------------------------------------------------------------------------- workflow


class WorkflowStarted(Event):
    type: Literal["workflow_started"] = "workflow_started"
    objective: str


class PlanCreated(Event):
    type: Literal["plan_created"] = "plan_created"
    rationale: str
    steps: list[dict[str, Any]]


class StepStarted(Event):
    type: Literal["step_started"] = "step_started"
    step_id: str
    kind: str
    title: str


class StepFinished(Event):
    type: Literal["step_finished"] = "step_finished"
    step_id: str
    status: str
    summary: str
    duration_s: float
    error: str | None = None


class WorkflowFinished(Event):
    type: Literal["workflow_finished"] = "workflow_finished"
    status: str
    duration_s: float
    usage: Usage


EventSink: TypeAlias = Callable[[Event], Awaitable[None] | None]


class EventBus:
    """Fan-out of events to sinks, in subscription order."""

    def __init__(self, sinks: Iterable[EventSink] = ()) -> None:
        self._sinks: list[EventSink] = list(sinks)

    def subscribe(self, sink: EventSink) -> None:
        self._sinks.append(sink)

    async def emit(self, event: Event) -> None:
        for sink in self._sinks:
            try:
                result = sink(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("Event sink %r failed on %s", sink, event.type)
