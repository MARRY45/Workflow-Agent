"""The tool-calling agent loop.

One *turn* = one LLM call. Each turn the model either

* answers in plain text  -> the run ends with ``stop_reason="final_answer"``;
* calls a **terminal** tool (e.g. ``complete_step``) with valid arguments
  -> the run ends with ``stop_reason="terminal_tool"`` and the validated arguments as
  structured output;
* calls regular tools -> they execute concurrently, their results are appended to the
  transcript in call order, and the next turn starts.

The loop is bounded by ``max_turns`` and by the shared :class:`UsageTracker` budget.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from workflow_agent.budget import UsageTracker
from workflow_agent.events import (
    AgentFinished,
    AgentStarted,
    EventBus,
    LLMCallCompleted,
    ToolCallFinished,
    ToolCallStarted,
)
from workflow_agent.llm.base import LLMClient
from workflow_agent.messages import Message, ToolCall, Usage
from workflow_agent.tools.base import ToolResult
from workflow_agent.tools.registry import ToolRegistry

DEFAULT_SYSTEM_PROMPT = """\
You are Workflow Agent, a meticulous technical research and software engineering assistant.

- Work step by step and use the available tools to gather facts and to check your work.
- Never invent tool results, file contents, package versions or test outcomes. If you need \
a fact, look it up; if you claim code works, run it.
- Read tool errors carefully and fix the root cause before retrying.
- When you have a complete answer, reply with it directly (without calling a tool). Cite \
sources (URLs, files) and state what you verified and how."""

StopReason = Literal["final_answer", "terminal_tool", "max_turns"]


@dataclass(frozen=True, slots=True)
class AgentConfig:
    name: str = "agent"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_turns: int = 12
    max_parallel_tools: int = 4

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if self.max_parallel_tools < 1:
            raise ValueError("max_parallel_tools must be >= 1")


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    """One executed tool call, kept for auditing and evidence checks."""

    turn: int
    call: ToolCall
    result: ToolResult
    duration_s: float
    tags: frozenset[str]


@dataclass(frozen=True, slots=True)
class AgentResult:
    output: str
    stop_reason: StopReason
    turns: int
    usage: Usage
    messages: tuple[Message, ...]
    tool_calls: tuple[ToolCallRecord, ...]
    terminal_output: dict[str, Any] | None = None


class Agent:
    """Runs one task to completion with an LLM and a tool registry."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry | None = None,
        *,
        config: AgentConfig | None = None,
        tracker: UsageTracker | None = None,
        events: EventBus | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.config = config or AgentConfig()
        self.tracker = tracker or UsageTracker()
        self.events = events or EventBus()

    async def run(self, task: str, *, context: Sequence[Message] = ()) -> AgentResult:
        """Run ``task``; ``context`` is inserted between the system prompt and the task."""
        name = self.config.name
        messages: list[Message] = [Message.system(self.config.system_prompt), *context]
        messages.append(Message.user(task))
        records: list[ToolCallRecord] = []
        usage = Usage()
        schemas = self.tools.schemas() or None
        await self.events.emit(AgentStarted(source=name, task=task))

        for turn in range(1, self.config.max_turns + 1):
            self.tracker.check()
            response = await self.llm.complete(messages, tools=schemas)
            usage = usage + response.usage
            self.tracker.record(response.usage)
            message = response.message
            messages.append(message)
            await self.events.emit(
                LLMCallCompleted(
                    source=name,
                    turn=turn,
                    finish_reason=response.finish_reason,
                    tool_calls=[call.name for call in message.tool_calls],
                    content=message.content,
                    usage=response.usage,
                )
            )

            if not message.tool_calls:
                return await self._finish(
                    "final_answer", message.content or "", turn, usage, messages, records
                )

            terminal = await self._handle_tool_calls(turn, message.tool_calls, messages, records)
            if terminal is not None:
                summary = terminal.get("summary")
                output = summary if isinstance(summary, str) else json.dumps(terminal)
                return await self._finish(
                    "terminal_tool", output, turn, usage, messages, records, terminal
                )

        last_text = next((m.content for m in reversed(messages) if m.role == "assistant"), None)
        return await self._finish(
            "max_turns", last_text or "", self.config.max_turns, usage, messages, records
        )

    # ------------------------------------------------------------------ internals

    async def _handle_tool_calls(
        self,
        turn: int,
        calls: Sequence[ToolCall],
        messages: list[Message],
        records: list[ToolCallRecord],
    ) -> dict[str, Any] | None:
        """Execute one batch of tool calls; return terminal output if the run should end."""
        is_terminal = [bool((t := self.tools.get(c.name)) and t.terminal) for c in calls]
        regular = [c for c, term in zip(calls, is_terminal, strict=True) if not term]
        results: list[ToolResult | None] = [None] * len(calls)

        semaphore = asyncio.Semaphore(self.config.max_parallel_tools)

        async def run_one(call: ToolCall) -> ToolCallRecord:
            async with semaphore:
                return await self._execute(turn, call)

        regular_records = await asyncio.gather(*(run_one(c) for c in regular))
        records.extend(regular_records)
        by_call = iter(regular_records)

        terminal_output: dict[str, Any] | None = None
        for index, (call, term) in enumerate(zip(calls, is_terminal, strict=True)):
            if not term:
                results[index] = next(by_call).result
            elif regular:
                results[index] = ToolResult.error(
                    f"{call.name!r} must be called on its own, after you have seen the results "
                    "of your other tool calls. Review them, then call it again."
                )
            elif terminal_output is not None:
                results[index] = ToolResult.error("ignored: the run was already completed")
            else:
                record = await self._execute(turn, call)
                records.append(record)
                results[index] = record.result
                if not record.result.is_error:
                    tool = self.tools.get(call.name)
                    assert tool is not None
                    validated = tool.validate(call.parsed_arguments())
                    terminal_output = validated.model_dump(mode="json")

        for call, result in zip(calls, results, strict=True):
            assert result is not None
            messages.append(
                Message.tool(tool_call_id=call.id, name=call.name, content=result.content)
            )
        return terminal_output

    async def _execute(self, turn: int, call: ToolCall) -> ToolCallRecord:
        name = self.config.name
        await self.events.emit(
            ToolCallStarted(
                source=name, turn=turn, call_id=call.id, tool=call.name, arguments=call.arguments
            )
        )
        started = time.monotonic()
        result = await self.tools.execute(call)
        duration = time.monotonic() - started
        tool = self.tools.get(call.name)
        await self.events.emit(
            ToolCallFinished(
                source=name,
                turn=turn,
                call_id=call.id,
                tool=call.name,
                is_error=result.is_error,
                duration_s=duration,
                output=result.content,
            )
        )
        return ToolCallRecord(
            turn=turn,
            call=call,
            result=result,
            duration_s=duration,
            tags=tool.tags if tool else frozenset(),
        )

    async def _finish(
        self,
        stop_reason: StopReason,
        output: str,
        turns: int,
        usage: Usage,
        messages: list[Message],
        records: list[ToolCallRecord],
        terminal_output: dict[str, Any] | None = None,
    ) -> AgentResult:
        await self.events.emit(
            AgentFinished(
                source=self.config.name,
                stop_reason=stop_reason,
                turns=turns,
                output=output,
                usage=usage,
            )
        )
        return AgentResult(
            output=output,
            stop_reason=stop_reason,
            turns=turns,
            usage=usage,
            messages=tuple(messages),
            tool_calls=tuple(records),
            terminal_output=terminal_output,
        )
