"""Deterministic :class:`LLMClient` double for tests and offline demos.

Two modes:

* **script** - an iterable of replies consumed in order (simple, sequential agents);
* **responder** - a function deciding the reply from the request, which is required when
  several agents call the client concurrently and ordering is not deterministic.
"""

from __future__ import annotations

import inspect
import itertools
import json
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

from workflow_agent.errors import LLMError
from workflow_agent.llm.base import LLMResponse, ToolChoice
from workflow_agent.messages import Message, ToolCall, Usage

Reply: TypeAlias = LLMResponse | Message | str
Responder: TypeAlias = Callable[["RecordedCall"], Reply | Awaitable[Reply]]

_DEFAULT_USAGE = Usage(prompt_tokens=100, completion_tokens=20, llm_calls=1)
_call_ids = itertools.count(1)


@dataclass(frozen=True, slots=True)
class RecordedCall:
    """What the client was asked; handed to responders and kept for assertions."""

    messages: tuple[Message, ...]
    tools: tuple[Mapping[str, Any], ...]
    tool_choice: ToolChoice | None

    @property
    def tool_names(self) -> list[str]:
        return [tool["function"]["name"] for tool in self.tools]

    @property
    def system_prompt(self) -> str:
        first = self.messages[0] if self.messages else None
        return first.content or "" if first is not None and first.role == "system" else ""

    @property
    def first_user_message(self) -> str:
        return next((m.content or "" for m in self.messages if m.role == "user"), "")

    @property
    def last_message(self) -> Message:
        return self.messages[-1]

    @property
    def turn(self) -> int:
        """0-based index of this call within its conversation (= assistant messages so far)."""
        return sum(1 for m in self.messages if m.role == "assistant")


def tool_call(
    name: str, arguments: Mapping[str, Any] | None = None, *, call_id: str = ""
) -> ToolCall:
    """Build a :class:`ToolCall` with a unique id."""
    return ToolCall(
        id=call_id or f"call_{next(_call_ids)}",
        name=name,
        arguments=json.dumps(dict(arguments or {})),
    )


def calls(*tool_calls: ToolCall, content: str | None = None) -> Message:
    """An assistant message requesting one or more tool calls."""
    return Message.assistant(content, tuple(tool_calls))


class ScriptedLLMClient:
    """Replays canned replies; records every request in :attr:`calls`."""

    def __init__(
        self,
        script: Iterable[Reply] | Responder,
        *,
        usage_per_call: Usage = _DEFAULT_USAGE,
    ) -> None:
        self._responder: Responder | None = script if callable(script) else None
        self._script: Iterator[Reply] | None = None if callable(script) else iter(script)
        self._usage = usage_per_call
        self.calls: list[RecordedCall] = []

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> LLMResponse:
        request = RecordedCall(tuple(messages), tuple(tools or ()), tool_choice)
        self.calls.append(request)

        if self._responder is not None:
            reply = self._responder(request)
            if inspect.isawaitable(reply):
                reply = await reply
        else:
            assert self._script is not None
            try:
                reply = next(self._script)
            except StopIteration:
                raise LLMError(
                    f"ScriptedLLMClient script exhausted after {len(self.calls) - 1} call(s)"
                ) from None
        return self._to_response(reply)

    def _to_response(self, reply: Reply) -> LLMResponse:
        if isinstance(reply, LLMResponse):
            return reply
        message = Message.assistant(reply) if isinstance(reply, str) else reply
        if message.role != "assistant":
            raise ValueError(f"scripted replies must be assistant messages, got {message.role!r}")
        finish = "tool_calls" if message.tool_calls else "stop"
        return LLMResponse(
            message=message, usage=self._usage, finish_reason=finish, model="scripted"
        )
