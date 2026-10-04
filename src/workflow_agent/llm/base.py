"""The LLM port: everything above this layer depends on :class:`LLMClient` only."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel, ConfigDict

from workflow_agent.messages import Message, Usage


@dataclass(frozen=True, slots=True)
class ForceTool:
    """``tool_choice`` value that forces the model to call one specific tool."""

    name: str


ToolChoice: TypeAlias = Literal["auto", "none", "required"] | ForceTool


class LLMResponse(BaseModel):
    """A single assistant turn plus accounting data."""

    model_config = ConfigDict(frozen=True)

    message: Message
    usage: Usage = Usage()
    finish_reason: str | None = None
    model: str | None = None


@runtime_checkable
class LLMClient(Protocol):
    """Minimal async chat-completion interface with tool calling."""

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> LLMResponse:
        """Return the next assistant message for ``messages``.

        ``tools`` uses the OpenAI function-tool schema format. Implementations raise
        :class:`workflow_agent.errors.LLMError` on unrecoverable failures.
        """
        ...


def tool_choice_to_openai(choice: ToolChoice) -> str | dict[str, Any]:
    if isinstance(choice, ForceTool):
        return {"type": "function", "function": {"name": choice.name}}
    return choice
