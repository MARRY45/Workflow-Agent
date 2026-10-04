"""Provider-neutral chat primitives.

The rest of the code base speaks in these types; conversion to the OpenAI-style wire
format that LiteLLM expects happens in exactly one place (:meth:`Message.to_openai`).
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    """A tool invocation requested by the model.

    ``arguments`` is kept as the raw JSON string the model produced: models do emit invalid
    JSON from time to time, and that must surface as a recoverable tool error rather than
    a crash while parsing the response.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: str = "{}"

    def parsed_arguments(self) -> dict[str, Any]:
        """Decode ``arguments``. Raises ``ValueError`` if it is not a JSON object."""
        raw = self.arguments.strip() or "{}"
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"tool arguments must be a JSON object, got {type(value).__name__}")
        return value


class Message(BaseModel):
    """One entry of a chat transcript."""

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls, content: str | None = None, tool_calls: tuple[ToolCall, ...] = ()
    ) -> Message:
        return cls(role="assistant", content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, *, tool_call_id: str, name: str, content: str) -> Message:
        return cls(role="tool", tool_call_id=tool_call_id, name=name, content=content)

    def to_openai(self) -> dict[str, Any]:
        """Serialize to the OpenAI chat-completions message format (used by LiteLLM)."""
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.name is not None and self.role == "tool":
            data["name"] = self.name
        return data


class Usage(BaseModel):
    """Token and cost accounting for one or more LLM calls.

    ``cost_usd`` is best-effort: it is 0.0 when LiteLLM has no pricing for the model.
    """

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    llm_calls: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            llm_calls=self.llm_calls + other.llm_calls,
        )
