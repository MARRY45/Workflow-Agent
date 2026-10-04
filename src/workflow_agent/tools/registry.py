"""Tool registry: lookup, schema export and *safe* execution of model-issued tool calls.

``execute`` never raises for problems caused by the model (unknown tool, malformed JSON,
invalid arguments, tool exceptions, timeouts). Those become error results the model can
read and recover from - a core property of a robust agent loop. Only cancellation
propagates.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from pydantic import BaseModel, ValidationError

from workflow_agent.errors import ToolDefinitionError, ToolError
from workflow_agent.messages import ToolCall
from workflow_agent.textutil import truncate_middle
from workflow_agent.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """An ordered, name-unique collection of tools."""

    def __init__(
        self,
        tools: Iterable[Tool] = (),
        *,
        default_timeout_s: float = 60.0,
        max_output_chars: int = 12_000,
    ) -> None:
        if default_timeout_s <= 0:
            raise ValueError("default_timeout_s must be positive")
        if max_output_chars < 100:
            raise ValueError("max_output_chars must be >= 100")
        self.default_timeout_s = default_timeout_s
        self.max_output_chars = max_output_chars
        self._tools: dict[str, Tool] = {}
        for item in tools:
            self.register(item)

    # ------------------------------------------------------------------ collection API

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ToolDefinitionError(f"duplicate tool name {tool.name!r}")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def filter(self, predicate: Callable[[Tool], bool]) -> ToolRegistry:
        """A new registry (same settings) with the tools matching ``predicate``."""
        return self._derive(t for t in self._tools.values() if predicate(t))

    def with_tools(self, *tools: Tool) -> ToolRegistry:
        """A new registry with ``tools`` appended."""
        return self._derive([*self._tools.values(), *tools])

    def _derive(self, tools: Iterable[Tool]) -> ToolRegistry:
        return ToolRegistry(
            tools,
            default_timeout_s=self.default_timeout_s,
            max_output_chars=self.max_output_chars,
        )

    # ------------------------------------------------------------------ execution

    async def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            available = ", ".join(self._tools) or "(none)"
            return ToolResult.error(f"unknown tool {call.name!r}. Available tools: {available}")

        try:
            arguments = call.parsed_arguments()
        except ValueError as exc:
            return ToolResult.error(
                f"arguments for {call.name!r} are not a valid JSON object ({exc}). "
                "Retry with a JSON object that matches the tool schema."
            )

        try:
            args = tool.validate(arguments)
        except ValidationError as exc:
            return ToolResult.error(
                f"invalid arguments for {call.name!r}:\n{format_validation_error(exc)}"
            )

        timeout_s = tool.timeout_s or self.default_timeout_s
        try:
            async with asyncio.timeout(timeout_s) as scope:
                output = await tool.run(args)
        except TimeoutError as exc:
            if scope.expired():
                return ToolResult.error(f"{call.name!r} timed out after {timeout_s:g}s")
            return ToolResult.error(f"TimeoutError: {exc}")
        except ToolError as exc:
            return ToolResult.error(str(exc))
        except Exception as exc:
            logger.exception("Tool %r raised an unexpected exception", call.name)
            return ToolResult.error(f"{type(exc).__name__}: {exc}")

        return self._to_result(output)

    def _to_result(self, output: Any) -> ToolResult:
        if isinstance(output, ToolResult):
            return ToolResult(
                content=truncate_middle(output.content, self.max_output_chars),
                is_error=output.is_error,
                metadata=output.metadata,
            )
        if output is None:
            text = "(no output)"
        elif isinstance(output, str):
            text = output
        elif isinstance(output, BaseModel):
            text = output.model_dump_json(indent=2)
        else:
            text = json.dumps(output, indent=2, ensure_ascii=False, default=str)
        return ToolResult(content=truncate_middle(text, self.max_output_chars))


def format_validation_error(exc: ValidationError) -> str:
    """Compact, model-readable rendering of a pydantic validation error."""
    lines = []
    for error in exc.errors(include_url=False):
        location = ".".join(str(part) for part in error["loc"]) or "<arguments>"
        lines.append(f"- {location}: {error['msg']}")
    return "\n".join(lines)
