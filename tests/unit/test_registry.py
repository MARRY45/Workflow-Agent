from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from workflow_agent.errors import ToolDefinitionError, ToolError
from workflow_agent.llm import tool_call
from workflow_agent.messages import ToolCall
from workflow_agent.tools import ToolRegistry, ToolResult, tool


class Report(BaseModel):
    ok: bool


@tool(tags={"math"})
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def fail_expected(reason: str) -> str:
    """Raise a ToolError."""
    raise ToolError(f"cannot do it: {reason}")


@tool
def fail_unexpected() -> str:
    """Raise a bug."""
    raise RuntimeError("boom")


@tool(timeout_s=0.05)
async def slow() -> str:
    """Sleep too long."""
    await asyncio.sleep(5)
    return "never"


@tool
async def inner_timeout() -> str:
    """Raise TimeoutError from inside the tool."""
    raise TimeoutError("upstream API timed out")


@tool
def structured() -> Report:
    """Return a model."""
    return Report(ok=True)


@tool
def nothing() -> None:
    """Return None."""


@tool
def big() -> str:
    """Return a large output."""
    return "a" * 1000 + "THE-END"


@tool
def custom() -> ToolResult:
    """Return a ToolResult."""
    return ToolResult(content="x" * 500, metadata={"k": 1})


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry(
        [
            add,
            fail_expected,
            fail_unexpected,
            slow,
            inner_timeout,
            structured,
            nothing,
            big,
            custom,
        ],
        max_output_chars=200,
    )


async def test_successful_call(registry: ToolRegistry) -> None:
    result = await registry.execute(tool_call("add", {"a": 2, "b": 3}))
    assert result == ToolResult(content="5")


async def test_unknown_tool_lists_alternatives(registry: ToolRegistry) -> None:
    result = await registry.execute(tool_call("nope"))
    assert result.is_error
    assert "unknown tool 'nope'" in result.content
    assert "add" in result.content


async def test_malformed_json_is_reported(registry: ToolRegistry) -> None:
    result = await registry.execute(ToolCall(id="1", name="add", arguments='{"a": 1,'))
    assert result.is_error
    assert "not a valid JSON object" in result.content


async def test_validation_errors_are_readable(registry: ToolRegistry) -> None:
    result = await registry.execute(tool_call("add", {"a": "x"}))
    assert result.is_error
    assert "- a: Input should be a valid integer" in result.content
    assert "- b: Field required" in result.content


async def test_tool_error_message_is_passed_through(registry: ToolRegistry) -> None:
    result = await registry.execute(tool_call("fail_expected", {"reason": "disk full"}))
    assert result == ToolResult.error("cannot do it: disk full")


async def test_unexpected_exception_is_contained(registry: ToolRegistry) -> None:
    result = await registry.execute(tool_call("fail_unexpected"))
    assert result.is_error
    assert result.content == "ERROR: RuntimeError: boom"


async def test_timeout(registry: ToolRegistry) -> None:
    result = await registry.execute(tool_call("slow"))
    assert result.is_error
    assert "timed out after 0.05s" in result.content


async def test_timeout_error_raised_by_tool_is_not_misreported(registry: ToolRegistry) -> None:
    result = await registry.execute(tool_call("inner_timeout"))
    assert result.content == "ERROR: TimeoutError: upstream API timed out"


async def test_output_serialization(registry: ToolRegistry) -> None:
    assert (await registry.execute(tool_call("structured"))).content == '{\n  "ok": true\n}'
    assert (await registry.execute(tool_call("nothing"))).content == "(no output)"


async def test_output_is_truncated_keeping_the_tail(registry: ToolRegistry) -> None:
    result = await registry.execute(tool_call("big"))
    assert len(result.content) <= 200
    assert result.content.endswith("THE-END")
    assert "characters truncated" in result.content

    custom_result = await registry.execute(tool_call("custom"))
    assert len(custom_result.content) <= 200
    assert custom_result.metadata == {"k": 1}


async def test_cancellation_propagates(registry: ToolRegistry) -> None:
    @tool
    async def hang() -> str:
        """Hang."""
        await asyncio.sleep(10)
        return ""

    reg = ToolRegistry([hang])
    task = asyncio.create_task(reg.execute(tool_call("hang")))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_collection_api(registry: ToolRegistry) -> None:
    assert "add" in registry
    assert registry.get("add") is add
    assert registry.get("missing") is None
    assert len(registry) == 9
    assert registry.names[0] == "add"
    assert [s["function"]["name"] for s in registry.schemas()] == registry.names

    math_only = registry.filter(lambda t: "math" in t.tags)
    assert math_only.names == ["add"]
    assert math_only.max_output_chars == 200
    assert math_only.with_tools(nothing).names == ["add", "nothing"]
    assert len(registry) == 9  # derived registries never mutate the original


def test_duplicate_names_are_rejected() -> None:
    with pytest.raises(ToolDefinitionError, match="duplicate"):
        ToolRegistry([add, add])


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [({"default_timeout_s": 0}, "timeout"), ({"max_output_chars": 10}, "max_output")],
)
def test_invalid_settings(kwargs: dict[str, float], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        ToolRegistry(**kwargs)  # type: ignore[arg-type]
