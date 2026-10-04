from __future__ import annotations

import asyncio
from typing import Literal

import pytest

from workflow_agent.agent import Agent, AgentConfig
from workflow_agent.budget import UsageTracker
from workflow_agent.errors import BudgetExceededError
from workflow_agent.events import Event, EventBus
from workflow_agent.llm import ScriptedLLMClient, calls, tool_call
from workflow_agent.messages import Message, ToolCall, Usage
from workflow_agent.tools import ToolRegistry, tool


@tool(tags={"math"})
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool(terminal=True)
def finish(status: Literal["success", "failed"], summary: str) -> str:
    """Finish the run."""
    return "recorded"


def make_agent(script, *tools, **kwargs) -> tuple[Agent, ScriptedLLMClient, list[Event]]:  # type: ignore[no-untyped-def]
    llm = ScriptedLLMClient(script)
    seen: list[Event] = []
    agent = Agent(llm, ToolRegistry(tools), events=EventBus([seen.append]), **kwargs)
    return agent, llm, seen


async def test_plain_answer_without_tools() -> None:
    agent, llm, events = make_agent(["The answer is 42."])
    result = await agent.run("question?")
    assert result.output == "The answer is 42."
    assert result.stop_reason == "final_answer"
    assert result.turns == 1
    assert result.usage.llm_calls == 1
    assert llm.calls[0].tools == ()  # no tools => no schemas sent
    assert [e.type for e in events] == ["agent_started", "llm_call", "agent_finished"]


async def test_tool_round_trip_builds_a_valid_transcript() -> None:
    call = tool_call("add", {"a": 2, "b": 3}, call_id="c1")
    agent, llm, events = make_agent([calls(call), "2 + 3 = 5"], add)
    result = await agent.run("add 2 and 3", context=[Message.user("prior context")])

    assert result.output == "2 + 3 = 5"
    assert result.turns == 2
    roles = [m.role for m in result.messages]
    assert roles == ["system", "user", "user", "assistant", "tool", "assistant"]
    tool_message = result.messages[4]
    assert (tool_message.tool_call_id, tool_message.name, tool_message.content) == (
        "c1",
        "add",
        "5",
    )
    assert llm.calls[0].tool_names == ["add"]

    (record,) = result.tool_calls
    assert record.call == call
    assert record.result.content == "5"
    assert record.tags == frozenset({"math"})
    assert [e.type for e in events] == [
        "agent_started", "llm_call", "tool_started", "tool_finished", "llm_call", "agent_finished",
    ]  # fmt: skip


async def test_parallel_tool_calls_run_concurrently_and_keep_order() -> None:
    active = peak = 0

    @tool
    async def wait(seconds: float, label: str) -> str:
        """Sleep then echo."""
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(seconds)
        active -= 1
        return label

    batch = calls(
        tool_call("wait", {"seconds": 0.10, "label": "slow"}),
        tool_call("wait", {"seconds": 0.01, "label": "fast"}),
        tool_call("wait", {"seconds": 0.05, "label": "mid"}),
    )
    agent, _, _ = make_agent([batch, "done"], wait, config=AgentConfig(max_parallel_tools=2))
    result = await agent.run("go")
    assert peak == 2  # bounded by max_parallel_tools
    assert [m.content for m in result.messages if m.role == "tool"] == ["slow", "fast", "mid"]


async def test_model_recovers_from_invalid_arguments() -> None:
    script = [calls(tool_call("add", {"a": "two"})), calls(tool_call("add", {"a": 2, "b": 2})), "4"]
    agent, llm, _ = make_agent(script, add)
    result = await agent.run("2+2")
    assert result.output == "4"
    first, second = result.tool_calls
    assert first.result.is_error
    assert "invalid arguments" in first.result.content
    assert not second.result.is_error
    # The error was shown to the model on the next turn.
    assert llm.calls[1].last_message.content == first.result.content


async def test_terminal_tool_ends_the_run_with_structured_output() -> None:
    script = [
        calls(tool_call("finish", {"status": "maybe", "summary": "x"})),  # invalid enum
        calls(tool_call("finish", {"status": "success", "summary": "All good."})),
    ]
    agent, _, _ = make_agent(script, add, finish)
    result = await agent.run("do it")
    assert result.stop_reason == "terminal_tool"
    assert result.terminal_output == {"status": "success", "summary": "All good."}
    assert result.output == "All good."
    assert result.turns == 2
    assert result.messages[-1].role == "tool"  # every tool call gets a result message


async def test_terminal_tool_must_be_called_alone() -> None:
    script = [
        calls(
            tool_call("add", {"a": 1, "b": 1}),
            tool_call("finish", {"status": "success", "summary": "early"}),
        ),
        calls(tool_call("finish", {"status": "success", "summary": "after review"})),
    ]
    agent, _, _ = make_agent(script, add, finish)
    result = await agent.run("do it")
    assert result.output == "after review"
    tool_messages = [m.content or "" for m in result.messages if m.role == "tool"]
    assert tool_messages[0] == "2"
    assert "must be called on its own" in tool_messages[1]


async def test_duplicate_terminal_calls_are_ignored() -> None:
    script = [
        calls(
            tool_call("finish", {"status": "success", "summary": "first"}),
            tool_call("finish", {"status": "failed", "summary": "second"}),
        )
    ]
    agent, _, _ = make_agent(script, finish)
    result = await agent.run("do it")
    assert result.terminal_output == {"status": "success", "summary": "first"}
    assert "already completed" in (result.messages[-1].content or "")


async def test_terminal_output_without_summary_is_serialized() -> None:
    @tool(terminal=True)
    def submit(value: int) -> str:
        """Submit a number."""
        return "ok"

    agent, _, _ = make_agent([calls(tool_call("submit", {"value": 7}))], submit)
    result = await agent.run("go")
    assert result.output == '{"value": 7}'


async def test_max_turns_stops_a_looping_model() -> None:
    script = [calls(tool_call("add", {"a": 1, "b": 1}), content=f"thinking {i}") for i in range(5)]
    agent, llm, _ = make_agent(script, add, config=AgentConfig(max_turns=3))
    result = await agent.run("loop")
    assert result.stop_reason == "max_turns"
    assert result.turns == 3
    assert result.output == "thinking 2"
    assert len(llm.calls) == 3


async def test_budget_is_enforced_before_each_call() -> None:
    tracker = UsageTracker(max_total_tokens=100)  # one scripted call costs 120 tokens
    agent, llm, _ = make_agent(
        [calls(tool_call("add", {"a": 1, "b": 1})), "never"], add, tracker=tracker
    )
    with pytest.raises(BudgetExceededError, match="token budget"):
        await agent.run("go")
    assert len(llm.calls) == 1
    assert tracker.usage.total_tokens == 120


async def test_failing_event_sinks_do_not_break_the_run(caplog: pytest.LogCaptureFixture) -> None:
    received: list[str] = []

    def broken(event: Event) -> None:
        raise RuntimeError("sink down")

    async def async_sink(event: Event) -> None:
        received.append(event.type)

    agent = Agent(ScriptedLLMClient(["ok"]), events=EventBus([broken, async_sink]))
    result = await agent.run("go")
    assert result.output == "ok"
    assert received == ["agent_started", "llm_call", "agent_finished"]
    assert "Event sink" in caplog.text


async def test_unknown_tool_call_is_reported_to_the_model() -> None:
    agent, _, _ = make_agent([calls(ToolCall(id="x", name="ghost")), "sorry"], add)
    result = await agent.run("go")
    assert result.tool_calls[0].result.is_error
    assert result.tool_calls[0].tags == frozenset()


@pytest.mark.parametrize("kwargs", [{"max_turns": 0}, {"max_parallel_tools": 0}])
def test_config_validation(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        AgentConfig(**kwargs)  # type: ignore[arg-type]


def test_usage_tracker() -> None:
    tracker = UsageTracker(max_cost_usd=0.01)
    tracker.record(Usage(prompt_tokens=1, cost_usd=0.004, llm_calls=1))
    assert tracker.exceeded() is None
    tracker.record(Usage(cost_usd=0.006, llm_calls=1))
    assert "cost budget" in (tracker.exceeded() or "")
    with pytest.raises(ValueError, match="positive"):
        UsageTracker(max_total_tokens=0)
    with pytest.raises(ValueError, match="positive"):
        UsageTracker(max_cost_usd=-1)
