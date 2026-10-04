from __future__ import annotations

import io
import json
from pathlib import Path

from workflow_agent.events import (
    AgentFinished,
    AgentStarted,
    LLMCallCompleted,
    PlanCreated,
    StepFinished,
    StepStarted,
    ToolCallFinished,
    ToolCallStarted,
    WorkflowFinished,
    WorkflowStarted,
)
from workflow_agent.messages import Usage
from workflow_agent.observability import ConsoleReporter, JsonlTraceWriter

USAGE = Usage(prompt_tokens=1000, completion_tokens=500, cost_usd=0.0042, llm_calls=3)


def render(event, verbose: bool = False) -> str:  # type: ignore[no-untyped-def]
    stream = io.StringIO()
    ConsoleReporter(stream, verbose=verbose)(event)
    return stream.getvalue()


def test_workflow_events() -> None:
    assert (
        render(WorkflowStarted(source="workflow", objective="Do\nthings"))
        == "▶ objective: Do things\n"
    )
    plan = PlanCreated(
        source="workflow",
        rationale="why",
        steps=[
            {"id": "a", "kind": "research", "title": "Find", "depends_on": []},
            {"id": "b", "kind": "verify", "title": "Check", "depends_on": ["a"]},
        ],
    )
    assert (
        render(plan)
        == "▶ plan (2 steps): why\n   1. [research] a: Find\n   2. [verify] b: Check (after a)\n"
    )
    assert (
        render(StepStarted(source="workflow", step_id="a", kind="research", title="Find"))
        == "→ a [research] started\n"
    )
    finished = StepFinished(
        source="workflow", step_id="a", status="success", summary="found it", duration_s=1.25
    )
    assert render(finished) == "✓ a: success (1.2s) - found it\n"
    failed = StepFinished(
        source="workflow", step_id="b", status="failed", summary="", duration_s=0, error="boom"
    )
    assert render(failed) == "✗ b: failed (0.0s) - boom\n"
    done = WorkflowFinished(source="workflow", status="partial", duration_s=3.0, usage=USAGE)
    assert render(done) == "■ workflow partial in 3.0s · 3 LLM calls · 1,500 tokens · $0.0042\n"


def test_tool_and_llm_events_respect_verbosity() -> None:
    started = ToolCallStarted(
        source="step:a", turn=1, call_id="c", tool="run_python", arguments='{"code": "1"}'
    )
    assert render(started) == '   · step:a → run_python({"code": "1"})\n'

    ok = ToolCallFinished(
        source="step:a",
        turn=1,
        call_id="c",
        tool="run_python",
        is_error=False,
        duration_s=0.1,
        output="exit code 0",
    )
    assert render(ok) == ""
    assert render(ok, verbose=True) == "   ✓ step:a ← run_python: exit code 0\n"
    error = ok.model_copy(update={"is_error": True, "output": "ERROR: bad"})
    assert render(error) == "   ✗ step:a ← run_python: ERROR: bad\n"

    llm = LLMCallCompleted(
        source="planner",
        turn=2,
        finish_reason="stop",
        tool_calls=[],
        content="thinking",
        usage=USAGE,
    )
    assert render(llm) == ""
    assert render(llm, verbose=True) == "   … planner (turn 2): thinking\n"

    agent_started = AgentStarted(source="agent", task="task")
    assert render(agent_started) == ""
    assert render(agent_started, verbose=True) == "▶ agent: task\n"
    step_agent_done = AgentFinished(
        source="step:a", stop_reason="terminal_tool", turns=2, output="", usage=USAGE
    )
    assert render(step_agent_done) == ""
    agent_done = step_agent_done.model_copy(update={"source": "agent"})
    assert render(agent_done) == "■ agent finished (terminal_tool) after 2 turn(s)\n"


def test_jsonl_trace_writer(tmp_path: Path) -> None:
    path = tmp_path / "traces" / "run.jsonl"
    with JsonlTraceWriter(path) as writer:
        writer(WorkflowStarted(source="workflow", objective="x"))
        writer(WorkflowFinished(source="workflow", status="success", duration_s=1, usage=USAGE))
    writer(WorkflowStarted(source="workflow", objective="after close"))  # ignored, no crash
    writer.close()  # idempotent

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["type"] for r in records] == ["workflow_started", "workflow_finished"]
    assert records[1]["usage"]["cost_usd"] == 0.0042
    assert "timestamp" in records[0]
