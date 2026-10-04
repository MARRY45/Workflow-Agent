"""Agent loop + real LiteLLM response parsing + real tools, with LiteLLM's mock transport."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import litellm

from workflow_agent.agent import Agent
from workflow_agent.llm import LiteLLMClient
from workflow_agent.tools import ToolRegistry, Workspace, execution_tools, filesystem_tools


async def test_agent_writes_and_runs_code_through_litellm(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    registry = ToolRegistry([*filesystem_tools(workspace), *execution_tools(workspace)])
    turns: list[dict[str, Any]] = []

    script = [
        _mock_call(
            "w1", "write_file", path="square.py", content="def square(x):\n    return x * x\n"
        ),
        _mock_call("r1", "run_python", code="from square import square\nprint(square(12))"),
        {"mock_response": "square(12) printed 144, so the function works."},
    ]

    async def scripted_litellm(**params: Any) -> Any:
        turns.append(params)
        return await litellm.acompletion(**params, **script[len(turns) - 1])

    agent = Agent(LiteLLMClient("gpt-4o-mini", completion_fn=scripted_litellm), registry)
    result = await agent.run("Write square() and prove it works.")

    assert result.stop_reason == "final_answer"
    assert "144" in result.output
    assert (tmp_path / "square.py").exists()
    run_record = result.tool_calls[1]
    assert run_record.call.name == "run_python"
    assert "144" in run_record.result.content
    assert result.usage.llm_calls == 3
    assert result.usage.cost_usd > 0

    # The transcript sent on the last turn is valid OpenAI wire format.
    wire = turns[-1]["messages"]
    assert [m["role"] for m in wire] == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assert wire[3]["tool_call_id"] == "w1"
    assert wire[2]["tool_calls"][0]["function"]["name"] == "write_file"
    assert {t["function"]["name"] for t in turns[0]["tools"]} == {
        "list_files", "read_file", "write_file", "run_python", "run_pytest",
    }  # fmt: skip


def _mock_call(call_id: str, name: str, **arguments: Any) -> dict[str, Any]:
    function = {"name": name, "arguments": json.dumps(arguments)}
    return {"mock_tool_calls": [{"id": call_id, "type": "function", "function": function}]}
