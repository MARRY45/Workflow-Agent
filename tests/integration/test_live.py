"""Smoke tests against a real provider. Deselected by default; run with

WORKFLOW_AGENT_LIVE_MODEL=gpt-4o-mini OPENAI_API_KEY=... pytest -m live
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from workflow_agent import Agent, LiteLLMClient, Workspace, default_registry
from workflow_agent.workflow import Planner

MODEL = os.environ.get("WORKFLOW_AGENT_LIVE_MODEL", "")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not MODEL, reason="set WORKFLOW_AGENT_LIVE_MODEL to run live tests"),
]


async def test_live_agent_uses_tools(tmp_path: Path) -> None:
    tools = default_registry(Workspace(tmp_path), include_web=False)
    agent = Agent(LiteLLMClient(MODEL), tools)
    result = await agent.run("Use run_python to compute 2**20 + 7 and reply with just the number.")
    assert "1048583" in result.output
    assert any(r.call.name == "run_python" for r in result.tool_calls)


async def test_live_planner_produces_a_valid_plan() -> None:
    planner = Planner(
        LiteLLMClient(MODEL),
        kind_tools={
            "research": ["fetch_url"],
            "code": ["write_file", "run_python"],
            "verify": ["run_pytest"],
        },
    )
    plan = await planner.plan(
        "Write a function that checks if a string is a palindrome and test it."
    )
    kinds = {step.kind for step in plan.steps}
    assert {"code", "verify"} <= kinds
