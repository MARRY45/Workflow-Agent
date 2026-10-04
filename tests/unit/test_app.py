from __future__ import annotations

from pathlib import Path

import pytest

from workflow_agent import Settings, app
from workflow_agent.llm import LiteLLMClient, ScriptedLLMClient


def test_builders_apply_settings(tmp_path: Path) -> None:
    settings = Settings(
        model="anthropic/claude-test",
        temperature=0.3,
        max_retries=5,
        max_concurrent_llm_requests=2,
        workspace_dir=tmp_path / "ws",
        max_agent_turns=7,
        max_plan_steps=4,
        step_concurrency=2,
        workflow_timeout_s=60,
        max_total_tokens=1000,
        max_cost_usd=0.5,
        tool_timeout_s=30,
        max_tool_output_chars=5000,
    )
    llm = app.build_llm(settings)
    assert isinstance(llm, LiteLLMClient)
    assert (llm.model, llm.temperature, llm.retry.max_retries) == ("anthropic/claude-test", 0.3, 5)

    tools = app.build_tools(settings)
    assert (tmp_path / "ws").is_dir()
    assert "fetch_url" in tools
    assert (tools.default_timeout_s, tools.max_output_chars) == (30, 5000)
    assert "fetch_url" not in app.build_tools(settings, include_web=False)

    engine = app.build_engine(settings, llm=ScriptedLLMClient([]))
    assert engine.config.max_plan_steps == 4
    assert engine.config.step_concurrency == 2
    assert engine.config.agent_max_turns == 7
    assert engine.config.timeout_s == 60
    assert engine.tracker.max_total_tokens == 1000
    assert engine.tracker.max_cost_usd == 0.5

    agent = app.build_agent(settings, llm=ScriptedLLMClient([]))
    assert agent.config.max_turns == 7
    assert "run_python" in agent.tools


async def test_run_workflow_entry_point(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from workflow_agent.demo import DemoModel

    (tmp_path / "SPEC.md").write_text("1. Requirement one.\n")
    monkeypatch.setattr(app, "build_llm", lambda settings: ScriptedLLMClient(DemoModel()))
    report = await app.run_workflow("Implement slugify", Settings(workspace_dir=tmp_path))
    assert report.status == "success"
    assert report.result("verify").evidence
