"""The examples run offline when given a scripted model."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from workflow_agent import ScriptedLLMClient, Settings
from workflow_agent.demo import DemoModel
from workflow_agent.llm import calls, tool_call

EXAMPLES = Path(__file__).parents[2] / "examples"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


async def test_custom_tool_example(tmp_path: Path) -> None:
    example = load("custom_tool")
    llm = ScriptedLLMClient(
        [
            calls(tool_call("describe_numbers", {"values": [1, 2, 3, 4]})),
            calls(tool_call("describe_numbers", {"values": [1]})),
            "mean 2.5",
        ]
    )
    result = await example.run("describe", llm, tmp_path)
    assert result.output == "mean 2.5"
    stats, too_short = result.tool_calls
    assert '"mean": 2.5' in stats.result.content
    assert too_short.result.is_error
    assert "at least two values" in too_short.result.content


async def test_research_and_verify_example(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    example = load("research_and_verify")
    monkeypatch.chdir(tmp_path)  # the example writes traces/ relative to the cwd
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "SPEC.md").write_text("1. A requirement.\n")
    report = await example.run(
        "objective", Settings(workspace_dir=tmp_path / "ws"), llm=ScriptedLLMClient(DemoModel())
    )
    assert report.status == "success"
    assert (tmp_path / "traces" / "research_and_verify.jsonl").stat().st_size > 0
