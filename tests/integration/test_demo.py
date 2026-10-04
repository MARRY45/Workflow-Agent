"""The offline demo exercises planner, engine and the *real* tools end to end."""

from __future__ import annotations

from pathlib import Path

import pytest

from workflow_agent import demo
from workflow_agent.events import Event, EventBus


async def test_demo_succeeds_with_real_execution(tmp_path: Path) -> None:
    events: list[Event] = []
    report = await demo.run_demo(tmp_path, events=EventBus([events.append]))

    assert report.status == "success"
    verify = report.result("verify")
    assert verify.evidence == [
        e for e in verify.evidence if e.startswith("run_pytest: exit code 0")
    ]
    assert "12 passed" in verify.summary
    assert report.result("implement").artifacts == ["slugify.py"]
    assert "meets every requirement" in report.answer
    assert sum(e.type == "tool_finished" for e in events) == 8


async def test_demo_catches_a_broken_implementation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug: forgets to strip leading/trailing hyphens. The verdict must come from pytest.
    broken = demo.IMPLEMENTATION.replace('.strip("-")', "")
    assert broken != demo.IMPLEMENTATION
    monkeypatch.setattr(demo, "IMPLEMENTATION", broken)

    report = await demo.run_demo(tmp_path)

    assert report.result("implement").status == "success"  # the smoke test still passes
    verify = report.result("verify")
    assert verify.status == "failed"
    assert "failed" in verify.summary
    assert verify.evidence[0].startswith("run_pytest: exit code 1: some tests failed")
    assert report.status == "partial"
    assert "NOT fully verified" in report.answer
