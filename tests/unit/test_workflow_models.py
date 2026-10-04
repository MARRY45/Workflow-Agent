from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from workflow_agent.messages import Usage
from workflow_agent.workflow import Plan, StepResult, WorkflowReport


def step(id: str, kind: str = "research", deps: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": id,
        "kind": kind,
        "title": f"Do {id}",
        "instructions": f"Instructions for {id}.",
        "depends_on": deps or [],
    }


def test_valid_plan_order_and_ancestors() -> None:
    plan = Plan(
        rationale="r",
        steps=[step("c", "verify", ["b"]), step("a"), step("b", "code", ["a"]), step("d")],  # type: ignore[list-item]
    )
    order = [s.id for s in plan.topological_order()]
    assert order.index("a") < order.index("b") < order.index("c")
    assert plan.ancestors("c") == {"a", "b"}
    assert plan.ancestors("a") == set()
    assert plan.get("d").kind == "research"
    with pytest.raises(KeyError):
        plan.get("zzz")


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ([step("a"), step("a")], "duplicate step ids: a"),
        ([step("a", deps=["a"])], "depends on itself"),
        ([step("a", deps=["ghost"])], "unknown step 'ghost'"),
        (
            [step("a", deps=["b"]), step("b", deps=["a"]), step("c")],
            "dependency cycle among steps: a, b",
        ),
        ([], "at least 1 item"),
        ([{**step("Bad-Id")}], "String should match pattern"),
        ([{**step("a"), "kind": "deploy"}], "Input should be 'research', 'code' or 'verify'"),
        ([{**step("a"), "priority": 1}], "Extra inputs are not permitted"),
    ],
)
def test_invalid_plans(steps: list[dict[str, Any]], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        Plan.model_validate({"rationale": "r", "steps": steps})


def test_step_result_ok() -> None:
    assert StepResult(step_id="a", status="success").ok
    assert StepResult(step_id="a", status="unverified").ok
    assert not StepResult(step_id="a", status="failed").ok
    assert not StepResult(step_id="a", status="skipped").ok


def test_report_markdown() -> None:
    plan = Plan.model_validate(
        {"rationale": "r", "steps": [step("impl", "code"), step("check", "verify", ["impl"])]}
    )
    report = WorkflowReport(
        objective="Build it",
        status="partial",
        plan=plan,
        results=[
            StepResult(step_id="impl", status="success", summary="Wrote x.py", artifacts=["x.py"]),
            StepResult(
                step_id="check",
                status="failed",
                summary="1 failed",
                evidence=["run_pytest: exit code 1"],
                error="tests failed",
            ),
        ],
        answer="It partially works.",
        usage=Usage(prompt_tokens=1000, completion_tokens=234, cost_usd=0.0123, llm_calls=5),
        duration_s=12.34,
    )
    md = report.to_markdown()
    assert (
        "**Status:** ⚠️ partial · 2 step(s) · 5 LLM call(s) · 1,234 tokens · $0.0123 · 12.3s" in md
    )
    assert "## Answer\n\nIt partially works." in md
    assert "| 1 | `impl` Do impl | code | ✅ success | 0 run(s) |" in md
    assert "| 2 | `check` Do check | verify | ❌ failed | 1 run(s) |" in md
    assert "**Artifacts:** `x.py`" in md
    assert "- `run_pytest: exit code 1`" in md
    assert "**Error:** tests failed" in md
    assert report.result("check").status == "failed"
    with pytest.raises(KeyError):
        report.result("nope")


def test_report_markdown_without_plan() -> None:
    report = WorkflowReport(
        objective="o",
        status="failed",
        plan=None,
        results=[],
        answer="",
        usage=Usage(),
        duration_s=0,
        error="planning failed",
    )
    md = report.to_markdown()
    assert "**Error:** planning failed" in md
    assert "_(no answer)_" in md
    assert "## Steps" not in md
