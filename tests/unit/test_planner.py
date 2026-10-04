from __future__ import annotations

import json
from typing import Any

import pytest

from workflow_agent.budget import UsageTracker
from workflow_agent.errors import BudgetExceededError, PlanningError
from workflow_agent.llm import ForceTool, ScriptedLLMClient, calls, tool_call
from workflow_agent.workflow import SUBMIT_PLAN, Planner

KIND_TOOLS = {
    "research": ["fetch_url"],
    "code": ["write_file", "run_python"],
    "verify": ["run_pytest"],
}


def step(id: str, kind: str = "research", deps: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": id,
        "kind": kind,
        "title": f"Do {id}",
        "instructions": f"Instructions for {id}.",
        "depends_on": deps or [],
    }


def submit(*steps: dict[str, Any]) -> Any:
    return calls(tool_call(SUBMIT_PLAN, {"rationale": "because", "steps": list(steps)}))


GOOD = (step("research"), step("impl", "code", ["research"]), step("check", "verify", ["impl"]))


async def test_plan_is_requested_with_forced_tool_choice() -> None:
    llm = ScriptedLLMClient([submit(*GOOD)])
    plan = await Planner(llm, kind_tools=KIND_TOOLS).plan("Build a thing")
    assert [s.id for s in plan.steps] == ["research", "impl", "check"]

    request = llm.calls[0]
    assert request.tool_choice == ForceTool(SUBMIT_PLAN)
    assert request.tool_names == [SUBMIT_PLAN]
    schema = request.tools[0]["function"]["parameters"]
    assert "$defs" not in json.dumps(schema)
    assert "Build a thing" in request.first_user_message
    system = request.system_prompt
    assert "- research:" in system
    assert "Tools: write_file, run_python" in system
    assert "at most 8" in system
    assert "must be checked by a `verify` step" in system


async def test_invalid_plan_is_repaired_with_feedback() -> None:
    cyclic = submit(step("a", deps=["b"]), step("b", deps=["a"]))
    llm = ScriptedLLMClient([cyclic, submit(step("a"), step("b", deps=["a"]))])
    plan = await Planner(llm, kind_tools=KIND_TOOLS).plan("x")
    assert [s.id for s in plan.steps] == ["a", "b"]
    feedback = llm.calls[1].last_message
    assert feedback.role == "tool"
    assert "dependency cycle among steps: a, b" in (feedback.content or "")
    assert "call `submit_plan` again" in (feedback.content or "")


@pytest.mark.parametrize(
    ("bad_plan", "problem"),
    [
        (submit(step("impl", "code")), "not covered by any verify step: impl"),
        (submit(*(step(f"s{i}") for i in range(9))), "the maximum is 8"),
        (submit({**step("a"), "kind": "verify"}), None),  # valid; used to check the next case
    ],
)
async def test_policy_violations_are_reported(bad_plan: Any, problem: str | None) -> None:
    llm = ScriptedLLMClient([bad_plan, submit(*GOOD)])
    await Planner(llm, kind_tools=KIND_TOOLS).plan("x")
    if problem is None:
        assert len(llm.calls) == 1
    else:
        assert problem in (llm.calls[1].last_message.content or "")


async def test_verification_policy_can_be_disabled_and_kinds_restricted() -> None:
    llm = ScriptedLLMClient([submit(step("impl", "code"))])
    planner = Planner(llm, kind_tools={"research": [], "code": []}, require_verification=False)
    plan = await planner.plan("x")
    assert plan.steps[0].kind == "code"
    assert "must be checked" not in llm.calls[0].system_prompt

    restricted = ScriptedLLMClient([submit(step("v", "verify")), submit(step("r"))])
    await Planner(restricted, kind_tools={"research": []}).plan("x")
    assert "disabled kind: v" in (restricted.calls[1].last_message.content or "")


async def test_json_in_content_is_accepted_when_tool_choice_is_ignored() -> None:
    body = json.dumps({"rationale": "r", "steps": [step("only")]})
    llm = ScriptedLLMClient([f"Here is the plan:\n```json\n{body}\n```"])
    plan = await Planner(llm, kind_tools=KIND_TOOLS).plan("x")
    assert plan.steps[0].id == "only"

    bare = ScriptedLLMClient([f"Plan: {body} -- done"])
    assert (await Planner(bare, kind_tools=KIND_TOOLS).plan("x")).steps[0].id == "only"


async def test_no_plan_and_wrong_tool_get_feedback() -> None:
    llm = ScriptedLLMClient(
        ["I would rather chat.", calls(tool_call("other_tool")), submit(step("a"))]
    )
    plan = await Planner(llm, kind_tools=KIND_TOOLS, max_attempts=3).plan("x")
    assert plan.steps[0].id == "a"
    assert llm.calls[1].last_message.role == "user"
    assert "no plan received" in (llm.calls[1].last_message.content or "")
    assert "unknown tool 'other_tool'" in (llm.calls[2].last_message.content or "")


async def test_gives_up_after_max_attempts() -> None:
    llm = ScriptedLLMClient([submit(step("a", deps=["a"]))] * 2)
    with pytest.raises(PlanningError, match="no valid plan after 2 attempt") as info:
        await Planner(llm, kind_tools=KIND_TOOLS, max_attempts=2).plan("x")
    assert "depends on itself" in str(info.value)


async def test_planner_respects_budget() -> None:
    tracker = UsageTracker(max_total_tokens=100)
    llm = ScriptedLLMClient(["no plan", submit(*GOOD)])
    with pytest.raises(BudgetExceededError):
        await Planner(llm, kind_tools=KIND_TOOLS, tracker=tracker).plan("x")
    assert len(llm.calls) == 1


@pytest.mark.parametrize("kwargs", [{"max_steps": 0}, {"max_attempts": 0}])
def test_invalid_planner_settings(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match=">= 1"):
        Planner(ScriptedLLMClient([]), kind_tools=KIND_TOOLS, **kwargs)  # type: ignore[arg-type]
