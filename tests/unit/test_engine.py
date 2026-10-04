from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import Any

import pytest

from workflow_agent.budget import UsageTracker
from workflow_agent.errors import LLMError
from workflow_agent.events import Event, EventBus
from workflow_agent.llm import RecordedCall, ScriptedLLMClient, calls, tool_call
from workflow_agent.llm.scripted import Reply
from workflow_agent.tools import ToolRegistry, tool
from workflow_agent.workflow import COMPLETE_STEP, SUBMIT_PLAN, Plan, WorkflowConfig, WorkflowEngine

STEP_RE = re.compile(r"# Your step: `(\w+)`")
CONCURRENCY = {"active": 0, "peak": 0}


@tool(tags={"research"})
async def lookup(topic: str) -> str:
    """Look something up."""
    return f"facts about {topic}"


@tool(tags={"research"})
async def slow_lookup(topic: str) -> str:
    """Slow lookup that records concurrency."""
    CONCURRENCY["active"] += 1
    CONCURRENCY["peak"] = max(CONCURRENCY["peak"], CONCURRENCY["active"])
    await asyncio.sleep(0.1)
    CONCURRENCY["active"] -= 1
    return f"slow facts about {topic}"


@tool(tags={"write"})
def save(name: str) -> str:
    """Save a file."""
    return f"saved {name}"


@tool(tags={"execution"})
async def execute(code: str) -> str:
    """Execute code."""
    return f"execute: exit code 0 [0.01s]\n{code}"


@tool(tags={"execution"})
async def hang() -> str:
    """Never finishes."""
    await asyncio.sleep(30)
    return "never"


TOOLS = ToolRegistry([lookup, slow_lookup, save, execute, hang])


def step(id: str, kind: str = "research", deps: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": id,
        "kind": kind,
        "title": f"Do {id}",
        "instructions": f"Instructions for {id}.",
        "depends_on": deps or [],
    }


def done(summary: str, status: str = "success", artifacts: list[str] | None = None) -> Reply:
    args = {"status": status, "summary": summary, "artifacts": artifacts or []}
    return calls(tool_call(COMPLETE_STEP, args))


def use(tool_name: str, /, **arguments: Any) -> Reply:
    return calls(tool_call(tool_name, arguments))


class Router:
    """Routes concurrent LLM calls to per-step scripts based on the prompt."""

    def __init__(
        self,
        steps: list[dict[str, Any]],
        scripts: dict[str, list[Any]],
        synthesis: str = "FINAL ANSWER",
    ) -> None:
        self.steps = steps
        self.scripts = scripts
        self.synthesis = synthesis
        self.step_calls: dict[str, list[RecordedCall]] = {}
        self.synthesis_requests: list[str] = []
        self.planner_calls = 0

    def __call__(self, call: RecordedCall) -> Reply:
        system = call.system_prompt
        if system.startswith("You are the planning component"):
            self.planner_calls += 1
            return calls(tool_call(SUBMIT_PLAN, {"rationale": "test plan", "steps": self.steps}))
        if system.startswith("You are the reporting component"):
            self.synthesis_requests.append(call.first_user_message)
            return self.synthesis
        match = STEP_RE.search(call.first_user_message)
        assert match, call.first_user_message
        step_id = match.group(1)
        self.step_calls.setdefault(step_id, []).append(call)
        reply = self.scripts[step_id][call.turn]
        if isinstance(reply, BaseException):
            raise reply
        if callable(reply):
            return reply(call)
        return reply


def make_engine(
    router: Router, *, tracker: UsageTracker | None = None, **config: Any
) -> tuple[WorkflowEngine, list[Event]]:
    events: list[Event] = []
    engine = WorkflowEngine(
        ScriptedLLMClient(router),
        TOOLS,
        config=WorkflowConfig(**config),
        tracker=tracker,
        events=EventBus([events.append]),
    )
    return engine, events


async def test_happy_path_research_code_verify() -> None:
    steps = [
        step("research"),
        step("impl", "code", ["research"]),
        step("check", "verify", ["impl"]),
    ]
    router = Router(
        steps,
        {
            "research": [
                use("lookup", topic="slugs"),
                done("Slugs are lowercase. Source: https://x"),
            ],
            "impl": [
                use("save", name="slug.py"),
                done("Implemented slug.py", artifacts=["slug.py"]),
            ],
            "check": [use("execute", code="pytest"), done("3 passed")],
        },
    )
    engine, events = make_engine(router)
    report = await engine.run("Build a slugify function")

    assert report.status == "success"
    assert report.error is None
    assert report.answer == "FINAL ANSWER"
    assert [r.status for r in report.results] == ["success"] * 3
    assert report.result("impl").artifacts == ["slug.py"]
    assert report.result("check").evidence == ["execute: exit code 0 [0.01s]"]
    assert report.result("research").evidence == []
    assert report.usage.llm_calls == 1 + 6 + 1  # planner + 3 steps x 2 turns + synthesis

    # Tools are scoped by step kind; complete_step is always available.
    tool_names = {sid: set(c[0].tool_names) for sid, c in router.step_calls.items()}
    assert tool_names["research"] == {"lookup", "slow_lookup", COMPLETE_STEP}
    assert tool_names["impl"] == {"lookup", "slow_lookup", "save", "execute", "hang", COMPLETE_STEP}
    assert tool_names["check"] == {"save", "execute", "hang", COMPLETE_STEP}

    # Dependency results are handed to dependents; the synthesizer sees evidence.
    impl_task = router.step_calls["impl"][0].first_user_message
    assert "Slugs are lowercase. Source: https://x" in impl_task
    assert "`research`: Do research [success]" in impl_task
    assert "Artifacts: slug.py" in router.step_calls["check"][0].first_user_message
    assert "Execution evidence:\n- execute: exit code 0" in router.synthesis_requests[0]

    types = [e.type for e in events]
    assert types[0] == "workflow_started"
    assert types[-1] == "workflow_finished"
    assert types.index("plan_created") < types.index("step_started")
    assert types.count("step_started") == types.count("step_finished") == 3


async def test_independent_steps_run_in_parallel() -> None:
    steps = [step("a"), step("b"), step("c"), step("merge", deps=["a", "b", "c"])]
    scripts = {sid: [use("slow_lookup", topic=sid), done(f"{sid} done")] for sid in "abc"}
    scripts["merge"] = [done("merged")]

    CONCURRENCY.update(active=0, peak=0)
    engine, _ = make_engine(Router(steps, scripts), require_verification=False, step_concurrency=3)
    report = await engine.run("x")
    assert report.status == "success"
    assert CONCURRENCY["peak"] == 3

    CONCURRENCY.update(active=0, peak=0)
    engine, _ = make_engine(Router(steps, scripts), require_verification=False, step_concurrency=1)
    await engine.run("x")
    assert CONCURRENCY["peak"] == 1


async def test_failed_step_skips_dependents_but_not_independent_branches() -> None:
    steps = [step("a"), step("b", deps=["a"]), step("c", deps=["b"]), step("other")]
    router = Router(
        steps, {"a": [done("could not find it", status="failed")], "other": [done("fine")]}
    )
    engine, events = make_engine(router)
    report = await engine.run("x")

    assert [r.status for r in report.results] == ["failed", "skipped", "skipped", "success"]
    assert report.result("b").error == "dependency not satisfied: a"
    assert report.result("c").error == "dependency not satisfied: b"
    assert report.status == "partial"
    assert set(router.step_calls) == {"a", "other"}  # skipped steps never call the LLM
    finished = [e for e in events if e.type == "step_finished"]
    assert len(finished) == 4


async def test_verify_without_execution_is_unverified() -> None:
    steps = [step("check", "verify")]
    engine, _ = make_engine(Router(steps, {"check": [done("Looks correct to me.")]}))
    report = await engine.run("x")
    result = report.result("check")
    assert result.status == "unverified"
    assert "without executing any code" in (result.error or "")
    assert report.status == "partial"


async def test_unverified_steps_still_feed_dependents() -> None:
    steps = [step("check", "verify"), step("after", deps=["check"])]
    router = Router(steps, {"check": [done("trust me")], "after": [done("ok")]})
    engine, _ = make_engine(router)
    report = await engine.run("x")
    assert [r.status for r in report.results] == ["unverified", "success"]


async def test_failed_tool_calls_are_not_evidence() -> None:
    steps = [step("check", "verify")]
    script = [use("execute"), done("verified")]  # missing required argument -> tool error
    engine, _ = make_engine(Router(steps, {"check": script}))
    report = await engine.run("x")
    assert report.result("check").status == "unverified"


async def test_llm_errors_are_contained_to_the_step() -> None:
    steps = [step("a"), step("b")]
    router = Router(steps, {"a": [LLMError("provider down")], "b": [done("fine")]})
    engine, _ = make_engine(router)
    report = await engine.run("x")
    assert report.result("a").status == "failed"
    assert report.result("a").error == "LLMError: provider down"
    assert report.result("b").status == "success"
    assert report.status == "partial"


async def test_turn_limit_and_plain_answers() -> None:
    steps = [step("loops"), step("plain"), step("empty")]
    router = Router(
        steps,
        {
            "loops": [use("lookup", topic="x")] * 5,
            "plain": ["Here is my plain-text result."],
            "empty": [""],
        },
    )
    engine, _ = make_engine(router, agent_max_turns=2)
    report = await engine.run("x")
    assert report.result("loops").status == "failed"
    assert "turn limit (2)" in (report.result("loops").error or "")
    assert report.result("plain").status == "success"
    assert report.result("plain").summary == "Here is my plain-text result."
    assert report.result("empty").status == "failed"
    assert report.result("empty").error == "the agent produced no result"


async def test_budget_exhaustion_aborts_the_run() -> None:
    steps = [step("first"), step("second", deps=["first"])]
    router = Router(
        steps, {"first": [use("lookup", topic="x"), done("done")], "second": [done("done")]}
    )
    tracker = UsageTracker(max_total_tokens=200)  # planner (120) + 1 step call (120) -> exhausted
    engine, _ = make_engine(router, tracker=tracker)
    report = await engine.run("x")

    assert report.result("first").status == "failed"
    assert "token budget exhausted" in (report.result("first").error or "")
    assert report.result("second").status == "skipped"
    assert report.status == "failed"
    assert "token budget exhausted" in (report.error or "")
    assert router.synthesis_requests == []  # no LLM call after the budget is gone
    assert "final synthesis was not generated" in report.answer


async def test_timeout_bounds_execution() -> None:
    steps = [step("slow", "verify"), step("later", deps=["slow"]), step("quick")]
    router = Router(steps, {"slow": [use("hang")], "quick": [done("fast")]})
    engine, events = make_engine(router, timeout_s=0.3)
    report = await engine.run("x")
    assert report.result("slow").status == "failed"
    assert report.result("later").status == "skipped"
    assert report.result("quick").status == "success"
    assert report.error == "workflow timed out after 0.3s"
    assert report.status == "partial"
    assert sum(e.type == "step_finished" for e in events) == 3


async def test_planning_failure_returns_a_failed_report() -> None:
    def broken(call: RecordedCall) -> Reply:
        return calls(tool_call(SUBMIT_PLAN, {"rationale": "r", "steps": []}))

    events: list[Event] = []
    engine = WorkflowEngine(
        ScriptedLLMClient(broken),
        TOOLS,
        config=WorkflowConfig(planner_attempts=2),
        events=EventBus([events.append]),
    )
    report = await engine.run("x")
    assert report.status == "failed"
    assert report.plan is None
    assert report.results == []
    assert (report.error or "").startswith("planning failed: PlanningError")
    assert report.usage.llm_calls == 2
    assert events[-1].type == "workflow_finished"


async def test_given_plan_skips_the_planner_and_synthesis_falls_back() -> None:
    steps = [step("only")]
    router = Router(steps, {"only": [done("result")]}, synthesis="")
    engine, _ = make_engine(router)
    plan = Plan.model_validate({"rationale": "manual", "steps": steps})
    report = await engine.run("x", plan=plan)
    assert router.planner_calls == 0
    assert report.status == "success"
    assert "final synthesis was not generated" in report.answer
    assert "**Do only** [success]: result" in report.answer


async def test_kind_tags_limit_planner_kinds() -> None:
    engine = WorkflowEngine(
        ScriptedLLMClient([]),
        TOOLS,
        config=WorkflowConfig(kind_tags={"research": frozenset({"research"})}),
    )
    assert set(engine.planner.kind_tools) == {"research"}
    assert engine.tools_for("research").names == ["lookup", "slow_lookup"]
    assert engine.tools_for("code").names == []


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"step_concurrency": 0}, "step_concurrency"),
        ({"timeout_s": 0}, "timeout_s"),
        ({"kind_tags": {"deploy": frozenset()}}, "unknown step kinds"),
    ],
)
def test_config_validation(kwargs: dict[str, Any], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        WorkflowConfig(**kwargs)


def test_router_helper_is_callable() -> None:
    assert isinstance(Router([], {}), Callable)  # type: ignore[arg-type]
