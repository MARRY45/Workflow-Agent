"""Workflow engine: plan -> concurrent DAG execution -> verification -> synthesis.

Execution model
---------------
Every step becomes an asyncio task that first awaits the results of its dependencies and
then (bounded by ``step_concurrency``) runs a fresh :class:`Agent` restricted to the tools
of its kind plus the terminal ``complete_step`` tool. Independent branches therefore run in
parallel without any explicit scheduling code.

Failure semantics
-----------------
* A failed or skipped step causes its dependents to be *skipped*; independent branches go on.
* A ``verify`` step that reports success without having executed any code is downgraded to
  ``unverified`` - claims of verification must be backed by execution evidence.
* An exhausted budget aborts the run: running steps fail at their next LLM call, pending
  ones are skipped, and the report is assembled without another LLM call.
* ``timeout_s`` bounds the whole execution phase.

``run`` returns a :class:`WorkflowReport` for every model/provider/budget failure and only
raises for cancellation or programming errors.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from workflow_agent.agent.loop import Agent, AgentConfig, AgentResult
from workflow_agent.budget import UsageTracker
from workflow_agent.errors import BudgetExceededError, LLMError, PlanningError
from workflow_agent.events import (
    EventBus,
    LLMCallCompleted,
    PlanCreated,
    StepFinished,
    StepStarted,
    WorkflowFinished,
    WorkflowStarted,
)
from workflow_agent.llm.base import LLMClient
from workflow_agent.messages import Message, Usage
from workflow_agent.textutil import one_line
from workflow_agent.tools.base import Tool
from workflow_agent.tools.registry import ToolRegistry
from workflow_agent.workflow.models import (
    STEP_KINDS,
    Plan,
    PlanStep,
    StepKind,
    StepResult,
    WorkflowReport,
    WorkflowStatus,
)
from workflow_agent.workflow.planner import Planner
from workflow_agent.workflow.prompts import (
    SYNTHESIZER_SYSTEM_PROMPT,
    render_step_task,
    render_synthesis_request,
    step_system_prompt,
)

logger = logging.getLogger(__name__)

COMPLETE_STEP = "complete_step"
EXECUTION_TAG = "execution"

DEFAULT_KIND_TAGS: Mapping[StepKind, frozenset[str]] = {
    "research": frozenset({"research", "read"}),
    "code": frozenset({"read", "write", "execution", "research"}),
    "verify": frozenset({"read", "write", "execution"}),
}


class StepOutcome(BaseModel):
    """Arguments of the terminal ``complete_step`` tool."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "failed"] = Field(
        description=(
            "'success' if the step achieved its goal (for verify steps: the checks passed), "
            "otherwise 'failed'."
        )
    )
    summary: str = Field(
        min_length=1,
        description=(
            "Complete, self-contained result for later steps and the final report: findings "
            "with sources, decisions, file paths, commands run and their exact outcomes."
        ),
    )
    artifacts: list[str] = Field(
        default_factory=list,
        description="Workspace-relative paths of files created or modified by this step.",
    )


async def _record_outcome(**_: object) -> str:
    return "Step result recorded."


COMPLETE_STEP_TOOL = Tool(
    name=COMPLETE_STEP,
    description=(
        "Finish this step and report its result. Call it exactly once, on its own, when the "
        "step is done or cannot be done."
    ),
    args_model=StepOutcome,
    fn=_record_outcome,
    terminal=True,
)


@dataclass(frozen=True, slots=True)
class WorkflowConfig:
    max_plan_steps: int = 8
    planner_attempts: int = 3
    step_concurrency: int = 3
    agent_max_turns: int = 12
    max_parallel_tools: int = 4
    require_verification: bool = True
    timeout_s: float | None = None
    max_dependency_chars: int = 6000
    kind_tags: Mapping[StepKind, frozenset[str]] = field(
        default_factory=lambda: dict(DEFAULT_KIND_TAGS)
    )

    def __post_init__(self) -> None:
        if self.step_concurrency < 1:
            raise ValueError("step_concurrency must be >= 1")
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        unknown = set(self.kind_tags) - set(STEP_KINDS)
        if unknown:
            raise ValueError(f"unknown step kinds in kind_tags: {sorted(unknown)}")


@dataclass
class _RunState:
    """Mutable state of one ``run`` (kept off the engine so runs can overlap)."""

    abort_reason: str | None = None
    started: set[str] = field(default_factory=set)


class WorkflowEngine:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        *,
        config: WorkflowConfig | None = None,
        tracker: UsageTracker | None = None,
        events: EventBus | None = None,
        planner_llm: LLMClient | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.config = config or WorkflowConfig()
        self.tracker = tracker or UsageTracker()
        self.events = events or EventBus()
        self.planner = Planner(
            planner_llm or llm,
            kind_tools={kind: self.tools_for(kind).names for kind in self.config.kind_tags},
            max_steps=self.config.max_plan_steps,
            max_attempts=self.config.planner_attempts,
            require_verification=self.config.require_verification,
            tracker=self.tracker,
            events=self.events,
        )

    def tools_for(self, kind: StepKind) -> ToolRegistry:
        """The tools a step of ``kind`` may use (without ``complete_step``)."""
        tags = self.config.kind_tags.get(kind, frozenset())
        return self.tools.filter(lambda tool: bool(tool.tags & tags))

    # ------------------------------------------------------------------ public API

    async def run(self, objective: str, *, plan: Plan | None = None) -> WorkflowReport:
        """Plan (unless ``plan`` is given), execute and report on ``objective``."""
        started = time.monotonic()
        usage_before = self.tracker.usage
        await self.events.emit(WorkflowStarted(source="workflow", objective=objective))

        if plan is None:
            try:
                plan = await self.planner.plan(objective)
            except (PlanningError, BudgetExceededError, LLMError) as exc:
                logger.warning("Planning failed: %s", exc)
                error = f"planning failed: {type(exc).__name__}: {exc}"
                return await self._report(objective, None, [], error, error, started, usage_before)

        await self.events.emit(
            PlanCreated(
                source="workflow",
                rationale=plan.rationale,
                steps=[step.model_dump() for step in plan.steps],
            )
        )
        state = _RunState()
        results = await self._execute(objective, plan, state)
        answer = await self._synthesize(objective, plan, results, state)
        return await self._report(
            objective, plan, results, answer, state.abort_reason, started, usage_before
        )

    # ------------------------------------------------------------------ execution

    async def _execute(self, objective: str, plan: Plan, state: _RunState) -> list[StepResult]:
        loop = asyncio.get_running_loop()
        futures: dict[str, asyncio.Future[StepResult]] = {
            step.id: loop.create_future() for step in plan.steps
        }
        semaphore = asyncio.Semaphore(self.config.step_concurrency)

        async def schedule(step: PlanStep) -> None:
            # shield(): if this task is cancelled (timeout) the *shared* future must survive.
            dependencies = [
                (plan.get(d), await asyncio.shield(futures[d])) for d in step.depends_on
            ]
            blocked = [dep.id for dep, result in dependencies if not result.ok]
            if blocked:
                result = _skipped(step, f"dependency not satisfied: {', '.join(blocked)}")
            else:
                async with semaphore:
                    if state.abort_reason:
                        result = _skipped(step, f"workflow aborted: {state.abort_reason}")
                    else:
                        state.started.add(step.id)
                        result = await self._run_step(objective, step, dependencies, state)
            if result.status == "skipped":
                await self._emit_step_finished(result)
            futures[step.id].set_result(result)

        try:
            async with asyncio.timeout(self.config.timeout_s), asyncio.TaskGroup() as group:
                for step in plan.steps:
                    group.create_task(schedule(step))
        except TimeoutError:
            state.abort_reason = f"workflow timed out after {self.config.timeout_s:g}s"
            logger.warning(state.abort_reason)

        results = []
        for step in plan.steps:
            future = futures[step.id]
            if future.done() and not future.cancelled():
                results.append(future.result())
                continue
            # Interrupted by the timeout: running steps failed, pending ones never started.
            if step.id in state.started:
                result = StepResult(step_id=step.id, status="failed", error=state.abort_reason)
            else:
                result = _skipped(step, f"workflow aborted: {state.abort_reason}")
            await self._emit_step_finished(result)
            results.append(result)
        return results

    async def _run_step(
        self,
        objective: str,
        step: PlanStep,
        dependencies: Sequence[tuple[PlanStep, StepResult]],
        state: _RunState,
    ) -> StepResult:
        started = time.monotonic()
        await self.events.emit(
            StepStarted(source="workflow", step_id=step.id, kind=step.kind, title=step.title)
        )
        agent = Agent(
            self.llm,
            self.tools_for(step.kind).with_tools(COMPLETE_STEP_TOOL),
            config=AgentConfig(
                name=f"step:{step.id}",
                system_prompt=step_system_prompt(step.kind),
                max_turns=self.config.agent_max_turns,
                max_parallel_tools=self.config.max_parallel_tools,
            ),
            tracker=self.tracker,
            events=self.events,
        )
        task = render_step_task(
            objective, step, dependencies, max_dependency_chars=self.config.max_dependency_chars
        )
        try:
            run = await agent.run(task)
        except BudgetExceededError as exc:
            state.abort_reason = str(exc)
            result = StepResult(step_id=step.id, status="failed", error=str(exc))
        except Exception as exc:  # provider failure or bug: contain it to this step
            logger.warning(
                "Step %s failed: %s", step.id, exc, exc_info=not isinstance(exc, LLMError)
            )
            result = StepResult(
                step_id=step.id, status="failed", error=f"{type(exc).__name__}: {exc}"
            )
        else:
            result = self._interpret(step, run)
        result = result.model_copy(update={"duration_s": time.monotonic() - started})
        await self._emit_step_finished(result)
        return result

    def _interpret(self, step: PlanStep, run: AgentResult) -> StepResult:
        evidence = [
            one_line(record.result.content.splitlines()[0] if record.result.content else "", 160)
            for record in run.tool_calls
            if EXECUTION_TAG in record.tags and not record.result.is_error
        ]
        error: str | None = None
        artifacts: list[str] = []
        if run.stop_reason == "terminal_tool" and run.terminal_output is not None:
            outcome = StepOutcome.model_validate(run.terminal_output)
            status: Literal["success", "unverified", "failed"] = outcome.status
            summary, artifacts = outcome.summary, outcome.artifacts
        elif run.stop_reason == "final_answer" and run.output.strip():
            status, summary = "success", run.output  # model answered without complete_step
        elif run.stop_reason == "max_turns":
            status, summary = "failed", run.output
            error = f"turn limit ({self.config.agent_max_turns}) reached before completion"
        else:
            status, summary, error = "failed", run.output, "the agent produced no result"

        if (
            status == "success"
            and step.kind == "verify"
            and self.config.require_verification
            and not evidence
        ):
            status = "unverified"
            error = "verify step reported success without executing any code"

        return StepResult(
            step_id=step.id,
            status=status,
            summary=summary,
            artifacts=artifacts,
            evidence=evidence,
            error=error,
            turns=run.turns,
            tool_calls=len(run.tool_calls),
            usage=run.usage,
        )

    async def _emit_step_finished(self, result: StepResult) -> None:
        await self.events.emit(
            StepFinished(
                source="workflow",
                step_id=result.step_id,
                status=result.status,
                summary=result.summary,
                duration_s=result.duration_s,
                error=result.error,
            )
        )

    # ------------------------------------------------------------------ reporting

    async def _synthesize(
        self, objective: str, plan: Plan, results: Sequence[StepResult], state: _RunState
    ) -> str:
        if state.abort_reason is None:
            request = render_synthesis_request(
                objective, plan, results, max_chars_per_step=self.config.max_dependency_chars
            )
            try:
                self.tracker.check()
                response = await self.llm.complete(
                    [Message.system(SYNTHESIZER_SYSTEM_PROMPT), Message.user(request)]
                )
                self.tracker.record(response.usage)
                await self.events.emit(
                    LLMCallCompleted(
                        source="synthesizer",
                        turn=1,
                        finish_reason=response.finish_reason,
                        tool_calls=[],
                        content=response.message.content,
                        usage=response.usage,
                    )
                )
                if response.message.content and response.message.content.strip():
                    return response.message.content
            except (BudgetExceededError, LLMError) as exc:
                logger.warning("Synthesis failed, using fallback summary: %s", exc)
        return _fallback_answer(plan, results, state.abort_reason)

    async def _report(
        self,
        objective: str,
        plan: Plan | None,
        results: list[StepResult],
        answer: str,
        error: str | None,
        started: float,
        usage_before: Usage,
    ) -> WorkflowReport:
        usage = _usage_delta(self.tracker.usage, usage_before)
        status = _overall_status(results, error)
        duration = time.monotonic() - started
        await self.events.emit(
            WorkflowFinished(source="workflow", status=status, duration_s=duration, usage=usage)
        )
        return WorkflowReport(
            objective=objective,
            status=status,
            plan=plan,
            results=results,
            answer=answer,
            usage=usage,
            duration_s=duration,
            error=error,
        )


def _skipped(step: PlanStep, reason: str) -> StepResult:
    return StepResult(step_id=step.id, status="skipped", error=reason)


def _overall_status(results: Sequence[StepResult], error: str | None) -> WorkflowStatus:
    if not results or not any(r.ok for r in results):
        return "failed"
    if error is None and all(r.status == "success" for r in results):
        return "success"
    return "partial"


def _usage_delta(after: Usage, before: Usage) -> Usage:
    return Usage(
        prompt_tokens=after.prompt_tokens - before.prompt_tokens,
        completion_tokens=after.completion_tokens - before.completion_tokens,
        cost_usd=max(after.cost_usd - before.cost_usd, 0.0),
        llm_calls=after.llm_calls - before.llm_calls,
    )


def _fallback_answer(plan: Plan, results: Sequence[StepResult], reason: str | None) -> str:
    """Deterministic answer used when the synthesizer cannot (or must not) be called."""
    lines = ["_The final synthesis was not generated" + (f" ({reason})" if reason else "") + "._"]
    lines.append("Step summaries:")
    for result in results:
        step = plan.get(result.step_id)
        detail = one_line(result.summary or result.error or "", 300)
        lines.append(f"- **{step.title}** [{result.status}]: {detail}")
    return "\n".join(lines)
