"""Workflow domain model: plans (DAGs of steps), step results and the final report."""

from __future__ import annotations

from collections import deque
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from workflow_agent.messages import Usage

StepKind: TypeAlias = Literal["research", "code", "verify"]
StepStatus: TypeAlias = Literal["success", "unverified", "failed", "skipped"]
WorkflowStatus: TypeAlias = Literal["success", "partial", "failed"]

STEP_KINDS: tuple[StepKind, ...] = ("research", "code", "verify")


class PlanStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(
        pattern=r"^[a-z][a-z0-9_]{0,39}$",
        description="Unique snake_case identifier, e.g. 'compare_libraries'.",
    )
    kind: StepKind = Field(description="research | code | verify (see the step kinds above).")
    title: str = Field(min_length=3, max_length=120, description="Short imperative title.")
    instructions: str = Field(
        min_length=10,
        description=(
            "Self-contained instructions: what to do, which inputs to use, and exactly what "
            "the result must contain (facts, files, test outcomes)."
        ),
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Ids of earlier steps whose results this step needs as input.",
    )


class Plan(BaseModel):
    """A validated DAG of steps."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rationale: str = Field(description="One or two sentences explaining the approach.")
    steps: list[PlanStep] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_graph(self) -> Plan:
        ids = [step.id for step in self.steps]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate step ids: {', '.join(duplicates)}")
        known = set(ids)
        for step in self.steps:
            for dep in step.depends_on:
                if dep == step.id:
                    raise ValueError(f"step {step.id!r} depends on itself")
                if dep not in known:
                    raise ValueError(f"step {step.id!r} depends on unknown step {dep!r}")
        order = self._kahn()
        if len(order) != len(self.steps):
            cyclic = sorted(known - {s.id for s in order})
            raise ValueError(f"dependency cycle among steps: {', '.join(cyclic)}")
        return self

    def _kahn(self) -> list[PlanStep]:
        indegree = {s.id: len(set(s.depends_on)) for s in self.steps}
        dependents: dict[str, list[str]] = {s.id: [] for s in self.steps}
        for step in self.steps:
            for dep in set(step.depends_on):
                dependents[dep].append(step.id)
        by_id = {s.id: s for s in self.steps}
        queue = deque(s.id for s in self.steps if indegree[s.id] == 0)
        order: list[PlanStep] = []
        while queue:
            current = queue.popleft()
            order.append(by_id[current])
            for nxt in dependents[current]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        return order

    def topological_order(self) -> list[PlanStep]:
        """Steps ordered so that every dependency precedes its dependents."""
        return self._kahn()

    def get(self, step_id: str) -> PlanStep:
        for step in self.steps:
            if step.id == step_id:
                return step
        raise KeyError(step_id)

    def ancestors(self, step_id: str) -> set[str]:
        """All steps ``step_id`` depends on, directly or transitively."""
        seen: set[str] = set()
        stack = list(self.get(step_id).depends_on)
        while stack:
            current = stack.pop()
            if current not in seen:
                seen.add(current)
                stack.extend(self.get(current).depends_on)
        return seen


class StepResult(BaseModel):
    step_id: str
    status: StepStatus
    summary: str = ""
    artifacts: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(
        default_factory=list, description="Headlines of successful code-execution tool calls."
    )
    error: str | None = None
    turns: int = 0
    tool_calls: int = 0
    usage: Usage = Usage()
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        """Whether dependents may build on this result."""
        return self.status in ("success", "unverified")


_STATUS_ICON = {"success": "✅", "unverified": "⚠️", "failed": "❌", "skipped": "⏭️", "partial": "⚠️"}


class WorkflowReport(BaseModel):
    objective: str
    status: WorkflowStatus
    plan: Plan | None
    results: list[StepResult]
    answer: str
    usage: Usage
    duration_s: float
    error: str | None = None

    def result(self, step_id: str) -> StepResult:
        for result in self.results:
            if result.step_id == step_id:
                return result
        raise KeyError(step_id)

    def to_markdown(self) -> str:
        usage = self.usage
        lines = [
            "# Workflow report",
            "",
            f"**Objective:** {self.objective}",
            "",
            f"**Status:** {_STATUS_ICON.get(self.status, '')} {self.status} · "
            f"{len(self.results)} step(s) · {usage.llm_calls} LLM call(s) · "
            f"{usage.total_tokens:,} tokens · ${usage.cost_usd:.4f} · {self.duration_s:.1f}s",
        ]
        if self.error:
            lines += ["", f"**Error:** {self.error}"]
        lines += ["", "## Answer", "", self.answer.strip() or "_(no answer)_"]
        if self.plan is None or not self.results:
            return "\n".join(lines) + "\n"

        lines += [
            "",
            "## Steps",
            "",
            "| # | Step | Kind | Status | Evidence |",
            "|---|---|---|---|---|",
        ]
        for index, result in enumerate(self.results, 1):
            step = self.plan.get(result.step_id)
            lines.append(
                f"| {index} | `{step.id}` {step.title} | {step.kind} | "
                f"{_STATUS_ICON[result.status]} {result.status} | {len(result.evidence)} run(s) |"
            )
        for index, result in enumerate(self.results, 1):
            step = self.plan.get(result.step_id)
            lines += ["", f"### {index}. `{step.id}` {step.title}", ""]
            lines.append(
                f"_{step.kind} · {result.status} · {result.turns} turn(s) · "
                f"{result.tool_calls} tool call(s) · {result.duration_s:.1f}s_"
            )
            if result.error:
                lines += ["", f"**Error:** {result.error}"]
            if result.summary.strip():
                lines += ["", result.summary.strip()]
            if result.artifacts:
                lines += ["", "**Artifacts:** " + ", ".join(f"`{a}`" for a in result.artifacts)]
            if result.evidence:
                lines += ["", "**Execution evidence:**", *(f"- `{e}`" for e in result.evidence)]
        return "\n".join(lines) + "\n"
