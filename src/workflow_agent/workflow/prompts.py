"""Prompt templates and renderers for the planner, step agents and synthesizer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from workflow_agent.textutil import truncate_middle
from workflow_agent.workflow.models import Plan, PlanStep, StepKind, StepResult

KIND_DESCRIPTIONS: Mapping[StepKind, str] = {
    "research": (
        "gather facts from the web, package indexes and workspace files; report findings "
        "with sources"
    ),
    "code": "write or modify code in the shared workspace and smoke-test it",
    "verify": (
        "independently check code or claims by *executing* something (tests, scripts) and "
        "report the exact outcome; a verify step without execution evidence is flagged"
    ),
}

PLANNER_SYSTEM_PROMPT = """\
You are the planning component of Workflow Agent, an autonomous system for multi-step \
technical research and code verification.

Decompose the user's objective into a small dependency graph (DAG) of steps. Every step \
is executed by an independent tool-using agent that sees only: the overall objective, the \
step's own instructions, and the final summaries of the steps it depends on.

Step kinds and the tools available to each:
{kinds}

Planning rules:
- Use the fewest steps that fully cover the objective (at most {max_steps}).
- Steps run in parallel unless one depends on another; declare only real data dependencies.
{verification_rule}\
- Instructions must be concrete and self-contained; say exactly what the step must deliver \
(facts with sources, file names, test results).
- Step ids are short snake_case names.

Submit the plan by calling the `submit_plan` tool."""

VERIFICATION_RULE = (
    "- Every `code` step must be checked by a `verify` step that depends on it (directly or "
    "transitively).\n"
)

STEP_SYSTEM_PROMPT = """\
You are an execution agent inside Workflow Agent, responsible for exactly ONE step of a \
larger plan.

Rules:
- Do real work with the tools. Never invent tool output, file contents, versions, URLs or \
test results.
- The workspace is shared with the other steps; always use workspace-relative paths.
- Prefer small, checkable actions. When a tool reports an error, read it and fix the root \
cause.
- {kind_guidance}
- When you are done (or the step cannot be completed), call `complete_step` exactly once, \
on its own. Its `summary` is the only thing later steps and the final report will see, so \
make it complete and self-contained: key findings with sources, decisions, file paths, \
commands you ran and their exact outcomes."""

KIND_GUIDANCE: Mapping[StepKind, str] = {
    "research": (
        "Research step: cite a source (URL or file) for every non-trivial fact, separate "
        "facts from your own inferences, and say so explicitly when something could not "
        "be found instead of guessing."
    ),
    "code": (
        "Code step: write clean, documented code into workspace files, smoke-test it with "
        "`run_python` before completing, and list every created or modified file in "
        "`artifacts`."
    ),
    "verify": (
        "Verify step: be skeptical; your job is to find defects. Write focused tests "
        "(normal, edge and error cases), RUN them, and report the exact pass/fail counts. "
        "Report status 'failed' if the code or claim does not hold. Do not change the code "
        "under test to make tests pass."
    ),
}

SYNTHESIZER_SYSTEM_PROMPT = """\
You are the reporting component of Workflow Agent. Using ONLY the step results provided, \
write the final answer to the objective in Markdown.

- Start with a direct answer or conclusion.
- Explain what was verified and how (cite the execution evidence and sources).
- Clearly flag anything that failed, was skipped or is unverified, and what that means for \
the conclusion.
- Do not introduce facts that are not supported by the step results."""


def planner_system_prompt(
    kind_tools: Mapping[StepKind, Sequence[str]], *, max_steps: int, require_verification: bool
) -> str:
    kinds = "\n".join(
        f"- {kind}: {KIND_DESCRIPTIONS[kind]}. Tools: {', '.join(tools) or '(none)'}"
        for kind, tools in kind_tools.items()
    )
    return PLANNER_SYSTEM_PROMPT.format(
        kinds=kinds,
        max_steps=max_steps,
        verification_rule=VERIFICATION_RULE if require_verification else "",
    )


def step_system_prompt(kind: StepKind) -> str:
    return STEP_SYSTEM_PROMPT.format(kind_guidance=KIND_GUIDANCE[kind])


def render_step_task(
    objective: str,
    step: PlanStep,
    dependencies: Sequence[tuple[PlanStep, StepResult]],
    *,
    max_dependency_chars: int,
) -> str:
    parts = [
        f"# Overall objective\n{objective}",
        f"# Your step: `{step.id}` ({step.kind})\n## {step.title}\n{step.instructions}",
    ]
    if dependencies:
        rendered = []
        for dep, result in dependencies:
            body = truncate_middle(result.summary.strip(), max_dependency_chars) or "(no summary)"
            extra = f"\nArtifacts: {', '.join(result.artifacts)}" if result.artifacts else ""
            rendered.append(f"## `{dep.id}`: {dep.title} [{result.status}]\n{body}{extra}")
        parts.append("# Results of the steps you depend on\n" + "\n\n".join(rendered))
    parts.append("Complete your step now, then call `complete_step`.")
    return "\n\n".join(parts)


def render_synthesis_request(
    objective: str, plan: Plan, results: Sequence[StepResult], *, max_chars_per_step: int
) -> str:
    sections = []
    for result in results:
        step = plan.get(result.step_id)
        lines = [f"## `{step.id}` ({step.kind}): {step.title}", f"Status: {result.status}"]
        if result.error:
            lines.append(f"Error: {result.error}")
        if result.evidence:
            lines.append("Execution evidence:\n" + "\n".join(f"- {e}" for e in result.evidence))
        if result.artifacts:
            lines.append(f"Artifacts: {', '.join(result.artifacts)}")
        lines.append(truncate_middle(result.summary.strip(), max_chars_per_step) or "(no summary)")
        sections.append("\n".join(lines))
    return (
        f"# Objective\n{objective}\n\n# Plan rationale\n{plan.rationale}\n\n"
        "# Step results\n\n" + "\n\n".join(sections) + "\n\nWrite the final answer now."
    )
