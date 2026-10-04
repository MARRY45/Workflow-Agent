"""Event sinks: a human-readable console reporter and a JSONL trace writer."""

from __future__ import annotations

import sys
from pathlib import Path
from types import TracebackType
from typing import TextIO

from workflow_agent.events import (
    AgentFinished,
    AgentStarted,
    Event,
    LLMCallCompleted,
    PlanCreated,
    StepFinished,
    StepStarted,
    ToolCallFinished,
    ToolCallStarted,
    WorkflowFinished,
    WorkflowStarted,
)
from workflow_agent.textutil import one_line

_STATUS_ICON = {"success": "✓", "unverified": "!", "failed": "✗", "skipped": "-", "partial": "!"}


class ConsoleReporter:
    """Writes one progress line per relevant event (stderr by default).

    Quiet by default: tool *results* and model text are shown only with ``verbose``,
    except tool errors, which are always shown.
    """

    def __init__(self, stream: TextIO | None = None, *, verbose: bool = False) -> None:
        self.stream = stream or sys.stderr
        self.verbose = verbose

    def __call__(self, event: Event) -> None:
        line = self.format(event)
        if line is not None:
            print(line, file=self.stream, flush=True)

    def format(self, event: Event) -> str | None:
        match event:
            case WorkflowStarted():
                return f"▶ objective: {one_line(event.objective, 200)}"
            case PlanCreated():
                lines = [f"▶ plan ({len(event.steps)} steps): {one_line(event.rationale, 160)}"]
                for index, step in enumerate(event.steps, 1):
                    after = (
                        f" (after {', '.join(step['depends_on'])})" if step["depends_on"] else ""
                    )
                    lines.append(
                        f"   {index}. [{step['kind']}] {step['id']}: {step['title']}{after}"
                    )
                return "\n".join(lines)
            case StepStarted():
                return f"→ {event.step_id} [{event.kind}] started"
            case StepFinished():
                icon = _STATUS_ICON.get(event.status, "?")
                detail = event.error or event.summary
                return (
                    f"{icon} {event.step_id}: {event.status} ({event.duration_s:.1f}s)"
                    f"{' - ' + one_line(detail, 140) if detail else ''}"
                )
            case ToolCallStarted():
                return f"   · {event.source} → {event.tool}({one_line(event.arguments, 100)})"
            case ToolCallFinished():
                if event.is_error:
                    return f"   ✗ {event.source} ← {event.tool}: {one_line(event.output, 160)}"
                if self.verbose:
                    return f"   ✓ {event.source} ← {event.tool}: {one_line(event.output, 160)}"
                return None
            case LLMCallCompleted():
                if self.verbose and event.content:
                    return (
                        f"   … {event.source} (turn {event.turn}): {one_line(event.content, 160)}"
                    )
                return None
            case AgentStarted():
                return f"▶ {event.source}: {one_line(event.task, 160)}" if self.verbose else None
            case AgentFinished():
                if event.source.startswith("step:") and not self.verbose:
                    return None
                return (
                    f"■ {event.source} finished ({event.stop_reason}) after {event.turns} turn(s)"
                )
            case WorkflowFinished():
                usage = event.usage
                return (
                    f"■ workflow {event.status} in {event.duration_s:.1f}s · "
                    f"{usage.llm_calls} LLM calls · {usage.total_tokens:,} tokens · "
                    f"${usage.cost_usd:.4f}"
                )
        return None


class JsonlTraceWriter:
    """Appends every event as one JSON line - a replayable, greppable run trace."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO | None = self.path.open("a", encoding="utf-8")

    def __call__(self, event: Event) -> None:
        if self._file is not None:
            self._file.write(event.model_dump_json() + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> JsonlTraceWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
