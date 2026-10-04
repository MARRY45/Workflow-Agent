"""Offline end-to-end demo: a scripted model drives the *real* tools.

No API key and no network are needed. A deterministic stand-in plays the planner, the step
agents and the synthesizer, but everything it triggers is real: files are written to the
workspace, the implementation is executed, and pytest runs in the sandbox. The verify
step's verdict is derived from the actual pytest result, never assumed.
"""

from __future__ import annotations

import re
from pathlib import Path

from workflow_agent.events import EventBus
from workflow_agent.llm.scripted import RecordedCall, Reply, ScriptedLLMClient, calls, tool_call
from workflow_agent.tools.builtin import default_registry
from workflow_agent.tools.workspace import Workspace
from workflow_agent.workflow.engine import COMPLETE_STEP, WorkflowEngine
from workflow_agent.workflow.models import WorkflowReport
from workflow_agent.workflow.planner import SUBMIT_PLAN

DEMO_OBJECTIVE = (
    "Implement a `slugify(text, max_length=None)` function that satisfies SPEC.md, "
    "and prove with executed tests that it meets every requirement."
)

SPEC = """\
# slugify - specification

1. The result contains only lowercase ASCII letters, digits and single hyphens.
2. Accented Latin letters are transliterated (é -> e, ü -> u, ş -> s) via Unicode NFKD.
3. Every run of other characters (spaces, punctuation, symbols) becomes one hyphen.
4. The result never starts or ends with a hyphen.
5. `max_length` truncates the slug without leaving a trailing hyphen; values < 1 raise ValueError.
6. Input without any letter or digit raises ValueError.
"""

IMPLEMENTATION = '''\
"""URL slug generation (see SPEC.md)."""

from __future__ import annotations

import re
import unicodedata

_SEPARATORS = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_length: int | None = None) -> str:
    """Convert ``text`` into a URL slug as specified in SPEC.md."""
    if max_length is not None and max_length < 1:
        raise ValueError("max_length must be >= 1")
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = _SEPARATORS.sub("-", ascii_text.lower()).strip("-")
    if not slug:
        raise ValueError(f"cannot build a slug from {text!r}")
    if max_length is not None:
        slug = slug[:max_length].rstrip("-")
    return slug
'''

SMOKE_TEST = 'from slugify import slugify\nprint(slugify("Hello, Workflow Agent!"))\n'

TESTS = """\
import pytest

from slugify import slugify


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello World", "hello-world"),
        ("  many   spaces  ", "many-spaces"),
        ("Crème Brûlée", "creme-brulee"),
        ("Göztepe Üsküdar", "goztepe-uskudar"),
        ("C++ / Python 3.12!", "c-python-3-12"),
        ("--already-slugged--", "already-slugged"),
    ],
)
def test_examples(text, expected):
    assert slugify(text) == expected


def test_output_alphabet():
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-")
    slug = slugify("Ünïcödé ✓ Tést 42")
    assert set(slug) <= allowed
    assert "--" not in slug


def test_max_length_does_not_leave_a_trailing_hyphen():
    assert slugify("hello world", max_length=6) == "hello"


def test_max_length_must_be_positive():
    with pytest.raises(ValueError):
        slugify("anything", max_length=0)


@pytest.mark.parametrize("text", ["", "   ", "!!! ???"])
def test_rejects_input_without_letters_or_digits(text):
    with pytest.raises(ValueError):
        slugify(text)
"""

_STEP_RE = re.compile(r"# Your step: `(\w+)`")
_PLAN = {
    "rationale": (
        "Extract the requirements from SPEC.md, implement them, then verify the "
        "implementation independently with executed pytest tests."
    ),
    "steps": [
        {
            "id": "read_spec",
            "kind": "research",
            "title": "Extract the slugify requirements",
            "instructions": "Read SPEC.md and list every testable requirement verbatim.",
            "depends_on": [],
        },
        {
            "id": "implement",
            "kind": "code",
            "title": "Implement slugify",
            "instructions": "Write slugify.py satisfying all requirements and smoke-test it.",
            "depends_on": ["read_spec"],
        },
        {
            "id": "verify",
            "kind": "verify",
            "title": "Verify slugify against the spec",
            "instructions": (
                "Write test_slugify.py covering every requirement (normal, edge and error "
                "cases), run it with pytest and report the exact outcome."
            ),
            "depends_on": ["implement"],
        },
    ],
}


def _last_tool_output(call: RecordedCall) -> str:
    return call.last_message.content or "" if call.last_message.role == "tool" else ""


def _complete(status: str, summary: str, artifacts: list[str] | None = None) -> Reply:
    args = {"status": status, "summary": summary, "artifacts": artifacts or []}
    return calls(tool_call(COMPLETE_STEP, args))


class DemoModel:
    """Deterministic responder standing in for an LLM (see module docstring)."""

    def __call__(self, call: RecordedCall) -> Reply:
        if SUBMIT_PLAN in call.tool_names:
            return calls(tool_call(SUBMIT_PLAN, _PLAN))
        if not call.tools:
            return self._synthesize(call.first_user_message)
        match = _STEP_RE.search(call.first_user_message)
        if match is None:
            raise ValueError("DemoModel: unrecognised request")
        handler = getattr(self, f"_step_{match.group(1)}")
        reply: Reply = handler(call.turn, _last_tool_output(call))
        return reply

    def _step_read_spec(self, turn: int, output: str) -> Reply:
        if turn == 0:
            return calls(tool_call("read_file", {"path": "SPEC.md"}))
        requirements = [line for line in output.splitlines() if re.match(r"^\d+\.", line)]
        summary = "Requirements from SPEC.md:\n" + "\n".join(requirements)
        return _complete("success", summary, [])

    def _step_implement(self, turn: int, output: str) -> Reply:
        if turn == 0:
            return calls(tool_call("write_file", {"path": "slugify.py", "content": IMPLEMENTATION}))
        if turn == 1:
            return calls(tool_call("run_python", {"code": SMOKE_TEST}))
        smoke_ok = "exit code 0" in output and "hello-workflow-agent" in output
        smoke = "hello-workflow-agent" if smoke_ok else "FAILED"
        summary = (
            "Implemented `slugify(text, max_length=None)` in slugify.py using NFKD "
            "transliteration, a single separator regex and hyphen trimming. Smoke test "
            f"`slugify('Hello, Workflow Agent!')` -> {smoke}."
        )
        return _complete("success" if smoke_ok else "failed", summary, ["slugify.py"])

    def _step_verify(self, turn: int, output: str) -> Reply:
        if turn == 0:
            return calls(tool_call("write_file", {"path": "test_slugify.py", "content": TESTS}))
        if turn == 1:
            return calls(tool_call("run_pytest", {"path": "test_slugify.py"}))
        passed = "all tests passed" in output
        counts = [ln for ln in output.splitlines() if re.search(r"\b\d+ (passed|failed|error)", ln)]
        verdict = counts[-1].strip(" =") if counts else "no pytest summary found"
        summary = (
            "Wrote test_slugify.py (examples incl. accented input, output alphabet, "
            f"max_length behaviour, error cases) and ran pytest: {verdict}."
        )
        return _complete("success" if passed else "failed", summary, ["test_slugify.py"])

    def _synthesize(self, request: str) -> str:
        statuses = re.findall(r"^Status: (\w+)", request, re.MULTILINE)
        evidence = re.findall(r"^- ((?:run_python|run_pytest): .+)$", request, re.MULTILINE)
        verified = bool(statuses) and all(s == "success" for s in statuses)
        lines = [
            "**`slugify()` is implemented in `slugify.py` and "
            + ("meets every requirement of SPEC.md.**" if verified else "is NOT fully verified.**"),
            "",
            "Execution evidence:",
            *(f"- `{item}`" for item in evidence),
        ]
        return "\n".join(lines)


async def run_demo(workspace_dir: str | Path, *, events: EventBus | None = None) -> WorkflowReport:
    """Seed the workspace with SPEC.md and run the demo workflow."""
    workspace = Workspace(workspace_dir)
    (workspace.root / "SPEC.md").write_text(SPEC, encoding="utf-8")
    engine = WorkflowEngine(
        ScriptedLLMClient(DemoModel()),
        default_registry(workspace, include_web=False),
        events=events,
    )
    return await engine.run(DEMO_OBJECTIVE)
