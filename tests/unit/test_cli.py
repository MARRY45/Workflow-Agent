from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from workflow_agent import __version__, app, cli
from workflow_agent.llm import RecordedCall, ScriptedLLMClient, calls, tool_call
from workflow_agent.llm.scripted import Reply
from workflow_agent.workflow import COMPLETE_STEP, SUBMIT_PLAN


def responder(call: RecordedCall) -> Reply:
    if SUBMIT_PLAN in call.tool_names:
        step = {
            "id": "look",
            "kind": "research",
            "title": "Look it up",
            "instructions": "Find the facts.",
            "depends_on": [],
        }
        return calls(tool_call(SUBMIT_PLAN, {"rationale": "simple", "steps": [step]}))
    if COMPLETE_STEP in call.tool_names:
        return calls(tool_call(COMPLETE_STEP, {"status": "success", "summary": "facts found"}))
    if call.tools:  # agent mode
        return "Agent answer."
    return "Synthesized answer."


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "build_llm", lambda settings: ScriptedLLMClient(responder))
    monkeypatch.setattr(cli, "_missing_credentials", lambda model: None)


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["--version"])
    assert info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_command_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main([])
    assert info.value.code == 2


def test_tools_listing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["tools", "--workspace", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "run_pytest" in out
    assert "fetch_url" in out
    assert "[execution]" in out

    assert cli.main(["tools", "--workspace", str(tmp_path), "--no-web", "--json"]) == 0
    schemas = json.loads(capsys.readouterr().out)
    names = [s["function"]["name"] for s in schemas]
    assert names == ["list_files", "read_file", "write_file", "run_python", "run_pytest"]


def test_demo_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    trace, report_md = tmp_path / "trace.jsonl", tmp_path / "out" / "report.md"
    code = cli.main(
        [
            "demo",
            "--workspace",
            str(tmp_path / "ws"),
            "--json",
            "--trace",
            str(trace),
            "--report",
            str(report_md),
            "-q",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""  # --quiet
    report = json.loads(captured.out)
    assert report["status"] == "success"
    assert [r["status"] for r in report["results"]] == ["success"] * 3
    assert (tmp_path / "ws" / "slugify.py").exists()
    assert (tmp_path / "ws" / "test_slugify.py").exists()
    types = [json.loads(line)["type"] for line in trace.read_text().splitlines()]
    assert types[0] == "workflow_started"
    assert types[-1] == "workflow_finished"
    assert "run_pytest: exit code 0" in report_md.read_text()


def test_demo_progress_goes_to_stderr(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["demo", "--workspace", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert "▶ plan (3 steps)" in captured.err
    assert "workspace:" in captured.err
    assert captured.out.startswith("# Workflow report")


@pytest.mark.usefixtures("scripted")
def test_run_workflow_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["run", "Find facts", "--workspace", str(tmp_path), "--no-web", "-q"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Synthesized answer." in out
    assert "`look` Look it up" in out


@pytest.mark.usefixtures("scripted")
def test_run_agent_mode_json_and_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("Question from stdin\n"))
    report = tmp_path / "agent.md"
    code = cli.main(
        [
            "run",
            "-",
            "--mode",
            "agent",
            "--workspace",
            str(tmp_path),
            "--json",
            "-q",
            "--report",
            str(report),
        ]
    )
    payload: dict[str, Any] = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["output"] == "Agent answer."
    assert payload["stop_reason"] == "final_answer"
    assert payload["usage"]["llm_calls"] == 1
    assert report.exists()


@pytest.mark.usefixtures("scripted")
def test_run_agent_mode_markdown(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "Q", "--mode", "agent", "--workspace", str(tmp_path), "-q"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Agent answer.")
    assert "_final_answer · 1 turn(s)" in out


@pytest.mark.usefixtures("scripted")
def test_agent_turn_limit_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    looping = ScriptedLLMClient(lambda call: calls(tool_call("list_files")))
    monkeypatch.setattr(app, "build_llm", lambda settings: looping)
    code = cli.main(
        ["run", "Q", "--mode", "agent", "--max-turns", "2", "--workspace", str(tmp_path), "-q"]
    )
    assert code == 1
    assert "max_turns" in capsys.readouterr().out


@pytest.mark.usefixtures("scripted")
@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["run", "   "], "objective is empty"),
        (["run", "x", "--max-plan-steps", "0"], "configuration error"),
    ],
)
def test_usage_errors(
    argv: list[str], message: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main([*argv, "--workspace", str(tmp_path)]) == 2
    assert message in capsys.readouterr().err


def test_missing_credentials_fail_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli, "_missing_credentials", lambda model: f"model {model!r} needs X_API_KEY"
    )
    assert cli.main(["run", "x", "-m", "some/model", "--workspace", str(tmp_path)]) == 2
    assert "needs X_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"keys_in_environment": True, "missing_keys": []}, None),
        (
            {"keys_in_environment": False, "missing_keys": ["OPENAI_API_KEY"]},
            "needs OPENAI_API_KEY",
        ),
        (
            {"keys_in_environment": False, "missing_keys": ["GOOGLE_API_KEY", "GEMINI_API_KEY"]},
            "GOOGLE_API_KEY or GEMINI_API_KEY",
        ),
        (
            {"keys_in_environment": False, "missing_keys": ["OLLAMA_API_BASE"]},
            None,
        ),  # has a default
        ({"keys_in_environment": False, "missing_keys": []}, None),
    ],
)
def test_missing_credentials_rules(
    env: dict[str, Any], expected: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litellm

    monkeypatch.setattr(litellm, "validate_environment", lambda model: env)
    result = cli._missing_credentials("m")
    assert result == expected if expected is None else expected in (result or "")


def test_missing_credentials_tolerates_unknown_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    def boom(model: str) -> dict[str, Any]:
        raise ValueError("unknown provider")

    monkeypatch.setattr(litellm, "validate_environment", boom)
    assert cli._missing_credentials("m") is None


def test_keyboard_interrupt_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def interrupted(args: Any) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_cmd_demo", interrupted)
    assert cli.main(["demo"]) == 130
    assert "interrupted" in capsys.readouterr().err
