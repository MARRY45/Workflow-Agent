from __future__ import annotations

import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

from workflow_agent.llm import tool_call
from workflow_agent.tools import ExecutionLimits, ToolRegistry, Workspace, execution_tools
from workflow_agent.tools.base import ToolResult

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX rlimits / process groups")


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path / "ws")


@pytest.fixture
def registry(workspace: Workspace) -> ToolRegistry:
    limits = ExecutionLimits(
        default_timeout_s=20, max_timeout_s=30, memory_bytes=512 << 20, max_output_bytes=4000
    )
    return ToolRegistry(execution_tools(workspace, limits=limits), max_output_chars=50_000)


async def run_python(registry: ToolRegistry, code: str, **kwargs: object) -> ToolResult:
    return await registry.execute(
        tool_call("run_python", {"code": textwrap.dedent(code), **kwargs})
    )


async def test_runs_code_and_captures_streams(registry: ToolRegistry) -> None:
    result = await run_python(
        registry, "import sys\nprint('hello')\nprint('warn', file=sys.stderr)"
    )
    assert not result.is_error
    assert result.content.startswith("run_python: exit code 0")
    assert "--- stdout ---\nhello" in result.content
    assert "--- stderr ---\nwarn" in result.content
    assert result.metadata == {"exit_code": 0, "timed_out": False}


async def test_failing_code_is_a_successful_tool_call(registry: ToolRegistry) -> None:
    result = await run_python(registry, "raise ValueError('bad input')")
    assert not result.is_error  # the tool worked; the *code* failed
    assert "exit code 1" in result.content
    assert "ValueError: bad input" in result.content


async def test_cwd_is_workspace_and_its_modules_are_importable(
    registry: ToolRegistry, workspace: Workspace
) -> None:
    (workspace.root / "helper.py").write_text("VALUE = 42\n")
    result = await run_python(registry, "import os, helper\nprint(os.getcwd(), helper.VALUE)")
    assert f"{workspace.root} 42" in result.content


async def test_secrets_do_not_leak_into_the_sandbox(
    registry: ToolRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    result = await run_python(
        registry, "import os\nprint(sorted(os.environ))\nprint(os.environ.get('OPENAI_API_KEY'))"
    )
    assert "sk-secret" not in result.content
    assert "OPENAI_API_KEY" not in result.content
    assert "PYTHONPATH" not in result.content


@posix_only
async def test_timeout_kills_the_process(registry: ToolRegistry) -> None:
    started = time.monotonic()
    result = await run_python(
        registry, "import time\nprint('started', flush=True)\ntime.sleep(30)", timeout_s=1
    )
    assert time.monotonic() - started < 10
    assert "timed out after 1s" in result.content
    assert "started" in result.content  # partial output survives the kill
    assert result.metadata["timed_out"] is True


async def test_requested_timeout_is_capped(registry: ToolRegistry) -> None:
    result = await run_python(registry, "print('ok')", timeout_s=10_000)
    assert "exit code 0" in result.content


@posix_only
async def test_memory_limit_is_enforced(registry: ToolRegistry) -> None:
    result = await run_python(registry, "blob = bytearray(2 * 1024 ** 3)\nprint('allocated')")
    assert "MemoryError" in result.content
    assert "allocated" not in result.content.split("--- stdout ---")[1].split("---")[0]


@posix_only
async def test_background_grandchildren_are_killed(
    registry: ToolRegistry, workspace: Workspace
) -> None:
    code = """
        import subprocess, sys
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        open("child.pid", "w").write(str(child.pid))
        print("parent done")
    """
    started = time.monotonic()
    result = await run_python(registry, code)
    assert "parent done" in result.content
    assert time.monotonic() - started < 10
    pid = int((workspace.root / "child.pid").read_text())
    for _ in range(50):  # the kill is asynchronous; give the kernel a moment
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"grandchild {pid} survived")


async def test_output_is_capped(registry: ToolRegistry) -> None:
    result = await run_python(registry, "print('x' * 20000)")
    assert "more bytes not captured" in result.content
    assert len(result.content) < 6000


async def test_unicode_output(registry: ToolRegistry) -> None:
    result = await run_python(registry, "print('Merhaba dünya ✓')")
    assert "Merhaba dünya ✓" in result.content


# --------------------------------------------------------------------------- pytest


async def test_pytest_reports_pass_and_fail(registry: ToolRegistry, workspace: Workspace) -> None:
    (workspace.root / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (workspace.root / "test_calc.py").write_text(
        "from calc import add\n\n"
        "def test_ok():\n    assert add(1, 2) == 3\n\n"
        "def test_bad():\n    assert add(1, 1) == 3\n"
    )
    result = await registry.execute(tool_call("run_pytest", {"path": "test_calc.py"}))
    assert "exit code 1: some tests failed" in result.content
    assert "1 failed, 1 passed" in result.content

    only_ok = await registry.execute(tool_call("run_pytest", {"extra_args": ["-k", "ok"]}))
    assert "exit code 0: all tests passed" in only_ok.content


async def test_pytest_no_tests(registry: ToolRegistry, workspace: Workspace) -> None:
    (workspace.root / "empty").mkdir()
    result = await registry.execute(tool_call("run_pytest", {"path": "empty"}))
    assert "exit code 5: no tests collected" in result.content


async def test_pytest_ignores_parent_project_configuration(tmp_path: Path) -> None:
    # A hostile/unrelated config above the workspace must not influence the run.
    (tmp_path / "pytest.ini").write_text("[pytest]\naddopts = --definitely-not-an-option\n")
    workspace = Workspace(tmp_path / "ws")
    (workspace.root / "test_x.py").write_text("def test_x():\n    assert True\n")
    registry = ToolRegistry(execution_tools(workspace))
    result = await registry.execute(tool_call("run_pytest"))
    assert "exit code 0: all tests passed" in result.content


async def test_pytest_rejects_bad_paths(registry: ToolRegistry) -> None:
    missing = await registry.execute(tool_call("run_pytest", {"path": "nope.py"}))
    assert "path not found" in missing.content
    outside = await registry.execute(tool_call("run_pytest", {"path": "../"}))
    assert "outside the workspace" in outside.content


def test_interpreter_is_the_current_one() -> None:
    assert Path(sys.executable).exists()
