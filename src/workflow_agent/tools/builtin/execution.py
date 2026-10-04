"""Code-execution tools: ``run_python`` and ``run_pytest``.

Isolation model (defence in depth, **not** a security sandbox):

* a fresh interpreter subprocess per call, ``cwd`` = workspace, no shell;
* a scrubbed environment - API keys and other secrets of the parent never leak in;
* POSIX rlimits on CPU time, address space and file size, applied by a bootstrap
  inside the child before user code runs;
* a wall-clock timeout that kills the whole process group (grandchildren included);
* bounded capture of stdout/stderr.

For untrusted workloads run the agent itself inside a container or VM.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import signal
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from workflow_agent.errors import ToolError
from workflow_agent.tools.base import Tool, ToolResult, tool
from workflow_agent.tools.workspace import Workspace

# Runs inside the child: apply rlimits, then hand over to the script or pytest.
_BOOTSTRAP = r"""
import sys

def _limit(name, value):
    try:
        import resource
    except ImportError:  # not POSIX
        return
    value = int(value)
    if value <= 0:
        return
    kind = getattr(resource, name)
    _soft, hard = resource.getrlimit(kind)
    if hard != resource.RLIM_INFINITY:
        value = min(value, hard)
    resource.setrlimit(kind, (value, value))

_, _cpu, _mem, _fsize, _mode, *_rest = sys.argv
_limit("RLIMIT_CPU", _cpu)
_limit("RLIMIT_AS", _mem)
_limit("RLIMIT_FSIZE", _fsize)
if _mode == "script":
    import runpy
    sys.argv = _rest
    runpy.run_path(_rest[0], run_name="__main__")
elif _mode == "pytest":
    import pytest
    sys.argv = ["pytest", *_rest]
    sys.exit(pytest.main(_rest))
else:
    sys.exit("unknown bootstrap mode: " + _mode)
"""

_PYTEST_EXIT_CODES = {
    0: "all tests passed",
    1: "some tests failed",
    2: "test run interrupted",
    3: "pytest internal error",
    4: "pytest usage error",
    5: "no tests collected",
}
_PYTEST_CONFIG_FILES = ("pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg")
_DRAIN_TIMEOUT_S = 2.0
_POLL_INTERVAL_S = 0.02


@dataclass(frozen=True, slots=True)
class ExecutionLimits:
    """Resource limits for executed code. ``0`` disables a limit."""

    default_timeout_s: float = 30.0
    max_timeout_s: float = 120.0
    memory_bytes: int = 1 << 30
    file_size_bytes: int = 50 << 20
    max_output_bytes: int = 64 << 10  # per stream


@dataclass(frozen=True, slots=True)
class ProcessResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float

    def describe_exit(self, timeout_s: float) -> str:
        if self.timed_out:
            return f"timed out after {timeout_s:g}s (process killed)"
        if self.exit_code is not None and self.exit_code < 0:
            try:
                name = signal.Signals(-self.exit_code).name
            except ValueError:
                name = f"signal {-self.exit_code}"
            return f"killed by {name} (exit code {self.exit_code})"
        return f"exit code {self.exit_code}"

    def render(self, headline: str) -> str:
        parts = [f"{headline} [{self.duration_s:.2f}s]"]
        parts.append("--- stdout ---\n" + (self.stdout.rstrip() or "(empty)"))
        if self.stderr.strip():
            parts.append("--- stderr ---\n" + self.stderr.rstrip())
        return "\n".join(parts)


class _CappedBuffer:
    """Accumulates a stream up to ``limit`` bytes, then keeps draining (and counting)."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.data = bytearray()
        self.dropped = 0

    async def consume(self, stream: asyncio.StreamReader) -> None:
        while chunk := await stream.read(65536):
            room = self.limit - len(self.data)
            if room > 0:
                self.data += chunk[:room]
            self.dropped += max(len(chunk) - max(room, 0), 0)

    def text(self) -> str:
        text = self.data.decode("utf-8", errors="replace")
        if self.dropped:
            text += f"\n... [{self.dropped} more bytes not captured]"
        return text


def _sandbox_env(workspace: Workspace) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": str(workspace.root),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    for key in ("TMPDIR", "SYSTEMROOT"):  # SYSTEMROOT is required on Windows
        if key in os.environ:
            env[key] = os.environ[key]
    return env


async def _wait_for_exit(proc: asyncio.subprocess.Process, timeout_s: float) -> bool:
    """Wait until the child itself has exited; ``False`` on timeout.

    ``Process.wait()`` is not used because it only resolves once every pipe is closed as
    well, so a background grandchild holding stdout would stall it until the timeout.
    ``returncode`` is set as soon as the child is reaped.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while proc.returncode is None:
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(_POLL_INTERVAL_S)
    return True


def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL the child's whole process group (POSIX) or the child itself (Windows)."""
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)  # start_new_session => pgid == pid
        elif proc.returncode is None:
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass  # already gone


async def run_sandboxed(
    mode: Literal["script", "pytest"],
    args: list[str],
    *,
    workspace: Workspace,
    timeout_s: float,
    limits: ExecutionLimits,
) -> ProcessResult:
    """Run the bootstrap in a child interpreter and capture its output."""
    cpu_seconds = math.ceil(timeout_s) + 1
    argv = [
        sys.executable, "-E", "-s", "-B", "-u", "-X", "utf8", "-c", _BOOTSTRAP,
        str(cpu_seconds), str(limits.memory_bytes), str(limits.file_size_bytes), mode, *args,
    ]  # fmt: skip
    started = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=workspace.root,
            env=_sandbox_env(workspace),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
    except OSError as exc:
        raise ToolError(f"could not start the Python subprocess: {exc}") from exc

    assert proc.stdout is not None
    assert proc.stderr is not None
    out, err = _CappedBuffer(limits.max_output_bytes), _CappedBuffer(limits.max_output_bytes)
    readers = asyncio.gather(out.consume(proc.stdout), err.consume(proc.stderr))
    try:
        timed_out = not await _wait_for_exit(proc, timeout_s)
        # Kill the group even after a normal exit: background grandchildren must not
        # outlive the call (or keep the pipes open).
        _kill_tree(proc)
        await _wait_for_exit(proc, _DRAIN_TIMEOUT_S)
        # A grandchild that escaped the group (setsid) may still hold the pipes open.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.shield(readers), _DRAIN_TIMEOUT_S)
    finally:
        _kill_tree(proc)  # covers cancellation mid-run
        readers.cancel()

    return ProcessResult(
        exit_code=proc.returncode,
        stdout=out.text(),
        stderr=err.text(),
        timed_out=timed_out,
        duration_s=time.monotonic() - started,
    )


def execution_tools(workspace: Workspace, *, limits: ExecutionLimits | None = None) -> list[Tool]:
    """Build ``run_python`` and ``run_pytest`` bound to ``workspace``."""
    lim = limits or ExecutionLimits()
    # The registry-level timeout must outlast the subprocess timeout so the subprocess
    # is killed (and its partial output reported) by us, not abandoned by the registry.
    registry_timeout = lim.max_timeout_s + 2 * _DRAIN_TIMEOUT_S + 5

    def clamp(timeout_s: float | None) -> float:
        return min(timeout_s or lim.default_timeout_s, lim.max_timeout_s)

    @tool(tags={"execution"}, timeout_s=registry_timeout)
    async def run_python(
        code: str,
        timeout_s: Annotated[float | None, Field(gt=0)] = None,
    ) -> ToolResult:
        """Execute a Python 3 script in a fresh, resource-limited subprocess.

        The working directory is the workspace, so workspace modules are importable.
        Print whatever you need to see; only stdout/stderr are returned.

        Args:
            code: Complete Python source code to run as __main__.
            timeout_s: Wall-clock limit in seconds (capped by the server).
        """
        timeout = clamp(timeout_s)
        with tempfile.TemporaryDirectory(prefix="wa-run-") as tmp:
            script = Path(tmp) / "snippet.py"
            script.write_text(code, encoding="utf-8")
            result = await run_sandboxed(
                "script", [str(script)], workspace=workspace, timeout_s=timeout, limits=lim
            )
        return ToolResult(
            content=result.render(f"run_python: {result.describe_exit(timeout)}"),
            metadata={"exit_code": result.exit_code, "timed_out": result.timed_out},
        )

    @tool(tags={"execution"}, timeout_s=registry_timeout)
    async def run_pytest(
        path: str = ".",
        extra_args: list[str] | None = None,
        timeout_s: Annotated[float | None, Field(gt=0)] = None,
    ) -> ToolResult:
        """Run pytest on a workspace file or directory and return its report.

        Args:
            path: Workspace-relative test file or directory.
            extra_args: Additional pytest arguments, e.g. ["-k", "edge_cases", "-x"].
            timeout_s: Wall-clock limit in seconds (capped by the server).
        """
        target = workspace.resolve(path)
        if not target.exists():
            raise ToolError(f"path not found: {path}")
        timeout = clamp(timeout_s)
        with tempfile.TemporaryDirectory(prefix="wa-pytest-") as tmp:
            args = ["-q", "--no-header", "-p", "no:cacheprovider", f"--rootdir={workspace.root}"]
            if not any((workspace.root / name).is_file() for name in _PYTEST_CONFIG_FILES):
                # Without a config of its own, pytest would walk up the directory tree and
                # could pick up an unrelated project's configuration.
                empty_config = Path(tmp) / "pytest.ini"
                empty_config.write_text("[pytest]\n", encoding="utf-8")
                args += ["-c", str(empty_config)]
            args += [workspace.relative(target), *(extra_args or [])]
            result = await run_sandboxed(
                "pytest", args, workspace=workspace, timeout_s=timeout, limits=lim
            )
        verdict = result.describe_exit(timeout)
        if not result.timed_out and result.exit_code in _PYTEST_EXIT_CODES:
            verdict += f": {_PYTEST_EXIT_CODES[result.exit_code]}"
        return ToolResult(
            content=result.render(f"run_pytest: {verdict}"),
            metadata={"exit_code": result.exit_code, "timed_out": result.timed_out},
        )

    return [run_python, run_pytest]
