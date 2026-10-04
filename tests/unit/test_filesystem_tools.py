from __future__ import annotations

from pathlib import Path

import pytest

from workflow_agent.errors import ConfigurationError, WorkspaceError
from workflow_agent.llm import tool_call
from workflow_agent.tools import ToolRegistry, Workspace, filesystem_tools


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path / "ws")


@pytest.fixture
def registry(workspace: Workspace) -> ToolRegistry:
    return ToolRegistry(filesystem_tools(workspace, max_read_bytes=1000, max_write_bytes=100))


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "a/../../x", "nul\x00byte"])
def test_workspace_rejects_escapes(workspace: Workspace, path: str) -> None:
    with pytest.raises(WorkspaceError):
        workspace.resolve(path)


def test_workspace_rejects_symlink_escape(workspace: Workspace, tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    (workspace.root / "link").symlink_to(secret)
    with pytest.raises(WorkspaceError, match="outside the workspace"):
        workspace.resolve("link")


def test_workspace_must_be_a_directory(tmp_path: Path) -> None:
    file = tmp_path / "file"
    file.write_text("x")
    with pytest.raises(ConfigurationError):
        Workspace(file)
    with pytest.raises(ConfigurationError):
        Workspace(tmp_path / "missing", create=False)


async def test_write_read_and_list(registry: ToolRegistry, workspace: Workspace) -> None:
    written = await registry.execute(
        tool_call("write_file", {"path": "pkg/mod.py", "content": "a\nb\nc\n"})
    )
    assert written.content == "Wrote 6 bytes to pkg/mod.py"
    assert (workspace.root / "pkg" / "mod.py").read_text() == "a\nb\nc\n"

    full = await registry.execute(tool_call("read_file", {"path": "pkg/mod.py"}))
    assert full.content == "[pkg/mod.py: lines 1-3 of 3]\na\nb\nc"

    window = await registry.execute(
        tool_call("read_file", {"path": "pkg/mod.py", "start_line": 2, "max_lines": 1})
    )
    assert window.content == "[pkg/mod.py: lines 2-2 of 3]\nb"

    (workspace.root / "__pycache__").mkdir()
    (workspace.root / "__pycache__" / "junk.pyc").write_bytes(b"\x00")
    listing = await registry.execute(tool_call("list_files"))
    assert listing.content.splitlines() == ["pkg/", "pkg/mod.py (6 B)"]


async def test_read_errors(registry: ToolRegistry, workspace: Workspace) -> None:
    (workspace.root / "bin.dat").write_bytes(b"abc\x00def")
    (workspace.root / "big.txt").write_text("x" * 2000)
    (workspace.root / "short.txt").write_text("one line")

    cases = {
        "missing.txt": "file not found",
        "bin.dat": "binary",
        "big.txt": "read limit",
        "../escape": "outside the workspace",
    }
    for path, message in cases.items():
        result = await registry.execute(tool_call("read_file", {"path": path}))
        assert result.is_error, path
        assert message in result.content, path

    past_end = await registry.execute(
        tool_call("read_file", {"path": "short.txt", "start_line": 5})
    )
    assert "past the end" in past_end.content


async def test_write_limits(registry: ToolRegistry, workspace: Workspace) -> None:
    too_big = await registry.execute(
        tool_call("write_file", {"path": "a.txt", "content": "x" * 101})
    )
    assert too_big.is_error
    assert "limit" in too_big.content
    (workspace.root / "dir").mkdir()
    onto_dir = await registry.execute(tool_call("write_file", {"path": "dir", "content": "x"}))
    assert "is a directory" in onto_dir.content


async def test_list_files_edge_cases(registry: ToolRegistry, workspace: Workspace) -> None:
    empty = await registry.execute(tool_call("list_files"))
    assert empty.content == "./ is empty"
    for i in range(5):
        (workspace.root / f"f{i}.txt").write_text("x")
    limited = await registry.execute(tool_call("list_files", {"max_entries": 2}))
    assert limited.content.splitlines()[-1] == "... (truncated at 2 entries)"
    not_dir = await registry.execute(tool_call("list_files", {"path": "f0.txt"}))
    assert "not a directory" in not_dir.content
