"""Workspace file tools: list, read, write."""

from __future__ import annotations

import os
from typing import Annotated

from pydantic import Field

from workflow_agent.errors import ToolError
from workflow_agent.tools.base import Tool, tool
from workflow_agent.tools.workspace import Workspace

_SKIP_DIRS = frozenset(
    {".git", ".hg", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
     ".pytest_cache", ".ruff_cache", ".tox"}
)  # fmt: skip
_BINARY_SNIFF_BYTES = 8192


def filesystem_tools(
    workspace: Workspace,
    *,
    max_read_bytes: int = 2_000_000,
    max_write_bytes: int = 1_000_000,
) -> list[Tool]:
    """Build ``list_files``, ``read_file`` and ``write_file`` bound to ``workspace``."""

    @tool(tags={"read"})
    def list_files(
        path: str = ".",
        max_entries: Annotated[int, Field(ge=1, le=1000)] = 200,
    ) -> str:
        """List files and directories in the workspace recursively.

        Args:
            path: Workspace-relative directory to list.
            max_entries: Maximum number of entries to return.
        """
        root = workspace.resolve(path)
        if not root.is_dir():
            raise ToolError(f"not a directory: {path}")
        entries: list[str] = []
        truncated = False
        for current, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
            for name in [*(f"{d}/" for d in dirs), *sorted(files)]:
                if len(entries) >= max_entries:
                    truncated = True
                    break
                full = os.path.join(current, name.rstrip("/"))
                rel = os.path.relpath(full, workspace.root).replace(os.sep, "/")
                if name.endswith("/"):
                    entries.append(f"{rel}/")
                else:
                    entries.append(f"{rel} ({os.path.getsize(full)} B)")
            if truncated:
                break
        if not entries:
            return f"{workspace.relative(root)}/ is empty"
        if truncated:
            entries.append(f"... (truncated at {max_entries} entries)")
        return "\n".join(entries)

    @tool(tags={"read"})
    def read_file(
        path: str,
        start_line: Annotated[int, Field(ge=1)] = 1,
        max_lines: Annotated[int, Field(ge=1, le=5000)] = 500,
    ) -> str:
        """Read a UTF-8 text file from the workspace.

        Args:
            path: Workspace-relative file path.
            start_line: 1-based line to start reading from.
            max_lines: Maximum number of lines to return.
        """
        target = workspace.resolve(path)
        if not target.is_file():
            raise ToolError(f"file not found: {path}")
        size = target.stat().st_size
        if size > max_read_bytes:
            raise ToolError(f"{path} is {size} bytes; the read limit is {max_read_bytes} bytes")
        data = target.read_bytes()
        if b"\x00" in data[:_BINARY_SNIFF_BYTES]:
            raise ToolError(f"{path} looks like a binary file")
        lines = data.decode("utf-8", errors="replace").splitlines()
        if not lines:
            return f"[{workspace.relative(target)}: empty file]"
        if start_line > len(lines):
            raise ToolError(f"start_line {start_line} is past the end ({len(lines)} lines)")
        end = min(start_line - 1 + max_lines, len(lines))
        body = "\n".join(lines[start_line - 1 : end])
        header = f"[{workspace.relative(target)}: lines {start_line}-{end} of {len(lines)}]"
        return f"{header}\n{body}"

    @tool(tags={"write"})
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a UTF-8 text file in the workspace (parent dirs are created).

        Args:
            path: Workspace-relative file path.
            content: Full new content of the file.
        """
        target = workspace.resolve(path)
        encoded = content.encode("utf-8")
        if len(encoded) > max_write_bytes:
            raise ToolError(f"content is {len(encoded)} bytes; the limit is {max_write_bytes}")
        if target.is_dir():
            raise ToolError(f"{path} is a directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded)
        return f"Wrote {len(encoded)} bytes to {workspace.relative(target)}"

    return [list_files, read_file, write_file]
