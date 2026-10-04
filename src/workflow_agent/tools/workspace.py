"""The workspace: a directory that confines every file the tools read, write or execute."""

from __future__ import annotations

from pathlib import Path

from workflow_agent.errors import ConfigurationError, WorkspaceError


class Workspace:
    """Resolves model-supplied paths and guarantees they stay inside :attr:`root`.

    Paths are resolved *after* following symlinks, so neither ``../`` sequences, absolute
    paths nor symlinks pointing outside the root can escape it.
    """

    def __init__(self, root: str | Path, *, create: bool = True) -> None:
        path = Path(root).expanduser()
        if create and not path.is_dir():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ConfigurationError(f"cannot create workspace {path}: {exc}") from exc
        if not path.is_dir():
            raise ConfigurationError(f"workspace {path} is not a directory")
        self.root = path.resolve()

    def __repr__(self) -> str:
        return f"Workspace({str(self.root)!r})"

    def resolve(self, path: str) -> Path:
        if "\x00" in path:
            raise WorkspaceError("path contains a NUL byte")
        candidate = (self.root / path).resolve()
        if not candidate.is_relative_to(self.root):
            raise WorkspaceError(f"path {path!r} is outside the workspace")
        return candidate

    def relative(self, path: Path) -> str:
        """Workspace-relative POSIX representation of ``path`` (``.`` for the root)."""
        return path.resolve().relative_to(self.root).as_posix() or "."
