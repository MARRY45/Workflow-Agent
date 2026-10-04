"""Built-in tools and the default tool set."""

from __future__ import annotations

from workflow_agent.tools.builtin.execution import ExecutionLimits, execution_tools
from workflow_agent.tools.builtin.filesystem import filesystem_tools
from workflow_agent.tools.builtin.web import HttpFetcher, web_tools
from workflow_agent.tools.registry import ToolRegistry
from workflow_agent.tools.workspace import Workspace


def default_registry(
    workspace: Workspace,
    *,
    fetcher: HttpFetcher | None = None,
    limits: ExecutionLimits | None = None,
    include_web: bool = True,
    default_timeout_s: float = 60.0,
    max_output_chars: int = 12_000,
) -> ToolRegistry:
    """Filesystem + execution tools, plus web research tools unless ``include_web`` is off."""
    tools = [*filesystem_tools(workspace), *execution_tools(workspace, limits=limits)]
    if include_web:
        tools += web_tools(fetcher or HttpFetcher())
    return ToolRegistry(
        tools, default_timeout_s=default_timeout_s, max_output_chars=max_output_chars
    )


__all__ = [
    "ExecutionLimits",
    "HttpFetcher",
    "default_registry",
    "execution_tools",
    "filesystem_tools",
    "web_tools",
]
