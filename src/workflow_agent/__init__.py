"""Workflow Agent: multi-step technical research and code verification with tool-calling LLMs.

Quick start::

    import asyncio
    from workflow_agent import Settings, run_workflow

    report = asyncio.run(run_workflow("Is httpx's AsyncClient safe to share between tasks?"))
    print(report.to_markdown())
"""

from __future__ import annotations

from workflow_agent._version import __version__
from workflow_agent.agent import Agent, AgentConfig, AgentResult
from workflow_agent.app import build_agent, build_engine, build_llm, build_tools, run_workflow
from workflow_agent.budget import UsageTracker
from workflow_agent.config import Settings
from workflow_agent.errors import (
    BudgetExceededError,
    ConfigurationError,
    LLMError,
    PlanningError,
    ToolDefinitionError,
    ToolError,
    WorkflowAgentError,
    WorkspaceError,
)
from workflow_agent.events import Event, EventBus
from workflow_agent.llm import LiteLLMClient, LLMClient, ScriptedLLMClient
from workflow_agent.messages import Message, ToolCall, Usage
from workflow_agent.observability import ConsoleReporter, JsonlTraceWriter
from workflow_agent.tools import Tool, ToolRegistry, ToolResult, Workspace, default_registry, tool
from workflow_agent.workflow import Plan, PlanStep, WorkflowConfig, WorkflowEngine, WorkflowReport

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResult",
    "BudgetExceededError",
    "ConfigurationError",
    "ConsoleReporter",
    "Event",
    "EventBus",
    "JsonlTraceWriter",
    "LLMClient",
    "LLMError",
    "LiteLLMClient",
    "Message",
    "Plan",
    "PlanStep",
    "PlanningError",
    "ScriptedLLMClient",
    "Settings",
    "Tool",
    "ToolCall",
    "ToolDefinitionError",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "Usage",
    "UsageTracker",
    "WorkflowAgentError",
    "WorkflowConfig",
    "WorkflowEngine",
    "WorkflowReport",
    "Workspace",
    "WorkspaceError",
    "__version__",
    "build_agent",
    "build_engine",
    "build_llm",
    "build_tools",
    "default_registry",
    "run_workflow",
    "tool",
]
