"""Composition root: wires settings into ready-to-use LLM clients, tools, agents and engines.

Library users who want full control construct the pieces directly; these helpers encode
the sensible defaults used by the CLI.
"""

from __future__ import annotations

from workflow_agent.agent.loop import Agent, AgentConfig
from workflow_agent.budget import UsageTracker
from workflow_agent.config import Settings
from workflow_agent.events import EventBus
from workflow_agent.llm.base import LLMClient
from workflow_agent.llm.litellm_client import LiteLLMClient, RetryPolicy
from workflow_agent.tools.builtin import ExecutionLimits, HttpFetcher, default_registry
from workflow_agent.tools.registry import ToolRegistry
from workflow_agent.tools.workspace import Workspace
from workflow_agent.workflow.engine import WorkflowConfig, WorkflowEngine
from workflow_agent.workflow.models import WorkflowReport


def build_llm(settings: Settings) -> LiteLLMClient:
    return LiteLLMClient(
        settings.model,
        temperature=settings.temperature,
        timeout_s=settings.request_timeout_s,
        retry=RetryPolicy(max_retries=settings.max_retries),
        max_concurrency=settings.max_concurrent_llm_requests,
    )


def build_tools(
    settings: Settings, *, workspace: Workspace | None = None, include_web: bool = True
) -> ToolRegistry:
    return default_registry(
        workspace or Workspace(settings.workspace_dir),
        fetcher=HttpFetcher(allow_private_network=settings.allow_private_network),
        limits=ExecutionLimits(),
        include_web=include_web,
        default_timeout_s=settings.tool_timeout_s,
        max_output_chars=settings.max_tool_output_chars,
    )


def build_tracker(settings: Settings) -> UsageTracker:
    return UsageTracker(
        max_total_tokens=settings.max_total_tokens, max_cost_usd=settings.max_cost_usd
    )


def build_engine(
    settings: Settings,
    *,
    llm: LLMClient | None = None,
    tools: ToolRegistry | None = None,
    events: EventBus | None = None,
) -> WorkflowEngine:
    return WorkflowEngine(
        llm or build_llm(settings),
        tools if tools is not None else build_tools(settings),
        config=WorkflowConfig(
            max_plan_steps=settings.max_plan_steps,
            step_concurrency=settings.step_concurrency,
            agent_max_turns=settings.max_agent_turns,
            timeout_s=settings.workflow_timeout_s,
        ),
        tracker=build_tracker(settings),
        events=events,
    )


def build_agent(
    settings: Settings,
    *,
    llm: LLMClient | None = None,
    tools: ToolRegistry | None = None,
    events: EventBus | None = None,
) -> Agent:
    return Agent(
        llm or build_llm(settings),
        tools if tools is not None else build_tools(settings),
        config=AgentConfig(max_turns=settings.max_agent_turns),
        tracker=build_tracker(settings),
        events=events,
    )


async def run_workflow(
    objective: str, settings: Settings | None = None, *, events: EventBus | None = None
) -> WorkflowReport:
    """One-call entry point: plan, execute and report on ``objective``."""
    return await build_engine(settings or Settings(), events=events).run(objective)
