"""Command-line interface.

    workflow-agent run "objective" [--mode workflow|agent] [--model ...] [options]
    workflow-agent demo            # offline end-to-end demo, no API key needed
    workflow-agent tools [--json]  # inspect the tool set

Progress goes to stderr; the final report (Markdown, or JSON with ``--json``) to stdout,
so the output can be piped or redirected cleanly.

Exit codes: 0 success, 1 failed/partial run, 2 usage or configuration error, 130 interrupted.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from workflow_agent import app
from workflow_agent._version import __version__
from workflow_agent.agent.loop import AgentResult
from workflow_agent.config import Settings
from workflow_agent.errors import ConfigurationError
from workflow_agent.events import EventBus
from workflow_agent.observability import ConsoleReporter, JsonlTraceWriter
from workflow_agent.tools.workspace import Workspace
from workflow_agent.workflow.models import WorkflowReport

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

DEFAULT_DEMO_WORKSPACE = Path("workspace/demo")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="workflow-agent",
        description="Automate multi-step technical research and code verification with LLM agents.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    run = commands.add_parser("run", help="run an objective with a real LLM (via LiteLLM)")
    run.add_argument("objective", help="what to research / build / verify ('-' reads stdin)")
    run.add_argument(
        "--mode",
        choices=["workflow", "agent"],
        default="workflow",
        help="workflow: plan + parallel steps + verification (default); agent: one agent loop",
    )
    run.add_argument("-m", "--model", help="LiteLLM model id, e.g. gpt-4o-mini, anthropic/...")
    run.add_argument("--workspace", type=Path, help="directory the tools may read/write/execute in")
    run.add_argument("--max-turns", type=int, help="LLM calls per agent")
    run.add_argument("--max-plan-steps", type=int, help="maximum steps in a plan")
    run.add_argument("--concurrency", type=int, help="steps executed in parallel")
    run.add_argument("--max-cost", type=float, metavar="USD", help="abort when spend reaches this")
    run.add_argument("--max-tokens", type=int, help="abort when total tokens reach this")
    run.add_argument("--timeout", type=float, metavar="SECONDS", help="workflow execution limit")
    run.add_argument("--no-web", action="store_true", help="disable fetch_url/pypi tools")
    run.add_argument(
        "--allow-private-network",
        action="store_true",
        help="let fetch_url reach localhost/private addresses (off by default)",
    )
    _add_output_options(run)

    demo = commands.add_parser("demo", help="offline end-to-end demo (no API key needed)")
    demo.add_argument("--workspace", type=Path, default=DEFAULT_DEMO_WORKSPACE)
    _add_output_options(demo)

    tools = commands.add_parser("tools", help="list the available tools")
    tools.add_argument("--workspace", type=Path, help="workspace the tools would be bound to")
    tools.add_argument("--no-web", action="store_true", help="exclude web tools")
    tools.add_argument("--json", action="store_true", help="print full JSON schemas")
    return parser


def _add_output_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("output")
    group.add_argument("--json", action="store_true", help="print the result as JSON")
    group.add_argument("--report", type=Path, metavar="FILE", help="also write a Markdown report")
    group.add_argument("--trace", type=Path, metavar="FILE", help="append events as JSON lines")
    group.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    group.add_argument("-v", "--verbose", action="count", default=0, help="more detail (-vv debug)")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(getattr(args, "verbose", 0))
    try:
        if args.command == "tools":
            return _cmd_tools(args)
        if args.command == "demo":
            return asyncio.run(_cmd_demo(args))
        return asyncio.run(_cmd_run(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    except (ValidationError, ConfigurationError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_USAGE


# ---------------------------------------------------------------------- commands


def _cmd_tools(args: argparse.Namespace) -> int:
    settings = Settings(**_drop_none({"workspace_dir": args.workspace}))
    registry = app.build_tools(settings, include_web=not args.no_web)
    if args.json:
        print(json.dumps(registry.schemas(), indent=2))
        return EXIT_OK
    width = max(len(name) for name in registry.names)
    for tool in registry:
        summary = tool.description.splitlines()[0]
        print(f"{tool.name:<{width}}  [{', '.join(sorted(tool.tags))}]  {summary}")
    return EXIT_OK


async def _cmd_demo(args: argparse.Namespace) -> int:
    from workflow_agent.demo import run_demo

    with _event_bus(args) as events:
        report = await run_demo(args.workspace, events=events)
    if not args.quiet:
        print(f"workspace: {Workspace(args.workspace).root}", file=sys.stderr)
    return _output_report(report, args)


async def _cmd_run(args: argparse.Namespace) -> int:
    objective = (sys.stdin.read() if args.objective == "-" else args.objective).strip()
    if not objective:
        print("error: the objective is empty", file=sys.stderr)
        return EXIT_USAGE
    settings = Settings(
        **_drop_none(
            {
                "model": args.model,
                "workspace_dir": args.workspace,
                "max_agent_turns": args.max_turns,
                "max_plan_steps": args.max_plan_steps,
                "step_concurrency": args.concurrency,
                "max_cost_usd": args.max_cost,
                "max_total_tokens": args.max_tokens,
                "workflow_timeout_s": args.timeout,
                "allow_private_network": True if args.allow_private_network else None,
            }
        )
    )
    _silence_litellm()
    problem = _missing_credentials(settings.model)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return EXIT_USAGE
    tools = app.build_tools(settings, include_web=not args.no_web)
    with _event_bus(args) as events:
        if args.mode == "agent":
            agent = app.build_agent(settings, tools=tools, events=events)
            return _output_agent_result(await agent.run(objective), args)
        report = await app.build_engine(settings, tools=tools, events=events).run(objective)
    return _output_report(report, args)


# ---------------------------------------------------------------------- output


def _output_report(report: WorkflowReport, args: argparse.Namespace) -> int:
    markdown = report.to_markdown()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(markdown, encoding="utf-8")
    print(report.model_dump_json(indent=2) if args.json else markdown)
    return EXIT_OK if report.status == "success" else EXIT_FAILED


def _output_agent_result(result: AgentResult, args: argparse.Namespace) -> int:
    usage = result.usage
    if args.json:
        payload: dict[str, Any] = {
            "output": result.output,
            "stop_reason": result.stop_reason,
            "turns": result.turns,
            "usage": usage.model_dump(),
            "tool_calls": [
                {"tool": r.call.name, "is_error": r.result.is_error, "duration_s": r.duration_s}
                for r in result.tool_calls
            ],
        }
        text = json.dumps(payload, indent=2)
    else:
        text = (
            f"{result.output.strip()}\n\n---\n_{result.stop_reason} · {result.turns} turn(s) · "
            f"{len(result.tool_calls)} tool call(s) · {usage.total_tokens:,} tokens · "
            f"${usage.cost_usd:.4f}_"
        )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return EXIT_OK if result.stop_reason != "max_turns" else EXIT_FAILED


# ---------------------------------------------------------------------- plumbing


@contextlib.contextmanager
def _event_bus(args: argparse.Namespace) -> Iterator[EventBus]:
    bus = EventBus()
    with contextlib.ExitStack() as stack:
        if not args.quiet:
            bus.subscribe(ConsoleReporter(verbose=args.verbose > 0))
        if args.trace:
            bus.subscribe(stack.enter_context(JsonlTraceWriter(args.trace)))
        yield bus


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("LiteLLM", "litellm", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))


def _missing_credentials(model: str) -> str | None:
    """Fail fast on a missing provider API key instead of burning retries on it.

    Only ``*_API_KEY`` variables count: other "missing" settings (e.g. OLLAMA_API_BASE)
    usually have working defaults.
    """
    import litellm

    try:
        env = litellm.validate_environment(model=model)
    except Exception:  # unknown provider: let the request itself decide
        return None
    missing = [str(key) for key in env.get("missing_keys") or []]
    if env.get("keys_in_environment") or not missing:
        return None
    if not all(key.endswith("_API_KEY") for key in missing):
        return None
    return f"model {model!r} needs {' or '.join(missing)} to be set in the environment"


def _silence_litellm() -> None:
    """Stop LiteLLM from printing its 'Give Feedback / Get Help' banner on errors."""
    import litellm

    litellm.suppress_debug_info = True


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
