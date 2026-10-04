"""Extend the agent with your own tool, then run a single agent loop.

    export ANTHROPIC_API_KEY=...
    python examples/custom_tool.py --model anthropic/claude-sonnet-4-5

Any typed (sync or async) function becomes a tool: the JSON schema shown to the model is
derived from its signature and docstring, and arguments are validated before the call.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
from pathlib import Path
from typing import Annotated

from pydantic import Field

from workflow_agent import (
    Agent,
    AgentConfig,
    AgentResult,
    ConsoleReporter,
    EventBus,
    LiteLLMClient,
    LLMClient,
    ToolError,
    Workspace,
    default_registry,
    tool,
)


@tool(tags={"analysis"})
def describe_numbers(
    values: Annotated[list[float], Field(min_length=1, description="The sample.")],
    precision: Annotated[int, Field(ge=0, le=10)] = 3,
) -> dict[str, float]:
    """Compute descriptive statistics (mean, median, stdev, min, max) of a numeric sample."""
    if len(values) < 2:
        raise ToolError("need at least two values to compute a standard deviation")
    return {
        "mean": round(statistics.fmean(values), precision),
        "median": round(statistics.median(values), precision),
        "stdev": round(statistics.stdev(values), precision),
        "min": min(values),
        "max": max(values),
    }


async def run(question: str, llm: LLMClient, workspace: Path) -> AgentResult:
    tools = default_registry(Workspace(workspace), include_web=False).with_tools(describe_numbers)
    agent = Agent(llm, tools, config=AgentConfig(max_turns=6), events=EventBus([ConsoleReporter()]))
    return await agent.run(question)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gpt-4o-mini", help="LiteLLM model id")
    parser.add_argument("--workspace", type=Path, default=Path("workspace/custom_tool"))
    args = parser.parse_args()

    question = (
        "Response times in ms were 120, 98, 143, 101, 99, 250, 97. Describe them with "
        "describe_numbers, then check with run_python whether 250 is more than two standard "
        "deviations above the mean. Answer concisely."
    )
    result = asyncio.run(run(question, LiteLLMClient(args.model), args.workspace))
    print(result.output)


if __name__ == "__main__":
    main()
