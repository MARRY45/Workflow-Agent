"""Run a full research -> code -> verify workflow against a real model.

    export OPENAI_API_KEY=...            # or any provider LiteLLM supports
    python examples/research_and_verify.py --model gpt-4o-mini

The report is printed as Markdown; every step's progress is streamed to stderr and the
full event trace is written to ``traces/research_and_verify.jsonl``.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from workflow_agent import (
    ConsoleReporter,
    EventBus,
    JsonlTraceWriter,
    LLMClient,
    Settings,
    WorkflowReport,
    build_engine,
    build_llm,
)

OBJECTIVE = """\
Find the latest released version of the `packaging` library on PyPI and how its
`packaging.version.Version` orders pre-releases (e.g. 1.0rc1) relative to final and
post-releases (1.0, 1.0.post1). Then write `sort_versions(versions: list[str]) -> list[str]`
in `versions.py` that sorts version strings according to PEP 440 using that library, and
verify it with pytest, including pre-release, post-release and invalid-input cases."""


async def run(
    objective: str, settings: Settings, *, llm: LLMClient | None = None
) -> WorkflowReport:
    with JsonlTraceWriter(Path("traces/research_and_verify.jsonl")) as trace:
        events = EventBus([ConsoleReporter(), trace])
        engine = build_engine(settings, llm=llm or build_llm(settings), events=events)
        return await engine.run(objective)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None, help="LiteLLM model id")
    parser.add_argument("--workspace", type=Path, default=Path("workspace/research_and_verify"))
    parser.add_argument("--max-cost", type=float, default=0.50, help="USD budget")
    args = parser.parse_args()

    overrides = {"workspace_dir": args.workspace, "max_cost_usd": args.max_cost}
    if args.model:
        overrides["model"] = args.model
    report = asyncio.run(run(OBJECTIVE, Settings(**overrides)))
    print(report.to_markdown())


if __name__ == "__main__":
    main()
