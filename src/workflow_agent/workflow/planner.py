"""LLM planner: objective -> validated :class:`Plan`.

Structured output is obtained through tool calling: the model is *forced* to call
``submit_plan`` whose parameter schema is the :class:`Plan` model. Validation failures
(schema errors, unknown dependencies, cycles, policy violations) are returned to the model
as the tool result so it can repair the plan, up to ``max_attempts`` times.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

from pydantic import ValidationError

from workflow_agent.budget import UsageTracker
from workflow_agent.errors import PlanningError
from workflow_agent.events import EventBus, LLMCallCompleted
from workflow_agent.llm.base import ForceTool, LLMClient
from workflow_agent.messages import Message
from workflow_agent.tools.base import Tool
from workflow_agent.tools.registry import format_validation_error
from workflow_agent.workflow.models import Plan, StepKind
from workflow_agent.workflow.prompts import planner_system_prompt

SUBMIT_PLAN = "submit_plan"
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


async def _accept(**_: object) -> str:
    return "Plan accepted."


class Planner:
    def __init__(
        self,
        llm: LLMClient,
        *,
        kind_tools: Mapping[StepKind, Sequence[str]],
        max_steps: int = 8,
        max_attempts: int = 3,
        require_verification: bool = True,
        tracker: UsageTracker | None = None,
        events: EventBus | None = None,
    ) -> None:
        if max_steps < 1 or max_attempts < 1:
            raise ValueError("max_steps and max_attempts must be >= 1")
        self.llm = llm
        self.kind_tools = kind_tools
        self.max_steps = max_steps
        self.max_attempts = max_attempts
        self.require_verification = require_verification
        self.tracker = tracker or UsageTracker()
        self.events = events or EventBus()
        self._submit_tool = Tool(
            name=SUBMIT_PLAN,
            description="Submit the complete execution plan for the objective.",
            args_model=Plan,
            fn=_accept,
        )

    async def plan(self, objective: str) -> Plan:
        system = planner_system_prompt(
            self.kind_tools,
            max_steps=self.max_steps,
            require_verification=self.require_verification,
        )
        messages = [Message.system(system), Message.user(f"Objective:\n{objective}")]
        schema = [self._submit_tool.schema()]
        problem = "no attempt made"

        for attempt in range(1, self.max_attempts + 1):
            self.tracker.check()
            response = await self.llm.complete(
                messages, tools=schema, tool_choice=ForceTool(SUBMIT_PLAN)
            )
            self.tracker.record(response.usage)
            reply = response.message
            messages.append(reply)
            await self.events.emit(
                LLMCallCompleted(
                    source="planner",
                    turn=attempt,
                    finish_reason=response.finish_reason,
                    tool_calls=[c.name for c in reply.tool_calls],
                    content=reply.content,
                    usage=response.usage,
                )
            )

            submit = next((c for c in reply.tool_calls if c.name == SUBMIT_PLAN), None)
            try:
                if submit is not None:
                    plan = Plan.model_validate(submit.parsed_arguments())
                elif reply.content and (raw := _extract_json(reply.content)) is not None:
                    plan = Plan.model_validate_json(raw)  # provider ignored tool_choice
                else:
                    raise ValueError(f"no plan received: call the `{SUBMIT_PLAN}` tool")
                self._check_policy(plan)
            except ValidationError as exc:
                problem = format_validation_error(exc)
            except ValueError as exc:
                problem = f"- {exc}"
            else:
                for call in reply.tool_calls:  # keep the transcript well-formed
                    messages.append(
                        Message.tool(tool_call_id=call.id, name=call.name, content="ok")
                    )
                return plan

            feedback = (
                f"ERROR: the plan was rejected:\n{problem}\n"
                f"Fix every problem and call `{SUBMIT_PLAN}` again with the full plan."
            )
            if reply.tool_calls:
                for call in reply.tool_calls:
                    content = feedback if call is submit else f"ERROR: unknown tool {call.name!r}"
                    messages.append(
                        Message.tool(tool_call_id=call.id, name=call.name, content=content)
                    )
            else:
                messages.append(Message.user(feedback))

        raise PlanningError(
            f"no valid plan after {self.max_attempts} attempt(s); last problem:\n{problem}"
        )

    def _check_policy(self, plan: Plan) -> None:
        if len(plan.steps) > self.max_steps:
            raise ValueError(
                f"the plan has {len(plan.steps)} steps; the maximum is {self.max_steps}"
            )
        unavailable = [s.id for s in plan.steps if s.kind not in self.kind_tools]
        if unavailable:
            raise ValueError(f"steps with a disabled kind: {', '.join(unavailable)}")
        if self.require_verification:
            verified: set[str] = set()
            for step in plan.steps:
                if step.kind == "verify":
                    verified |= plan.ancestors(step.id)
            unchecked = [s.id for s in plan.steps if s.kind == "code" and s.id not in verified]
            if unchecked:
                raise ValueError(
                    "code step(s) not covered by any verify step: "
                    f"{', '.join(unchecked)}; add a verify step that depends on each of them"
                )


def _extract_json(text: str) -> str | None:
    """Best-effort extraction of a JSON object from free text."""
    fenced = _FENCE_RE.search(text)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        json.loads(candidate)
    except ValueError:
        return None
    return candidate
