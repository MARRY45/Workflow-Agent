from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import litellm
import pytest

from workflow_agent.errors import LLMError
from workflow_agent.llm import ForceTool, LiteLLMClient, RetryPolicy
from workflow_agent.messages import Message

ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
}


def _raw_response(content: Any = "ok", tool_calls: Any = None) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
        model="fake",
    )


class _Recorder:
    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.params: list[dict[str, Any]] = []

    async def __call__(self, **params: Any) -> Any:
        self.params.append(params)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


async def _no_sleep(_: float) -> None:
    return None


# --------------------------------------------------------------------- real LiteLLM (mocked I/O)


async def test_text_completion_through_real_litellm() -> None:
    client = LiteLLMClient("gpt-4o-mini", extra_params={"mock_response": "Hello there"})
    response = await client.complete([Message.user("hi")])
    assert response.message.role == "assistant"
    assert response.message.content == "Hello there"
    assert response.message.tool_calls == ()
    assert response.usage.llm_calls == 1
    assert response.usage.total_tokens > 0
    assert response.usage.cost_usd > 0  # priced from LiteLLM's local cost map


async def test_tool_call_completion_through_real_litellm() -> None:
    mock_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
        }
    ]
    client = LiteLLMClient("gpt-4o-mini", extra_params={"mock_tool_calls": mock_calls})
    response = await client.complete([Message.user("1+2?")], tools=[ADD_TOOL])
    (call,) = response.message.tool_calls
    assert call.id == "call_1"
    assert call.name == "add"
    assert call.parsed_arguments() == {"a": 1, "b": 2}


# --------------------------------------------------------------------------- request building


async def test_params_include_tools_and_forced_choice() -> None:
    fake = _Recorder(_raw_response())
    client = LiteLLMClient("m", temperature=0.2, max_tokens=50, completion_fn=fake)
    await client.complete([Message.user("x")], tools=[ADD_TOOL], tool_choice=ForceTool("add"))
    params = fake.params[0]
    assert params["model"] == "m"
    assert params["temperature"] == 0.2
    assert params["max_tokens"] == 50
    assert params["tools"] == [ADD_TOOL]
    assert params["tool_choice"] == {"type": "function", "function": {"name": "add"}}
    assert params["num_retries"] == 0
    assert params["messages"] == [{"role": "user", "content": "x"}]


async def test_tool_choice_is_omitted_without_tools() -> None:
    fake = _Recorder(_raw_response())
    client = LiteLLMClient("m", temperature=None, completion_fn=fake)
    await client.complete([Message.user("x")], tool_choice="required")
    assert "tools" not in fake.params[0]
    assert "tool_choice" not in fake.params[0]
    assert "temperature" not in fake.params[0]


# --------------------------------------------------------------------------- response parsing


async def test_list_content_and_missing_tool_call_id_are_normalized() -> None:
    raw_call = SimpleNamespace(id=None, function=SimpleNamespace(name="add", arguments=None))
    fake = _Recorder(
        _raw_response(content=[{"type": "text", "text": "a"}, "b"], tool_calls=[raw_call])
    )
    response = await LiteLLMClient("m", completion_fn=fake).complete([Message.user("x")])
    assert response.message.content == "ab"
    (call,) = response.message.tool_calls
    assert call.id.startswith("call_")
    assert call.arguments == "{}"
    assert response.usage.prompt_tokens == 3
    assert response.usage.cost_usd == 0.0  # fake response has no pricing


async def test_empty_choices_raise() -> None:
    fake = _Recorder(SimpleNamespace(choices=[]))
    with pytest.raises(LLMError, match="no choices"):
        await LiteLLMClient("m", completion_fn=fake).complete([Message.user("x")])


# --------------------------------------------------------------------------------- retries


def _rate_limit() -> Exception:
    return litellm.RateLimitError("slow down", llm_provider="openai", model="m")


async def test_transient_errors_are_retried_with_backoff() -> None:
    delays: list[float] = []

    async def record_sleep(seconds: float) -> None:
        delays.append(seconds)

    fake = _Recorder(
        _rate_limit(), litellm.Timeout("t", model="m", llm_provider="openai"), _raw_response()
    )
    client = LiteLLMClient(
        "m",
        completion_fn=fake,
        retry=RetryPolicy(max_retries=3, initial_backoff_s=1.0),
        sleep=record_sleep,
    )
    response = await client.complete([Message.user("x")])
    assert response.message.content == "ok"
    assert len(fake.params) == 3
    assert len(delays) == 2
    assert 0.0 <= delays[0] <= 1.0
    assert 0.0 <= delays[1] <= 2.0


async def test_retries_exhausted_raise_retryable_llm_error() -> None:
    fake = _Recorder(_rate_limit(), _rate_limit())
    client = LiteLLMClient(
        "m", completion_fn=fake, retry=RetryPolicy(max_retries=1), sleep=_no_sleep
    )
    with pytest.raises(LLMError, match="after 2 attempt") as info:
        await client.complete([Message.user("x")])
    assert info.value.retryable is True


async def test_non_transient_errors_fail_fast() -> None:
    fake = _Recorder(litellm.BadRequestError("bad", model="m", llm_provider="openai"))
    client = LiteLLMClient("m", completion_fn=fake, sleep=_no_sleep)
    with pytest.raises(LLMError, match="BadRequestError") as info:
        await client.complete([Message.user("x")])
    assert info.value.retryable is False
    assert len(fake.params) == 1


def test_backoff_is_capped() -> None:
    policy = RetryPolicy(initial_backoff_s=1.0, max_backoff_s=5.0)
    assert all(0.0 <= policy.backoff(10) <= 5.0 for _ in range(50))


# ------------------------------------------------------------------------------ concurrency


async def test_concurrent_requests_are_bounded() -> None:
    active = peak = 0

    async def slow(**_: Any) -> Any:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return _raw_response()

    client = LiteLLMClient("m", completion_fn=slow, max_concurrency=2)
    await asyncio.gather(*(client.complete([Message.user(str(i))]) for i in range(6)))
    assert peak == 2


def test_invalid_concurrency_rejected() -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        LiteLLMClient("m", max_concurrency=0)
