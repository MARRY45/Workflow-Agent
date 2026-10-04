"""LiteLLM adapter: one :class:`LLMClient` implementation for 100+ providers.

Responsibilities kept here (and nowhere else):

* translating our message/tool types to LiteLLM's OpenAI-style parameters and back;
* retrying transient failures (rate limits, timeouts, 5xx) with jittered exponential backoff;
* bounding concurrent requests so parallel steps cannot stampede a provider;
* best-effort cost accounting via LiteLLM's pricing table.

``litellm`` is imported lazily because importing it takes seconds; commands that never
call a model (``--help``, ``tools``, ``demo``) stay fast.
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from workflow_agent.errors import LLMError
from workflow_agent.llm.base import LLMResponse, ToolChoice, tool_choice_to_openai
from workflow_agent.messages import Message, ToolCall, Usage

logger = logging.getLogger(__name__)

CompletionFn = Callable[..., Awaitable[Any]]
SleepFn = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Retry schedule for transient provider errors."""

    max_retries: int = 3
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 30.0

    def backoff(self, attempt: int) -> float:
        """Full-jitter exponential backoff for the 0-based ``attempt``."""
        ceiling = min(self.max_backoff_s, self.initial_backoff_s * (2**attempt))
        return random.uniform(0.0, ceiling)


def _litellm() -> Any:
    import litellm

    return litellm


def _retryable_exceptions() -> tuple[type[BaseException], ...]:
    litellm = _litellm()
    return (
        litellm.RateLimitError,
        litellm.APIConnectionError,
        # litellm.Timeout derives from openai.APITimeoutError, *not* litellm.APIConnectionError.
        litellm.Timeout,
        litellm.ServiceUnavailableError,
        litellm.InternalServerError,
    )


class LiteLLMClient:
    """Async chat client backed by ``litellm.acompletion``."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
        timeout_s: float = 120.0,
        retry: RetryPolicy | None = None,
        max_concurrency: int = 4,
        drop_params: bool = True,
        extra_params: Mapping[str, Any] | None = None,
        completion_fn: CompletionFn | None = None,
        sleep: SleepFn = asyncio.sleep,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.retry = retry or RetryPolicy()
        self.drop_params = drop_params
        self.extra_params = dict(extra_params or {})
        self._completion_fn = completion_fn
        self._sleep = sleep
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> LLMResponse:
        params = self._build_params(messages, tools, tool_choice)
        async with self._semaphore:
            raw = await self._call_with_retries(params)
        return self._parse(raw)

    # ------------------------------------------------------------------ internals

    def _build_params(
        self,
        messages: Sequence[Message],
        tools: Sequence[Mapping[str, Any]] | None,
        tool_choice: ToolChoice | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_openai() for message in messages],
            "timeout": self.timeout_s,
            # Unsupported params (e.g. temperature on some reasoning models) are dropped
            # instead of failing the request; retries are handled by us, not LiteLLM.
            "drop_params": self.drop_params,
            "num_retries": 0,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if tools:
            params["tools"] = [dict(tool) for tool in tools]
            if tool_choice is not None:
                params["tool_choice"] = tool_choice_to_openai(tool_choice)
        params.update(self.extra_params)
        return params

    async def _call_with_retries(self, params: dict[str, Any]) -> Any:
        completion = self._completion_fn or _litellm().acompletion
        retryable = _retryable_exceptions()
        attempt = 0
        while True:
            try:
                return await completion(**params)
            except retryable as exc:
                if attempt >= self.retry.max_retries:
                    raise LLMError(
                        f"{type(exc).__name__} from {self.model} after {attempt + 1} attempt(s): "
                        f"{exc}",
                        retryable=True,
                    ) from exc
                delay = self.retry.backoff(attempt)
                logger.warning(
                    "Transient LLM error (%s); retry %d/%d in %.1fs",
                    type(exc).__name__,
                    attempt + 1,
                    self.retry.max_retries,
                    delay,
                )
                await self._sleep(delay)
                attempt += 1
            except Exception as exc:
                raise LLMError(f"{type(exc).__name__} from {self.model}: {exc}") from exc

    def _parse(self, raw: Any) -> LLMResponse:
        choices = getattr(raw, "choices", None)
        if not choices:
            raise LLMError(f"{self.model} returned no choices")
        choice = choices[0]
        raw_message = choice.message

        tool_calls = tuple(
            ToolCall(
                id=getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:12]}",
                name=tc.function.name or "",
                arguments=tc.function.arguments or "{}",
            )
            for tc in (getattr(raw_message, "tool_calls", None) or ())
        )
        return LLMResponse(
            message=Message.assistant(_content_to_text(raw_message.content), tool_calls),
            usage=self._usage(raw),
            finish_reason=getattr(choice, "finish_reason", None),
            model=getattr(raw, "model", None),
        )

    def _usage(self, raw: Any) -> Usage:
        usage = getattr(raw, "usage", None)
        try:
            cost = float(_litellm().completion_cost(completion_response=raw) or 0.0)
        except Exception:  # unknown model pricing, custom endpoints, test doubles ...
            cost = 0.0
        return Usage(
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cost_usd=max(cost, 0.0),
            llm_calls=1,
        )


def _content_to_text(content: Any) -> str | None:
    """Normalize provider content (``str`` or a list of content parts) to plain text."""
    if content is None:
        return None
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        parts = [
            part.get("text", "") if isinstance(part, Mapping) else str(part) for part in content
        ]
        return "".join(parts) or None
    return str(content)
