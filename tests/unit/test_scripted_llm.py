from __future__ import annotations

import pytest

from workflow_agent.errors import LLMError
from workflow_agent.llm import (
    LLMClient,
    LLMResponse,
    RecordedCall,
    ScriptedLLMClient,
    calls,
    tool_call,
)
from workflow_agent.messages import Message, Usage


async def test_script_mode_replays_in_order_and_records_calls() -> None:
    client = ScriptedLLMClient([calls(tool_call("t", {"x": 1})), "done"])
    assert isinstance(client, LLMClient)

    first = await client.complete([Message.system("sys"), Message.user("go")], tools=[_schema("t")])
    assert first.message.tool_calls[0].name == "t"
    assert first.message.tool_calls[0].parsed_arguments() == {"x": 1}
    assert first.finish_reason == "tool_calls"

    second = await client.complete([Message.user("go")])
    assert second.message.content == "done"
    assert second.finish_reason == "stop"

    recorded = client.calls[0]
    assert recorded.system_prompt == "sys"
    assert recorded.first_user_message == "go"
    assert recorded.tool_names == ["t"]
    assert recorded.turn == 0


async def test_exhausted_script_raises() -> None:
    client = ScriptedLLMClient(["only"])
    await client.complete([Message.user("a")])
    with pytest.raises(LLMError, match="exhausted"):
        await client.complete([Message.user("b")])


async def test_responder_mode_sync_and_async() -> None:
    def sync_responder(call: RecordedCall) -> str:
        return f"echo:{call.last_message.content}"

    async def async_responder(call: RecordedCall) -> LLMResponse:
        return LLMResponse(message=Message.assistant("async"), usage=Usage(llm_calls=1))

    assert (
        await ScriptedLLMClient(sync_responder).complete([Message.user("hi")])
    ).message.content == "echo:hi"
    assert (
        await ScriptedLLMClient(async_responder).complete([Message.user("hi")])
    ).message.content == "async"


async def test_non_assistant_reply_is_rejected() -> None:
    client = ScriptedLLMClient([Message.user("wrong")])
    with pytest.raises(ValueError, match="assistant"):
        await client.complete([Message.user("x")])


def test_tool_call_ids_are_unique() -> None:
    assert tool_call("a").id != tool_call("a").id
    assert tool_call("a", call_id="fixed").id == "fixed"


def _schema(name: str) -> dict[str, object]:
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
