from __future__ import annotations

import pytest

from workflow_agent.messages import Message, ToolCall, Usage


def test_assistant_with_tool_calls_serializes_to_openai_format() -> None:
    call = ToolCall(id="c1", name="read_file", arguments='{"path": "a.py"}')
    data = Message.assistant("thinking", (call,)).to_openai()
    assert data == {
        "role": "assistant",
        "content": "thinking",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
            }
        ],
    }


def test_tool_message_serialization_includes_call_id_and_name() -> None:
    data = Message.tool(tool_call_id="c1", name="read_file", content="ok").to_openai()
    assert data == {"role": "tool", "content": "ok", "tool_call_id": "c1", "name": "read_file"}


def test_system_and_user_messages() -> None:
    assert Message.system("s").to_openai() == {"role": "system", "content": "s"}
    assert Message.user("u").to_openai() == {"role": "user", "content": "u"}


@pytest.mark.parametrize("raw", ["", "  ", "{}"])
def test_empty_arguments_parse_to_empty_dict(raw: str) -> None:
    assert ToolCall(id="x", name="t", arguments=raw).parsed_arguments() == {}


@pytest.mark.parametrize(
    ("raw", "error"),
    [("[1, 2]", "JSON object"), ('"text"', "JSON object"), ("{not json", "Expecting")],
)
def test_invalid_arguments_raise_value_error(raw: str, error: str) -> None:
    # json.JSONDecodeError is a ValueError subclass, so callers handle one exception type.
    with pytest.raises(ValueError, match=error):
        ToolCall(id="x", name="t", arguments=raw).parsed_arguments()


def test_usage_addition() -> None:
    total = Usage(prompt_tokens=10, completion_tokens=2, cost_usd=0.5, llm_calls=1) + Usage(
        prompt_tokens=5, completion_tokens=3, cost_usd=0.25, llm_calls=2
    )
    assert total == Usage(prompt_tokens=15, completion_tokens=5, cost_usd=0.75, llm_calls=3)
    assert total.total_tokens == 20
