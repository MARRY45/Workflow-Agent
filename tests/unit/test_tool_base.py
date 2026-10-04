from __future__ import annotations

from typing import Annotated, Any, Literal

import pytest
from pydantic import BaseModel, Field, ValidationError

from workflow_agent.errors import ToolDefinitionError
from workflow_agent.tools import Tool, tool
from workflow_agent.tools.base import parse_docstring


class Point(BaseModel):
    x: int
    y: int = 0


class Node(BaseModel):
    value: int
    children: list[Node] = []


def test_schema_is_derived_from_signature_and_docstring() -> None:
    @tool(tags={"demo"})
    def search(
        query: str,
        limit: Annotated[int, Field(ge=1, le=50, description="Max hits.")] = 10,
        mode: Literal["fast", "exact"] = "fast",
    ) -> str:
        """Search the index.

        Longer explanation that is still part of the summary.

        Args:
            query: What to look for. This description
                continues on a second line.
            limit: Ignored because the Annotated description wins.

        Returns:
            Matching documents.
        """
        return query

    schema = search.schema()
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "search"
    assert fn["description"].startswith("Search the index.")
    assert "Longer explanation" in fn["description"]
    assert "Returns" not in fn["description"]
    params = fn["parameters"]
    assert params["required"] == ["query"]
    assert params["additionalProperties"] is False
    assert params["properties"]["query"] == {
        "type": "string",
        "description": "What to look for. This description continues on a second line.",
    }
    assert params["properties"]["limit"]["description"] == "Max hits."
    assert params["properties"]["limit"]["maximum"] == 50
    assert params["properties"]["mode"]["enum"] == ["fast", "exact"]
    assert "title" not in params
    assert search.tags == frozenset({"demo"})


def test_nested_models_are_inlined_and_property_names_preserved() -> None:
    @tool
    def plot(point: Point, title: str, default: Point | None = None) -> str:
        """Plot a point."""
        return title

    params = plot.schema()["function"]["parameters"]
    assert "$defs" not in params
    assert params["properties"]["point"]["properties"]["x"] == {"type": "integer"}
    assert params["properties"]["title"] == {"type": "string"}  # a property called "title"
    # A property called "default" is a schema too: its $ref must be inlined.
    any_of = params["properties"]["default"]["anyOf"]
    assert {"type": "null"} in any_of
    assert any("properties" in option for option in any_of)
    assert "$ref" not in str(params)


def test_recursive_models_are_rejected() -> None:
    @tool
    def walk(tree: Node) -> int:
        """Walk a tree."""
        return tree.value

    with pytest.raises(ToolDefinitionError, match="recursive"):
        walk.schema()


@pytest.mark.parametrize(
    ("fn", "error"),
    [
        (lambda *args: None, "regular or keyword-only"),
        (lambda **kwargs: None, "regular or keyword-only"),
    ],
)
def test_variadic_parameters_are_rejected(fn: Any, error: str) -> None:
    with pytest.raises(ToolDefinitionError, match=error):
        Tool.from_function(fn, name="bad", description="x")


def test_missing_annotation_is_rejected() -> None:
    def untyped(value):  # type: ignore[no-untyped-def]
        """Untyped."""

    with pytest.raises(ToolDefinitionError, match="type annotation"):
        tool(untyped)


@pytest.mark.parametrize("name", ["", "has space", "x" * 65, "dots.not.allowed"])
def test_invalid_names_are_rejected(name: str) -> None:
    def fn() -> None:
        """Doc."""

    with pytest.raises(ToolDefinitionError, match="invalid tool name"):
        Tool.from_function(fn, name=name)


def test_description_is_required() -> None:
    def undocumented() -> None:
        pass

    with pytest.raises(ToolDefinitionError, match="description"):
        tool(undocumented)


async def test_sync_and_async_functions_run() -> None:
    @tool
    def add(a: int, b: int = 1) -> int:
        """Add."""
        return a + b

    @tool
    async def shout(text: str) -> str:
        """Shout."""
        return text.upper()

    assert await add.run(add.validate({"a": 2})) == 3
    assert await shout.run(shout.validate({"text": "hi"})) == "HI"


def test_validation_rejects_extra_and_wrong_types() -> None:
    @tool
    def add(a: int) -> int:
        """Add."""
        return a

    with pytest.raises(ValidationError):
        add.validate({"a": 1, "unexpected": True})
    with pytest.raises(ValidationError):
        add.validate({"a": "not a number"})


def test_parse_docstring_handles_sections_and_types() -> None:
    summary, params = parse_docstring(
        """Do things.

        Args:
            path (str): The path.
            *items: Ignored star prefix.
            depth: Line one
                line two.

        Raises:
            ValueError: never documented as a param.
        """
    )
    assert summary == "Do things."
    assert params == {
        "path": "The path.",
        "items": "Ignored star prefix.",
        "depth": "Line one line two.",
    }
