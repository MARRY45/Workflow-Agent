"""Tool abstraction: typed Python callables exposed to the model through JSON Schema.

A tool's argument schema is derived from its signature (type hints + defaults + docstring),
so the schema the model sees and the validation applied to its arguments can never drift
apart: both come from the same pydantic model.

    @tool(tags={"read"})
    def read_file(path: str, max_lines: int = 500) -> str:
        '''Read a text file from the workspace.

        Args:
            path: Workspace-relative file path.
            max_lines: Maximum number of lines to return.
        '''
"""

from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any, get_args, get_origin, get_type_hints, overload

from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo

from workflow_agent.errors import ToolDefinitionError

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SECTION_RE = re.compile(r"^(Args|Arguments|Parameters|Returns|Raises|Yields|Examples?|Notes?):$")
_PARAM_RE = re.compile(r"^(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")
# Keywords whose values are data, not sub-schemas; never rewrite inside them.
_DATA_KEYS = frozenset({"default", "examples", "enum", "const"})
# Keywords mapping arbitrary names to sub-schemas.
_SCHEMA_MAPS = frozenset({"properties", "patternProperties", "$defs", "definitions"})


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Outcome of a tool call as shown to the model.

    ``is_error`` marks *tool* failures (bad arguments, timeouts, exceptions). A script that
    ran and exited non-zero is a successful tool call whose content reports the failure.
    """

    content: str
    is_error: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def error(cls, message: str, **metadata: Any) -> ToolResult:
        return cls(content=f"ERROR: {message}", is_error=True, metadata=metadata)


class Tool:
    """A named, schema-validated capability the model can invoke."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        args_model: type[BaseModel],
        fn: Callable[..., Any],
        timeout_s: float | None = None,
        tags: Iterable[str] = (),
        terminal: bool = False,
    ) -> None:
        if not _NAME_RE.fullmatch(name):
            raise ToolDefinitionError(f"invalid tool name {name!r}: use [A-Za-z0-9_-]{{1,64}}")
        if not description.strip():
            raise ToolDefinitionError(f"tool {name!r} needs a description")
        if timeout_s is not None and timeout_s <= 0:
            raise ToolDefinitionError(f"tool {name!r}: timeout_s must be positive")
        self.name = name
        self.description = description.strip()
        self.args_model = args_model
        self.fn = fn
        self.timeout_s = timeout_s
        self.tags = frozenset(tags)
        self.terminal = terminal
        self._is_async = inspect.iscoroutinefunction(fn)
        self._schema: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r}, tags={sorted(self.tags)})"

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        timeout_s: float | None = None,
        tags: Iterable[str] = (),
        terminal: bool = False,
    ) -> Tool:
        tool_name = fn.__name__ if name is None else name
        summary, param_docs = parse_docstring(inspect.getdoc(fn) or "")
        try:
            hints = get_type_hints(fn, include_extras=True)
        except Exception as exc:  # unresolvable forward references
            raise ToolDefinitionError(f"cannot resolve type hints of {tool_name!r}: {exc}") from exc

        fields: dict[str, Any] = {}
        for param in inspect.signature(fn).parameters.values():
            if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                raise ToolDefinitionError(
                    f"tool {tool_name!r}: parameter {param.name!r} must be a regular or "
                    "keyword-only parameter (no *args, **kwargs or positional-only)"
                )
            if param.name not in hints:
                raise ToolDefinitionError(
                    f"tool {tool_name!r}: parameter {param.name!r} needs a type annotation"
                )
            annotation = hints[param.name]
            default = ... if param.default is param.empty else param.default
            doc = param_docs.get(param.name)
            if doc and not _has_description(annotation):
                fields[param.name] = (annotation, Field(default=default, description=doc))
            else:
                fields[param.name] = (annotation, default)

        args_model: type[BaseModel] = create_model(
            f"{_camel(tool_name)}Args",
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )
        return cls(
            name=tool_name,
            description=description or summary or "",
            args_model=args_model,
            fn=fn,
            timeout_s=timeout_s,
            tags=tags,
            terminal=terminal,
        )

    # ------------------------------------------------------------------ usage

    def schema(self) -> dict[str, Any]:
        """OpenAI function-tool schema (``$ref``s inlined, titles stripped)."""
        if self._schema is None:
            parameters = _strip_titles(_inline_refs(self.args_model.model_json_schema()))
            self._schema = {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": parameters,
                },
            }
        return self._schema

    def validate(self, arguments: Mapping[str, Any]) -> BaseModel:
        """Validate raw arguments; raises ``pydantic.ValidationError``."""
        return self.args_model.model_validate(arguments)

    async def run(self, args: BaseModel) -> Any:
        """Invoke the underlying callable with already-validated arguments.

        Sync callables run in a worker thread so they never block the event loop. Note that
        a thread cannot be cancelled: on timeout it keeps running in the background.
        """
        kwargs = {name: getattr(args, name) for name in type(args).model_fields}
        if self._is_async:
            return await self.fn(**kwargs)
        return await asyncio.to_thread(self.fn, **kwargs)


@overload
def tool(fn: Callable[..., Any], /) -> Tool: ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    timeout_s: float | None = None,
    tags: Iterable[str] = (),
    terminal: bool = False,
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    timeout_s: float | None = None,
    tags: Iterable[str] = (),
    terminal: bool = False,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Decorator turning a (sync or async) function into a :class:`Tool`."""

    def decorate(func: Callable[..., Any]) -> Tool:
        return Tool.from_function(
            func,
            name=name,
            description=description,
            timeout_s=timeout_s,
            tags=tags,
            terminal=terminal,
        )

    return decorate(fn) if fn is not None else decorate


# ---------------------------------------------------------------------- helpers


def parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into (summary, {param: description})."""
    summary: list[str] = []
    params: dict[str, str] = {}
    section: str | None = None
    param_indent: int | None = None
    current: str | None = None

    for line in doc.splitlines():
        stripped = line.strip()
        header = _SECTION_RE.match(stripped)
        if header:
            section = "args" if header.group(1) in ("Args", "Arguments", "Parameters") else "other"
            param_indent, current = None, None
            continue
        if section is None:
            summary.append(line)
            continue
        if section != "args" or not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        match = _PARAM_RE.match(stripped)
        if match and (param_indent is None or indent <= param_indent):
            param_indent = indent
            current = match.group(1).lstrip("*")
            params[current] = match.group(2).strip()
        elif current is not None:
            params[current] = f"{params[current]} {stripped}".strip()

    return "\n".join(summary).strip(), params


def _has_description(annotation: Any) -> bool:
    if get_origin(annotation) is not Annotated:
        return False
    return any(
        isinstance(meta, FieldInfo) and meta.description for meta in get_args(annotation)[1:]
    )


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[_\-]", name) if part) or "Tool"


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace local ``$ref``s with their definitions.

    Several providers reject ``$defs``/``$ref`` in tool parameters, so the schema is
    flattened. Recursive models cannot be flattened and are rejected.
    """
    defs: dict[str, Any] = schema.get("$defs", {})

    def resolve(node: Any, seen: frozenset[str]) -> Any:
        if isinstance(node, list):
            return [resolve(item, seen) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            key = ref.removeprefix("#/$defs/")
            if key in seen:
                raise ToolDefinitionError(f"recursive schema {key!r} is not supported")
            if key not in defs:
                raise ToolDefinitionError(f"unresolvable schema reference {ref!r}")
            target = resolve(defs[key], seen | {key})
            siblings = {k: v for k, v in node.items() if k != "$ref"}
            return {**target, **_map_schema(siblings, lambda sub: resolve(sub, seen))}
        return _map_schema(
            {k: v for k, v in node.items() if k != "$defs"}, lambda sub: resolve(sub, seen)
        )

    result: dict[str, Any] = resolve(schema, frozenset())
    return result


def _strip_titles(node: Any) -> Any:
    """Drop pydantic's auto-generated ``title`` annotations (noise for the model)."""
    if isinstance(node, list):
        return [_strip_titles(item) for item in node]
    if not isinstance(node, dict):
        return node
    return _map_schema({k: v for k, v in node.items() if k != "title"}, _strip_titles)


def _map_schema(node: dict[str, Any], fn: Callable[[Any], Any]) -> dict[str, Any]:
    """Apply ``fn`` to every sub-schema of one JSON-Schema object.

    Name-to-schema maps (``properties``) are walked by value, so a property that happens to
    be called ``title`` or ``default`` is treated as a schema, not as a keyword; data-valued
    keywords (``default``, ``enum``, ...) are copied untouched.
    """
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _DATA_KEYS:
            out[key] = value
        elif key in _SCHEMA_MAPS and isinstance(value, dict):
            out[key] = {name: fn(sub) for name, sub in value.items()}
        else:
            out[key] = fn(value)
    return out
