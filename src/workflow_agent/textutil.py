"""Small text helpers shared by tools, agents and reporters."""

from __future__ import annotations

import re

_WHITESPACE = re.compile(r"\s+")


def truncate_middle(text: str, limit: int) -> str:
    """Shorten ``text`` to about ``limit`` characters, keeping its head and tail.

    Tool output is truncated in the middle because both ends tend to matter: the head has
    context, the tail has the verdict (exit codes, pytest summaries, tracebacks).
    """
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    marker = f"\n\n... [{len(text) - limit} characters truncated] ...\n\n"
    keep = max(limit - len(marker), 0)
    head = keep * 3 // 5
    tail = keep - head
    return text[:head] + marker + (text[-tail:] if tail else "")


def one_line(text: str | None, limit: int = 120) -> str:
    """Collapse whitespace and cut to ``limit`` characters (for logs and event previews)."""
    if not text:
        return ""
    flat = _WHITESPACE.sub(" ", text).strip()
    return flat if len(flat) <= limit else flat[: max(limit - 1, 0)] + "…"
