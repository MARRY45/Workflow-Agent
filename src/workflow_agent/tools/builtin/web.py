"""Research tools: ``fetch_url`` and ``pypi_package_info``.

:class:`HttpFetcher` guards against SSRF by default: every hop (including redirects) must
resolve to public IP addresses only, so a model cannot be steered into probing
``localhost``, cloud metadata endpoints or the private network. (A DNS answer can still
change between the check and the connection; deployments that need hard guarantees
should also enforce egress rules at the network layer.)
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import socket
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from typing import Annotated, Any, ClassVar
from urllib.parse import urljoin, urlsplit

import httpx
from pydantic import Field

from workflow_agent._version import __version__
from workflow_agent.errors import ToolError
from workflow_agent.tools.base import Tool, tool

Resolver = Callable[[str, int], Awaitable[list[str]]]

_PACKAGE_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,98}[A-Za-z0-9])?$")
_TEXTUAL_TYPES = ("text/", "application/xml", "application/javascript", "application/x-yaml")


async def system_resolver(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return [str(info[4][0]) for info in infos]


def is_public_address(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])  # drop IPv6 zone id
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


@dataclass(frozen=True, slots=True)
class FetchedResource:
    url: str
    status: int
    content_type: str
    text: str
    truncated: bool


class HttpFetcher:
    """Small, guarded HTTP GET client used by the research tools."""

    def __init__(
        self,
        *,
        allow_private_network: bool = False,
        timeout_s: float = 20.0,
        max_bytes: int = 2_000_000,
        max_redirects: int = 5,
        cache_size: int = 32,
        transport: httpx.AsyncBaseTransport | None = None,
        resolver: Resolver = system_resolver,
    ) -> None:
        self.allow_private_network = allow_private_network
        self.timeout_s = timeout_s
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self._transport = transport
        self._resolver = resolver
        self._cache: OrderedDict[str, FetchedResource] = OrderedDict()
        self._cache_size = cache_size

    async def get(self, url: str) -> FetchedResource:
        if url in self._cache:
            self._cache.move_to_end(url)
            return self._cache[url]
        resource = await self._get_uncached(url)
        if resource.status < 400:
            self._cache[url] = resource
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return resource

    async def _get_uncached(self, url: str) -> FetchedResource:
        headers = {
            "User-Agent": f"workflow-agent/{__version__} (research bot)",
            "Accept": "text/html,application/json,text/plain;q=0.9,*/*;q=0.5",
        }
        async with httpx.AsyncClient(
            transport=self._transport,
            timeout=self.timeout_s,
            follow_redirects=False,  # redirects are followed manually so every hop is checked
            headers=headers,
        ) as client:
            current = url
            for _ in range(self.max_redirects + 1):
                await self._check_url(current)
                try:
                    async with client.stream("GET", current) as response:
                        if response.is_redirect:
                            location = response.headers.get("location")
                            if not location:
                                raise ToolError(f"redirect without Location header from {current}")
                            current = urljoin(current, location)
                            continue
                        body, truncated = await self._read_capped(response)
                        return FetchedResource(
                            url=str(response.url),
                            status=response.status_code,
                            content_type=response.headers.get("content-type", "").lower(),
                            text=body.decode(response.encoding or "utf-8", errors="replace"),
                            truncated=truncated,
                        )
                except httpx.HTTPError as exc:
                    raise ToolError(
                        f"request to {current} failed: {type(exc).__name__}: {exc}"
                    ) from exc
        raise ToolError(f"too many redirects (>{self.max_redirects}) starting at {url}")

    async def _check_url(self, url: str) -> None:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            raise ToolError(f"only http(s) URLs are allowed, got {url!r}")
        if not parts.hostname:
            raise ToolError(f"URL has no host: {url!r}")
        if self.allow_private_network:
            return
        port = parts.port or (443 if parts.scheme == "https" else 80)
        try:
            ipaddress.ip_address(parts.hostname.split("%", 1)[0])
            addresses = [parts.hostname]  # IP literal: nothing to resolve
        except ValueError:
            try:
                addresses = await self._resolver(parts.hostname, port)
            except OSError as exc:
                raise ToolError(f"cannot resolve host {parts.hostname!r}: {exc}") from exc
        if not addresses:
            raise ToolError(f"host {parts.hostname!r} did not resolve")
        blocked = [a for a in addresses if not is_public_address(a)]
        if blocked:
            raise ToolError(
                f"refusing to fetch {parts.hostname!r}: it resolves to non-public address "
                f"{blocked[0]} (private network access is disabled)"
            )

    async def _read_capped(self, response: httpx.Response) -> tuple[bytes, bool]:
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
            size += len(chunk)
            if size >= self.max_bytes:
                return b"".join(chunks)[: self.max_bytes], True
        return b"".join(chunks), False


# ---------------------------------------------------------------------- HTML -> text


class _HTMLToText(HTMLParser):
    _SKIP = frozenset({"script", "style", "noscript", "svg", "template", "iframe"})
    _BLOCK = frozenset(
        {"p", "div", "br", "tr", "section", "article", "header", "footer", "main", "nav",
         "table", "ul", "ol", "blockquote", "hr", "dl", "dt", "dd", "figure", "form"}
    )  # fmt: skip
    _HEADINGS: ClassVar[dict[str, int]] = {f"h{level}": level for level in range(1, 7)}

    def __init__(self, base_url: str, max_links: int) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.max_links = max_links
        self.title = ""
        self.links: dict[str, str] = {}
        self._out: list[str] = []
        self._skip_depth = 0
        self._pre_depth = 0
        self._in_title = False
        self._href: str | None = None
        self._anchor_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self._HEADINGS:
            self._out.append("\n\n" + "#" * self._HEADINGS[tag] + " ")
        elif tag == "li":
            self._out.append("\n- ")
        elif tag == "pre":
            self._pre_depth += 1
            self._out.append("\n```\n")
        elif tag in self._BLOCK:
            self._out.append("\n")
        elif tag == "a":
            self._href = dict(attrs).get("href")
            self._anchor_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(self._skip_depth - 1, 0)
        elif tag == "title":
            self._in_title = False
        elif tag == "pre":
            self._pre_depth = max(self._pre_depth - 1, 0)
            self._out.append("\n```\n")
        elif tag in self._HEADINGS or tag in self._BLOCK:
            self._out.append("\n")
        elif tag == "a" and self._href is not None:
            self._record_link(self._href, " ".join(self._anchor_text))
            self._href = None

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
            return
        if self._href is not None:
            self._anchor_text.append(data.strip())
        if self._pre_depth:
            self._out.append(data)
            return
        text = re.sub(r"\s+", " ", data)
        if not self._out or self._out[-1][-1:].isspace():
            text = text.lstrip()  # collapse whitespace across chunk boundaries too
        if text:
            self._out.append(text)

    def _record_link(self, href: str, text: str) -> None:
        if len(self.links) >= self.max_links or href.startswith(("#", "javascript:", "mailto:")):
            return
        absolute = urljoin(self.base_url, href).split("#", 1)[0]
        if absolute.startswith(("http://", "https://")) and absolute not in self.links:
            self.links[absolute] = re.sub(r"\s+", " ", text).strip()

    def text(self) -> str:
        raw = "".join(self._out)
        raw = re.sub(r"[ \t]+\n", "\n", raw)
        return re.sub(r"\n{3,}", "\n\n", raw).strip()


def html_to_text(html: str, *, base_url: str = "", max_links: int = 40) -> str:
    """Readable plain-text rendering of an HTML page, with its outgoing links listed."""
    parser = _HTMLToText(base_url, max_links)
    parser.feed(html)
    parser.close()
    parts = []
    title = re.sub(r"\s+", " ", parser.title).strip()
    if title:
        parts.append(f"Title: {title}")
    parts.append(parser.text())
    if parser.links:
        lines = [
            f"- {text} <{url}>" if text else f"- <{url}>" for url, text in parser.links.items()
        ]
        parts.append("Links:\n" + "\n".join(lines))
    return "\n\n".join(parts)


def render_resource(resource: FetchedResource) -> str:
    content_type = resource.content_type.split(";", 1)[0].strip()
    if content_type in ("text/html", "application/xhtml+xml"):
        return html_to_text(resource.text, base_url=resource.url)
    if content_type == "application/json" or content_type.endswith("+json"):
        try:
            return json.dumps(json.loads(resource.text), indent=2, ensure_ascii=False)
        except ValueError:
            return resource.text
    if not content_type or content_type.startswith(_TEXTUAL_TYPES):
        return resource.text
    raise ToolError(f"unsupported content type {content_type!r} at {resource.url}")


# ---------------------------------------------------------------------- tools


def web_tools(fetcher: HttpFetcher) -> list[Tool]:
    """Build ``fetch_url`` and ``pypi_package_info`` backed by ``fetcher``."""

    @tool(tags={"research", "network"})
    async def fetch_url(
        url: str,
        start_char: Annotated[int, Field(ge=0)] = 0,
        max_chars: Annotated[int, Field(ge=500, le=20_000)] = 8000,
    ) -> str:
        """Fetch a web page or API endpoint (HTTP GET) and return it as readable text.

        HTML is converted to text with its links listed at the end; JSON is pretty-printed.
        Long documents can be paged through with start_char.

        Args:
            url: Absolute http(s) URL.
            start_char: Offset into the extracted text, for reading long pages in chunks.
            max_chars: Maximum number of characters to return.
        """
        resource = await fetcher.get(url)
        if resource.status >= 400:
            preview = resource.text[:300].strip()
            raise ToolError(f"HTTP {resource.status} for {resource.url}: {preview}")
        text = render_resource(resource)
        chunk = text[start_char : start_char + max_chars]
        end = start_char + len(chunk)
        header = [
            f"URL: {resource.url}",
            f"Content-Type: {resource.content_type or 'unknown'}",
            f"Characters {start_char}-{end} of {len(text)}"
            + (" (download truncated)" if resource.truncated else ""),
        ]
        if end < len(text):
            header.append(f"More content available: call again with start_char={end}.")
        return "\n".join(header) + "\n\n" + chunk

    @tool(tags={"research", "network"})
    async def pypi_package_info(package: str) -> str:
        """Look up a Python package on PyPI: latest version, metadata, dependencies, releases.

        Args:
            package: Distribution name as published on PyPI, e.g. "httpx".
        """
        if not _PACKAGE_RE.fullmatch(package):
            raise ToolError(f"invalid package name {package!r}")
        resource = await fetcher.get(f"https://pypi.org/pypi/{package}/json")
        if resource.status == 404:
            raise ToolError(f"package {package!r} was not found on PyPI")
        if resource.status >= 400:
            raise ToolError(f"PyPI returned HTTP {resource.status} for {package!r}")
        try:
            data = json.loads(resource.text)
        except ValueError as exc:
            raise ToolError(f"PyPI returned invalid JSON for {package!r}") from exc
        return _summarize_pypi(data)

    return [fetch_url, pypi_package_info]


def _summarize_pypi(data: dict[str, Any]) -> str:
    info: dict[str, Any] = data.get("info") or {}
    releases: dict[str, list[dict[str, Any]]] = data.get("releases") or {}

    dated: list[tuple[datetime, str, bool]] = []
    for version, files in releases.items():
        stamps: list[str] = [
            str(f["upload_time_iso_8601"]) for f in files if f.get("upload_time_iso_8601")
        ]
        if stamps:
            uploaded = min(datetime.fromisoformat(s.replace("Z", "+00:00")) for s in stamps)
            dated.append((uploaded, version, all(f.get("yanked") for f in files)))
    dated.sort(reverse=True)

    urls = dict(info.get("project_urls") or {})
    if info.get("home_page"):
        urls.setdefault("Homepage", info["home_page"])
    requires = info.get("requires_dist") or []

    lines = [
        f"Package: {info.get('name', '?')}",
        f"Latest version: {info.get('version', '?')}",
        f"Summary: {info.get('summary') or '-'}",
        f"Requires-Python: {info.get('requires_python') or '-'}",
        f"License: {info.get('license_expression') or (info.get('license') or '-')[:100]}",
        f"Author: {info.get('author') or info.get('author_email') or '-'}",
    ]
    if urls:
        lines.append("Project URLs:\n" + "\n".join(f"- {k}: {v}" for k, v in urls.items()))
    if requires:
        shown = "\n".join(f"- {r}" for r in requires[:25])
        more = f"\n- ... and {len(requires) - 25} more" if len(requires) > 25 else ""
        lines.append(f"Dependencies ({len(requires)}):\n{shown}{more}")
    if dated:
        recent = "\n".join(
            f"- {version} ({uploaded.date().isoformat()}){' [yanked]' if yanked else ''}"
            for uploaded, version, yanked in dated[:10]
        )
        lines.append(f"Recent releases ({len(dated)} total):\n{recent}")
    return "\n".join(lines)
