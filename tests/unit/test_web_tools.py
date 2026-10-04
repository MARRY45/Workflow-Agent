from __future__ import annotations

import httpx
import pytest

from workflow_agent.errors import ToolError
from workflow_agent.llm import tool_call
from workflow_agent.tools import HttpFetcher, ToolRegistry, web_tools
from workflow_agent.tools.builtin.web import (
    FetchedResource,
    html_to_text,
    is_public_address,
    render_resource,
)

PUBLIC_IP = "93.184.216.34"

HTML = """
<html><head><title> Example   Docs </title><style>body{}</style></head>
<body>
  <nav><a href="/guide">Guide</a> <a href="#top">Top</a> <a href="mailto:x@y.z">Mail</a></nav>
  <h1>Install</h1>
  <p>Run   the
     command:</p>
  <pre>pip install example
  --upgrade</pre>
  <ul><li>fast</li><li>typed</li></ul>
  <script>alert("x")</script>
  <a href="https://other.example/page#frag">Other site</a>
</body></html>
"""


def make_fetcher(handler, dns: dict[str, str] | None = None, **kwargs) -> HttpFetcher:  # type: ignore[no-untyped-def]
    table = {"example.com": PUBLIC_IP, "pypi.org": PUBLIC_IP, **(dns or {})}

    async def resolver(host: str, port: int) -> list[str]:
        if host not in table:
            raise OSError("Name or service not known")
        return [table[host]]

    return HttpFetcher(transport=httpx.MockTransport(handler), resolver=resolver, **kwargs)


@pytest.mark.parametrize(
    ("address", "public"),
    [
        (PUBLIC_IP, True),
        ("8.8.8.8", True),
        ("127.0.0.1", False),
        ("10.1.2.3", False),
        ("192.168.0.1", False),
        ("169.254.169.254", False),  # cloud metadata endpoint
        ("100.64.0.1", False),  # carrier-grade NAT
        ("::1", False),
        ("fe80::1%eth0", False),
        ("::ffff:127.0.0.1", False),  # IPv4-mapped loopback
        ("224.0.0.1", False),
        ("2606:4700:4700::1111", True),
    ],
)
def test_is_public_address(address: str, public: bool) -> None:
    assert is_public_address(address) is public


def test_html_to_text_extracts_structure() -> None:
    text = html_to_text(HTML, base_url="https://example.com/docs/")
    assert text.startswith("Title: Example Docs")
    assert "# Install" in text
    assert "Run the command:" in text
    assert "```\npip install example\n  --upgrade\n```" in text
    assert "- fast" in text
    assert "- typed" in text
    assert "alert" not in text
    assert "body{}" not in text
    assert "- Guide <https://example.com/guide>" in text
    assert "- Other site <https://other.example/page>" in text
    assert "mailto" not in text
    assert "#top" not in text


def test_render_resource_by_content_type() -> None:
    def res(content_type: str, text: str) -> FetchedResource:
        return FetchedResource(
            url="https://example.com",
            status=200,
            content_type=content_type,
            text=text,
            truncated=False,
        )

    assert render_resource(res("application/json; charset=utf-8", '{"a":1}')) == '{\n  "a": 1\n}'
    assert render_resource(res("application/json", "not json")) == "not json"
    assert render_resource(res("text/plain", "plain")) == "plain"
    assert render_resource(res("", "unknown")) == "unknown"
    with pytest.raises(ToolError, match="unsupported content type"):
        render_resource(res("image/png", "\x89PNG"))


async def test_fetch_url_html_with_paging() -> None:
    fetcher = make_fetcher(lambda request: httpx.Response(200, html=HTML))
    registry = ToolRegistry(web_tools(fetcher))

    first = await registry.execute(
        tool_call("fetch_url", {"url": "https://example.com/docs/", "max_chars": 500})
    )
    assert not first.is_error
    assert "URL: https://example.com/docs/" in first.content
    assert "Title: Example Docs" in first.content

    total = len(html_to_text(HTML, base_url="https://example.com/docs/"))
    offset = await registry.execute(
        tool_call(
            "fetch_url", {"url": "https://example.com/docs/", "start_char": 100, "max_chars": 500}
        )
    )
    assert f"Characters 100-{min(600, total)} of {total}" in offset.content


async def test_fetch_url_reports_more_content() -> None:
    fetcher = make_fetcher(lambda request: httpx.Response(200, text="x" * 3000))
    registry = ToolRegistry(web_tools(fetcher))
    result = await registry.execute(
        tool_call("fetch_url", {"url": "https://example.com/", "max_chars": 1000})
    )
    assert "Characters 0-1000 of 3000" in result.content
    assert "call again with start_char=1000" in result.content


async def test_private_hosts_are_blocked() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, text="secret")

    fetcher = make_fetcher(handler, dns={"internal.corp": "10.0.0.5"})
    for url in [
        "http://127.0.0.1:8080/admin",
        "http://internal.corp/",
        "http://169.254.169.254/latest/meta-data",
    ]:
        with pytest.raises(ToolError, match="non-public address"):
            await fetcher.get(url)
    assert calls == []  # nothing was ever sent


async def test_redirect_to_private_host_is_blocked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "http://127.0.0.1/admin"})

    with pytest.raises(ToolError, match=r"non-public address 127\.0\.0\.1"):
        await make_fetcher(handler).get("https://example.com/")


async def test_redirects_are_followed_and_capped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(301, headers={"Location": "/final"})
        if request.url.path == "/loop":
            return httpx.Response(302, headers={"Location": "/loop"})
        return httpx.Response(200, text="arrived")

    fetcher = make_fetcher(handler, max_redirects=3)
    resource = await fetcher.get("https://example.com/start")
    assert resource.url == "https://example.com/final"
    assert resource.text == "arrived"
    with pytest.raises(ToolError, match="too many redirects"):
        await fetcher.get("https://example.com/loop")


async def test_private_network_can_be_allowed() -> None:
    fetcher = make_fetcher(
        lambda request: httpx.Response(200, text="local"), allow_private_network=True
    )
    assert (await fetcher.get("http://127.0.0.1:8000/")).text == "local"


@pytest.mark.parametrize(
    ("url", "error"),
    [
        ("ftp://example.com/x", "only http"),
        ("file:///etc/passwd", "only http"),
        ("https:///nohost", "no host"),
        ("https://unknown.invalid/", "cannot resolve"),
    ],
)
async def test_bad_urls(url: str, error: str) -> None:
    with pytest.raises(ToolError, match=error):
        await make_fetcher(lambda request: httpx.Response(200)).get(url)


async def test_large_downloads_are_truncated_and_transport_errors_wrapped() -> None:
    big = make_fetcher(lambda request: httpx.Response(200, text="y" * 5000), max_bytes=1000)
    resource = await big.get("https://example.com/")
    assert resource.truncated
    assert len(resource.text) == 1000

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(ToolError, match="ConnectError"):
        await make_fetcher(boom).get("https://example.com/")


async def test_successful_responses_are_cached() -> None:
    hits = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.path)
        return httpx.Response(200 if request.url.path != "/missing" else 404, text="x")

    fetcher = make_fetcher(handler, cache_size=1)
    await fetcher.get("https://example.com/a")
    await fetcher.get("https://example.com/a")
    await fetcher.get("https://example.com/missing")
    await fetcher.get("https://example.com/missing")
    await fetcher.get("https://example.com/b")  # evicts /a
    await fetcher.get("https://example.com/a")
    assert hits == ["/a", "/missing", "/missing", "/b", "/a"]


async def test_fetch_url_http_errors_become_tool_errors() -> None:
    registry = ToolRegistry(
        web_tools(make_fetcher(lambda request: httpx.Response(404, text="Not here")))
    )
    result = await registry.execute(tool_call("fetch_url", {"url": "https://example.com/x"}))
    assert result.is_error
    assert "HTTP 404" in result.content


PYPI_PAYLOAD = {
    "info": {
        "name": "demo",
        "version": "2.0.0",
        "summary": "A demo package",
        "requires_python": ">=3.9",
        "license_expression": "MIT",
        "author": "Jane",
        "project_urls": {"Source": "https://github.com/x/demo"},
        "home_page": "https://demo.dev",
        "requires_dist": [f"dep{i}>=1" for i in range(27)],
    },
    "releases": {
        "1.0.0": [{"upload_time_iso_8601": "2023-01-02T10:00:00.000000Z", "yanked": False}],
        "1.5.0": [{"upload_time_iso_8601": "2024-03-04T10:00:00.000000Z", "yanked": True}],
        "2.0.0": [{"upload_time_iso_8601": "2025-05-06T10:00:00.000000Z", "yanked": False}],
        "0.0.1": [],
    },
}


async def test_pypi_package_info() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/pypi/demo/json":
            return httpx.Response(200, json=PYPI_PAYLOAD)
        return httpx.Response(404, json={"message": "Not Found"})

    registry = ToolRegistry(web_tools(make_fetcher(handler)))
    result = await registry.execute(tool_call("pypi_package_info", {"package": "demo"}))
    content = result.content
    assert "Latest version: 2.0.0" in content
    assert "License: MIT" in content
    assert "- Source: https://github.com/x/demo" in content
    assert "- Homepage: https://demo.dev" in content
    assert "Dependencies (27):" in content
    assert "... and 2 more" in content
    releases = content.split("Recent releases (3 total):\n")[1].splitlines()
    assert releases == [
        "- 2.0.0 (2025-05-06)",
        "- 1.5.0 (2024-03-04) [yanked]",
        "- 1.0.0 (2023-01-02)",
    ]

    missing = await registry.execute(tool_call("pypi_package_info", {"package": "nope"}))
    assert "was not found on PyPI" in missing.content
    invalid = await registry.execute(tool_call("pypi_package_info", {"package": "../etc"}))
    assert "invalid package name" in invalid.content


async def test_pypi_invalid_json() -> None:
    registry = ToolRegistry(web_tools(make_fetcher(lambda r: httpx.Response(200, text="<html>"))))
    result = await registry.execute(tool_call("pypi_package_info", {"package": "demo"}))
    assert "invalid JSON" in result.content
