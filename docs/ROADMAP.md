# Roadmap and phased delivery

The prototype was built in six phases. Each phase ended with the same gate: `ruff`
(lint + format), `mypy --strict`, and the full test suite green, before it was committed.

| Phase | Scope | Exit criteria | Status |
|---|---|---|---|
| 0. Foundation | src layout, packaging, settings, error hierarchy, tooling | lint/type/test gate runs | ✅ |
| 1. LLM layer | message model, `LLMClient` port, LiteLLM adapter, scripted client | real LiteLLM parsing tested offline via `mock_response`/`mock_tool_calls`; retry and concurrency tests | ✅ |
| 2. Tool system | `@tool` schema derivation, registry, workspace, file/exec/web tools | sandbox tests (timeout kill, env scrubbing, memory limit, orphaned processes); SSRF tests; live PyPI smoke check | ✅ |
| 3. Agent loop | turn loop, parallel tools, terminal tools, budgets, events | transcript validity, self-correction, turn and budget limits; agent + real LiteLLM + real tools integration test | ✅ |
| 4. Workflow | DAG plan model, planner repair loop, engine, evidence rule, synthesis | parallelism, failure propagation, unverified downgrade, budget abort, timeout tests | ✅ |
| 5. Interfaces | CLI, console/JSONL sinks, composition root, offline demo | demo runs real pytest; a broken implementation is caught; CLI exit codes | ✅ |
| 6. Hardening & docs | public API, examples, CI matrix, docs, packaging check | examples run offline under test; suite green on Python 3.11, 3.12 and 3.13; the built wheel's CLI runs the demo in a clean venv | ✅ |

## Defects caught by the per-phase gates

Bugs the phase gates found and fixed during development:

* `litellm.Timeout` is not a subclass of `litellm.APIConnectionError` (it derives from
  openai's), so timeouts were not retried.
* `asyncio.subprocess.Process.wait()` only resolves after **all pipes close**, so a
  background grandchild stalled `run_python` until its timeout. Exit is now detected from
  the return code, then the process group is killed.
* Awaiting a dependency's future from a cancelled task cancelled the **shared** future
  (workflow timeout). Dependency futures are now shielded.
* IP-literal URLs went through DNS resolution in the SSRF guard; they are now checked
  directly.
* `Workspace` leaked `FileExistsError` for a file path, and `name=""` silently fell back
  to the function name in `@tool`.
* A missing API key surfaced as a retryable `InternalServerError`; the CLI now checks
  credentials before the run.

## Next phases

| Phase | Idea | Why |
|---|---|---|
| 7. Search | pluggable `web_search` tool (Tavily / Brave / SearXNG) | research beyond known URLs |
| 8. Repair loop | on a failed `verify`, re-plan a fix step with the failing evidence | close the loop from detection to correction |
| 9. Context management | summarise or compact long transcripts; token-aware truncation | long-horizon tasks |
| 10. Isolation | container (Docker/gVisor) execution backend behind the same tool interface | real security boundary |
| 11. Persistence | checkpoint step results; resume and replay from JSONL traces | long or interrupted runs |
| 12. Interop | expose tools via MCP; consume MCP servers as tool sources | ecosystem integration |
| 13. Evaluation | benchmark suite of research/verification tasks with graded outcomes | measure prompt and model changes |
| 14. Streaming UI | stream events over SSE/WebSocket to a small web dashboard | operator visibility |
