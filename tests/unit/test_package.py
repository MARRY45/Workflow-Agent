from __future__ import annotations

import pytest

import workflow_agent
from workflow_agent import Settings, ToolError, WorkflowAgentError, WorkspaceError


def test_version_is_exposed() -> None:
    assert workflow_agent.__version__ == "0.1.0"


def test_error_hierarchy() -> None:
    assert issubclass(WorkspaceError, ToolError)
    assert issubclass(ToolError, WorkflowAgentError)


def test_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKFLOW_AGENT_MODEL", "anthropic/claude-test")
    monkeypatch.setenv("WORKFLOW_AGENT_MAX_COST_USD", "0.25")
    settings = Settings()
    assert settings.model == "anthropic/claude-test"
    assert settings.max_cost_usd == 0.25
    assert settings.allow_private_network is False


def test_settings_validate_ranges(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKFLOW_AGENT_TEMPERATURE", "5")
    with pytest.raises(ValueError, match="temperature"):
        Settings()
