"""Runtime settings, loaded from ``WORKFLOW_AGENT_*`` environment variables.

Provider credentials (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ...) are intentionally *not*
modelled here: LiteLLM reads them from the environment directly.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunables in one place. CLI flags override these values."""

    model_config = SettingsConfigDict(env_prefix="WORKFLOW_AGENT_", extra="ignore")

    # --- LLM -----------------------------------------------------------------------------
    model: str = Field(default="gpt-4o-mini", description="Any LiteLLM model identifier.")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    request_timeout_s: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    max_concurrent_llm_requests: int = Field(default=4, ge=1)

    # --- Agent / workflow ------------------------------------------------------------------
    max_agent_turns: int = Field(default=12, ge=1, description="LLM calls per agent run.")
    max_plan_steps: int = Field(default=8, ge=1, le=20)
    step_concurrency: int = Field(default=3, ge=1)
    workflow_timeout_s: float | None = Field(default=None, gt=0)

    # --- Budgets ---------------------------------------------------------------------------
    max_total_tokens: int | None = Field(default=None, gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0)

    # --- Tools -----------------------------------------------------------------------------
    workspace_dir: Path = Path("./workspace")
    tool_timeout_s: float = Field(default=60.0, gt=0)
    max_tool_output_chars: int = Field(default=12_000, ge=500)
    allow_private_network: bool = Field(
        default=False, description="Allow fetch_url to reach loopback/private addresses."
    )
