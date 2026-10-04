"""Shared pytest configuration."""

from __future__ import annotations

import os

# Never let LiteLLM fetch its model-cost map over the network during tests.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
