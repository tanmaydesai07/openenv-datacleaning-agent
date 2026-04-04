"""
FastAPI server for the Data Cleaning environment.

Environment Variables:
    DATA_CLEAN_TASKS_PATH: Path to tasks directory (default: /app/env/tasks)
    DATA_CLEAN_MAX_STEPS: Maximum tool calls per episode (default: 30)
"""

import json
import os
from typing import Any, Dict

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from pydantic import field_validator

from .environment import DataCleanEnvironment

TASKS_PATH = os.environ.get("DATA_CLEAN_TASKS_PATH", "/app/env/tasks")
MAX_STEPS = int(os.environ.get("DATA_CLEAN_MAX_STEPS", "30"))


def _env_factory():
    """Create a new DataCleanEnvironment instance for each session."""
    return DataCleanEnvironment(
        tasks_path=TASKS_PATH,
        max_steps=MAX_STEPS,
    )


class DataCleanCallToolAction(CallToolAction):
    """CallToolAction that accepts JSON strings for arguments (web UI sends strings)."""

    @field_validator("arguments", mode="before")
    @classmethod
    def parse_arguments(cls, v: Any) -> Dict[str, Any]:
        if isinstance(v, str):
            return json.loads(v)
        return v


app = create_app(
    _env_factory,
    DataCleanCallToolAction,
    CallToolObservation,
    env_name="data_clean_env",
)
