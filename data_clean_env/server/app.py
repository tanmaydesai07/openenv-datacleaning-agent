"""
FastAPI server for the Data Cleaning environment.

Environment Variables:
    DATA_CLEAN_TASKS_PATH: Path to tasks directory (default: /app/env/tasks)
    DATA_CLEAN_MAX_STEPS: Maximum tool calls per episode (default: 30)

Usage:
    uvicorn data_clean_env.server.app:app --host 0.0.0.0 --port 8000
"""

import os

# Import create_app from the correct location
try:
    from openenv.core.env_server import create_app
except ImportError:
    from openenv.core.env_server.http_server import create_app

# Use the base CallToolAction - not a subclass - so the server's 
# MCP action routing (list_tools, call_tool) works correctly.
# The serialization.py checks `action_cls in _MCP_ACTION_TYPES.values()`
# which requires an exact match, not a subclass.
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from .environment import DataCleanEnvironment

DEFAULT_TASKS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tasks")
TASKS_PATH = os.environ.get("DATA_CLEAN_TASKS_PATH", DEFAULT_TASKS_PATH)
if not os.path.exists(TASKS_PATH):
    # Fall back to packaged tasks if the provided env var path is invalid.
    TASKS_PATH = DEFAULT_TASKS_PATH
MAX_STEPS = int(os.environ.get("DATA_CLEAN_MAX_STEPS", "30"))


def _env_factory():
    """Create a new DataCleanEnvironment instance for each session."""
    return DataCleanEnvironment(
        tasks_path=TASKS_PATH,
        max_steps=MAX_STEPS,
    )


# Create the FastAPI app
# IMPORTANT: Use base CallToolAction, not a custom subclass.
# The server's deserialize_action() only recognizes MCP actions
# (list_tools, call_tool) when action_cls is exactly one of the
# MCP action types or the base Action class.
app = create_app(
    _env_factory,
    CallToolAction,  # Use base class, not custom subclass
    CallToolObservation,
    env_name="data_clean_env",
)

# Explicit health/info routes for platform checks.
@app.get("/health")
def health():
    return {
        "status": "ok",
        "env_name": "data_clean_env",
        "tasks_path": TASKS_PATH,
    }


@app.get("/")
def root():
    return {
        "name": "data_clean_env",
        "status": "running",
        "health": "/health",
    }


def main():
    """Entry point for running the server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
