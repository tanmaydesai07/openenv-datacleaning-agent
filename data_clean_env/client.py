"""
Client for the Data Cleaning environment.

This client connects to a running Data Cleaning environment server and provides
a Python interface for interacting with it via MCP tools. Async by default.

Example:
    >>> from envs.data_clean_env import DataCleanEnv
    >>>
    >>> async with DataCleanEnv(base_url="http://localhost:8000") as env:
    ...     await env.reset()
    ...     tools = await env.list_tools()
    ...     result = await env.call_tool("read_file", path="easy_messy.csv")
    ...     print(result)
    ...     result = await env.call_tool("submit_cleaned_file", path="cleaned.csv")
"""

from typing import Any, Dict

from openenv.core.mcp_client import MCPToolClient

from .models import DataCleanState


class DataCleanEnv(MCPToolClient):
    """
    Client for the Data Cleaning environment.

    Inherits all functionality from MCPToolClient:
    - list_tools(): Discover available tools
    - call_tool(name, **kwargs): Call a tool by name
    - reset(**kwargs): Reset the environment
    - step(action): Execute an action
    """

    def __init__(self, *args, **kwargs):
        # Increase message timeout to 300s for large file writes
        kwargs.setdefault("message_timeout_s", 300.0)
        super().__init__(*args, **kwargs)

    def _parse_state(self, payload: Dict[str, Any]) -> DataCleanState:
        """Parse state response into DataCleanState with all task fields."""
        return DataCleanState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_level=payload.get("task_level", ""),
            messy_file_path=payload.get("messy_file_path", ""),
            clean_file_path=payload.get("clean_file_path", ""),
            task_description=payload.get("task_description", ""),
            workspace_dir=payload.get("workspace_dir", ""),
            submitted=payload.get("submitted", False),
        )
