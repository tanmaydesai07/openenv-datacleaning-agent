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

from openenv.core.mcp_client import MCPToolClient


class DataCleanEnv(MCPToolClient):
    """
    Client for the Data Cleaning environment.

    Inherits all functionality from MCPToolClient:
    - list_tools(): Discover available tools
    - call_tool(name, **kwargs): Call a tool by name
    - reset(**kwargs): Reset the environment
    - step(action): Execute an action
    """

    pass  # MCPToolClient provides all needed functionality
