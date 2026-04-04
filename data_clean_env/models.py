"""
State types for the Data Cleaning environment.

Data Cleaning Agent evaluates LLMs on their ability to identify and fix
data quality issues in CSV files using tool calls (read_file, run_python,
write_file, submit_cleaned_file).

This environment uses the MCP protocol for tool interactions. Use
``CallToolAction`` and ``ListToolsAction`` from ``openenv.core.env_server.mcp_types``
to interact with the environment.
"""

from openenv.core.env_server import State


# Tool names - defined statically to avoid circular imports
AVAILABLE_TOOLS = ["read_file", "run_python", "write_file", "submit_cleaned_file"]


class DataCleanState(State):
    """
    Internal environment state for tracking the current episode.

    All fields are set during reset() and are essential for episode tracking.

    Attributes:
        task_level: Difficulty level (easy, medium, hard)
        messy_file_path: Path to the input file with data issues
        clean_file_path: Path to the expected cleaned output
        task_description: Description of the cleaning task
        workspace_dir: Working directory for agent operations
        submitted: Whether the agent has submitted a cleaned file
        # Inherited from State: episode_id, step_count
    """

    task_level: str = ""
    messy_file_path: str = ""
    clean_file_path: str = ""
    task_description: str = ""
    workspace_dir: str = ""
    submitted: bool = False
