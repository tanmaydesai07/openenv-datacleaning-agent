"""
Data Cleaning Environment Implementation.

A data cleaning environment that evaluates LLMs on their ability to identify
and fix data quality issues in CSV files using tool calls.
"""

import logging
import os
import random
import shutil
import tempfile
import uuid
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.env_server.types import Action, Observation

from ..models import AVAILABLE_TOOLS, DataCleanState
from .rewards import compute_reward
from .tools import DataCleanTools

logger = logging.getLogger(__name__)


class DataCleanEnvironment(MCPEnvironment):
    """
    Data cleaning environment for RL training.

    Evaluates agents on their ability to clean messy CSV files by:
    - Reading messy input files
    - Analyzing data quality issues
    - Running Python code to clean data
    - Writing cleaned output files
    - Submitting cleaned files for grading

    Args:
        tasks_path: Path to the directory containing task files (default: ./tasks)
        max_steps: Maximum number of tool calls per episode (default: 30)
    """

    def __init__(
        self,
        tasks_path: str = "./tasks",
        max_steps: int = 30,
    ):
        # Create MCP server and define tools inline
        mcp = FastMCP("data_clean_env")

        self.tasks_path = tasks_path
        self.max_steps = max_steps

        self.task_configs = self._load_task_configs()
        logger.info(f"Loaded {len(self.task_configs)} task configurations")

        # This will be initialized in reset()
        self._tools: Optional[DataCleanTools] = None
        self._workspace_dir: Optional[str] = None

        # Register tools with FastMCP
        @mcp.tool
        def read_file(path: str) -> str:
            """
            Read a file from the workspace.

            Args:
                path: Relative path to the file in the workspace

            Returns:
                File contents or error message
            """
            if self._tools is None:
                return "Error: Environment not initialized. Call reset() first."
            return self._tools.read_file(path)

        @mcp.tool
        def run_python(code: str) -> str:
            """
            Execute Python code in the workspace environment.

            The code runs with access to pandas, numpy, and standard library.
            Use this to analyze data, identify issues, and perform cleaning operations.

            Args:
                code: Python code to execute

            Returns:
                stdout/stderr from execution or error message
            """
            if self._tools is None:
                return "Error: Environment not initialized. Call reset() first."
            return self._tools.run_python(code)

        @mcp.tool
        def write_file(path: str, content: str) -> str:
            """
            Write content to a file in the workspace.

            Args:
                path: Relative path to the file in the workspace
                content: Content to write to the file

            Returns:
                Success message or error
            """
            if self._tools is None:
                return "Error: Environment not initialized. Call reset() first."
            return self._tools.write_file(path, content)

        @mcp.tool
        def submit_cleaned_file(path: str) -> str:
            """
            Submit a cleaned file as the final answer.

            This terminates the episode and triggers grading.

            Args:
                path: Relative path to the cleaned file

            Returns:
                Confirmation message
            """
            if self._tools is None:
                return "Error: Environment not initialized. Call reset() first."
            return self._tools.submit_cleaned_file(path)

        # Pass the MCP server to the base class
        super().__init__(mcp)

        # Shuffle dataset for sequential selection
        self._shuffled_tasks = self.task_configs.copy()
        random.shuffle(self._shuffled_tasks)
        self._task_index = 0

        self._state = DataCleanState()

    def _load_task_configs(self) -> List[Dict[str, Any]]:
        """Load task configurations from the tasks directory."""
        configs = []

        # Easy task - customer data
        configs.append(
            {
                "level": "easy",
                "description": (
                    "Clean a customer dataset with 5 rows. "
                    "Issues: duplicate rows, missing values. "
                    "Remove duplicates and handle missing values appropriately."
                ),
                "messy_file": os.path.join(self.tasks_path, "easy_messy.csv"),
                "clean_file": os.path.join(self.tasks_path, "easy_clean.csv"),
            }
        )

        # Medium task - sales records
        configs.append(
            {
                "level": "medium",
                "description": (
                    "Clean a sales records dataset with 20 rows. "
                    "Issues: inconsistent date formats, inconsistent casing, "
                    "invalid email formats, extra whitespace, wrong data types. "
                    "Normalize dates to YYYY-MM-DD, standardize casing to title case, "
                    "validate/fix emails, trim whitespace, convert amounts to floats."
                ),
                "messy_file": os.path.join(self.tasks_path, "medium_messy.csv"),
                "clean_file": os.path.join(self.tasks_path, "medium_clean.csv"),
            }
        )

        # Hard task - product inventory
        configs.append(
            {
                "level": "hard",
                "description": (
                    "Clean a product inventory dataset with 50 rows. "
                    "Issues: missing product IDs, negative quantities, invalid prices, "
                    "inconsistent date formats, inconsistent category casing, "
                    "extra whitespace, inconsistent name casing, mixed price formats (USD suffix). "
                    "Fix all issues: generate missing IDs, set negative quantities to 0, "
                    "set invalid prices to 0, normalize dates to YYYY-MM-DD, "
                    "standardize category casing to title case, trim whitespace, "
                    "standardize name casing to title case, extract numeric prices."
                ),
                "messy_file": os.path.join(self.tasks_path, "hard_messy.csv"),
                "clean_file": os.path.join(self.tasks_path, "hard_clean.csv"),
            }
        )

        # Domain-specific tasks (generated by generate_training_data.py)
        domain_tasks = [
            {
                "level": "medium",
                "description": (
                    "Clean HR employee records. Issues: name casing inconsistencies, "
                    "salary format variations ($, USD), date format mismatches, "
                    "email corruption, missing values, duplicate entries. "
                    "Standardize all formats and fix data quality issues."
                ),
                "messy_file": os.path.join(self.tasks_path, "hr_messy.csv"),
                "clean_file": os.path.join(self.tasks_path, "hr_clean.csv"),
            },
            {
                "level": "medium",
                "description": (
                    "Clean healthcare patient appointment records. Issues: "
                    "inconsistent name casing, date format variations, "
                    "condition/doctor/status casing issues, missing insurance IDs. "
                    "Normalize all fields to consistent formats."
                ),
                "messy_file": os.path.join(self.tasks_path, "healthcare_messy.csv"),
                "clean_file": os.path.join(self.tasks_path, "healthcare_clean.csv"),
            },
            {
                "level": "hard",
                "description": (
                    "Clean financial transaction records. Issues: "
                    "merchant/category casing, price formats ($, USD, whitespace), "
                    "date inconsistencies, payment method variations, "
                    "status casing, missing values. Clean and standardize all data."
                ),
                "messy_file": os.path.join(self.tasks_path, "finance_messy.csv"),
                "clean_file": os.path.join(self.tasks_path, "finance_clean.csv"),
            },
            {
                "level": "hard",
                "description": (
                    "Clean logistics shipping records. Issues: "
                    "origin/destination casing, carrier name variations, "
                    "date format inconsistencies, status casing, missing weights. "
                    "Normalize location names and all other fields."
                ),
                "messy_file": os.path.join(self.tasks_path, "logistics_messy.csv"),
                "clean_file": os.path.join(self.tasks_path, "logistics_clean.csv"),
            },
            {
                "level": "medium",
                "description": (
                    "Clean student grade records. Issues: "
                    "name casing, subject casing variations, date format issues, "
                    "status casing inconsistencies, missing credits. "
                    "Standardize all fields to proper formats."
                ),
                "messy_file": os.path.join(self.tasks_path, "education_messy.csv"),
                "clean_file": os.path.join(self.tasks_path, "education_clean.csv"),
            },
        ]

        # Only add domain tasks if the files exist
        for task in domain_tasks:
            if os.path.exists(task["messy_file"]) and os.path.exists(
                task["clean_file"]
            ):
                configs.append(task)

        return configs

    def _get_next_task(self) -> Dict[str, Any]:
        """Get the next task using sequential shuffle selection."""
        if self._task_index >= len(self._shuffled_tasks):
            random.shuffle(self._shuffled_tasks)
            self._task_index = 0

        task = self._shuffled_tasks[self._task_index]
        self._task_index += 1
        return task

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset the environment for a new episode.

        Creates a temporary workspace and copies the messy file.

        Returns:
            Initial observation with the task description
        """
        # Clean up previous workspace
        if self._workspace_dir and os.path.exists(self._workspace_dir):
            shutil.rmtree(self._workspace_dir, ignore_errors=True)

        # Get next task
        task = self._get_next_task()

        # Create workspace
        self._workspace_dir = tempfile.mkdtemp(prefix="dataclean_")

        # Copy messy file to workspace
        messy_filename = os.path.basename(task["messy_file"])
        workspace_messy_path = os.path.join(self._workspace_dir, messy_filename)
        shutil.copy(task["messy_file"], workspace_messy_path)

        # Initialize tools
        self._tools = DataCleanTools(self._workspace_dir, workspace_messy_path)

        # Update state
        self._state = DataCleanState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_level=task["level"],
            messy_file_path=workspace_messy_path,
            clean_file_path=task["clean_file"],
            task_description=task["description"],
            workspace_dir=self._workspace_dir,
            submitted=False,
        )

        logger.info(f"Reset episode {self._state.episode_id} with {task['level']} task")

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "task_level": task["level"],
                "task_description": task["description"],
                "messy_file": messy_filename,
                "workspace_dir": self._workspace_dir,
                "tool_result": "",
                "step_count": 0,
                "available_tools": AVAILABLE_TOOLS.copy(),
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Handle non-MCP actions. Returns an error since this env is MCP-only.
        """
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction or CallToolAction for MCP interactions."
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Execute a step in the environment.

        Delegates to base class for MCP actions. Handles submit_cleaned_file
        reward computation and max-step termination.
        """
        self._state.step_count += 1

        # Let the base class handle MCP actions
        obs = super().step(action, timeout_s=timeout_s, **kwargs)

        # Check if submit_cleaned_file was called
        if (
            isinstance(action, CallToolAction)
            and action.tool_name == "submit_cleaned_file"
        ):
            self._state.submitted = True
            submitted_path = self._tools.submitted_file_path if self._tools else None

            reward = compute_reward(submitted_path, self._state.clean_file_path)

            logger.info(
                f"Episode {self._state.episode_id} ended: "
                f"level={self._state.task_level}, reward={reward:.2f}"
            )

            return Observation(
                done=True,
                reward=reward,
                metadata={
                    **obs.metadata,
                    "submitted_file": submitted_path,
                    "expected_file": self._state.clean_file_path,
                    "final_reward": reward,
                },
            )

        # Check for max steps
        if self._state.step_count >= self.max_steps:
            logger.info(
                f"Episode {self._state.episode_id} terminated: max steps reached"
            )
            return Observation(
                done=True,
                reward=0.0,
                metadata={
                    **obs.metadata,
                    "error": f"Max steps ({self.max_steps}) reached without submitting file.",
                },
            )

        return obs

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Async step used by the WebSocket handler.

        Mirrors step() logic but delegates to MCPEnvironment.step_async.
        """
        self._state.step_count += 1

        obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)

        if (
            isinstance(action, CallToolAction)
            and action.tool_name == "submit_cleaned_file"
        ):
            self._state.submitted = True
            submitted_path = self._tools.submitted_file_path if self._tools else None

            reward = compute_reward(submitted_path, self._state.clean_file_path)

            logger.info(
                f"Episode {self._state.episode_id} ended: "
                f"level={self._state.task_level}, reward={reward:.2f}"
            )

            return Observation(
                done=True,
                reward=reward,
                metadata={
                    **obs.metadata,
                    "submitted_file": submitted_path,
                    "expected_file": self._state.clean_file_path,
                    "final_reward": reward,
                },
            )

        if self._state.step_count >= self.max_steps:
            logger.info(
                f"Episode {self._state.episode_id} terminated: max steps reached"
            )
            return Observation(
                done=True,
                reward=0.0,
                metadata={
                    **obs.metadata,
                    "error": f"Max steps ({self.max_steps}) reached without submitting file.",
                },
            )

        return obs

    @property
    def state(self) -> DataCleanState:
        """Get the current environment state."""
        return self._state

    def __del__(self):
        """Cleanup workspace on deletion."""
        if self._workspace_dir and os.path.exists(self._workspace_dir):
            try:
                shutil.rmtree(self._workspace_dir)
            except Exception:
                pass
