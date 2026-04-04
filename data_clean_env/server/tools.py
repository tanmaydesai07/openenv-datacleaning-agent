"""
Tool implementations for the Data Cleaning environment.

Provides file operations and Python code execution for data cleaning tasks.
"""

import os
import subprocess
import tempfile
from typing import Any, Dict, Tuple


class DataCleanTools:
    """
    Tool implementations for data cleaning tasks.

    Args:
        workspace_dir: Working directory for agent operations
        messy_file_path: Path to the input messy file
    """

    def __init__(self, workspace_dir: str, messy_file_path: str):
        self.workspace_dir = workspace_dir
        self.messy_file_path = messy_file_path
        self.submitted_file_path = None

    def execute_tool(
        self, tool_name: str, tool_args: Dict[str, Any]
    ) -> Tuple[str, bool]:
        """
        Execute a tool and return its result.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool

        Returns:
            Tuple of (result_string, is_final_action)
        """
        if tool_name == "read_file":
            return self.read_file(**tool_args), False
        elif tool_name == "run_python":
            return self.run_python(**tool_args), False
        elif tool_name == "write_file":
            return self.write_file(**tool_args), False
        elif tool_name == "submit_cleaned_file":
            return self.submit_cleaned_file(**tool_args), True
        else:
            return f"Error: Unknown tool '{tool_name}'", False

    def read_file(self, path: str) -> str:
        """
        Read a file from the workspace.

        Args:
            path: Relative path to the file in the workspace

        Returns:
            File contents or error message
        """
        # Resolve path relative to workspace
        if not os.path.isabs(path):
            full_path = os.path.join(self.workspace_dir, path)
        else:
            # Security check: ensure path is within workspace
            full_path = os.path.abspath(path)
            if not full_path.startswith(os.path.abspath(self.workspace_dir)):
                return f"Error: Path '{path}' is outside workspace directory"

        if not os.path.exists(full_path):
            return f"Error: File '{path}' not found"

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def run_python(self, code: str) -> str:
        """
        Execute Python code in a sandboxed environment.

        Args:
            code: Python code to execute

        Returns:
            stdout/stderr from execution or error message
        """
        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=self.workspace_dir
        ) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Execute with timeout and capture output
            result = subprocess.run(
                ["python", temp_file],
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )

            output = ""
            if result.stdout:
                output += f"stdout:\n{result.stdout}\n"
            if result.stderr:
                output += f"stderr:\n{result.stderr}\n"
            if result.returncode != 0:
                output += f"\nExit code: {result.returncode}"

            return output if output else "Code executed successfully (no output)"

        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out (10s limit)"
        except Exception as e:
            return f"Error executing code: {str(e)}"
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass

    def write_file(self, path: str, content: str) -> str:
        """
        Write content to a file in the workspace.

        Args:
            path: Relative path to the file in the workspace
            content: Content to write

        Returns:
            Success message or error
        """
        # Resolve path relative to workspace
        if not os.path.isabs(path):
            full_path = os.path.join(self.workspace_dir, path)
        else:
            # Security check: ensure path is within workspace
            full_path = os.path.abspath(path)
            if not full_path.startswith(os.path.abspath(self.workspace_dir)):
                return f"Error: Path '{path}' is outside workspace directory"

        try:
            # Ensure directory exists (only if path has a directory component)
            dir_path = os.path.dirname(full_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"File written successfully: {path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

    def submit_cleaned_file(self, path: str) -> str:
        """
        Submit a cleaned file as the final answer.

        Args:
            path: Relative path to the cleaned file

        Returns:
            Confirmation message
        """
        # Resolve path relative to workspace
        if not os.path.isabs(path):
            full_path = os.path.join(self.workspace_dir, path)
        else:
            full_path = os.path.abspath(path)
            if not full_path.startswith(os.path.abspath(self.workspace_dir)):
                return f"Error: Path '{path}' is outside workspace directory"

        if not os.path.exists(full_path):
            return f"Error: File '{path}' not found. Please create it first."

        self.submitted_file_path = full_path
        return f"Cleaned file submitted: {path}"
