#!/usr/bin/env python3
"""
Data Cleaning Agent inference script using OpenAI-compatible API.

This script evaluates an LLM on data cleaning tasks by connecting to the
Data Cleaning environment and using tool calls to clean messy CSV files.

Environment Variables:
    API_BASE_URL: Base URL for the OpenAI-compatible API
    MODEL_NAME: Name of the model to use
    HF_TOKEN: HuggingFace API token (alternative to API_KEY)
    API_KEY: API key for the model service

Usage:
    python inference.py

Output Format:
    Emits structured logs following the hackathon [START], [STEP], [END] format.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List
import sys
from openai import OpenAI, RateLimitError

# Add data_clean_env to path
sys.path.insert(0, str(Path(__file__).parent))

from data_clean_env.client import DataCleanEnv
from data_clean_env.models import AVAILABLE_TOOLS

# Import types needed for actions
try:
    from openenv.core.env_server.mcp_types import CallToolAction
except ImportError:
    # Fallback for local development
    class CallToolAction:
        def __init__(self, tool_name: str, arguments: dict):
            self.tool_name = tool_name
            self.arguments = arguments

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b:novita")

MAX_EPISODES = 3
MAX_TOKENS = 4096
VERBOSE = True

SYSTEM_PROMPT = """You are a data cleaning expert. You MUST follow this exact workflow:

STEP 1: Read the messy CSV file using read_file(path)
STEP 2: Run a Python cleaning script using run_python(code) that:
  - Reads the CSV with pandas
  - Fixes all identified issues
  - Saves the result using df.to_csv('cleaned_output.csv', index=False)
STEP 3: Submit IMMEDIATELY using submit_cleaned_file(path='cleaned_output.csv')

CRITICAL RULES:
- After run_python saves the file with df.to_csv(), you MUST submit immediately
- Do NOT use write_file — it is too slow for large files
- Do NOT run extra verification steps after saving
- Do NOT run more than 2 Python commands total
- The file path MUST be exactly 'cleaned_output.csv'
"""


def _tools_to_openai_format(tools) -> List[dict]:
    """Convert MCP tools to OpenAI function-calling format."""
    openai_tools = []
    for tool in tools:
        properties = {}
        required = []
        # Use input_schema (snake_case) - the attribute name in openenv Tool class
        schema = tool.input_schema
        if schema and "properties" in schema:
            for name, prop_schema in schema["properties"].items():
                properties[name] = {
                    "type": prop_schema.get("type", "string"),
                    "description": prop_schema.get("description", ""),
                }
            required = schema.get("required", [])

        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )
    return openai_tools


# ---------------------------------------------------------------------------
# Gameplay
# ---------------------------------------------------------------------------


async def play_episode(
    env: DataCleanEnv,
    client: OpenAI,
    tools: List[dict],
    episode_num: int,
) -> Dict[str, Any]:
    """Play a single data cleaning episode."""
    tool_names = [t["function"]["name"] for t in tools]

    result = await env.reset()

    # Get task info from state (metadata is excluded from serialization)
    state = await env.state()
    task_id = str(uuid.uuid4())[:8]

    # State contains task info that we need
    task_level = getattr(state, "task_level", "unknown") or "unknown"
    task_description = getattr(state, "task_description", "") or ""
    messy_file_path = getattr(state, "messy_file_path", "") or ""
    # Extract just the filename from the full path
    messy_file = os.path.basename(messy_file_path) if messy_file_path else ""

    # [START] log - required hackathon format
    print(
        f"[START] task_id={task_id}, episode={episode_num}, level={task_level}, messy_file={messy_file}"
    )

    if VERBOSE:
        print(f"\n{'=' * 60}")
        print(f"Episode {episode_num}")
        print(f"{'=' * 60}")
        print(f"Level: {task_level}")
        print(f"Task: {task_description}")
        print(f"Messy file: {messy_file}")

    chat_history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Task: {task_description}\n\nThe messy data file is located at: {messy_file}\n\nPlease clean this file and submit the cleaned version.",
        },
    ]
    step_count = 0

    while not result.done:
        step_count += 1
        if VERBOSE:
            print(f"\n--- Step {step_count} ---")

        # Retry loop for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=chat_history,
                    tools=tools,
                    tool_choice="auto",
                    max_completion_tokens=MAX_TOKENS,
                )
                break
            except RateLimitError as e:
                wait_time = 30 * (attempt + 1)
                print(
                    f"[WARN] Rate limit hit, waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    print(
                        f"[ERROR] Rate limit exceeded after {max_retries} retries. Submitting best effort."
                    )
                    # Force submit
                    tool_name = "submit_cleaned_file"
                    tool_args = {"path": "cleaned_output.csv"}
                    tool_call_id = "none"
                    response = None
                    break

        # If rate limit exhausted, skip to action execution with forced submit
        if response is None:
            pass  # tool_name, tool_args, tool_call_id already set above
        else:
            message = response.choices[0].message

            if not message.tool_calls:
                # Fallback if no tool call
                tool_name = "submit_cleaned_file"
                tool_args = {"path": "cleaned.csv"}
                tool_call_id = "none"
            else:
                tool_call_obj = message.tool_calls[0]
                tool_name = tool_call_obj.function.name
                tool_args = json.loads(tool_call_obj.function.arguments)
                tool_call_id = tool_call_obj.id

            chat_history.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args)
                                if tool_call_id == "none"
                                else tool_call_obj.function.arguments,
                            },
                        }
                    ],
                }
            )

        # [STEP] log - required hackathon format
        args_str_log = json.dumps(tool_args)
        if len(args_str_log) > 200:
            args_str_log = args_str_log[:200] + "..."
        print(
            f"[STEP] task_id={task_id}, step={step_count}, tool={tool_name}, args={args_str_log}"
        )

        if VERBOSE:
            args_str = json.dumps(tool_args)
            if len(args_str) > 100:
                args_str = args_str[:100] + "..."
            print(f"Tool: {tool_name}({args_str})")

        if tool_name not in tool_names:
            tool_name = "submit_cleaned_file"
            tool_args = {"path": "cleaned.csv"}

        # Execute action
        action = CallToolAction(tool_name=tool_name, arguments=tool_args)
        step_result = await env.step(action)
        obs = step_result.observation
        result_text = (
            obs.result if hasattr(obs, "result") else str(getattr(obs, "metadata", {}))
        )

        if not step_result.done:
            chat_history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": str(result_text) or "No result",
                }
            )

        if VERBOSE:
            result_preview = str(result_text)[:300]
            print(f"Result: {result_preview}...")

        # Update result for while loop condition
        result = step_result

    reward = result.reward or 0.0

    # [END] log - required hackathon format
    print(
        f"[END] task_id={task_id}, episode={episode_num}, level={task_level}, reward={reward:.4f}, steps={step_count}"
    )

    if VERBOSE:
        outcome = "SUCCESS" if reward > 0.7 else "PARTIAL" if reward > 0.3 else "FAILED"
        print(f"\nResult: {outcome}")
        print(f"  Reward: {reward:.2f}")

    return {
        "episode": episode_num,
        "task_id": task_id,
        "level": task_level,
        "reward": reward,
        "steps": step_count,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


async def async_main() -> None:
    if not API_KEY:
        raise SystemExit("API_KEY (or HF_TOKEN) must be set to query the model.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Connect to environment
    # In production (HF Spaces), this will connect to the running server
    # For local testing, use from_docker_image
    env_url = os.getenv("ENV_URL", "http://localhost:8000")

    async with DataCleanEnv(base_url=env_url) as env:
        # Discover tools via MCP and convert to OpenAI format
        mcp_tools = await env.list_tools()
        tools = _tools_to_openai_format(mcp_tools)

        if VERBOSE:
            tool_names = [t["function"]["name"] for t in tools]
            print(f"API: {API_BASE_URL}")
            print(f"Model: {MODEL_NAME}")
            print(f"Tools: {tool_names}")

        results = []
        for episode_num in range(1, MAX_EPISODES + 1):
            episode_result = await play_episode(env, client, tools, episode_num)
            results.append(episode_result)

        # Summary
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")

        avg_reward = sum(r["reward"] for r in results) / len(results)
        avg_steps = sum(r["steps"] for r in results) / len(results)

        print(f"Episodes: {len(results)}")
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Average steps: {avg_steps:.1f}")

        # Print breakdown by level
        by_level = {}
        for r in results:
            level = r["level"]
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(r["reward"])

        for level, rewards in sorted(by_level.items()):
            avg = sum(rewards) / len(rewards)
            print(f"  {level}: {avg:.2f} ({len(rewards)} episodes)")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
