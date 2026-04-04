#!/usr/bin/env python3
"""
GRPO Training for Data Cleaning Agent

Trains an LLM to clean messy CSV files using reinforcement learning with
Group Relative Policy Optimization (GRPO). The model learns to:
- Read and analyze messy data
- Write Python cleaning code
- Submit cleaned files for grading

Based on the OpenEnv GRPO pattern (see tutorial/examples/wordle.py).

Usage:
    # Option 1: Colocated vLLM (1 GPU)
    python train_grpo.py --vllm-mode colocate

    # Option 2: Separate vLLM server (2 GPUs)
    # Terminal 1:
    CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 8000
    # Terminal 2:
    CUDA_VISIBLE_DEVICES=1 python train_grpo.py --vllm-mode server --vllm-server-url http://localhost:8000

    # Option 3: Local environment + colocated vLLM
    # Start env server:
    uvicorn data_clean_env.server.app:app --port 8001
    # Then train:
    python train_grpo.py --vllm-mode colocate --env-url http://localhost:8001
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset
from transformers import AutoTokenizer

# TRL imports — install with: pip install trl[vllm]
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

# Add data_clean_env to path
sys.path.insert(0, str(Path(__file__).parent))

from data_clean_env.client import DataCleanEnv
from data_clean_env.models import AVAILABLE_TOOLS

try:
    from openenv.core.env_server.mcp_types import CallToolAction
except ImportError:

    class CallToolAction:
        def __init__(self, tool_name: str, arguments: dict):
            self.tool_name = tool_name
            self.arguments = arguments


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a data cleaning expert. Your task is to identify and fix data quality issues in CSV files.

Available tools:
- read_file(path): Read a file from the workspace
- run_python(code): Execute Python code (pandas, numpy available)
- write_file(path, content): Write content to a file
- submit_cleaned_file(path): Submit your cleaned file for grading

Process:
1. Read the messy input file
2. Analyze data quality issues
3. Write and run Python code to clean the data
4. Write the cleaned file
5. Submit the cleaned file

Always use tools. Reply with tool calls only. Wrap tool calls in [TOOL_CALL] blocks."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GRPO training for Data Cleaning Agent."
    )
    parser.add_argument(
        "--tokenizer-id",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model identifier for tokenizer.",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model identifier for GRPOTrainer.",
    )
    parser.add_argument(
        "--env-url",
        type=str,
        default="http://localhost:8000",
        help="URL for the data cleaning environment server.",
    )
    parser.add_argument(
        "--dataset-prompt",
        default="Clean the messy data file and submit the cleaned version.",
        help="Prompt text used to seed the training dataset.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=500,
        help="Number of entries in the synthetic training dataset.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Maximum tool call steps per episode.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens per model generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for rollout generation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-6,
        help="Learning rate for GRPO training.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=32,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Warmup steps for scheduler.",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=1,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="Number of rollout generations per prompt.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for training outputs and checkpoints.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run name for logging.",
    )
    parser.add_argument(
        "--vllm-mode",
        choices=("colocate", "server"),
        default="colocate",
        help="vLLM execution mode.",
    )
    parser.add_argument(
        "--vllm-server-url",
        type=str,
        default="http://localhost:8000",
        help="URL for vLLM server (used when --vllm-mode=server).",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Logging frequency.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable verbose debugging.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Tool schema conversion
# ---------------------------------------------------------------------------


def build_tool_schemas() -> List[Dict[str, Any]]:
    """Build OpenAI-style tool schemas for the data cleaning tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from the workspace",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the file in the workspace",
                        }
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_python",
                "description": "Execute Python code in the workspace environment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute",
                        }
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file in the workspace",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the file",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_cleaned_file",
                "description": "Submit a cleaned file as the final answer. This terminates the episode.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the cleaned file",
                        }
                    },
                    "required": ["path"],
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------


def extract_tool_call(text: str) -> tuple[str, dict] | None:
    """Extract a tool call from model output.

    Supports formats:
    - [TOOL_CALL] tool_name(arg1=val1, arg2=val2)
    - JSON: {"tool": "name", "args": {...}}
    - Plain: tool_name(arg1=val1)
    """
    import re

    # Try [TOOL_CALL] format
    match = re.search(r"\[TOOL_CALL\]\s*(\w+)\((.*?)\)", text, re.DOTALL)
    if match:
        tool_name = match.group(1)
        args_str = match.group(2)
        args = {}
        for m in re.finditer(
            r"(\w+)\s*=\s*(.+?)(?:,\s*(?=\w+=)|$)", args_str, re.DOTALL
        ):
            key = m.group(1)
            val = m.group(2).strip().strip("\"'")
            args[key] = val
        return tool_name, args

    # Try JSON format
    match = re.search(r'\{[^{}]*"tool"[^{}]*\}', text)
    if match:
        try:
            obj = json.loads(match.group())
            return obj.get("tool"), obj.get("args", {})
        except json.JSONDecodeError:
            pass

    # Try plain function call
    match = re.search(r"(\w+)\((.*?)\)", text, re.DOTALL)
    if match:
        tool_name = match.group(1)
        if tool_name in AVAILABLE_TOOLS:
            args_str = match.group(2)
            args = {}
            for m in re.finditer(
                r"(\w+)\s*=\s*(.+?)(?:,\s*(?=\w+=)|$)", args_str, re.DOTALL
            ):
                key = m.group(1)
                val = m.group(2).strip().strip("\"'")
                args[key] = val
            return tool_name, args

    return None


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def rollout_once(
    trainer: GRPOTrainer,
    env,  # SyncEnvClient wrapper
    tokenizer: AutoTokenizer,
    dataset_prompt: str,
    system_prompt: str,
    max_steps: int,
    tool_schemas: List[Dict[str, Any]],
    debug: bool = False,
) -> Dict[str, Any]:
    """Play one episode and collect training data."""
    result = env.reset()
    obs = result.observation

    task_level = obs.metadata.get("task_level", "unknown")
    task_description = obs.metadata.get("task_description", "")
    messy_file = obs.metadata.get("messy_file", "")
    workspace_dir = obs.metadata.get("workspace_dir", "")

    prompt_ids: List[int] = []
    completion_ids: List[int] = []
    logprobs: List[float] = []
    step_rewards: List[float] = []
    tool_call_success: List[float] = []
    step_count = 0
    tools_used: List[str] = []

    # Build initial prompt
    user_prompt = (
        f"Task: {task_description}\n\n"
        f"The messy data file is at: {messy_file}\n"
        f"Workspace directory: {workspace_dir}\n\n"
        f"Clean this data file and submit the cleaned version."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for step in range(max_steps):
        if result.done:
            break

        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Generate model response
        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        step_prompt_ids = rollout_outputs["prompt_ids"]
        step_completion_ids = rollout_outputs["completion_ids"]
        step_logprobs = rollout_outputs["logprobs"]

        prompt_ids.extend(step_prompt_ids)
        completion_ids.extend(step_completion_ids)
        logprobs.extend(step_logprobs)

        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            step_completion_ids, skip_special_tokens=True
        )

        if debug:
            print(f"\n  Step {step + 1} completion: {completion_text[:200]}...")

        # Extract tool call
        tool_call = extract_tool_call(completion_text)

        if tool_call is None:
            # Fallback: submit if we've done enough steps
            if step >= 2:
                tool_name = "submit_cleaned_file"
                tool_args = {"path": "cleaned.csv"}
            else:
                tool_name = "read_file"
                tool_args = {"path": messy_file}
            success = 0.0
        else:
            tool_name, tool_args = tool_call
            success = 1.0 if tool_name in AVAILABLE_TOOLS else 0.0

        tool_call_success.append(success)
        tools_used.append(tool_name)

        if debug:
            print(f"  Tool: {tool_name}({tool_args})")

        # Execute action
        action = CallToolAction(tool_name=tool_name, arguments=tool_args)
        result = env.step(action)
        obs = result.observation
        step_count += 1

        # Get tool result for feedback
        tool_result = ""
        if hasattr(obs, "result"):
            tool_result = str(obs.result)
        elif obs.metadata:
            tool_result = str(obs.metadata.get("tool_result", ""))

        # Step reward: small positive for valid tool calls
        step_r = 0.01 * success
        step_rewards.append(step_r)

        # Add tool result to conversation
        messages.append(
            {
                "role": "assistant",
                "content": completion_text,
            }
        )
        if not result.done:
            messages.append(
                {
                    "role": "user",
                    "content": f"Tool result from {tool_name}:\n{tool_result[:1000]}\n\nContinue cleaning the data.",
                }
            )

    # Final reward from environment
    final_reward = float(result.reward or 0.0)

    if debug:
        print(
            f"  Final reward: {final_reward:.2f}, Steps: {step_count}, Tools: {tools_used}"
        )

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "final_reward": final_reward,
        "step_rewards": step_rewards,
        "tool_call_success": tool_call_success,
        "task_level": task_level,
        "steps": step_count,
        "tools_used": tools_used,
    }


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def reward_final(completions: list[str], **kwargs) -> list[float]:
    """Primary reward: final environment reward (0-1 scale)."""
    rewards = kwargs.get("final_reward")
    if rewards is None:
        return [0.0] * len(completions)
    return [float(r) for r in rewards]


def reward_tool_accuracy(completions: list[str], **kwargs) -> list[float]:
    """Reward for making valid tool calls (fraction of successful calls)."""
    successes = kwargs.get("tool_call_success")
    if successes is None:
        return [0.0] * len(completions)
    results = []
    for s in successes:
        avg = sum(s) / len(s) if s else 0.0
        results.append(avg * 0.1)  # Scale down to not dominate
    return results


def reward_efficiency(completions: list[str], **kwargs) -> list[float]:
    """Reward for completing the task in fewer steps (only if final reward > 0)."""
    final_rewards = kwargs.get("final_reward")
    steps_list = kwargs.get("steps")
    if final_rewards is None or steps_list is None:
        return [0.0] * len(completions)
    results = []
    for fr, steps in zip(final_rewards, steps_list):
        if fr > 0:
            # Bonus for efficiency: max 15 steps, fewer is better
            efficiency = max(0.0, 1.0 - (steps / 15.0)) * 0.05
            results.append(efficiency)
        else:
            results.append(0.0)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def sanitize_name(name: str) -> str:
    return name.replace("/", "-")


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Data Cleaning Agent — GRPO Training")
    print("=" * 60)
    print(f"Model: {args.model_id}")
    print(f"Environment: {args.env_url}")
    print(f"vLLM mode: {args.vllm_mode}")
    print(f"Max steps/episode: {args.max_steps}")
    print(f"Num generations: {args.num_generations}")
    print()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Environment client - use sync wrapper for training loop
    env = DataCleanEnv(base_url=args.env_url).sync()

    # Test connection
    print("Testing environment connection...")
    try:
        result = env.reset()
        print(
            f"  Connected! Task level: {result.observation.metadata.get('task_level')}"
        )
    except Exception as e:
        print(f"  ERROR: Could not connect to environment: {e}")
        print(f"  Make sure the server is running at {args.env_url}")
        sys.exit(1)

    # Tool schemas
    tool_schemas = build_tool_schemas()
    tool_names = [t["function"]["name"] for t in tool_schemas]
    print(f"  Tools: {tool_names}")
    print()

    # Training dataset
    dataset = Dataset.from_dict({"prompt": [args.dataset_prompt] * args.dataset_size})
    print(f"Dataset: {args.dataset_size} prompts")

    # Output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_output = (
        Path("outputs") / f"data-clean-grpo-{sanitize_name(args.model_id)}-{timestamp}"
    )
    output_dir = Path(args.output_dir or default_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # GRPO config
    grpo_config = GRPOConfig(
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_url
        if args.vllm_mode == "server"
        else None,
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        warmup_steps=args.warmup_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        logging_steps=args.logging_steps,
        report_to="tensorboard",
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    grpo_config.run_name = args.run_name or f"data-clean-{timestamp}"
    grpo_config.project = f"data-clean-{sanitize_name(args.model_id)}"

    # Rollout function
    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        episode_prompt_ids: list[list[int]] = []
        episode_completion_ids: list[list[int]] = []
        episode_logprobs: list[list[float]] = []
        final_rewards: list[float] = []
        tool_successes: list[list[float]] = []
        steps_list: list[int] = []

        for prompt_text in prompts:
            episode = rollout_once(
                trainer=trainer,
                env=env,
                tokenizer=tokenizer,
                dataset_prompt=prompt_text,
                system_prompt=SYSTEM_PROMPT,
                max_steps=args.max_steps,
                tool_schemas=tool_schemas,
                debug=args.debug,
            )
            episode_prompt_ids.append(episode["prompt_ids"])
            episode_completion_ids.append(episode["completion_ids"])
            episode_logprobs.append(episode["logprobs"])
            final_rewards.append(episode["final_reward"])
            tool_successes.append(episode["tool_call_success"])
            steps_list.append(episode["steps"])

        return {
            "prompt_ids": episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "logprobs": episode_logprobs,
            "final_reward": final_rewards,
            "tool_call_success": tool_successes,
            "steps": steps_list,
        }

    # Create trainer
    trainer = GRPOTrainer(
        model=args.model_id,
        processing_class=tokenizer,
        reward_funcs=[
            reward_final,
            reward_tool_accuracy,
            reward_efficiency,
        ],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )

    print("Starting GRPO training...")
    print(f"Output dir: {output_dir}")
    print(f"Reward functions: final_reward, tool_accuracy, efficiency")
    print()

    try:
        trainer.train()
    finally:
        env.close()

    print()
    print("=" * 60)
    print("Training complete!")
    print(f"Model saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
