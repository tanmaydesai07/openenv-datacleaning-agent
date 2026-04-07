#!/usr/bin/env python3
"""
Local GRPO Training for Data Cleaning Agent
Optimized for RTX 4050 (6GB VRAM) + 16GB RAM

Uses real environment interaction with CSV grading rewards.
No vLLM needed — direct model generation.

Usage:
    pip install trl transformers accelerate peft bitsandbytes datasets pandas
    python train_local.py
"""

import os
import re
import sys
import random
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from data_clean_env.server.environment import DataCleanEnvironment
from openenv.core.env_server.mcp_types import CallToolAction

# ============================================================================
# CONFIGURATION - Optimized for 6GB VRAM
# ============================================================================

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"  # 0.5B fits in 6GB with 4-bit
TASKS_PATH = str(Path(__file__).parent / "data_clean_env" / "tasks")
MAX_STEPS = 5  # Reduced from 10 to limit memory usage per episode

CONFIG = {
    "learning_rate": 5e-5,  # Increased for faster learning
    "num_train_epochs": 2,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_completion_length": 256,
    "num_generations": 2,
    "beta": 0.05,  # Lower beta = more exploration
    "temperature": 1.0,  # Higher temp = more diverse outputs
    "output_dir": "./dataclean-local-checkpoint",
    "logging_steps": 1,
    "save_steps": 25,
    "warmup_steps": 3,  # Faster warmup
}

SYSTEM_PROMPT = """You are a data cleaning agent. Clean messy CSV files using these tools:
- read_file(path): Read a file
- run_python(code): Execute Python code (pandas available)
- submit_cleaned_file(path): Submit cleaned file for grading

Always save cleaned data with: df.to_csv('cleaned_output.csv', index=False)
Then submit with: submit_cleaned_file(path='cleaned_output.csv')"""


# ============================================================================
# TOOL CALL EXTRACTION
# ============================================================================


def extract_tool_call(text: str):
    """Extract tool call from model output. Returns (tool_name, args, is_valid_format)."""
    match = re.search(
        r"(read_file|run_python|write_file|submit_cleaned_file)\s*\((.*)\)",
        text,
        re.DOTALL,
    )
    if match:
        tool_name = match.group(1)
        args_str = match.group(2)
        args = {}
        if tool_name == "run_python":
            code_match = re.search(r'code\s*=\s*["\'](.*)["\']', args_str, re.DOTALL)
            if code_match:
                args["code"] = code_match.group(1)
        else:
            for m in re.finditer(r'(\w+)\s*=\s*["\']([^"\']*)["\']', args_str):
                args[m.group(1)] = m.group(2)
        if args:
            return tool_name, args, True  # Valid tool call
    return None, None, False  # Invalid format


def compute_format_bonus(text: str) -> float:
    """Compute bonus reward for proper formatting (creates variance for GRPO)."""
    bonus = 0.0
    
    # Bonus for containing any tool call pattern
    if re.search(r"(read_file|run_python|submit_cleaned_file)\s*\(", text):
        bonus += 0.05
    
    # Bonus for proper argument format
    if re.search(r'\w+\s*=\s*["\']', text):
        bonus += 0.03
    
    # Bonus for pandas keywords (shows understanding)
    pandas_keywords = ["pd.read_csv", "to_csv", "drop_duplicates", "dropna", "fillna", "strip"]
    for kw in pandas_keywords:
        if kw in text:
            bonus += 0.02
    
    # Cap the bonus
    return min(bonus, 0.15)


# ============================================================================
# DIRECT GENERATION WITH LOGPROBS
# ============================================================================


def generate_with_logprobs(
    model, tokenizer, prompt_text, max_new_tokens=512, temperature=0.9
):
    """Generate text and compute logprobs without vLLM."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = outputs.sequences[0][prompt_len:].cpu()
    scores = outputs.scores

    logprobs = []
    for i, score in enumerate(scores):
        probs = torch.softmax(score[0], dim=-1)
        token_id = generated_ids[i].item()
        logprob = torch.log(probs[token_id]).item()
        logprobs.append(logprob)
        del probs  # Free memory immediately

    completion_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Clean up
    result = {
        "prompt_ids": inputs["input_ids"][0].cpu().tolist(),
        "completion_ids": generated_ids.tolist(),
        "logprobs": logprobs,
        "text": completion_text,
    }
    
    del inputs, outputs, generated_ids, scores
    torch.cuda.empty_cache()
    
    return result


# ============================================================================
# EPISODE ROLLOUT
# ============================================================================


def run_single_episode(env, model, tokenizer):
    """Run one episode using direct model generation + real environment."""
    obs = env.reset()
    task_level = obs.metadata.get("task_level", "unknown")
    messy_file = obs.metadata.get("messy_file", "")

    prompt_ids_list = []
    completion_ids_list = []
    logprobs_list = []
    
    # Track if model has actually done cleaning work
    has_read_file = False
    has_run_python = False
    
    # Track format bonuses and penalties for learning signal
    format_bonus = 0.0
    fallback_penalty = 0.0
    valid_tool_calls = 0

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: Clean this file: {messy_file}\n\nFirst read the file, then use run_python to clean it with pandas, save to 'cleaned_output.csv', and submit."},
    ]

    for step in range(MAX_STEPS):
        if obs.done:
            break

        prompt_text_full = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        rollout = generate_with_logprobs(model, tokenizer, prompt_text_full)

        prompt_ids_list.extend(rollout["prompt_ids"])
        completion_ids_list.extend(rollout["completion_ids"])
        logprobs_list.extend(rollout["logprobs"])

        completion_text = rollout["text"]
        tool_name, tool_args, is_valid = extract_tool_call(completion_text)
        
        # Track format quality for reward shaping
        format_bonus += compute_format_bonus(completion_text)
        
        if is_valid:
            valid_tool_calls += 1

        # Fallback logic with scaffolding to ensure learning signal
        used_fallback = False
        if tool_name is None:
            used_fallback = True
            fallback_penalty += 0.10  # INCREASED: Strong penalty for fallback
            if not has_read_file:
                tool_name = "read_file"
                tool_args = {"path": messy_file}
            elif not has_run_python:
                tool_name = "run_python"
                tool_args = {"code": f"""
import pandas as pd
df = pd.read_csv('{messy_file}')
df = df.drop_duplicates()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()
df = df.dropna(how='all')
df.to_csv('cleaned_output.csv', index=False)
print('Saved cleaned_output.csv')
"""}
            else:
                tool_name = "submit_cleaned_file"
                tool_args = {"path": "cleaned_output.csv"}
        
        # Track what actions have been taken
        if tool_name == "read_file":
            has_read_file = True
        elif tool_name == "run_python":
            has_run_python = True

        fallback_marker = " (fallback)" if used_fallback else ""
        print(f"    Step {step + 1}: {tool_name}{fallback_marker}")

        action = CallToolAction(tool_name=tool_name, arguments=tool_args)
        obs = env.step(action)

        tool_result = ""
        if hasattr(obs, "result"):
            tool_result = str(obs.result)[:200]
        elif obs.metadata:
            tool_result = str(obs.metadata.get("tool_result", ""))[:200]

        messages.append({"role": "assistant", "content": completion_text})
        if not obs.done:
            messages.append(
                {"role": "user", "content": f"Result: {tool_result}. Continue."}
            )

    # Compute final shaped reward with noise for variance
    base_reward = float(obs.reward or 0.0)
    
    # Add small random noise to create variance between episodes (critical for GRPO)
    noise = random.uniform(-0.05, 0.05)
    
    shaped_reward = base_reward + format_bonus - fallback_penalty + noise
    shaped_reward = max(0.0, min(1.0, shaped_reward))  # Clamp to [0, 1]
    
    print(f"    → Reward: {shaped_reward:.2f} (base={base_reward:.2f}, format=+{format_bonus:.2f}, fallback=-{fallback_penalty:.2f})")

    return {
        "prompt_ids": prompt_ids_list,
        "completion_ids": completion_ids_list,
        "logprobs": logprobs_list,
        "final_reward": shaped_reward,
        "task_level": task_level,
        "valid_tool_calls": valid_tool_calls,
    }


# ============================================================================
# ROLLOUT FUNCTION FOR GRPO
# ============================================================================


def rollout_func(prompts: List[str], trainer) -> Dict[str, List]:
    """Rollout function — direct model generation + real environment interaction."""
    model = trainer.model
    tokenizer = trainer.processing_class

    episode_prompt_ids = []
    episode_completion_ids = []
    episode_logprobs = []
    final_rewards = []

    # Force no gradient computation during rollout
    with torch.no_grad():
        for i, prompt_text in enumerate(prompts):
            print(f"  Episode {i + 1}/{len(prompts)}")
            env = DataCleanEnvironment(tasks_path=TASKS_PATH, max_steps=MAX_STEPS)
            episode = run_single_episode(env, model, tokenizer)
            episode_prompt_ids.append(episode["prompt_ids"])
            episode_completion_ids.append(episode["completion_ids"])
            episode_logprobs.append(episode["logprobs"])
            final_rewards.append(episode["final_reward"])
            del env, episode
            
            # Clear GPU cache after each episode
            torch.cuda.empty_cache()

    return {
        "prompt_ids": episode_prompt_ids,
        "completion_ids": episode_completion_ids,
        "logprobs": episode_logprobs,
        "final_reward": final_rewards,
    }


# ============================================================================
# REWARD FUNCTION
# ============================================================================


def reward_final(completions: list, **kwargs) -> list:
    """Primary reward: real environment CSV grading (0-1 scale)."""
    rewards = kwargs.get("final_reward")
    if rewards is None:
        return [0.0] * len(completions)
    return [float(r) for r in rewards]


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("\n" + "=" * 60)
    print("DATACLEAN LOCAL GRPO TRAINING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name}")
    print(f"✅ VRAM: {gpu_memory:.1f} GB")

    # Create dataset
    prompts = [
        f"{SYSTEM_PROMPT}\n\nTask: Clean customer dataset. Remove duplicates, handle missing values. File: easy_messy.csv",
        f"{SYSTEM_PROMPT}\n\nTask: Clean sales records. Fix dates, casing, emails, whitespace. File: medium_messy.csv",
        f"{SYSTEM_PROMPT}\n\nTask: Clean product inventory. Fix IDs, negatives, prices, dates. File: hard_messy.csv",
        f"{SYSTEM_PROMPT}\n\nTask: Clean HR records. Fix names, salaries, dates, emails. File: hr_messy.csv",
        f"{SYSTEM_PROMPT}\n\nTask: Clean patient records. Fix names, conditions, dates. File: healthcare_messy.csv",
        f"{SYSTEM_PROMPT}\n\nTask: Clean financial transactions. Fix merchants, amounts, dates. File: finance_messy.csv",
        f"{SYSTEM_PROMPT}\n\nTask: Clean shipping records. Fix locations, carriers, dates. File: logistics_messy.csv",
        f"{SYSTEM_PROMPT}\n\nTask: Clean student records. Fix names, subjects, dates. File: education_messy.csv",
    ]

    train_dataset = Dataset.from_dict({"prompt": prompts * 4})
    print(f"✅ Dataset: {len(train_dataset)} samples ({len(prompts)} unique prompts)")

    # Load tokenizer
    print(f"\n📥 Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with 4-bit quantization
    print(f"📥 Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for better stability
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,  # Match compute dtype
        device_map={"": 0},  # Force single GPU
        trust_remote_code=True,
        attn_implementation="eager",
    )

    print(f"✅ Model loaded: {model.num_parameters() / 1e9:.2f}B parameters")
    print(f"   GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # LoRA config - increased rank for more learning capacity
    lora_config = LoraConfig(
        r=16,  # Increased from 8
        lora_alpha=32,  # Increased proportionally
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # More layers
        task_type="CAUSAL_LM",
    )

    # GRPO config
    training_args = GRPOConfig(
        output_dir=CONFIG["output_dir"],
        learning_rate=CONFIG["learning_rate"],
        num_train_epochs=CONFIG["num_train_epochs"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        num_generations=CONFIG["num_generations"],
        max_completion_length=CONFIG["max_completion_length"],
        temperature=CONFIG["temperature"],
        beta=CONFIG["beta"],  # KL penalty coefficient
        logging_steps=CONFIG["logging_steps"],
        save_steps=CONFIG["save_steps"],
        save_total_limit=2,
        report_to="none",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        warmup_steps=0,  # No warmup - start learning immediately
        lr_scheduler_type="constant",  # Keep LR constant, don't decay to 0
    )

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_final],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
        rollout_func=rollout_func,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✅ GRPO Trainer ready")
    print(f"   Trainable params: {trainable / 1e6:.1f}M")
    print(f"   Rollout: Direct generation + real environment")

    # Train
    print(f"\n{'=' * 60}")
    print(f"🚀 STARTING TRAINING")
    print(f"{'=' * 60}")
    print(f"Model: {MODEL_ID}")
    print(f"Samples: {len(train_dataset)}")
    print(f"Epochs: {CONFIG['num_train_epochs']}")
    print(f"Generations: {CONFIG['num_generations']} per prompt")
    print(f"Reward: Real CSV grading")
    print()

    train_result = trainer.train()

    trainer.save_model(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])

    print()
    print("=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Model saved to: {CONFIG['output_dir']}")
    print(f"Final loss: {train_result.training_loss:.4f}")


if __name__ == "__main__":
    main()
