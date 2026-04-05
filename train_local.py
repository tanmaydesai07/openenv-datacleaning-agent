#!/usr/bin/env python3
"""
Local GRPO Training for Data Cleaning Agent
Optimized for RTX 4050 (6GB VRAM) + 16GB RAM

This script uses:
- 4-bit quantization (QLoRA) to fit in 6GB VRAM
- Small model (Qwen2.5-0.5B or 1.5B)
- Gradient checkpointing for memory efficiency
- Heuristic reward function (no env server needed)

Usage:
    pip install -r requirements-train.txt
    python train_local.py
"""

import os
import sys
import torch
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================================
# CONFIGURATION - Optimized for 6GB VRAM
# ============================================================================

# Use smallest model for 6GB VRAM
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # Only 0.5B params, fits easily in 6GB

# Training config - very conservative for low VRAM
CONFIG = {
    "learning_rate": 1e-4,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_completion_length": 256,      # Short completions
    "num_generations": 1,               # Single generation per prompt
    "beta": 0.1,
    "temperature": 0.7,
    "output_dir": "./dataclean-local-checkpoint",
    "logging_steps": 1,
    "save_steps": 50,
    "warmup_steps": 5,
}

# System prompt for data cleaning
SYSTEM_PROMPT = """You are a data cleaning expert. Clean messy CSV files using these tools:

Tools:
- read_file(path): Read the messy CSV file
- run_python(code): Run pandas code to clean data
- submit_cleaned_file(path): Submit cleaned file

Steps:
1. read_file("messy.csv") - read the data
2. run_python(code) - clean with pandas (drop_duplicates, fillna, strip, etc.)
3. submit_cleaned_file("cleaned_output.csv") - submit result

Always save cleaned data to 'cleaned_output.csv' before submitting."""


def check_gpu():
    """Check GPU availability and memory."""
    print("=" * 60)
    print("GPU CHECK")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Training will be very slow on CPU.")
        print("   Install CUDA: https://developer.nvidia.com/cuda-downloads")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"✅ GPU: {gpu_name}")
    print(f"✅ VRAM: {gpu_memory:.1f} GB")
    
    if gpu_memory < 4:
        print("⚠️  Warning: Less than 4GB VRAM. Training may fail.")
        print("   Consider using CPU offloading or smaller model.")
    
    return True


def create_training_dataset():
    """Create training dataset from task files."""
    print("\n" + "=" * 60)
    print("CREATING TRAINING DATASET")
    print("=" * 60)
    
    tasks_path = Path(__file__).parent / "data_clean_env" / "tasks"
    
    # Find messy CSV files
    messy_files = list(tasks_path.glob("*_messy.csv")) + list(tasks_path.glob("*messy*.csv"))
    
    if not messy_files:
        print("⚠️  No messy CSV files found. Creating sample prompts...")
        # Fallback prompts
        prompts = [
            f"{SYSTEM_PROMPT}\n\nTask: Clean customer data with duplicates. File: easy_messy.csv",
            f"{SYSTEM_PROMPT}\n\nTask: Clean sales data with missing values. File: medium_messy.csv",
            f"{SYSTEM_PROMPT}\n\nTask: Clean employee records with inconsistent formats. File: hr_messy.csv",
        ]
    else:
        print(f"✅ Found {len(messy_files)} messy CSV files")
        prompts = []
        for f in messy_files:
            task_name = f.stem.replace("_messy", "").replace("messy_", "")
            prompt = f"{SYSTEM_PROMPT}\n\nTask: Clean the {task_name} dataset. File: {f.name}"
            prompts.append(prompt)
    
    # Repeat for more samples
    prompts = prompts * 4  # ~12-20 samples
    
    dataset = Dataset.from_dict({"prompt": prompts})
    print(f"✅ Created dataset with {len(dataset)} samples")
    
    return dataset


def reward_fn(completions, **kwargs):
    """
    Heuristic reward function - no environment server needed!
    
    Rewards the model for:
    1. Using correct tool sequence (read → run_python → submit)
    2. Using pandas cleaning operations
    3. Saving to correct output file
    """
    rewards = []
    
    for completion in completions:
        if isinstance(completion, list):
            text = " ".join([str(t) for t in completion])
        else:
            text = str(completion)
        
        reward = 0.0
        
        # Reward for reading file first
        if "read_file" in text:
            reward += 0.15
        
        # Reward for using Python/pandas
        if "run_python" in text:
            reward += 0.25
            # Bonus for actual cleaning operations
            cleaning_ops = [
                "drop_duplicates", "fillna", "dropna", "strip", 
                "to_datetime", "str.lower", "str.upper", "str.title",
                "replace", "astype", "rename"
            ]
            for op in cleaning_ops:
                if op in text:
                    reward += 0.03
        
        # Reward for saving cleaned data
        if "to_csv" in text:
            reward += 0.15
            if "cleaned" in text.lower():
                reward += 0.1
        
        # Reward for submitting
        if "submit_cleaned_file" in text:
            reward += 0.2
            if "cleaned_output.csv" in text:
                reward += 0.1
        
        # Cap reward at 1.0
        reward = min(1.0, reward)
        rewards.append(reward)
    
    return rewards


def load_model_and_tokenizer():
    """Load model with 4-bit quantization for low VRAM."""
    print("\n" + "=" * 60)
    print("LOADING MODEL")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # float16 for older GPUs
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization
    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Memory usage
    memory_used = torch.cuda.memory_allocated() / 1e9
    print(f"✅ Model loaded!")
    print(f"   Parameters: {model.num_parameters() / 1e9:.2f}B")
    print(f"   GPU Memory: {memory_used:.2f} GB")
    
    return model, tokenizer


def main():
    """Main training function."""
    print("\n" + "=" * 60)
    print("DATACLEAN LOCAL TRAINING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Check GPU
    has_gpu = check_gpu()
    if not has_gpu:
        print("\n⚠️  Continuing without GPU (will be slow)...")
    
    # Create dataset
    train_dataset = create_training_dataset()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # LoRA config for memory-efficient fine-tuning
    print("\n" + "=" * 60)
    print("CONFIGURING LORA")
    print("=" * 60)
    
    lora_config = LoraConfig(
        r=8,                    # Low rank for less memory
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    
    print(f"✅ LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}")
    
    # GRPO training config
    print("\n" + "=" * 60)
    print("CONFIGURING TRAINER")
    print("=" * 60)
    
    training_args = GRPOConfig(
        output_dir=CONFIG["output_dir"],
        learning_rate=CONFIG["learning_rate"],
        num_train_epochs=CONFIG["num_train_epochs"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        num_generations=CONFIG["num_generations"],
        max_completion_length=CONFIG["max_completion_length"],
        temperature=CONFIG["temperature"],
        logging_steps=CONFIG["logging_steps"],
        save_steps=CONFIG["save_steps"],
        save_total_limit=2,
        report_to="none",               # No wandb/tensorboard
        fp16=True,                      # Use fp16 for speed
        gradient_checkpointing=True,    # Save memory
        warmup_steps=CONFIG["warmup_steps"],
        optim="adamw_torch",            # Standard optimizer
    )
    
    print(f"✅ Training config:")
    print(f"   Epochs: {CONFIG['num_train_epochs']}")
    print(f"   Batch size: {CONFIG['per_device_train_batch_size']}")
    print(f"   Gradient accumulation: {CONFIG['gradient_accumulation_steps']}")
    print(f"   Generations per prompt: {CONFIG['num_generations']}")
    print(f"   Max completion length: {CONFIG['max_completion_length']}")
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
    )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Trainer initialized")
    print(f"   Trainable params: {trainable_params / 1e6:.1f}M")
    
    # Start training
    print("\n" + "=" * 60)
    print("🚀 STARTING TRAINING")
    print("=" * 60)
    print(f"   Samples: {len(train_dataset)}")
    print(f"   Steps: {len(train_dataset) // CONFIG['gradient_accumulation_steps']}")
    print(f"   Expected time: ~30-60 minutes on RTX 4050")
    print()
    
    try:
        train_result = trainer.train()
        
        # Save model
        print("\n" + "=" * 60)
        print("SAVING MODEL")
        print("=" * 60)
        
        trainer.save_model(CONFIG["output_dir"])
        tokenizer.save_pretrained(CONFIG["output_dir"])
        
        print(f"✅ Model saved to: {CONFIG['output_dir']}")
        print(f"\n🎉 Training complete!")
        print(f"   Final loss: {train_result.training_loss:.4f}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\nTroubleshooting:")
        print("1. Reduce max_completion_length to 128")
        print("2. Use smaller model: Qwen/Qwen2.5-0.5B-Instruct")
        print("3. Close other GPU applications")
        raise


if __name__ == "__main__":
    main()
