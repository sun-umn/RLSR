#!/usr/bin/env python3
"""
Merge LoRA adapter with base model to create a full model checkpoint.
Usage: python merge_lora.py <checkpoint_path> [--output_dir <output_path>]
"""

import argparse
import os
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_checkpoint(checkpoint_path, output_dir=None):
    """Merge LoRA adapter with base model."""
    checkpoint_path = Path(checkpoint_path)

    # Check if it's a LoRA checkpoint
    adapter_config_path = checkpoint_path / "adapter_config.json"
    if not adapter_config_path.exists():
        print(f"Error: {checkpoint_path} is not a LoRA checkpoint (no adapter_config.json found)")
        return None

    # Read adapter config to get base model
    with open(adapter_config_path) as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        print("Error: Could not find base_model_name_or_path in adapter_config.json")
        return None

    print(f"Base model: {base_model_name}")
    print(f"Loading base model...")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        device_map="cpu",  # Load on CPU for merging
        trust_remote_code=True
    )

    print(f"Loading LoRA adapter from {checkpoint_path}...")
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, str(checkpoint_path))

    print("Merging LoRA weights with base model...")
    # Merge and unload
    model = model.merge_and_unload()

    # Set output directory
    if output_dir is None:
        output_dir = checkpoint_path.parent / f"{checkpoint_path.name}-merged"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to {output_dir}...")
    # Save merged model
    model.save_pretrained(output_dir, safe_serialization=True)

    # Copy tokenizer
    print("Copying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    print(f"✓ Successfully merged and saved to: {output_dir}")
    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("checkpoint_path", type=str, help="Path to LoRA checkpoint")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: <checkpoint_path>-merged)")

    args = parser.parse_args()

    merge_lora_checkpoint(args.checkpoint_path, args.output_dir)


if __name__ == "__main__":
    main()
