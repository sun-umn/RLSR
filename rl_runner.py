"""
RLSR Training Script using TRL's GRPOTrainer

Builds on the RLCR codebase (the RLCR/ submodule provides the config dataclasses
and the base reward functions) with TRL's native GRPOTrainer, extended by
BatchRLSRGRPOTrainer for the batch-level RLSR selection reward.
"""

import sys
import os
import glob

# Get absolute paths
_current_dir = os.path.dirname(os.path.abspath(__file__))

# Add current directory FIRST (for local reward_fns.py / evaluation.py)
sys.path.insert(0, _current_dir)

# Add RLCR to path for imports (at position 1, so local files take precedence)
sys.path.insert(1, os.path.join(_current_dir, 'RLCR'))

from dataclasses import dataclass, field
from typing import Optional

from RLCR.arguments import GRPOScriptArguments, GRPOConfig, ModelConfig
from trl import TrlParser, get_peft_config, GRPOTrainer


@dataclass
class RLSRScriptArguments(GRPOScriptArguments):
    """RLCR's script arguments extended with the batch-level RLSR reward parameters."""

    batch_rlsr_lambda: float = field(
        default=1.0,
        metadata={"help": "λ multiplier on the positive branch of the batch RLSR reward: +λ·α̂ if correct, -α̂ if incorrect."},
    )
    batch_rlsr_max_weight: Optional[float] = field(
        default=None,
        metadata={"help": "Clip α̂ at this value. None/null = no clipping (α̂ is naturally bounded by H_n, the harmonic ceiling of the generation pool)."},
    )
from transformers import set_seed
import logging
import transformers
import datasets
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from RLCR.reward_fns import (
    format_reward,
    accuracy_reward,
    brier_reward,
    mean_confidence_reward,
    confidence_one_or_zero,
)
from reward_fns import batch_rlsr_reward_placeholder
from trainers import BatchRLSRGRPOTrainer
from evaluation import (
    AURCEvaluationCallback,
    set_global_aurc_callback,
    aurc_data_collector_reward,
)

# Import from local dataset_processing (not RLCR's)
import importlib.util
_dp_spec = importlib.util.spec_from_file_location(
    "local_dataset_processing",
    os.path.join(_current_dir, "dataset_processing.py")
)
_local_dp = importlib.util.module_from_spec(_dp_spec)
_dp_spec.loader.exec_module(_local_dp)
process_dataset = _local_dp.process_dataset

import torch

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency safety
    yaml = None


logger = logging.getLogger(__name__)


def _named_partial(func, *partial_args, name=None, **partial_kwargs):
    """Return a callable that behaves like functools.partial but keeps a __name__."""

    def wrapper(*args, **kwargs):
        merged_kwargs = {**partial_kwargs, **kwargs}
        return func(*partial_args, *args, **merged_kwargs)

    wrapper.__name__ = name or getattr(func, "__name__", "reward_func")
    return wrapper


def _load_deepspeed_gradient_accumulation_steps() -> int | None:
    """Read gradient accumulation steps from the active DeepSpeed config."""

    if yaml is None:
        logger.warning("PyYAML not available; skipping DeepSpeed gradient accumulation alignment")
        return None

    config_candidates = []
    env_config = os.environ.get("ACCELERATE_CONFIG_FILE") or os.environ.get("ACCELERATE_CONFIG")
    if env_config:
        config_candidates.append(env_config)

    repo_config = os.path.join(os.path.dirname(__file__), "RLCR", "deepspeed.yaml")
    config_candidates.append(repo_config)

    for path in config_candidates:
        if not path or not os.path.exists(path):
            continue

        try:
            with open(path, "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.warning(f"Failed to read DeepSpeed config '{path}': {exc}")
            continue

        ds_config = config.get("deepspeed_config")
        if not isinstance(ds_config, dict):
            continue

        gas = ds_config.get("gradient_accumulation_steps")
        if isinstance(gas, str):
            if gas.lower() == "auto":
                return None
            if gas.isdigit():
                return int(gas)
        elif isinstance(gas, int):
            return gas

    return None


def logger_setup(script_args, training_args, model_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")


def model_init(model_args, training_args):
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    use_cache = False if training_args.gradient_checkpointing else True
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=use_cache,
    )
    return model_kwargs


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)
    logger_setup(script_args, training_args, model_args)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    def _has_valid_deepspeed_shards(path: str) -> bool:
        """Return True if `path` looks like a valid DeepSpeed checkpoint directory.

        Transformers + DeepSpeed expects `path/global_step*/` to contain model-state shards:
          - ZeRO-3: zero_pp_rank_*_mp_rank_*_model_states.pt
          - Non-ZeRO: mp_rank_*_model_states.pt
        If those are missing, resuming will crash inside DeepSpeed.
        """
        try:
            step_dirs = sorted(glob.glob(os.path.join(path, "global_step*")))
            if not step_dirs:
                return False
            for sd in step_dirs:
                if glob.glob(os.path.join(sd, "zero_pp_rank_*_mp_rank_*_model_states.pt")):
                    return True
                if glob.glob(os.path.join(sd, "mp_rank_*_model_states.pt")):
                    return True
            return False
        except Exception as exc:
            logger.warning(f"Failed to inspect DeepSpeed checkpoint at '{path}': {exc}")
            return False

    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        if _has_valid_deepspeed_shards(last_checkpoint):
            logger.info(f"Checkpoint detected, resuming training at last_checkpoint='{last_checkpoint}'.")
        else:
            logger.warning(
                "Found checkpoint directory but no DeepSpeed shards; skipping auto-resume: %s",
                last_checkpoint,
            )
            last_checkpoint = None

    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Map long-prompt patterns to base patterns for the RLCR reward functions
    # (the "long" refers to prompt content, not output structure)
    FORMAT_PATTERN_MAP = {
        "tabc_long": "tabc",
    }
    base_format_pattern = FORMAT_PATTERN_MAP.get(script_args.format_pattern, script_args.format_pattern)

    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "format": _named_partial(
            format_reward,
            format_pattern=base_format_pattern,
        ),
        "accuracy": _named_partial(
            accuracy_reward,
            format_pattern=base_format_pattern,
        ),
        "brier": _named_partial(
            brier_reward,
            format_pattern=base_format_pattern,
        ),
        "mean_confidence": mean_confidence_reward,
        "confidence_one_or_zero": confidence_one_or_zero,
        "aurc_collector": aurc_data_collector_reward(base_format_pattern),
        "batch_rlsr": batch_rlsr_reward_placeholder,  # Placeholder - actual computation in BatchRLSRGRPOTrainer
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    # Add AURC collector if evaluation is enabled (it returns NaN so doesn't affect training)
    if training_args.eval_strategy != "no" and "aurc_collector" not in script_args.reward_funcs:
        reward_funcs.append(REWARD_FUNCS_REGISTRY["aurc_collector"])
        logger.info("*** Added AURC data collector reward function (returns NaN, only collects eval data) ***")

    dataset = process_dataset(dataset, script_args)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    model_init_kwargs = model_init(model_args, training_args)
    training_args.model_init_kwargs = model_init_kwargs

    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project

    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split]
    if script_args.train_subset_size is not None:
        train_dataset = train_dataset.select(range(script_args.train_subset_size))
    if script_args.eval_subset_size is not None:
        eval_dataset = eval_dataset.select(range(script_args.eval_subset_size))

    #############################
    # Get PEFT config for LoRA (unused for the full-finetune reproduction configs)
    #############################
    peft_config = get_peft_config(model_args)
    if peft_config is not None:
        logger.info(f"*** Using PEFT with config: {peft_config} ***")

    #############################
    # Setup callbacks
    #############################
    callbacks = []

    # Add AURC evaluation callback if evaluation is enabled
    if training_args.eval_strategy != "no":
        aurc_eval_callback = AURCEvaluationCallback(format_pattern=script_args.format_pattern)
        set_global_aurc_callback(aurc_eval_callback)
        callbacks.append(aurc_eval_callback)
        logger.info("*** AURC evaluation callback enabled - will compute AURC metrics during eval ***")

    #############################
    # Initialize Trainer
    #############################
    if "batch_rlsr" in script_args.reward_funcs:
        # Use BatchRLSRGRPOTrainer for the batch-level RLSR selection reward
        logger.info(f"*** Using BatchRLSRGRPOTrainer with λ={script_args.batch_rlsr_lambda}, "
                   f"max_weight={script_args.batch_rlsr_max_weight} (None = no clipping), "
                   f"format={script_args.format_pattern} ***")

        trainer = BatchRLSRGRPOTrainer(
            model=model_args.model_name_or_path,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
            peft_config=peft_config,
            callbacks=callbacks,
            # Batch-level RLSR specific args
            batch_level_reward_funcs=["batch_rlsr"],
            batch_rlsr_lambda=script_args.batch_rlsr_lambda,
            batch_rlsr_max_weight=script_args.batch_rlsr_max_weight,
            batch_rlsr_format_pattern=script_args.format_pattern,
        )
    else:
        # Use standard GRPOTrainer
        trainer = GRPOTrainer(
            model=model_args.model_name_or_path,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
            peft_config=peft_config,
            callbacks=callbacks,
        )

    # Workaround for TRL 0.23.1 compatibility issue
    # GRPOTrainer expects current_gradient_accumulation_steps but it's not always set
    if not hasattr(trainer, 'current_gradient_accumulation_steps'):
        trainer.current_gradient_accumulation_steps = training_args.gradient_accumulation_steps
        logger.info(f"*** Set current_gradient_accumulation_steps={trainer.current_gradient_accumulation_steps} ***")

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = script_args.train_subset_size or len(train_dataset)

    try:
        trainer.save_state()
    except:
        print("Failed to save state, please debug")
        pass

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["rlsr"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    # Monkey-patch GRPOConfig to fix parent class interference with generation_batch_size/steps_per_generation
    def patched_post_init(self):
        # Save user-configured values before parent class modifies them
        user_generation_batch_size = self.generation_batch_size
        user_steps_per_generation = self.steps_per_generation

        # The DeepSpeed config's gradient_accumulation_steps overrides the training
        # YAML's value (DeepSpeed wins at runtime, so make the config agree upfront)
        ds_gradient_accumulation_steps = _load_deepspeed_gradient_accumulation_steps()
        if (
            ds_gradient_accumulation_steps is not None
            and self.gradient_accumulation_steps != ds_gradient_accumulation_steps
        ):
            logger.info(
                "Overriding gradient_accumulation_steps to match DeepSpeed config: %s -> %s",
                self.gradient_accumulation_steps,
                ds_gradient_accumulation_steps,
            )
            self.gradient_accumulation_steps = ds_gradient_accumulation_steps

        # Call the parent trl.GRPOConfig.__post_init__() to get all TRL initialization
        import trl
        trl.GRPOConfig.__post_init__(self)

        # Restore user values (override what newer TRL version may have set)
        self.generation_batch_size = user_generation_batch_size
        self.steps_per_generation = user_steps_per_generation

        # Now run the rest of the original RLCR __post_init__ logic
        print("Post init on the config")

        num_processes = self.world_size
        if self.generation_batch_size is not None and self.steps_per_generation is not None:
            raise ValueError(
                "'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time"
            )

        if self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps

        if self.generation_batch_size is None:
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation

        if self.generation_batch_size % (self.per_device_train_batch_size * num_processes) != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size "
                f"({self.per_device_train_batch_size * num_processes})."
            )

        self.steps_per_generation = self.generation_batch_size // (self.per_device_train_batch_size * num_processes)

        possible_values = [
            n_gen for n_gen in range(2, self.generation_batch_size + 1) if (self.generation_batch_size) % n_gen == 0
        ]

        if self.num_generations not in possible_values:
            raise ValueError(
                f"The effective train batch size ({num_processes} x {self.per_device_train_batch_size} x "
                f"{self.steps_per_generation}) must be evenly divisible by the number of generations per "
                f"prompt ({self.num_generations}). Given the current effective train batch size, the valid values for "
                f"the number of generations are: {possible_values}."
            )

        if self.eval_strategy != "no":
            global_eval_batch_size = self.per_device_eval_batch_size * num_processes
            possible_values = [
                n_gen for n_gen in range(2, global_eval_batch_size + 1) if (global_eval_batch_size) % n_gen == 0
            ]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {self.per_device_eval_batch_size}) must be "
                    f"evenly divisible by the number of generations per prompt ({self.num_generations}). Given the "
                    "current global eval batch size, the valid values for the number of generations are: "
                    f"{possible_values}."
                )

    GRPOConfig.__post_init__ = patched_post_init

    parser = TrlParser((RLSRScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Normalize scale_rewards: accept bool-style values from existing configs
    _sr = str(training_args.scale_rewards).lower()
    if _sr in ("false", "0", "no"):
        training_args.scale_rewards = "none"
    elif _sr in ("true", "1", "yes"):
        training_args.scale_rewards = "group"

    # Add missing attributes expected by newer TRL version
    if not hasattr(model_args, 'use_dora'):
        model_args.use_dora = False  # We're using standard LoRA, not DoRA

    main(script_args, training_args, model_args)
