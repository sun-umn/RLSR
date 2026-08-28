"""
Batch-Level RLSR Trainer

Extends TRL's GRPOTrainer to support reward functions that require
global batch visibility for cross-sample ranking.

CRITICAL: This trainer handles the TRL reward gathering correctly:
- TRL calls gather(rewards_per_func) AFTER _calculate_rewards returns
- For batch-level rewards, we compute globally, then each process stores its LOCAL slice
- The parent's gather reconstructs the full tensor correctly

RLSR selection reward (the paper's Lifted-AURC formulation):
    R_RLSR(ŷ, s, y) = {
        +λ · α̂_i    if ŷ = y  (correct prediction)
        -α̂_i        if ŷ ≠ y  (incorrect prediction)
    }

Where α̂_i = H_n - H_{n-r_i} (harmonic weight based on confidence rank)
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Union
import logging

from trl import GRPOTrainer
from accelerate.utils import gather_object, gather

logger = logging.getLogger(__name__)


class BatchRLSRGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer with support for batch-level reward computation.

    For reward functions marked as 'batch_level', this trainer gathers
    all completions across processes before computing rewards, enabling
    cross-sample ranking for RLSR-style rewards.
    """

    def __init__(
        self,
        *args,
        batch_level_reward_funcs: list[str] = None,
        batch_rlsr_lambda: float = 1.0,
        batch_rlsr_max_weight: float | None = None,
        batch_rlsr_format_pattern: str = "tabc",
        **kwargs
    ):
        """
        Args:
            batch_level_reward_funcs: Names of reward functions that require
                global batch visibility (e.g., ["batch_rlsr"])
            batch_rlsr_lambda: Positive scale (λ) for correct predictions
            batch_rlsr_max_weight: Clip α̂ at this value; None = no clipping
                (α̂ is naturally bounded by H_n, the harmonic ceiling of the pool)
            batch_rlsr_format_pattern: Format pattern for parsing (tabc, tabc_long, etc.)

        Environment Variables:
            BATCH_RLSR_TIE_BREAKING: How to handle tied confidence values.
                - "average" (default): Tied samples get average of their ordinal weights.
                  Preserves total weight sum, theoretically sound.
                - "bisect": Tied samples get the same rank (min rank of the tie group).
        """
        # TRL hardcodes max_num_batched_tokens=4096 in GRPOTrainer, but with large
        # gradient_accumulation_steps, max_num_seqs can exceed that (e.g., 6*1024=6144).
        # Additionally, vLLM sizes its sampled_token_ids pinned buffer to max_model_len,
        # so max_num_seqs must not exceed max_model_len to avoid buffer overflow.
        # Monkey-patch vLLM's LLM.__init__ to fix both issues.
        import vllm.entrypoints.llm as _vllm_llm
        _orig_LLM_init = _vllm_llm.LLM.__init__

        def _patched_LLM_init(self_llm, *a, **kw):
            # Cap max_num_seqs to max_model_len (vLLM buffer overflow bug)
            max_model_len = kw.get("max_model_len")
            if max_model_len is not None and "max_num_seqs" in kw:
                kw["max_num_seqs"] = min(kw["max_num_seqs"], max_model_len)
            # Ensure max_num_batched_tokens >= max_num_seqs (TRL hardcodes 4096)
            if "max_num_batched_tokens" in kw and "max_num_seqs" in kw:
                kw["max_num_batched_tokens"] = max(kw["max_num_batched_tokens"], kw["max_num_seqs"])
            _orig_LLM_init(self_llm, *a, **kw)

        _vllm_llm.LLM.__init__ = _patched_LLM_init
        try:
            super().__init__(*args, **kwargs)
        finally:
            _vllm_llm.LLM.__init__ = _orig_LLM_init

        self.batch_level_reward_funcs = set(batch_level_reward_funcs or [])
        self.batch_rlsr_lambda = batch_rlsr_lambda
        self.batch_rlsr_max_weight = batch_rlsr_max_weight
        self.batch_rlsr_format_pattern = batch_rlsr_format_pattern

        # Tie-breaking method: "average" or "bisect"
        self.batch_rlsr_tie_breaking = os.environ.get("BATCH_RLSR_TIE_BREAKING", "average").lower()
        if self.batch_rlsr_tie_breaking not in ("average", "bisect"):
            logger.warning(
                f"Unknown BATCH_RLSR_TIE_BREAKING='{self.batch_rlsr_tie_breaking}', "
                f"defaulting to 'average'"
            )
            self.batch_rlsr_tie_breaking = "average"

        # Precompute harmonic numbers for efficiency (up to 50k samples)
        self._harmonic_cache = self._precompute_harmonic(50000)

        logger.info(
            f"BatchRLSRGRPOTrainer initialized with λ={batch_rlsr_lambda}, "
            f"max_weight={batch_rlsr_max_weight} (None = no clipping), "
            f"format={batch_rlsr_format_pattern}, tie_breaking={self.batch_rlsr_tie_breaking}"
        )

    def _precompute_harmonic(self, max_n: int) -> np.ndarray:
        """Precompute H_0 to H_{max_n}."""
        harmonic = np.zeros(max_n + 1)
        for k in range(1, max_n + 1):
            harmonic[k] = harmonic[k - 1] + 1.0 / k
        return harmonic

    def _get_harmonic(self, k: int) -> float:
        """Get H_k, extending cache if needed."""
        if k < 0:
            return 0.0
        while k >= len(self._harmonic_cache):
            new_val = self._harmonic_cache[-1] + 1.0 / len(self._harmonic_cache)
            self._harmonic_cache = np.append(self._harmonic_cache, new_val)
        return self._harmonic_cache[k]

    def _compute_batch_rlsr_rewards(
        self,
        all_completions: list,  # list[list[dict]] - TRL conversational format
        all_answers: list[str],
        all_sources: list[str] | None,
        format_pattern: str,
    ) -> np.ndarray:
        """
        Compute batch-level RLSR rewards across all B×N samples.

        Args:
            all_completions: All completions from all processes (TRL format)
            all_answers: All ground truth answers
            all_sources: Dataset source for each sample (e.g., 'hotpot')
            format_pattern: Pattern for parsing (tabc, tabc_long, etc.)

        Returns:
            np.ndarray of rewards for ALL samples (length = world_size * local_batch)
        """
        # Import from RLCR - these are the actual functions used in the codebase
        from RLCR.reward_fns import accuracy_reward, mean_confidence_reward

        n = len(all_completions)
        if n == 0:
            return np.array([])

        # Step 1: Compute correctness and confidence using existing functions
        # These functions expect completions in TRL format: list[list[dict]]
        # accuracy_reward returns list of 0.0 or 1.0
        # mean_confidence_reward returns list of floats in [0, 1]

        # Map format pattern to base pattern (handle tabc_long -> tabc)
        base_format_pattern = format_pattern
        if format_pattern in ["tabc_long", "tabc_long_2decimal"]:
            base_format_pattern = "tabc"

        correctness_scores = accuracy_reward(
            base_format_pattern,
            all_completions,
            all_answers,
            source=all_sources
        )
        confidence_scores = mean_confidence_reward(
            all_completions,
            all_answers
        )

        # Convert to numpy arrays
        correct = np.array([c > 0.5 for c in correctness_scores], dtype=bool)
        confidences = np.array(confidence_scores, dtype=np.float64)

        # Handle any NaN or invalid confidence values
        confidences = np.nan_to_num(confidences, nan=0.5)
        confidences = np.clip(confidences, 0.0, 1.0)

        # Step 2: Compute weights α̂_i based on confidence ranking
        H_n = self._get_harmonic(n)

        if self.batch_rlsr_tie_breaking == "bisect":
            # Bisect method: tied samples get the same rank (min rank of tie group)
            # This matches soft_running_reweighted_risk behavior
            # rank = number of elements strictly less than confidence + 1
            sorted_confs = np.sort(confidences)
            ranks = np.searchsorted(sorted_confs, confidences, side='left') + 1
            alphas = np.array([H_n - self._get_harmonic(n - r) for r in ranks])
        else:
            # Average method: tied samples get average of their ordinal weights
            # This preserves the total weight sum (theoretically sound)
            ordinal_weights = np.array([H_n - self._get_harmonic(n - r) for r in range(1, n + 1)])

            sort_idx = np.argsort(confidences)
            sorted_confs = confidences[sort_idx]

            # Find unique confidence values and their counts
            unique_confs, inverse_idx, counts = np.unique(
                sorted_confs, return_inverse=True, return_counts=True
            )

            # Compute average weight for each tie group
            alphas = np.zeros(n)
            pos = 0
            for group_idx, count in enumerate(counts):
                # Average the ordinal weights for positions this group occupies
                group_weights = ordinal_weights[pos:pos + count]
                avg_weight = group_weights.mean()
                # Assign to all samples in this tie group
                mask = (inverse_idx == group_idx)
                alphas[sort_idx[mask]] = avg_weight
                pos += count

        # Clamp α̂ if a max_weight is configured (None = keep the natural
        # harmonic ceiling H_n)
        if self.batch_rlsr_max_weight is not None:
            alphas = np.clip(alphas, 0, self.batch_rlsr_max_weight)

        # Step 4: λ from config
        n_correct = correct.sum()
        batch_acc = n_correct / n if n > 0 else 0.5
        lambda_val = self.batch_rlsr_lambda

        # Step 5: Apply RLSR formula
        # R = +λ·α̂  if correct
        # R = -α̂    if incorrect
        rewards = np.where(
            correct,
            lambda_val * alphas,  # Correct: +λ·α̂
            -alphas               # Incorrect: -α̂
        )

        # Log statistics periodically
        if hasattr(self, 'state') and self.state.global_step % 10 == 0:
            n_incorrect = n - n_correct
            logger.info(
                f"BatchRLSR: n={n}, correct={n_correct}, incorrect={n_incorrect}, "
                f"batch_acc={batch_acc:.3f}, λ={lambda_val:.3f}, "
                f"α̂ range=[{alphas.min():.3f}, {alphas.max():.3f}], "
                f"reward range=[{rewards.min():.3f}, {rewards.max():.3f}]"
            )

        return rewards

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        """
        Override to handle batch-level reward functions.

        CRITICAL LOGIC:
        1. For batch-level rewards: gather all data, compute globally, store LOCAL slice
        2. For other rewards: compute normally (local only)
        3. Parent's gather() at line 1057 reconstructs full tensor from local slices

        This ensures the gather() call produces correct [world_size * local_batch, num_funcs] tensor.
        """
        device = self.accelerator.device
        local_batch_size = len(prompts)
        rewards_per_func = torch.zeros(local_batch_size, len(self.reward_funcs), device=device)

        # Prepare reward kwargs (same as parent)
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        reward_kwargs["trainer_state"] = self.state

        # Check if any reward function is batch-level
        has_batch_level = any(
            name in self.batch_level_reward_funcs
            for name in self.reward_func_names
        )

        # Gather data for batch-level rewards (all processes must participate)
        all_completions = None
        all_answers = None
        all_sources = None

        if has_batch_level:
            # Gather all completions across processes
            # completions is list[list[dict]] in TRL format
            all_completions = gather_object(completions)

            # Flatten: [[GPU0_comps], [GPU1_comps], ...] → flat list
            if all_completions and isinstance(all_completions[0], list):
                # Check if doubly nested (gather of list of list)
                if all_completions[0] and isinstance(all_completions[0][0], list):
                    all_completions = [c for gpu_comps in all_completions for c in gpu_comps]
                # else already flat list of completions

            # Gather answers
            local_answers = reward_kwargs.get("answer", [None] * local_batch_size)
            all_answers = gather_object(local_answers)
            if all_answers and isinstance(all_answers[0], list):
                all_answers = [a for sublist in all_answers for a in sublist]

            # Gather sources (for hotpot vs math handling)
            local_sources = reward_kwargs.get("source", [None] * local_batch_size)
            all_sources = gather_object(local_sources)
            if all_sources and isinstance(all_sources[0], list):
                all_sources = [s for sublist in all_sources for s in sublist]

        # Import profiling context for compatibility
        try:
            from trl.trainer.grpo_trainer import profiling_context
        except ImportError:
            # Fallback if import fails
            from contextlib import nullcontext
            def profiling_context(trainer, name):
                return nullcontext()

        # Compute rewards for each function
        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if reward_func_name in self.batch_level_reward_funcs:
                    # ============ BATCH-LEVEL REWARD ============
                    # Compute globally on ALL processes (deterministic, same result everywhere)
                    global_rewards = self._compute_batch_rlsr_rewards(
                        all_completions,
                        all_answers,
                        all_sources,
                        self.batch_rlsr_format_pattern,
                    )

                    # Extract LOCAL slice for this process
                    # After parent's gather(), this reconstructs the full global tensor
                    process_idx = self.accelerator.process_index
                    start_idx = process_idx * local_batch_size
                    end_idx = start_idx + local_batch_size
                    local_rewards = global_rewards[start_idx:end_idx]

                    rewards_per_func[:, i] = torch.tensor(
                        local_rewards,
                        dtype=torch.float32,
                        device=device
                    )

                elif isinstance(reward_func, nn.Module):
                    # ============ NEURAL REWARD MODEL ============
                    # Copy from parent implementation for compatibility
                    try:
                        from trl.trainer.grpo_trainer import is_conversational, apply_chat_template
                    except ImportError:
                        # Fallback for older TRL versions
                        def is_conversational(x):
                            return isinstance(x.get("prompt"), list)
                        def apply_chat_template(x, proc):
                            return {"text": str(x)}

                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]

                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True,
                        padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)

                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]

                else:
                    # ============ RULE-BASED REWARD ============
                    output_reward_func = reward_func(
                        prompts=prompts,
                        completions=completions,
                        completion_ids=completion_ids_list,
                        **reward_kwargs
                    )
                    output_reward_func = [
                        reward if reward is not None else torch.nan
                        for reward in output_reward_func
                    ]
                    rewards_per_func[:, i] = torch.tensor(
                        output_reward_func,
                        dtype=torch.float32,
                        device=device
                    )

        # Handle NaN warnings (same as parent)
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items()
                if key != "trainer_state"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather rewards across processes (same as parent)
        # For batch-level rewards: local slices → full global tensor
        # For other rewards: local values → full global tensor
        rewards_per_func = gather(rewards_per_func)

        return rewards_per_func
