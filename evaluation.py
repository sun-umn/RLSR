"""
RLSR evaluation.

Extends RLCR's evaluation with AURC (Area Under Risk-Coverage) metrics for
selective prediction:
  - during training, via AURCEvaluationCallback + aurc_data_collector_reward;
  - post-hoc, via `python evaluation.py --config <config.json>` (same JSON
    format as RLCR/evaluation.py), which runs RLCR's own pipeline and then
    computes AURC/selective metrics from the saved outputs. Driven by eval.sh.
"""

import os
import sys

# Root first (local eval/ package overrides RLCR's), then the RLCR submodule.
_root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root_dir)
sys.path.insert(1, os.path.join(_root_dir, 'RLCR'))

# Load the local system_prompts.py into the module cache so RLCR's eval code
# resolves prompts from the release copy.
import importlib.util
_sp_spec = importlib.util.spec_from_file_location(
    "system_prompts", os.path.join(_root_dir, "system_prompts.py"))
_local_sp = importlib.util.module_from_spec(_sp_spec)
_sp_spec.loader.exec_module(_local_sp)
sys.modules['system_prompts'] = _local_sp

import numpy as np
from transformers import TrainerCallback


def compute_aurc(correctness: np.ndarray, confidences: np.ndarray) -> float:
    """
    Compute Area Under the Risk-Coverage curve using trapezoidal integration.

    Args:
        correctness: array-like of 0/1, 1 if the prediction was correct else 0
        confidences: array-like confidence scores (higher = more confident)

    Returns:
        aurc: scalar area under risk-coverage curve (lower is better)
    """
    if len(correctness) == 0:
        return 0.0

    correct = np.asarray(correctness, dtype=float)
    conf = np.asarray(confidences, dtype=float)
    n = len(conf)

    # Sort by confidence descending
    order = np.argsort(-conf)
    correct_sorted = correct[order]

    # Cumulative errors among top-k accepts
    cum_errors = np.cumsum(1.0 - correct_sorted)
    k = np.arange(1, n + 1)
    risk = cum_errors / k
    coverage = k / n

    # Numerical integration using trapezoidal rule
    aurc = np.trapz(risk, coverage)
    return float(aurc)


def compute_selective_metrics(correctness: np.ndarray,
                              confidences: np.ndarray,
                              coverage_levels: list = [0.1, 0.25, 0.5]) -> dict:
    """
    Compute selective prediction metrics for AURC analysis.

    Args:
        correctness: Binary array (1 if correct, 0 if incorrect)
        confidences: Confidence scores
        coverage_levels: Coverage levels to compute accuracy/risk at

    Returns:
        Dictionary of AURC-specific metrics
    """
    n = len(confidences)
    if n == 0:
        return {}

    metrics = {}

    # Sort by confidence descending
    order = np.argsort(-confidences)
    correct_sorted = correctness[order]

    # Core AURC metric
    metrics["aurc"] = compute_aurc(correctness, confidences)

    # Selective accuracy and risk at different coverage levels
    for cov in coverage_levels:
        k = max(1, int(n * cov))

        # Accuracy when accepting top k% most confident predictions
        selective_acc = float(np.mean(correct_sorted[:k]))
        metrics[f"selective_accuracy@{int(cov*100)}%"] = selective_acc

        # Risk (error rate) at this coverage
        selective_risk = 1.0 - selective_acc
        metrics[f"selective_risk@{int(cov*100)}%"] = selective_risk

    # Compute coverage needed to achieve target accuracy levels
    target_accuracies = [0.90, 0.95, 0.99]
    for target_acc in target_accuracies:
        # Find minimum coverage needed to achieve target accuracy
        cum_correct = np.cumsum(correct_sorted)
        k_values = np.arange(1, n + 1)
        running_acc = cum_correct / k_values

        # Find first k where accuracy >= target
        achieves_target = running_acc >= target_acc
        if np.any(achieves_target):
            min_k = np.argmax(achieves_target) + 1
            coverage_needed = min_k / n
            metrics[f"coverage_for_{int(target_acc*100)}%_accuracy"] = float(coverage_needed)
        else:
            # Target accuracy not achievable
            metrics[f"coverage_for_{int(target_acc*100)}%_accuracy"] = 1.0

    # Confidence statistics for correct vs incorrect predictions
    correct_mask = correctness.astype(bool)
    if np.any(correct_mask):
        metrics["mean_confidence_on_correct"] = float(np.mean(confidences[correct_mask]))
        metrics["median_confidence_on_correct"] = float(np.median(confidences[correct_mask]))

    if np.any(~correct_mask):
        metrics["mean_confidence_on_incorrect"] = float(np.mean(confidences[~correct_mask]))
        metrics["median_confidence_on_incorrect"] = float(np.median(confidences[~correct_mask]))

    # Confidence gap: how well does confidence discriminate?
    if np.any(correct_mask) and np.any(~correct_mask):
        conf_gap = metrics["mean_confidence_on_correct"] - metrics["mean_confidence_on_incorrect"]
        metrics["confidence_discrimination_gap"] = float(conf_gap)

    return metrics


# ==============================================================================
# AURC Evaluation Callback for Training
# ==============================================================================

class AURCEvaluationCallback(TrainerCallback):
    """
    Callback to compute AURC metrics during training evaluation.

    This callback works with TRL's GRPOTrainer to collect correctness and confidence
    data during evaluation and compute AURC metrics.

    Usage:
        callback = AURCEvaluationCallback(format_pattern="tabc")
        set_global_aurc_callback(callback)

        # Add to trainer callbacks
        trainer = GRPOTrainer(..., callbacks=[callback, ...])
    """

    def __init__(self, format_pattern="tabc"):
        """
        Args:
            format_pattern: The format pattern used for completions (e.g., "tabc")
        """
        self.format_pattern = format_pattern
        self.eval_data = {
            'correctness': [],
            'confidences': []
        }
        self._in_evaluation = False

    def on_evaluate(self, args, state, control, **kwargs):
        """Called at the start of evaluation."""
        self._in_evaluation = True
        # Clear previous evaluation data
        self.eval_data = {
            'correctness': [],
            'confidences': []
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Called when the trainer logs metrics.
        During evaluation, we compute and add AURC metrics here.
        """
        if not self._in_evaluation or logs is None:
            return

        # Check if this is an evaluation log (has 'eval_' prefix)
        if not any(k.startswith('eval_') for k in logs.keys()):
            return

        # If we have collected evaluation data, compute AURC
        if len(self.eval_data['correctness']) > 0:
            aurc_metrics = compute_selective_metrics(
                np.array(self.eval_data['correctness']),
                np.array(self.eval_data['confidences'])
            )

            # Add AURC metrics to logs with 'eval_' prefix
            for key, value in aurc_metrics.items():
                logs[f'eval_{key}'] = value

            # Mark that evaluation is done
            self._in_evaluation = False


# Global instance for easy access from reward functions
_global_aurc_callback = None


def set_global_aurc_callback(callback):
    """Set the global AURC callback instance."""
    global _global_aurc_callback
    _global_aurc_callback = callback


def get_global_aurc_callback():
    """Get the global AURC callback instance."""
    return _global_aurc_callback


def aurc_data_collector_reward(format_pattern):
    """
    A special reward function that collects AURC data during evaluation.

    This function:
    1. Returns NaN (doesn't contribute to training rewards)
    2. During evaluation, extracts correctness and confidence for AURC computation
    3. Stores data in the global AURC callback

    Args:
        format_pattern: The format pattern (e.g., "tabc")

    Returns:
        A reward function compatible with GRPOTrainer that returns NaN
    """
    import re
    from RLCR.reward_fns import accuracy_reward as original_accuracy_reward

    def reward_func(completions, answer, source=None, **kwargs):
        # Get the global callback
        callback = get_global_aurc_callback()

        # If we're in evaluation mode and have a callback, collect data
        if callback and callback._in_evaluation:
            # Compute correctness using the same logic as accuracy_reward
            correctness_rewards = original_accuracy_reward(format_pattern, completions, answer, source, **kwargs)

            # Extract confidence for each completion
            confidence_pattern = r"<confidence>(.*?)</confidence>"
            completion_contents = [completion[0]["content"] for completion in completions]

            for completion_text, correctness in zip(completion_contents, correctness_rewards):
                # Extract confidence
                confidence_matches = re.findall(
                    confidence_pattern, completion_text, re.DOTALL | re.MULTILINE
                )
                if confidence_matches:
                    try:
                        conf = float(confidence_matches[-1])  # Get last confidence
                        # Clip to [0, 1]
                        conf = max(0.0, min(conf, 1.0))
                        # Store in callback
                        callback.eval_data['correctness'].append(correctness)
                        callback.eval_data['confidences'].append(conf)
                    except:
                        pass  # Skip if confidence parsing fails

        # Return NaN so this doesn't contribute to the reward
        # (torch.nanmean will ignore NaN values)
        return [float('nan')] * len(completions)

    # Set a distinct name
    reward_func.__name__ = "aurc_collector"
    return reward_func


# ==============================================================================
# Post-hoc evaluation (wraps RLCR's evaluation.py, adds AURC metrics)
# ==============================================================================

def compute_aurc_metrics_from_dataset(dataset, config_name: str) -> dict:
    """Extract correctness/confidence from an evaluated dataset and compute AURC metrics."""
    evals_key = f"{config_name}-evals"
    conf_key = f"{config_name}-confidence_levels"

    if evals_key not in dataset.column_names or conf_key not in dataset.column_names:
        print(f"Warning: Could not find evaluation results for {config_name}")
        return {}

    evals = dataset[evals_key]  # List of lists (per sample, per generation)
    confidences = dataset[conf_key]

    correctness_flat = np.array([c for eval_list in evals for c in eval_list])
    confidence_flat = np.array([c for conf_list in confidences for c in conf_list])

    return compute_selective_metrics(correctness_flat, confidence_flat)


def main():
    import json
    import argparse
    import datasets as hf_datasets

    parser = argparse.ArgumentParser(
        description="RLSR evaluation with AURC metrics (wraps RLCR's evaluation)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to JSON config file (same format as RLCR/evaluation.py)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    global_config = config[0]
    model_configs = config[1:]
    store_name = global_config.get("store_name")
    log_path = global_config.get("log_path")

    # Snapshot pre-existing metrics: when RLCR's pipeline skips a cached model
    # config, it still rewrites metrics.json (empty), clobbering earlier results.
    preexisting_metrics = {}
    if log_path is not None and os.path.exists(os.path.join(log_path, "metrics.json")):
        with open(os.path.join(log_path, "metrics.json")) as f:
            preexisting_metrics = json.load(f)

    # Step 1: run RLCR's evaluation pipeline
    from eval.eval_args import GlobalArgs, LocalConfig
    from RLCR import evaluation as rlcr_eval

    rlcr_global_args = GlobalArgs(**global_config)
    local_configs = [LocalConfig(**c) for c in model_configs]

    # RLCR resolves local dataset paths (data/...) relative to its own root.
    original_cwd = os.getcwd()
    os.chdir(os.path.join(os.path.dirname(__file__), "RLCR"))

    # Source-level patches to RLCR's main():
    #  - save the output dataset when the store does not exist yet (upstream's
    #    `if updated:` skips saving on fresh runs, but the AURC pass below needs
    #    the store; when the store already exists and nothing ran, skip the save
    #    — save_to_disk refuses to overwrite the dataset it was loaded from);
    #  - serve the generator tensor-parallel when EVAL_TP is set (eval.sh sets
    #    it to the number of GPUs it was given).
    import inspect
    original_main_func = rlcr_eval.main
    try:
        source = inspect.getsource(rlcr_eval.main)
        modified_source = source.replace(
            "if updated:",
            "if updated or not os.path.exists(global_args.store_name):")
        # Code datasets: the ans_at_end reprompt caps answers at 50 tokens, which
        # truncates code into syntax errors; eval.sh raises it via env for
        # mbpp (QA datasets keep the original 50).
        ans_max_tokens = os.environ.get("ANS_REPROMPT_MAX_TOKENS")
        if ans_max_tokens:
            modified_source = modified_source.replace(
                "SamplingParams(n = 1, temperature = 0, max_tokens=50)",
                f"SamplingParams(n = 1, temperature = 0, max_tokens={int(ans_max_tokens)})")
        modified_source = modified_source.replace(
            "LLM(model=config.model,gpu_memory_utilization=global_args.gpu_memory_utilization)",
            "LLM(model=config.model,gpu_memory_utilization=global_args.gpu_memory_utilization,"
            "tensor_parallel_size=int(__import__('os').environ.get('EVAL_TP','1')))"
        )
        exec(modified_source, rlcr_eval.__dict__)
        print("Patched RLCR evaluation (force-save dataset, EVAL_TP support)")
    except Exception as e:
        print(f"Warning: could not patch RLCR's main(): {e}")

    try:
        rlcr_eval.main(rlcr_global_args, local_configs)
    finally:
        rlcr_eval.main = original_main_func
        os.chdir(original_cwd)

    # Step 2: compute AURC metrics from the saved outputs. Never abort here —
    # step 3 must still run so pre-existing metrics survive upstream's rewrite.
    all_aurc_metrics = {}
    try:
        evaluated_dataset = hf_datasets.load_from_disk(store_name)
        for model_config in model_configs:
            config_name = model_config.get("name")
            aurc_metrics = compute_aurc_metrics_from_dataset(evaluated_dataset, config_name)
            if aurc_metrics:
                all_aurc_metrics[config_name] = aurc_metrics
                print(f"\nAURC metrics for {config_name}:")
                for k, v in sorted(aurc_metrics.items()):
                    print(f"  {k:.<50} {v:.4f}")
    except Exception as e:
        print(f"Error: could not compute AURC metrics from {store_name}: {e}")

    # Step 3: merge AURC metrics into RLCR's metrics.json + write aurc_metrics.json
    if log_path is not None:
        rlcr_metrics_path = os.path.join(log_path, "metrics.json")
        rlcr_metrics = {}
        if os.path.exists(rlcr_metrics_path):
            with open(rlcr_metrics_path) as f:
                rlcr_metrics = json.load(f)

        enhanced = {
            name: {
                **preexisting_metrics.get(name, {}),
                **rlcr_metrics.get(name, {}),
                **all_aurc_metrics.get(name, {}),
            }
            for name in set(preexisting_metrics) | set(rlcr_metrics) | set(all_aurc_metrics)
        }
        with open(rlcr_metrics_path, "w") as f:
            json.dump(enhanced, f, indent=4)
        with open(os.path.join(log_path, "aurc_metrics.json"), "w") as f:
            json.dump(all_aurc_metrics, f, indent=4)
        print(f"\nMetrics (RLCR + AURC) saved to: {rlcr_metrics_path}")


if __name__ == "__main__":
    main()
