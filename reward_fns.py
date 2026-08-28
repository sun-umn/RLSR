"""
Custom reward functions for RLSR training.

This module extends RLCR's reward functions. Import RLCR's original functions
when needed:
    from RLCR.reward_fns import format_reward, accuracy_reward, brier_reward
"""


# ==============================================================================
# Batch-Level RLSR Reward Placeholder
# ==============================================================================

def batch_rlsr_reward_placeholder(completions, answer, **kwargs):
    """
    Placeholder for batch-level RLSR reward.

    This function is a no-op because the actual reward computation
    happens in BatchRLSRGRPOTrainer._calculate_rewards().

    It exists only to register the reward function name with TRL.

    IMPORTANT: TRL uses the function's __name__ attribute for logging.

    The actual RLSR reward formula is:
        R_RLSR(ŷ, s, y) = {
            +λ · α̂_i    if ŷ = y  (correct prediction)
            -α̂_i        if ŷ ≠ y  (incorrect prediction)
        }

    Where α̂_i = H_n - H_{n-r_i} (harmonic weight based on confidence rank)
    """
    # Return zeros - actual computation happens in BatchRLSRGRPOTrainer
    return [0.0] * len(completions)


# Give it a proper name for TRL's reward_func_names
batch_rlsr_reward_placeholder.__name__ = "batch_rlsr"
