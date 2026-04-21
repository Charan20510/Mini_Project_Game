"""Per-level single-map PPO training for the L1-L10 demo.

Each level gets its own isolated checkpoint directory and tuned hyperparameters.
Anti-forgetting / curriculum / EMA-teacher machinery from the phased pipeline
is disabled — one map, one policy.
"""

from .level_configs import build_configs_for_level, LEVEL_TIER
from .train_single import best_ckpt_for_level, train_single_level

__all__ = [
    "LEVEL_TIER",
    "best_ckpt_for_level",
    "build_configs_for_level",
    "train_single_level",
]
