"""Thin wrapper around ``train_rainbow`` for one-level demo training.

Usage:

    from Bobby_Carrot.rl_models.single_level import train_single_level
    train_single_level(level_num=3)

Produces ``checkpoints_level3/rainbow/rainbow_best.pt`` (and ``rainbow_final.pt``),
which can be fed straight into ``evaluate_agent`` for rendering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..rainbow import train_rainbow
from .level_configs import build_configs_for_level, LEVEL_TIER


def train_single_level(
    level_num: int,
    checkpoint_root: str = ".",
    total_timesteps_override: Optional[int] = None,
):
    """Train Rainbow DQN on a single normal level.

    Parameters
    ----------
    level_num : int
        Target level in 1..10.
    checkpoint_root : str
        Directory under which per-level checkpoints/logs are created.
    total_timesteps_override : Optional[int]
        Override the tier default. Useful if a level needs extra budget.
    """
    rb_cfg, train_cfg, level_cfg, icm_cfg = build_configs_for_level(
        level_num, checkpoint_root=checkpoint_root
    )
    if total_timesteps_override is not None:
        train_cfg.total_timesteps = int(total_timesteps_override)

    tier = LEVEL_TIER[level_num]
    print(
        f"\n{'='*72}\n"
        f"  Single-Level Rainbow DQN: normal-{level_num:02d} (tier {tier})\n"
        f"  budget={train_cfg.total_timesteps:,} steps | "
        f"lr={rb_cfg.lr} | "
        f"v=[{rb_cfg.v_min},{rb_cfg.v_max}] | "
        f"n_step={rb_cfg.n_step} | "
        f"icm={'on' if icm_cfg.enabled else 'off'} | "
        f"early_stop={train_cfg.early_stop_success:.0%}\n"
        f"  ckpt={train_cfg.checkpoint_dir}\n"
        f"{'='*72}"
    )

    return train_rainbow(
        rainbow_config=rb_cfg,
        train_config=train_cfg,
        level_config=level_cfg,
        icm_config=icm_cfg,
    )


def best_ckpt_for_level(level_num: int, checkpoint_root: str = ".") -> str:
    """Return the best available checkpoint path for a level.

    Prefers ``rainbow_best.pt`` (saved at peak rolling success) over
    ``rainbow_final.pt`` (last step of training).
    """
    root = Path(checkpoint_root) / f"checkpoints_level{level_num}" / "rainbow"
    best = root / "rainbow_best.pt"
    if best.exists():
        return str(best)
    final = root / "rainbow_final.pt"
    if final.exists():
        return str(final)
    raise FileNotFoundError(f"No checkpoint found under {root}")
