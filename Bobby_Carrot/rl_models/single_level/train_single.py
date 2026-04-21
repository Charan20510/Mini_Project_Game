"""Thin wrapper around ``train_ppo`` for one-level demo training.

Usage:

    from Bobby_Carrot.rl_models.single_level import train_single_level
    train_single_level(level_num=3)

Produces ``checkpoints_level3/ppo/ppo_best.pt`` (and ``ppo_final.pt``), which
can be fed straight into ``evaluate_agent`` for rendering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..ppo import train_ppo
from .level_configs import build_configs_for_level, LEVEL_TIER


def train_single_level(
    level_num: int,
    checkpoint_root: str = ".",
    resume_path: Optional[str] = None,
    total_timesteps_override: Optional[int] = None,
):
    """Train PPO on a single normal level.

    Parameters
    ----------
    level_num : int
        Target level in 1..10.
    checkpoint_root : str
        Directory under which per-level checkpoints/logs are created.
    resume_path : Optional[str]
        If given, warm-start from this checkpoint (rare — each level is normally
        trained from scratch).
    total_timesteps_override : Optional[int]
        Override the tier default. Useful if a level needs extra budget.
    """
    ppo_cfg, train_cfg, level_cfg, icm_cfg = build_configs_for_level(
        level_num, checkpoint_root=checkpoint_root
    )
    if total_timesteps_override is not None:
        train_cfg.total_timesteps = int(total_timesteps_override)

    tier = LEVEL_TIER[level_num]
    print(
        f"\n{'='*72}\n"
        f"  Single-Level PPO: normal-{level_num:02d} (tier {tier})\n"
        f"  budget={train_cfg.total_timesteps:,} steps | "
        f"lr={ppo_cfg.lr} | ent={ppo_cfg.entropy_coeff} | "
        f"icm={'on' if icm_cfg.enabled else 'off'} | "
        f"early_stop={train_cfg.early_stop_success:.0%}\n"
        f"  ckpt={train_cfg.checkpoint_dir}\n"
        f"{'='*72}"
    )

    return train_ppo(
        ppo_config=ppo_cfg,
        train_config=train_cfg,
        level_config=level_cfg,
        icm_config=icm_cfg,
        resume_path=resume_path,
    )


def best_ckpt_for_level(level_num: int, checkpoint_root: str = ".") -> str:
    """Return path to ``ppo_best.pt`` for a level, falling back to ``ppo_final.pt``."""
    root = Path(checkpoint_root) / f"checkpoints_level{level_num}" / "ppo"
    best = root / "ppo_best.pt"
    if best.exists():
        return str(best)
    final = root / "ppo_final.pt"
    if final.exists():
        print(f"[single_level] ppo_best.pt missing for L{level_num} — using ppo_final.pt")
        return str(final)
    raise FileNotFoundError(f"No checkpoint found under {root}")
