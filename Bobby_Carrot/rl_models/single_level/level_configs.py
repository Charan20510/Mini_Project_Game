"""Per-level hyperparameter presets for single-map PPO demo runs.

Four tiers, chosen by mechanic complexity:

  Tier A1 (L1):       floor + carrot only. Tiny map, fast convergence.
  Tier A2 (L2-L3):    + crumble-intro (1 to 3 stacked one-use bridges).
                      Needs longer budget, higher entropy floor, and ICM to
                      avoid the collapse-to-oscillation failure mode observed
                      on the prior "tier A" preset (avg_r ~150, success 0%).
  Tier B  (L4-L7):    + crumble chains + first arrows. Modest headroom.
  Tier C  (L8-L10):   + conveyor belts. Needs ICM and a bigger time budget.

All anti-forgetting / curriculum knobs are disabled. Single level = single
policy = no forgetting to fight.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from ..config import ICMConfig, LevelConfig, RainbowConfig, TrainingConfig


# level_num -> tier letter
LEVEL_TIER: Dict[int, str] = {
    1: "A1",
    2: "A2", 3: "A2",
    4: "B", 5: "B", 6: "B", 7: "B",
    8: "C", 9: "C", 10: "C",
}


def build_configs_for_level(
    level_num: int,
    checkpoint_root: str = ".",
) -> Tuple[RainbowConfig, TrainingConfig, LevelConfig, ICMConfig]:
    """Build the four configs needed by `train_rainbow` for a single normal level.

    Parameters
    ----------
    level_num : int
        Normal level number (1-10).
    checkpoint_root : str
        Directory under which ``checkpoints_level{N}/`` and ``logs_level{N}/``
        will be created.
    """
    if level_num not in LEVEL_TIER:
        raise ValueError(
            f"level_num must be 1-10 for the demo preset, got {level_num}"
        )

    tier = LEVEL_TIER[level_num]
    root = Path(checkpoint_root)

    level_cfg = LevelConfig(
        train_levels=[("normal", level_num)],
        test_levels=[("normal", level_num)],
    )

    # Shared defaults across all tiers. Every curriculum / anti-forgetting knob
    # is neutralised so the training loop degenerates to "one-level PPO".
    train_cfg = TrainingConfig(
        device="auto",
        checkpoint_dir=root / f"checkpoints_level{level_num}",
        log_dir=root / f"logs_level{level_num}",
        curriculum=False,
        curriculum_start_levels=1,
        curriculum_promotion_window=100,
        curriculum_promotion_threshold=2.0,   # Unreachable -> no promotion path
        curriculum_add_levels=0,
        level_history_window=60,
        curriculum_mastery_floor=0.0,
        curriculum_min_quota=0.0,
        curriculum_max_level_weight=1.0,
        curriculum_dwell_windows=999,
        curriculum_fallback_threshold=2.0,
        curriculum_fallback_windows=999,
        entropy_boost_steps=0,
        entropy_boost_multiplier=1.0,
        regression_trigger_drop=0.99,          # Effectively disabled
        teacher_ema_decay=1.0,                 # Teacher never moves -> no drift
        teacher_kl_coef=0.0,                   # No KL anchor needed
        teacher_kl_mastery_coef=0.0,
        curriculum_retention_floor=0.0,        # No retention gate
        observation_mode="full",
        reward_scale=1.0,
        reset_policy_head_on_resume=False,
        checkpoint_every=20_000,
        eval_interval=25_000,
        eval_episodes_per_level=10,
        log_interval=2_000,
        # Early-stopping: require > 95 continuous successes in a row to interrupt before total_timesteps
        early_stop_success=1.0,
        early_stop_window=96,
        early_stop_min_timesteps=20_000,
        # Same cosine LR decay shape as phased runs, a bit gentler at the end.
        lr_decay_final_fraction=0.25,
        lr_decay_min_multiplier=0.3,
    )

    if tier == "A1":
        # Pure carrot-collection (L1). Fast; the previous tuning worked.
        train_cfg.total_timesteps = 150_000
        train_cfg.max_steps_per_episode = 300
        train_cfg.early_stop_success = 0.95
        train_cfg.early_stop_window = 50
        rb_cfg = RainbowConfig(
            lr=1e-4,
            buffer_size=50_000,
            batch_size=64,
            learning_starts=5_000,
            target_update_freq=500,
            n_step=3,
            v_min=-200.0,
            v_max=600.0,
            cnn_channels=[32, 64, 64, 64],
        )
        icm_cfg = ICMConfig(enabled=False, intrinsic_reward_scale=0.0)

    elif tier == "A2":
        # L2 / L3 — crumble-intro.
        #
        # Root causes of prior failure (logs: avg_r=572 while success=0%):
        #   1. v_min=-120 / v_max=250 did not bracket actual returns.
        #      finish=300 alone exceeded v_max, saturating all atoms into
        #      the wrong region and destroying the Q-value learning signal.
        #   2. learning_starts=5000 → only ~16 random episodes before
        #      training; the entire buffer was negative experiences,
        #      causing monotonic Q-collapse (Q: 58→2 over 24k steps).
        #   3. batch_size=32 → high-variance distributional updates early.
        #   4. n_step=5 → deeply negative bootstrapped targets under sparse
        #      rewards, reinforcing Q-underestimation.
        #   5. No early stopping / best-model save → final ckpt = last ckpt,
        #      not peak performance.
        train_cfg.total_timesteps = 500_000
        train_cfg.max_steps_per_episode = 350
        train_cfg.eval_interval = 10_000
        train_cfg.checkpoint_every = 10_000
        train_cfg.lr_decay_final_fraction = 0.20
        train_cfg.lr_decay_min_multiplier = 0.3
        train_cfg.early_stop_success = 0.95
        train_cfg.early_stop_window = 50
        train_cfg.early_stop_min_timesteps = 20_000
        rb_cfg = RainbowConfig(
            lr=1e-4,
            buffer_size=80_000,
            batch_size=64,
            learning_starts=8_000,
            target_update_freq=1_000,
            n_step=3,
            # Return range: finish=300 + 13 carrots×15×1.5≈293 + approach≈100 → ~700 max
            # Minimum: death(-50) + step+progress penalties → ~-200 min
            v_min=-200.0,
            v_max=700.0,
            per_beta_anneal_steps=200_000,
            cnn_channels=[32, 64, 64, 64],
        )
        icm_cfg = ICMConfig(enabled=True, intrinsic_reward_scale=0.01)

    elif tier == "B":
        train_cfg.total_timesteps = 300_000
        train_cfg.max_steps_per_episode = 600
        rb_cfg = RainbowConfig(
            lr=6.25e-5,
            buffer_size=200_000,
            batch_size=32,
            cnn_channels=[32, 64, 64, 64],
        )
        icm_cfg = ICMConfig(enabled=True, intrinsic_reward_scale=0.01)

    else:  # tier C
        train_cfg.total_timesteps = 500_000
        train_cfg.max_steps_per_episode = 800
        rb_cfg = RainbowConfig(
            lr=6.25e-5,
            buffer_size=300_000,
            batch_size=32,
            cnn_channels=[32, 64, 64, 64],
        )
        icm_cfg = ICMConfig(enabled=True, intrinsic_reward_scale=0.02)

    return rb_cfg, train_cfg, level_cfg, icm_cfg
