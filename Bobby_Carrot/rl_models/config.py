"""Unified configuration dataclasses for all RL algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class LevelConfig:
    """Defines which levels are used for training and testing."""

    train_levels: List[Tuple[str, int]] = field(default_factory=lambda: (
        [("normal", i) for i in range(1, 26)]   # normal 1-25
    ))
    test_levels: List[Tuple[str, int]] = field(default_factory=lambda: (
        [("normal", i) for i in range(26, 31)]  # normal 26-30
    ))


@dataclass
class TrainingConfig:
    """Shared training orchestration settings."""

    total_timesteps: int = 3_000_000     # 25 levels need ~120k steps each
    seed: int = 42
    device: str = "auto"  # "auto", "cuda", "cpu"
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_every: int = 100_000
    eval_interval: int = 50_000
    eval_episodes_per_level: int = 20
    log_interval: int = 2_000
    log_dir: Path = Path("logs")

    # Curriculum settings
    curriculum: bool = True
    curriculum_start_levels: int = 5      # Start with levels 1-5
    curriculum_promotion_window: int = 120
    curriculum_promotion_threshold: float = 0.55  # 55% success to promote (25 levels is harder)
    curriculum_add_levels: int = 3        # Add 3 levels per promotion
    # Anti-forgetting: keep mastered levels practiced even when new levels are
    # failing hard, and require 2 consecutive windows above threshold before
    # admitting the next batch of levels.
    level_history_window: int = 80
    curriculum_mastery_floor: float = 0.55  # Min sampling weight for a mastered (≥75%) level
    curriculum_min_quota: float = 0.10    # Every active level must get ≥ this fraction (10% with 25 levels)
    curriculum_dwell_windows: int = 2     # Highest level must stay ≥ threshold this many windows
    # On each promotion, boost entropy_coeff for N steps to force exploration
    # on the newly-added levels before the LR schedule tightens it.
    entropy_boost_steps: int = 80_000
    entropy_boost_multiplier: float = 2.0
    # Cosine decay of LR over the last fraction of training (0.0 disables).
    lr_decay_final_fraction: float = 0.25  # Decay over last 25%
    lr_decay_min_multiplier: float = 0.3   # LR floor = lr * 0.3

    # Observation
    observation_mode: str = "full"
    max_steps_per_episode: int = 1500     # Level 26 has 65 carrots — needs headroom
    reward_scale: float = 1.0

    # Adaptive exploration: force random actions on levels the agent hasn't learned
    exploration_epsilon: float = 0.25
    exploration_success_threshold: float = 0.3

    # Multi-env (vectorized)
    n_envs: int = 1  # Single env for simplicity

    # Transfer learning: reset policy head when resuming from a different phase.
    # Default False — action semantics (L/R/U/D) are identical across all Bobby
    # Carrot levels, so discarding learned action preferences is almost always
    # harmful. Only set True if the action space *semantics* change.
    reset_policy_head_on_resume: bool = False


@dataclass
class PPOConfig:
    """PPO-specific hyperparameters."""

    lr: float = 3e-4           # Full LR — cold start on new 25-level split
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2    # Default PPO clip
    value_coeff: float = 0.5
    entropy_coeff: float = 0.15  # High start — exploring 25 levels from scratch
    max_grad_norm: float = 0.5
    rollout_length: int = 4096   # Larger rollout covers more level variety per update
    n_epochs: int = 4
    minibatch_size: int = 128    # Scaled with rollout length
    normalize_advantages: bool = True
    entropy_min: float = 0.04    # Floor that keeps exploration alive on hard late levels

    # Network architecture
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 64, 64])
    hidden_dim: int = 256


@dataclass
class RainbowConfig:
    """Rainbow DQN hyperparameters (all 6 enhancements)."""

    lr: float = 6.25e-5
    gamma: float = 0.99
    batch_size: int = 32
    buffer_size: int = 200_000   # Deeper replay to cover 25-level diversity
    learning_starts: int = 5_000 # Wait for diverse level coverage before first update
    target_update_freq: int = 2_000
    max_grad_norm: float = 10.0

    # Double DQN — enabled by default (no extra param needed)

    # Dueling
    hidden_dim: int = 256

    # PER (Prioritized Experience Replay)
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_anneal_steps: int = 100_000
    per_epsilon: float = 1e-6

    # NoisyNet
    noisy_std: float = 0.5

    # N-step returns
    n_step: int = 5              # Wider return window helps credit assignment across crumble chains

    # C51 (Distributional)
    atom_size: int = 51
    v_min: float = -120.0        # Extra headroom for death + stranding combo penalties
    v_max: float = 250.0         # Level 26's 65 carrots pushes returns higher

    # Network architecture
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 64, 64])


@dataclass
class ICMConfig:
    """Intrinsic Curiosity Module hyperparameters."""

    enabled: bool = True   # Enabled by default for crumble-heavy levels
    lr: float = 1e-3
    feature_dim: int = 128
    intrinsic_reward_scale: float = 0.01
    forward_loss_weight: float = 0.2
    inverse_loss_weight: float = 0.8
    reward_running_mean_decay: float = 0.99
