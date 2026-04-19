"""Unified configuration dataclasses for all RL algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class LevelConfig:
    """Defines which levels are used for training and testing."""

    train_levels: List[Tuple[str, int]] = field(default_factory=lambda: (
        [("normal", i) for i in range(1, 8)]   # normal 1-7
    ))
    test_levels: List[Tuple[str, int]] = field(default_factory=lambda: (
        [("normal", i) for i in range(8, 11)]  # normal 8-10
    ))


@dataclass
class TrainingConfig:
    """Shared training orchestration settings."""

    total_timesteps: int = 500_000
    seed: int = 42
    device: str = "auto"  # "auto", "cuda", "cpu"
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_every: int = 50_000
    eval_interval: int = 25_000
    eval_episodes_per_level: int = 10
    log_interval: int = 2_000
    log_dir: Path = Path("logs")

    # Curriculum settings
    curriculum: bool = True
    curriculum_start_levels: int = 3      # Start with levels 1-3
    curriculum_promotion_window: int = 100
    curriculum_promotion_threshold: float = 0.6  # 60% success to promote
    curriculum_add_levels: int = 2        # Add 2 levels per promotion

    # Observation
    observation_mode: str = "full"
    max_steps_per_episode: int = 500
    reward_scale: float = 1.0  # Full reward magnitude; 0.1 was drowning the completion signal

    # Adaptive exploration: force random actions on levels the agent hasn't learned
    exploration_epsilon: float = 0.15   # Probability of random action on failing levels
    exploration_success_threshold: float = 0.3  # Level success below this triggers exploration

    # Multi-env (vectorized)
    n_envs: int = 1  # Single env for simplicity


@dataclass
class PPOConfig:
    """PPO-specific hyperparameters."""

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.04  # Higher entropy for exploration on complex levels
    max_grad_norm: float = 0.5
    rollout_length: int = 256
    n_epochs: int = 4
    minibatch_size: int = 64
    normalize_advantages: bool = True

    # Network architecture
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 64, 64])
    hidden_dim: int = 256


@dataclass
class RainbowConfig:
    """Rainbow DQN hyperparameters (all 6 enhancements)."""

    lr: float = 6.25e-5
    gamma: float = 0.99
    batch_size: int = 32
    buffer_size: int = 100_000
    learning_starts: int = 2_000
    target_update_freq: int = 1_000
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
    n_step: int = 3

    # C51 (Distributional)
    atom_size: int = 51
    v_min: float = -100.0
    v_max: float = 200.0

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
