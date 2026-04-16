"""PPO (Proximal Policy Optimization) agent and training loop.

Implements:
- Shared CNN encoder → Policy head + Value head
- Clipped surrogate objective with entropy bonus
- GAE (Generalised Advantage Estimation)
- Action masking (invalid actions get -inf logits)
- Multi-level curriculum training
- Optional ICM integration
"""

from __future__ import annotations

import copy
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import PPOConfig, TrainingConfig, ICMConfig, LevelConfig
from .networks import CNNEncoder, ObservationPreprocessor, PolicyHead, ValueHead
from .buffers import RolloutBuffer


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent(nn.Module):
    """PPO actor-critic agent with shared CNN backbone and action masking."""

    def __init__(self, config: PPOConfig, n_actions: int = 4) -> None:
        super().__init__()
        self.config = config
        self.n_actions = n_actions

        self.encoder = CNNEncoder(
            channel_sizes=config.cnn_channels,
            hidden_dim=config.hidden_dim,
        )
        self.policy = PolicyHead(config.hidden_dim, n_actions)
        self.value = ValueHead(config.hidden_dim)

    def forward(self, obs: torch.Tensor):
        """Not used directly — use select_action or evaluate_actions."""
        features = self.encoder(obs)
        return self.policy(features), self.value(features)

    @torch.no_grad()
    def select_action(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[int, float, float]:
        """Select action for a single observation with optional masking.

        Returns: (action, log_prob, value)
        """
        features = self.encoder(obs.unsqueeze(0))

        # Pass mask to policy head (will be None if no masking)
        mask = action_mask.unsqueeze(0) if action_mask is not None else None
        dist = self.policy(features, action_mask=mask)
        value = self.value(features)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(value.item())

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for a batch of observations.

        Returns: (log_probs, values, entropy)
        """
        features = self.encoder(obs)
        dist = self.policy(features, action_mask=action_masks)
        values = self.value(features)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values, entropy


# ---------------------------------------------------------------------------
# PPO Training Loop
# ---------------------------------------------------------------------------

def train_ppo(
    ppo_config: PPOConfig,
    train_config: TrainingConfig,
    level_config: LevelConfig,
    icm_config: Optional[ICMConfig] = None,
    resume_path: Optional[str] = None,
) -> PPOAgent:
    """Full PPO training with curriculum learning, action masking, and optional ICM.

    Returns the trained PPOAgent.
    """
    import sys
    _here = Path(__file__).resolve().parent.parent.parent
    game_python = _here / "Game_Python"
    if str(game_python) not in sys.path:
        sys.path.insert(0, str(game_python))
    from bobby_carrot.rl_env import BobbyCarrotEnv  # type: ignore

    # Device setup
    if train_config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(train_config.device)
    print(f"[PPO] Using device: {device}")

    # Seed
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)

    # Create agent
    agent = PPOAgent(ppo_config).to(device)
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt['agent_state_dict'])
        print(f"[PPO] Loaded weights from {resume_path}")
    optimizer = optim.Adam(agent.parameters(), lr=ppo_config.lr, eps=1e-5)
    preprocessor = ObservationPreprocessor(device)

    # Optional ICM
    icm_module = None
    icm_optimizer = None
    if icm_config and icm_config.enabled:
        from .icm import ICMModule
        icm_module = ICMModule(icm_config, ppo_config.hidden_dim).to(device)
        icm_optimizer = optim.Adam(icm_module.parameters(), lr=icm_config.lr)
        print(f"[PPO] ICM enabled (scale={icm_config.intrinsic_reward_scale})")

    # Curriculum
    all_train_levels = list(level_config.train_levels)
    if train_config.curriculum:
        active_levels = all_train_levels[:min(train_config.curriculum_start_levels, len(all_train_levels))]
    else:
        active_levels = all_train_levels

    # Create env for first level
    current_level = active_levels[0]
    env = BobbyCarrotEnv(
        map_kind=current_level[0],
        map_number=current_level[1],
        observation_mode=train_config.observation_mode,
        include_inventory=True,
        headless=True,
        max_steps=train_config.max_steps_per_episode,
    )

    # Determine obs_dim from a reset
    dummy_obs = env.reset()
    obs_dim = len(dummy_obs)

    # Rollout buffer (stores raw int16 obs + action masks)
    rollout = RolloutBuffer(
        rollout_length=ppo_config.rollout_length,
        obs_dim=obs_dim,
        n_actions=agent.n_actions,
        gamma=ppo_config.gamma,
        gae_lambda=ppo_config.gae_lambda,
    )

    # Logging
    log_dir = Path(train_config.log_dir) / "ppo"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(train_config.checkpoint_dir) / "ppo"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training_log.csv"
    csv_handle = open(log_file, "w", newline="")
    csv_writer = csv.writer(csv_handle)
    csv_writer.writerow([
        "timestep", "episode", "avg_reward", "avg_success",
        "policy_loss", "value_loss", "entropy", "clip_fraction",
        "active_levels", "elapsed_sec",
    ])

    # Training state
    obs_raw = env.reset()
    obs_tensor = preprocessor(obs_raw)
    action_mask_np = env.get_valid_actions()
    action_mask_tensor = torch.tensor(action_mask_np, dtype=torch.bool, device=device)
    done = False
    total_timesteps = 0
    episode_count = 0
    episode_reward = 0.0
    episode_rewards: List[float] = []
    episode_successes: List[float] = []
    curriculum_window: List[float] = []
    best_avg_success = 0.0
    level_cycle_idx = 0  # Round-robin level index for equal exposure
    start_time = time.time()

    print(f"[PPO] Starting training for {train_config.total_timesteps} timesteps")
    print(f"[PPO] Active levels: {len(active_levels)} / {len(all_train_levels)}")
    print(f"[PPO] Observation dim: {obs_dim} | Channels: {preprocessor.num_channels()}")

    while total_timesteps < train_config.total_timesteps:
        # ── Collect rollout ───────────────────────────────────
        rollout.reset()
        agent.eval()

        for _step in range(ppo_config.rollout_length):
            action, log_prob, value = agent.select_action(obs_tensor, action_mask_tensor)

            next_obs_raw, reward, done, info = env.step(action)

            # ICM intrinsic reward
            if icm_module is not None:
                next_tensor = preprocessor(next_obs_raw)
                with torch.no_grad():
                    enc_obs = agent.encoder(obs_tensor.unsqueeze(0))
                    enc_next = agent.encoder(next_tensor.unsqueeze(0))
                intrinsic = icm_module.intrinsic_reward(
                    enc_obs, enc_next,
                    torch.tensor([action], device=device),
                )
                reward += icm_config.intrinsic_reward_scale * intrinsic

            # Store in rollout (raw int16 obs for correct preprocessing later)
            rollout.add(
                obs_raw.astype(np.int16), action, reward, done,
                log_prob, value, action_mask_np,
            )
            episode_reward += reward
            total_timesteps += 1

            if done:
                success = 1.0 if info.get("level_completed", False) else 0.0
                episode_rewards.append(episode_reward)
                episode_successes.append(success)
                curriculum_window.append(success)
                if len(curriculum_window) > train_config.curriculum_promotion_window:
                    curriculum_window = curriculum_window[-train_config.curriculum_promotion_window:]
                episode_count += 1
                episode_reward = 0.0

                # Round-robin level cycling for equal exposure across all levels
                level_cycle_idx = (level_cycle_idx + 1) % len(active_levels)
                current_level = active_levels[level_cycle_idx]
                env.set_map(map_kind=current_level[0], map_number=current_level[1])
                obs_raw = env.reset()
                obs_tensor = preprocessor(obs_raw)
                action_mask_np = env.get_valid_actions()
                action_mask_tensor = torch.tensor(action_mask_np, dtype=torch.bool, device=device)
            else:
                obs_raw = next_obs_raw
                obs_tensor = preprocessor(obs_raw)
                action_mask_np = env.get_valid_actions()
                action_mask_tensor = torch.tensor(action_mask_np, dtype=torch.bool, device=device)

        # ── Compute GAE ───────────────────────────────────────
        with torch.no_grad():
            last_tensor = preprocessor(obs_raw)
            last_features = agent.encoder(last_tensor.unsqueeze(0))
            last_value = float(agent.value(last_features).item())

        rollout.compute_gae(last_value, done)

        # ── PPO Update ────────────────────────────────────────
        agent.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        update_count = 0

        for _epoch in range(ppo_config.n_epochs):
            for batch in rollout.get_batches(ppo_config.minibatch_size):
                b_obs_raw = batch["observations"]
                b_obs = preprocessor.process_numpy_batch(b_obs_raw.astype(np.int16))
                b_actions = torch.tensor(batch["actions"], dtype=torch.long, device=device)
                b_old_log_probs = torch.tensor(batch["log_probs"], dtype=torch.float32, device=device)
                b_advantages = torch.tensor(batch["advantages"], dtype=torch.float32, device=device)
                b_returns = torch.tensor(batch["returns"], dtype=torch.float32, device=device)
                b_action_masks = torch.tensor(batch["action_masks"], dtype=torch.bool, device=device)

                if ppo_config.normalize_advantages and b_advantages.numel() > 1:
                    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

                log_probs, values, entropy = agent.evaluate_actions(
                    b_obs, b_actions, action_masks=b_action_masks,
                )

                # Clipped surrogate
                ratio = (log_probs - b_old_log_probs).exp()
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - ppo_config.clip_ratio, 1.0 + ppo_config.clip_ratio) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, b_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + ppo_config.value_coeff * value_loss
                    + ppo_config.entropy_coeff * entropy_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), ppo_config.max_grad_norm)
                optimizer.step()

                # ICM update
                if icm_module is not None and icm_optimizer is not None:
                    enc_obs = agent.encoder(b_obs).detach()
                    # Need next obs — approximate: shift by 1 within batch
                    # For proper ICM, we'd store next_obs in rollout; using detached encoder
                    if enc_obs.size(0) > 1:
                        icm_loss = icm_module.compute_loss(
                            enc_obs[:-1], enc_obs[1:],
                            b_actions[:-1],
                        )
                        icm_optimizer.zero_grad()
                        icm_loss.backward()
                        icm_optimizer.step()

                # Track metrics
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > ppo_config.clip_ratio).float().mean().item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_clip_frac += clip_frac
                update_count += 1

        # ── Logging ───────────────────────────────────────────
        if total_timesteps % train_config.log_interval < ppo_config.rollout_length:
            avg_reward = float(np.mean(episode_rewards[-50:])) if episode_rewards else 0.0
            avg_success = float(np.mean(episode_successes[-50:])) if episode_successes else 0.0
            avg_pl = total_policy_loss / max(1, update_count)
            avg_vl = total_value_loss / max(1, update_count)
            avg_ent = total_entropy / max(1, update_count)
            avg_cf = total_clip_frac / max(1, update_count)
            elapsed = time.time() - start_time

            print(
                f"[PPO] t={total_timesteps:>7d} | ep={episode_count:>4d} | "
                f"avg_r={avg_reward:>7.2f} | success={avg_success:>5.2%} | "
                f"p_loss={avg_pl:.4f} | v_loss={avg_vl:.4f} | "
                f"ent={avg_ent:.4f} | clip={avg_cf:.3f} | "
                f"levels={len(active_levels)} | {elapsed:.0f}s"
            )
            csv_writer.writerow([
                total_timesteps, episode_count, f"{avg_reward:.4f}", f"{avg_success:.4f}",
                f"{avg_pl:.6f}", f"{avg_vl:.6f}", f"{avg_ent:.6f}", f"{avg_cf:.4f}",
                len(active_levels), f"{elapsed:.1f}",
            ])
            csv_handle.flush()

        # ── Curriculum Promotion ──────────────────────────────
        if train_config.curriculum and len(curriculum_window) >= train_config.curriculum_promotion_window:
            rolling_success = float(np.mean(curriculum_window))
            if (
                rolling_success >= train_config.curriculum_promotion_threshold
                and len(active_levels) < len(all_train_levels)
            ):
                n_add = min(train_config.curriculum_add_levels, len(all_train_levels) - len(active_levels))
                old_count = len(active_levels)
                active_levels = all_train_levels[:old_count + n_add]
                curriculum_window.clear()
                print(
                    f"[PPO] Curriculum promotion: {old_count} -> {len(active_levels)} levels "
                    f"(rolling_success={rolling_success:.2%})"
                )

        # ── Checkpointing ─────────────────────────────────────
        if total_timesteps % train_config.checkpoint_every < ppo_config.rollout_length:
            ckpt_path = ckpt_dir / f"ppo_{total_timesteps}.pt"
            torch.save({
                "agent_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "total_timesteps": total_timesteps,
                "episode_count": episode_count,
                "active_levels": active_levels,
                "config": {
                    "ppo": ppo_config,
                    "train": train_config,
                    "level": level_config,
                },
            }, ckpt_path)

            # Save best model
            recent_success = float(np.mean(episode_successes[-100:])) if len(episode_successes) >= 10 else 0.0
            if recent_success > best_avg_success:
                best_avg_success = recent_success
                best_path = ckpt_dir / "ppo_best.pt"
                torch.save({
                    "agent_state_dict": agent.state_dict(),
                    "total_timesteps": total_timesteps,
                    "episode_count": episode_count,
                    "best_success": best_avg_success,
                }, best_path)
                print(f"[PPO] New best model saved (success={best_avg_success:.2%})")

        # ── Periodic Evaluation ───────────────────────────────
        if total_timesteps % train_config.eval_interval < ppo_config.rollout_length:
            _run_eval(
                agent, preprocessor, level_config.test_levels,
                train_config, device, total_timesteps,
            )

    # Final save
    final_path = ckpt_dir / "ppo_final.pt"
    torch.save({
        "agent_state_dict": agent.state_dict(),
        "total_timesteps": total_timesteps,
        "episode_count": episode_count,
    }, final_path)
    print(f"[PPO] Training complete. Final model saved to {final_path}")

    csv_handle.close()
    env.close()
    return agent


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _run_eval(
    agent: PPOAgent,
    preprocessor: ObservationPreprocessor,
    test_levels: List[Tuple[str, int]],
    train_config: TrainingConfig,
    device: torch.device,
    timestep: int,
) -> Dict[str, float]:
    """Run deterministic evaluation on test levels."""
    import sys
    _here = Path(__file__).resolve().parent.parent.parent
    game_python = _here / "Game_Python"
    if str(game_python) not in sys.path:
        sys.path.insert(0, str(game_python))
    from bobby_carrot.rl_env import BobbyCarrotEnv  # type: ignore

    agent.eval()
    total_success = 0
    total_episodes = 0
    total_reward = 0.0
    level_results: List[str] = []

    for kind, num in test_levels:
        env = BobbyCarrotEnv(
            map_kind=kind, map_number=num,
            observation_mode=train_config.observation_mode,
            include_inventory=True, headless=True,
            max_steps=train_config.max_steps_per_episode,
        )
        level_successes = 0
        for _ in range(train_config.eval_episodes_per_level):
            obs_raw = env.reset()
            done = False
            ep_reward = 0.0
            info: Dict[str, object] = {}
            while not done:
                obs_t = preprocessor(obs_raw)
                # Get action mask for eval too
                mask_np = env.get_valid_actions()
                mask_t = torch.tensor(mask_np, dtype=torch.bool, device=device)
                with torch.no_grad():
                    features = agent.encoder(obs_t.unsqueeze(0))
                    dist = agent.policy(features, action_mask=mask_t.unsqueeze(0))
                    action = int(dist.probs.argmax(dim=-1).item())
                obs_raw, reward, done, info = env.step(action)
                ep_reward += reward
            total_reward += ep_reward
            if info.get("level_completed", False):
                total_success += 1
                level_successes += 1
            total_episodes += 1
        env.close()
        level_results.append(f"{kind}-{num}:{level_successes}/{train_config.eval_episodes_per_level}")

    avg_success = total_success / max(1, total_episodes)
    avg_reward = total_reward / max(1, total_episodes)
    print(
        f"[PPO-EVAL] t={timestep} | test_success={avg_success:.2%} "
        f"| test_reward={avg_reward:.2f} | episodes={total_episodes} "
        f"| {', '.join(level_results)}"
    )
    return {"success_rate": avg_success, "avg_reward": avg_reward}
