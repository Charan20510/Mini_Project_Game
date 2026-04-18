"""Multi-level evaluation harness for trained RL agents.

Loads a saved checkpoint, runs deterministic episodes on all test levels,
and reports per-level metrics.

Usage:
    python -m Bobby_Carrot.rl_models.evaluate --algo ppo --checkpoint checkpoints/ppo/ppo_final.pt
    python -m Bobby_Carrot.rl_models.evaluate --algo rainbow --checkpoint checkpoints/rainbow/rainbow_final.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch

# Ensure imports work
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
_GAME_PYTHON = _PROJECT_ROOT / "Game_Python"
if str(_GAME_PYTHON) not in sys.path:
    sys.path.insert(0, str(_GAME_PYTHON))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Bobby_Carrot.rl_models.config import PPOConfig, RainbowConfig, LevelConfig
from Bobby_Carrot.rl_models.networks import ObservationPreprocessor
from Bobby_Carrot.rl_models.ppo import PPOAgent
from Bobby_Carrot.rl_models.rainbow import RainbowAgent


def evaluate_agent(
    algo: str,
    checkpoint_path: str,
    levels: List[Tuple[str, int]],
    episodes_per_level: int = 10,
    max_steps: int = 300,
    observation_mode: str = "full",
    device_str: str = "auto",
    render: bool = False,
    render_fps: float = 4.0,
    save_frames: bool = False,
    frames_dir: str = "frames",
) -> Dict[str, Any]:
    """Evaluate a trained agent across multiple levels.

    Returns a dict with per-level and aggregate metrics.
    """
    from bobby_carrot.rl_env import BobbyCarrotEnv  # type: ignore

    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create agent with matching config from checkpoint if available
    if algo == "ppo":
        saved_config = None
        if "config" in ckpt and "ppo" in ckpt["config"]:
            saved_config = ckpt["config"]["ppo"]
        agent = PPOAgent(saved_config or PPOConfig()).to(device)
        agent.load_state_dict(ckpt["agent_state_dict"])
        agent.eval()
    elif algo == "rainbow":
        agent = RainbowAgent(RainbowConfig()).to(device)
        agent.load_state_dict(ckpt["online_state_dict"])
        agent.eval()
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    preprocessor = ObservationPreprocessor(device)

    # Setup frames directory if saving frames
    frames_path = Path(frames_dir)
    if save_frames:
        frames_path.mkdir(parents=True, exist_ok=True)

    # Per-level results
    results: Dict[str, Dict[str, float]] = {}
    all_successes = []
    all_rewards = []
    all_steps = []
    all_collected = []

    print(f"\n{'='*70}")
    print(f"  Evaluating {algo.upper()} on {len(levels)} levels × {episodes_per_level} episodes")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")

    print(f"{'Level':<15} {'Success%':>10} {'Collected%':>12} {'Avg Reward':>12} {'Avg Steps':>10}")
    print("-" * 60)

    for kind, num in levels:
        level_key = f"{kind}-{num:02d}"
        successes = []
        rewards = []
        steps = []
        collected = []

        env = BobbyCarrotEnv(
            map_kind=kind,
            map_number=num,
            observation_mode=observation_mode,
            include_inventory=True,
            headless=not render,
            max_steps=max_steps,
        )

        for ep in range(episodes_per_level):
            obs_raw = env.reset()
            done = False
            ep_reward = 0.0
            ep_steps = 0
            info = {}

            if render:
                env.render()
                if save_frames:
                    import pygame
                    frame_path = frames_path / f"{level_key}_ep{ep+1}_step{ep_steps:04d}.png"
                    if getattr(env, '_screen', None) is not None:
                        pygame.image.save(env._screen, str(frame_path))

            while not done:
                obs_t = preprocessor(obs_raw)
                # Use action masking for consistent eval
                mask_np = env.get_valid_actions()
                mask_t = torch.tensor(mask_np, dtype=torch.bool, device=device)
                with torch.no_grad():
                    if algo == "ppo":
                        features = agent.encoder(obs_t.unsqueeze(0))
                        dist = agent.policy(features, action_mask=mask_t.unsqueeze(0))
                        action = int(dist.probs.argmax(dim=-1).item())
                    else:  # rainbow
                        q = agent.q_values(obs_t.unsqueeze(0))
                        action = int(q.argmax(dim=-1).item())

                obs_raw, reward, done, info = env.step(action)
                ep_reward += reward
                ep_steps += 1

                if render:
                    env.render()
                    if save_frames:
                        import pygame
                        frame_path = frames_path / f"{level_key}_ep{ep+1}_step{ep_steps:04d}.png"
                        if getattr(env, '_screen', None) is not None:
                            pygame.image.save(env._screen, str(frame_path))
                    import time
                    time.sleep(1.0 / render_fps)

            successes.append(1.0 if info.get("level_completed", False) else 0.0)
            collected.append(1.0 if info.get("all_collected", False) else 0.0)
            rewards.append(ep_reward)
            steps.append(ep_steps)

        env.close()

        level_success = float(np.mean(successes))
        level_collected = float(np.mean(collected))
        level_reward = float(np.mean(rewards))
        level_steps = float(np.mean(steps))

        results[level_key] = {
            "success_rate": level_success,
            "collection_rate": level_collected,
            "avg_reward": level_reward,
            "avg_steps": level_steps,
        }

        all_successes.extend(successes)
        all_rewards.extend(rewards)
        all_steps.extend(steps)
        all_collected.extend(collected)

        print(
            f"{level_key:<15} {level_success:>9.1%} {level_collected:>11.1%} "
            f"{level_reward:>11.2f} {level_steps:>9.1f}"
        )

    # Aggregate
    print("-" * 60)
    agg_success = float(np.mean(all_successes))
    agg_collected = float(np.mean(all_collected))
    agg_reward = float(np.mean(all_rewards))
    agg_steps = float(np.mean(all_steps))
    print(
        f"{'AGGREGATE':<15} {agg_success:>9.1%} {agg_collected:>11.1%} "
        f"{agg_reward:>11.2f} {agg_steps:>9.1f}"
    )
    print()

    return {
        "per_level": results,
        "aggregate": {
            "success_rate": agg_success,
            "collection_rate": agg_collected,
            "avg_reward": agg_reward,
            "avg_steps": agg_steps,
            "total_episodes": len(all_successes),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained Bobby Carrot RL agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--algo", type=str, required=True, choices=["ppo", "rainbow"],
                   help="Algorithm to evaluate")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to checkpoint file (.pt)")
    p.add_argument("--episodes", type=int, default=10,
                   help="Episodes per level")
    p.add_argument("--max-steps", type=int, default=300,
                   help="Max steps per episode")
    p.add_argument("--device", type=str, default="auto",
                   help="Device: auto, cuda, cpu")
    p.add_argument("--render", action="store_true",
                   help="Render episodes visually")
    p.add_argument("--render-fps", type=float, default=4.0,
                   help="Frames per second for rendering")
    p.add_argument("--save-frames", action="store_true",
                   help="Save rendered frames to disk as PNG images")
    p.add_argument("--frames-dir", type=str, default="frames",
                   help="Directory to save the rendered frames")

    # Level selection
    p.add_argument("--eval-set", type=str, default="test", choices=["test", "train", "all"],
                   help="Which level set to evaluate on")
    p.add_argument("--levels", type=str, default=None,
                   help="Comma-separated list of levels (e.g. 'normal-1,egg-5')")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    level_config = LevelConfig()

    # Determine levels
    if args.levels:
        levels = []
        for spec in args.levels.split(","):
            spec = spec.strip()
            if "-" in spec:
                kind, num = spec.rsplit("-", 1)
                levels.append((kind, int(num)))
            else:
                levels.append(("normal", int(spec)))
    elif args.eval_set == "test":
        levels = level_config.test_levels
    elif args.eval_set == "train":
        levels = level_config.train_levels
    else:
        levels = level_config.train_levels + level_config.test_levels

    evaluate_agent(
        algo=args.algo,
        checkpoint_path=args.checkpoint,
        levels=levels,
        episodes_per_level=args.episodes,
        max_steps=args.max_steps,
        device_str=args.device,
        render=args.render,
        render_fps=args.render_fps,
        save_frames=args.save_frames,
        frames_dir=args.frames_dir,
    )


if __name__ == "__main__":
    main()
