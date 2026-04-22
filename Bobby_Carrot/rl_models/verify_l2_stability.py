"""One-command Level-2 stability verifier (greedy policy).

Pass criterion:
- For N consecutive windows, greedy success must be >= threshold.
- Each window evaluates episodes_per_window episodes on normal-2.

Example:
    python -m Bobby_Carrot.rl_models.verify_l2_stability \
      --checkpoint checkpoints_level2/ppo/ppo_best.pt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from Bobby_Carrot.rl_models.evaluate import evaluate_agent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Verify Level-2 greedy stability over consecutive windows",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True, help="Path to PPO checkpoint")
    p.add_argument("--windows", type=int, default=10, help="Required consecutive windows")
    p.add_argument("--episodes", type=int, default=20, help="Episodes per window")
    p.add_argument("--threshold", type=float, default=0.95, help="Minimum success rate per window")
    p.add_argument("--max-steps", type=int, default=350, help="Episode max steps")
    p.add_argument("--device", type=str, default="auto", help="Device: auto/cpu/cuda")
    p.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility")
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = build_parser().parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print("=" * 76)
    print(" Level-2 Greedy Stability Verification")
    print("=" * 76)
    print(f"checkpoint={ckpt}")
    print(
        f"criterion: {args.windows} consecutive windows, "
        f"each >= {args.threshold:.0%} over {args.episodes} episodes"
    )
    print("-" * 76)

    consecutive = 0
    rates: list[float] = []

    for win_idx in range(1, args.windows + 1):
        set_seed(args.seed + win_idx)
        res = evaluate_agent(
            algo="ppo",
            checkpoint_path=str(ckpt),
            levels=[("normal", 2)],
            episodes_per_level=args.episodes,
            max_steps=args.max_steps,
            observation_mode="full",
            device_str=args.device,
            use_mcts=False,
        )
        success = float(res["aggregate"]["success_rate"])
        rates.append(success)

        if success >= args.threshold:
            consecutive += 1
        else:
            consecutive = 0

        print(
            f"window {win_idx:02d}: success={success:.1%} | "
            f"streak={consecutive}/{args.windows}"
        )

    passed = len(rates) >= args.windows and all(r >= args.threshold for r in rates)
    print("-" * 76)
    print("PASS" if passed else "FAIL")
    print(
        "summary: "
        + ", ".join(f"w{i+1}={r:.1%}" for i, r in enumerate(rates))
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
