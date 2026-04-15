"""Neural network components shared across all RL algorithms.

Includes:
- ObservationPreprocessor: converts raw env observations to multi-channel tensors
- CNNEncoder: shared convolutional backbone
- PolicyHead / ValueHead: for PPO
- NoisyLinear: factorised Gaussian noise layer for NoisyNet
- DuelingDistributionalHead: combined Dueling + C51 head for Rainbow
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Observation Preprocessing
# ---------------------------------------------------------------------------

# Tile category mapping (must match rl_env.py tile IDs)
_NUM_OBS_CHANNELS = 10


class ObservationPreprocessor:
    """Converts raw 16x16 tile-grid observations into multi-channel float tensors.

    Channel layout (10 channels total):
        0: Walkable ground  (tile >= 18)
        1: Carrot            (tile == 19)
        2: Egg               (tile == 45)
        3: Finish tile       (tile == 44)
        4: Crumble tile      (tile == 30, active crumble)
        5: Hazard / collapsed (tile == 31 or 46)
        6: Key pickup        (tile in {32, 34, 36})
        7: Door locked       (tile in {33, 35, 37})
        8: Agent position    (1 at agent pos)
        9: Inventory info    (remaining targets normalised + key flags)
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @staticmethod
    def num_channels() -> int:
        return _NUM_OBS_CHANNELS

    def __call__(self, obs: np.ndarray) -> torch.Tensor:
        """Convert a single observation array to (C, 16, 16) float tensor."""
        return self.process_single(obs)

    def process_single(self, obs: np.ndarray) -> torch.Tensor:
        """Process one observation → (C, 16, 16)."""
        obs = np.asarray(obs, dtype=np.int16)
        channels = np.zeros((_NUM_OBS_CHANNELS, 16, 16), dtype=np.float32)

        # First 2 values are always (px, py)
        px, py = int(obs[0]), int(obs[1])

        # Determine inventory offset
        # Full mode: [px, py, <tiles 256>] or [px, py, inv..., <tiles 256>]
        # We handle both cases
        obs_len = len(obs)
        if obs_len >= 2 + 4 + 256:
            # With inventory: [px, py, key_gray, key_yellow, key_red, remaining_bucket, tiles...]
            inv = obs[2:6]
            tiles = obs[6:6 + 256]
        elif obs_len >= 2 + 256:
            # Without inventory: [px, py, tiles...]
            inv = np.zeros(4, dtype=np.int16)
            tiles = obs[2:2 + 256]
        else:
            # Local/compact mode — create a blank 16×16 and fill what we can
            inv = np.zeros(4, dtype=np.int16)
            tiles = np.zeros(256, dtype=np.int16)
            # For local observations, we embed available tile data at agent pos
            remaining = obs[2:] if obs_len > 2 else np.array([], dtype=np.int16)
            # We'll still set agent position channel — the local tiles are partial
            # so the CNN must handle sparse input
            local_size = int(math.isqrt(len(remaining))) if len(remaining) > 0 else 0
            if local_size > 0:
                half = local_size // 2
                for idx, val in enumerate(remaining):
                    dy = idx // local_size - half
                    dx = idx % local_size - half
                    gx, gy = px + dx, py + dy
                    if 0 <= gx < 16 and 0 <= gy < 16:
                        tiles[gx + gy * 16] = val

        # Fill tile channels
        for y in range(16):
            for x in range(16):
                tile = int(tiles[x + y * 16])
                if tile >= 18:
                    channels[0, y, x] = 1.0
                if tile == 19:
                    channels[1, y, x] = 1.0
                if tile == 45:
                    channels[2, y, x] = 1.0
                if tile == 44:
                    channels[3, y, x] = 1.0
                if tile == 30:
                    channels[4, y, x] = 1.0
                if tile == 31 or tile == 46:
                    channels[5, y, x] = 1.0
                if tile in (32, 34, 36):
                    channels[6, y, x] = 1.0
                if tile in (33, 35, 37):
                    channels[7, y, x] = 1.0

        # Agent position channel
        if 0 <= px < 16 and 0 <= py < 16:
            channels[8, py, px] = 1.0

        # Inventory channel — broadcast key/remaining info
        key_gray = float(inv[0]) if len(inv) > 0 else 0.0
        key_yellow = float(inv[1]) if len(inv) > 1 else 0.0
        key_red = float(inv[2]) if len(inv) > 2 else 0.0
        remaining = float(inv[3]) / 5.0 if len(inv) > 3 else 0.0
        # Encode as a spatial pattern: top-left quadrant = keys, bottom-right = remaining
        channels[9, 0:8, 0:8] = key_gray * 0.33 + key_yellow * 0.33 + key_red * 0.34
        channels[9, 8:16, 8:16] = remaining

        return torch.from_numpy(channels).to(self.device)

    def process_batch(self, obs_list: List[np.ndarray]) -> torch.Tensor:
        """Process a list of observations → (B, C, 16, 16)."""
        tensors = [self.process_single(o) for o in obs_list]
        return torch.stack(tensors)

    def process_numpy_batch(self, obs_array: np.ndarray) -> torch.Tensor:
        """Process a numpy array of shape (B, obs_dim) → (B, C, 16, 16)."""
        return self.process_batch([obs_array[i] for i in range(obs_array.shape[0])])


# ---------------------------------------------------------------------------
# CNN Encoder (shared backbone)
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    """Shared CNN backbone for processing 16x16 multi-channel grid observations.

    Architecture:
        Conv2d(in, 32, 3, pad=1) → BN → ReLU
        Conv2d(32, 64, 3, pad=1) → BN → ReLU
        Conv2d(64, 64, 3, stride=2) → BN → ReLU
        Flatten → Linear(64*7*7, hidden_dim)  → ReLU
    """

    def __init__(
        self,
        in_channels: int = _NUM_OBS_CHANNELS,
        channel_sizes: List[int] | None = None,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if channel_sizes is None:
            channel_sizes = [32, 64, 64]

        layers: list[nn.Module] = []
        prev_ch = in_channels
        for i, ch in enumerate(channel_sizes):
            stride = 2 if i == len(channel_sizes) - 1 else 1
            layers.extend([
                nn.Conv2d(prev_ch, ch, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ])
            prev_ch = ch

        self.conv = nn.Sequential(*layers)

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 16, 16)
            conv_out = self.conv(dummy)
            self._conv_flat_size = int(conv_out.view(1, -1).shape[1])

        self.fc = nn.Sequential(
            nn.Linear(self._conv_flat_size, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x shape: (B, C, 16, 16) → (B, hidden_dim)."""
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        return self.fc(h)


# ---------------------------------------------------------------------------
# PPO Network Heads
# ---------------------------------------------------------------------------

class PolicyHead(nn.Module):
    """Categorical policy head for discrete action space."""

    def __init__(self, input_dim: int, n_actions: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, n_actions)

    def forward(self, features: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.linear(features)
        return torch.distributions.Categorical(logits=logits)


class ValueHead(nn.Module):
    """Scalar value head for V(s)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


# ---------------------------------------------------------------------------
# NoisyNet Linear Layer
# ---------------------------------------------------------------------------

class NoisyLinear(nn.Module):
    """Factorised NoisyNet linear layer.

    Replaces standard nn.Linear with learnable noise parameters for
    exploration without ε-greedy. Uses factorised Gaussian noise for
    memory efficiency.
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Factorised noise — not learnable, registered as buffers
        self.weight_epsilon: torch.Tensor
        self.bias_epsilon: torch.Tensor
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self._init_parameters()
        self.reset_noise()

    def _init_parameters(self) -> None:
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# ---------------------------------------------------------------------------
# Dueling + Distributional (C51) Head for Rainbow DQN
# ---------------------------------------------------------------------------

class DuelingDistributionalHead(nn.Module):
    """Combined Dueling + C51 distributional head.

    Outputs per-action categorical distributions over atoms.
    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    where V and A are distributional (each outputs atom_size logits).
    """

    def __init__(
        self,
        input_dim: int,
        n_actions: int = 4,
        atom_size: int = 51,
        v_min: float = -100.0,
        v_max: float = 200.0,
        hidden_dim: int = 256,
        noisy_std: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max

        # Support atoms
        self.register_buffer(
            "support",
            torch.linspace(v_min, v_max, atom_size),
        )

        # Value stream (Noisy)
        self.v_fc = NoisyLinear(input_dim, hidden_dim, noisy_std)
        self.v_out = NoisyLinear(hidden_dim, atom_size, noisy_std)

        # Advantage stream (Noisy)
        self.a_fc = NoisyLinear(input_dim, hidden_dim, noisy_std)
        self.a_out = NoisyLinear(hidden_dim, n_actions * atom_size, noisy_std)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns log-probabilities over atoms for each action.

        Output shape: (B, n_actions, atom_size)
        """
        batch_size = features.size(0)

        # Value stream
        v = F.relu(self.v_fc(features))
        v = self.v_out(v).view(batch_size, 1, self.atom_size)

        # Advantage stream
        a = F.relu(self.a_fc(features))
        a = self.a_out(a).view(batch_size, self.n_actions, self.atom_size)

        # Dueling combination
        q_atoms = v + a - a.mean(dim=1, keepdim=True)

        # Log-softmax over atoms
        log_probs = F.log_softmax(q_atoms, dim=-1)
        return log_probs

    def q_values(self, features: torch.Tensor) -> torch.Tensor:
        """Compute expected Q-values from the distributional output.

        Returns shape: (B, n_actions)
        """
        log_probs = self.forward(features)
        probs = log_probs.exp()
        return (probs * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=-1)

    def reset_noise(self) -> None:
        self.v_fc.reset_noise()
        self.v_out.reset_noise()
        self.a_fc.reset_noise()
        self.a_out.reset_noise()
