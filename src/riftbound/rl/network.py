"""Policy/value network over the flat observation and action encodings.

The model is deliberately small: the observation is ~1.1k floats and the
action table ~800 entries, so a two-layer MLP trunk (~1M parameters) is
plenty and trains at interactive speed on CPU. Illegal actions are masked
out of the policy by setting their logits to -inf before the softmax, so
the distribution only ever puts mass on legal action indices.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

# Logit value used to mask out illegal actions. Large-but-finite avoids
# NaNs from (-inf) - (-inf) arithmetic inside log_softmax backward.
MASK_LOGIT = -1e9


def masked_distribution(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """A categorical over legal actions only.

    `logits`: (..., A) raw policy head output.
    `mask`:   (..., A) bool, True where the action is legal.
    """
    masked = torch.where(mask, logits, torch.full_like(logits, MASK_LOGIT))
    return Categorical(logits=masked)


class PolicyValueNet(nn.Module):
    """MLP trunk with a masked policy head and a tanh value head in [-1, 1]."""

    def __init__(self, obs_size: int, action_size: int, hidden: int = 512) -> None:
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.hidden = hidden
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, action_size)
        self.value_head = nn.Linear(hidden, 1)
        # Near-uniform initial policy and near-zero initial values keep early
        # PPO updates well-conditioned.
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (raw policy logits (..., A), value (...,) in [-1, 1])."""
        h = self.trunk(obs)
        return self.policy_head(h), torch.tanh(self.value_head(h)).squeeze(-1)

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, mask: torch.Tensor, deterministic: bool = False
    ) -> tuple[int, float, float]:
        """Single-state helper: returns (action index, log-prob, value)."""
        logits, value = self(obs.unsqueeze(0))
        dist = masked_distribution(logits, mask.unsqueeze(0))
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())
