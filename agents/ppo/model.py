import torch
import torch.nn as nn
from typing import Tuple


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    Attributes:
        actor: Neural network for policy (action probabilities).
        critic: Neural network for value estimation.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        """
        Initialize the Actor-Critic network.

        Args:
            obs_dim: Dimension of observation space.
            act_dim: Dimension of action space.
            hidden_dim: Number of units in hidden layers.
        """
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for both actor and critic.

        Args:
            x: Input observation tensor.

        Returns:
            Tuple of (action probabilities, value estimate).
        """
        return self.actor(x), self.critic(x)
