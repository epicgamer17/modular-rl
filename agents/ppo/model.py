import torch
import torch.nn as nn
import numpy as np
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
        
        def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
            return layer

        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
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
