from dataclasses import dataclass
from typing import Optional, Any
import torch
from torch.distributions import Distribution


@dataclass
class InferenceOutput:
    """
    The strict contract for data yielded to MCTS/Actor.
    Contains semantic, interpreted values (Expected Value, Distributions).
    """

    network_state: Any  # Opaque state (hidden_state, reward_hidden, etc.)
    value: float | torch.Tensor  # Expected Value (Scalar)
    policy: Distribution  # Action Distribution
    reward: Optional[float | torch.Tensor] = None  # Expected Reward (Scalar)
    chance: Optional[Distribution] = None  # Chance Distribution (for Stochastic MuZero)
    to_play: Optional[int | torch.Tensor] = None  # To Play (Scalar/Class Index)
    extras: Optional[dict] = (
        None  # Opaque extras (e.g., for logging or specialized heads)
    )


@dataclass
class UnrollOutput:
    """
    The strict contract for data yielded to the Learner.
    Contains raw logits for mathematically stable loss computation.
    """

    values: torch.Tensor  # [B, T+1, ...] Logits
    policies: torch.Tensor  # [B, T+1, ...] Logits
    rewards: torch.Tensor  # [B, T, ...] Logits
    to_plays: Optional[torch.Tensor] = None  # [B, T+1, ...] Logits

    # Optional components for specialized agents (Dreamer, Stochastic MuZero)
    latents: Optional[torch.Tensor] = None
    latents_afterstates: Optional[torch.Tensor] = None
    chance_logits: Optional[torch.Tensor] = None
    chance_values: Optional[torch.Tensor] = None
    extras: Optional[dict] = None
