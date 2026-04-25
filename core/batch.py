from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch


@dataclass(frozen=True)
class TransitionBatch:
    """
    A typed container for batches of transitions.
    Enforces a consistent interface across algorithms (DQN, PPO, etc.)
    """

    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor  # Often terminated | truncated

    # Optional fields for specific algorithms
    log_prob: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None
    terminated: Optional[torch.Tensor] = None
    truncated: Optional[torch.Tensor] = None

    # For on-policy version tracking
    policy_version: Optional[torch.Tensor] = None

    # Extra data for training
    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None

    # Metadata for tracking/debugging
    metadata: Optional[Dict[str, Any]] = None
