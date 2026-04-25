from dataclasses import dataclass, fields
from typing import Optional, Dict, Any
import torch


# TODO: should this be a runtime value?
@dataclass(frozen=True)
class TransitionBatch:
    """
    A typed container for batches of transitions.
    Enforces a consistent interface across algorithms (DQN, PPO, etc.)
    """

    obs: Optional[torch.Tensor] = None
    action: Optional[torch.Tensor] = None
    reward: Optional[torch.Tensor] = None
    next_obs: Optional[torch.Tensor] = None
    done: Optional[torch.Tensor] = None  # Often terminated | truncated

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

    def to_dict(self, drop_none: bool = True) -> Dict[str, Any]:
        """Returns a shallow dict of fields. Tensors are not cloned."""
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        if drop_none:
            out = {k: v for k, v in out.items() if v is not None}
        return out
