from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
import torch
from torch import Tensor
from torch.distributions import Distribution
from modules.world_models.inference_output import InferenceOutput


@dataclass
class InferenceResult:
    """
    Standardized data contract for all policy-providing components.
    Used as the strict input type for all ActionSelectors.

    Fields mirror InferenceOutput semantics:
    - value: scalar state-value estimate
    - q_values: Q-value estimates per action
    - policy: policy distribution or semantic policy object
    - action: specific selected action (if any)
    - reward: predicted reward
    - to_play: current player index
    - extras: logging/debugging dictionary
    """

    recurrent_state: Any = None
    value: Optional[Tensor] = None
    q_values: Optional[Tensor] = None
    policy: Optional[Union[Distribution, Any]] = None
    action: Optional[Tensor] = None
    reward: Optional[Tensor] = None
    to_play: Optional[Tensor] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def logits(self) -> Optional[Tensor]:
        if hasattr(self.policy, "logits"):
            return self.policy.logits
        return None

    @property
    def probs(self) -> Optional[Tensor]:
        if hasattr(self.policy, "probs"):
            return self.policy.probs
        return None

    @property
    def action_dim(self) -> Optional[Tensor]:
        """Returns whichever action tensor is available (for shape queries)."""
        if self.q_values is not None:
            return self.q_values
        if self.logits is not None:
            return self.logits
        if self.probs is not None:
            return self.probs
        return None

    @classmethod
    def from_inference_output(cls, output: InferenceOutput) -> "InferenceResult":
        """
        Converts a standard InferenceOutput (network results)
        into a standardized InferenceResult for selectors.
        """
        recurrent_state = getattr(output, "recurrent_state", None)
        q_values = getattr(output, "q_values", None)
        policy = getattr(output, "policy", None)
        action = getattr(output, "action", None)
        value = getattr(output, "value", None)
        if value is not None and not isinstance(value, torch.Tensor):
            value = torch.tensor([value])

        reward = getattr(output, "reward", None)
        if reward is not None and not isinstance(reward, torch.Tensor):
            reward = torch.tensor([reward])

        to_play = getattr(output, "to_play", None)
        if to_play is not None and not isinstance(to_play, torch.Tensor):
            to_play = torch.tensor([to_play])

        extras = getattr(output, "extras", None) or {}

        assert (
            q_values is not None or policy is not None
        ), "InferenceOutput must contain q_values or a policy distribution"

        return cls(
            recurrent_state=recurrent_state,
            value=value,
            q_values=q_values,
            policy=policy,
            action=action,
            reward=reward,
            to_play=to_play,
            extras=extras,
        )
