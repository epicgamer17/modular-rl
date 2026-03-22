from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
import torch
from torch import Tensor
from torch.distributions import Distribution, Categorical
from modules.models.inference_output import InferenceOutput


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

    @logits.setter
    def logits(self, value: Tensor):
        """Updates the policy with new logits."""
        self.policy = Categorical(logits=value)

    @property
    def probs(self) -> Optional[Tensor]:
        if hasattr(self.policy, "probs"):
            return self.policy.probs
        return None

    @probs.setter
    def probs(self, value: Optional[Tensor]):
        """Updates the policy with new probabilities, or clears it if None."""
        if value is None:
            # If we are clearing probs, we might still have logits if it's a Categorical.
            # However, usually this is called to force re-evaluation from logits.
            # For now, we just clear the policy if we don't have a way to keep only logits.
            # But the Categorical distribution in PyTorch doesn't easily let us clear one side.
            # If the user sets probs=None, they likely want to rely on the current logits.
            # But since this is a data contract, we'll just allow setting it to None if needed.
            if hasattr(self.policy, "probs"):
                # We can't easily 'unset' probs on an existing Categorical.
                # If someone sets result.probs = None, they might be trying to 'dirty' the object.
                # In our case, we'll just allow it to stay as is or set policy to None.
                pass
        else:
            self.policy = Categorical(probs=value)

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
