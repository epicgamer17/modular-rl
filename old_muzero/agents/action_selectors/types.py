from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import torch
from old_muzero.modules.world_models.inference_output import InferenceOutput


@dataclass
class InferenceResult:
    """
    Standardized data contract for all policy-providing components.
    Used as the strict input type for all ActionSelectors.

    Fields mirror InferenceOutput semantics:
    - logits: raw pre-softmax policy logits (from network policy head)
    - probs: normalized probabilities (MCTS visit counts, or softmax of logits)
    - q_values: Q-value estimates per action
    - value: scalar state-value estimate
    - reward: predicted reward
    - to_play: current player index
    """

    value: Optional[torch.Tensor] = None
    q_values: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    probs: Optional[torch.Tensor] = None
    reward: Optional[torch.Tensor] = None
    to_play: Optional[torch.Tensor] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def action_dim(self) -> Optional[torch.Tensor]:
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
        q_values = getattr(output, "q_values", None)
        logits = None
        probs = None

        policy = getattr(output, "policy", None)
        if policy is not None:
            logits = getattr(policy, "logits", None)
            if logits is None:
                probs = getattr(policy, "probs", None)

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
            q_values is not None or logits is not None or probs is not None
        ), "InferenceOutput must contain q_values or a policy with logits/probs"

        return cls(
            value=value,
            q_values=q_values,
            logits=logits,
            probs=probs,
            reward=reward,
            to_play=to_play,
            extra_metadata=extras,
        )
