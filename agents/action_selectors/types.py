from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
import torch
from torch import Tensor
from torch.distributions import Distribution, Categorical
from modules.models.inference_output import InferenceOutput


@dataclass(frozen=True)
class InferenceResult:
    """
    Standardized, immutable data contract for all policy-providing components.
    Used as the strict input type for all ActionSelectors.
    
    Fields are explicit tensors. The ActionSelector is responsible for 
    instantiating PyTorch Distributions if they are needed for sampling.
    """
    value: Optional[torch.Tensor] = None
    q_values: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None  # Explicit pre-softmax logits
    probs: Optional[torch.Tensor] = None   # Explicit probabilities
    action: Optional[torch.Tensor] = None  # Specific pre-selected action (e.g. from MCTS)
    
    # Generic container for algorithm-specific data (recurrent_state, v, to_play, etc)
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Fail fast if the contract is violated upon creation
        if (self.q_values is None and 
            self.logits is None and 
            self.probs is None and 
            self.action is None):
            raise ValueError(
                "InferenceResult must contain at least one action-derivable tensor "
                "(q_values, logits, probs, or action)."
            )

    @classmethod
    def from_inference_output(cls, output: InferenceOutput) -> "InferenceResult":
        """
        Converts a standard InferenceOutput (network results)
        into a standardized, frozen InferenceResult for selectors.
        """
        q_values = getattr(output, "q_values", None)
        policy = getattr(output, "policy", None) # Often a Distribution
        action = getattr(output, "action", None)
        value = getattr(output, "value", None)
        
        # Extract explicit tensors from policy distribution if it exists
        logits = None
        probs = None
        if policy is not None:
            if hasattr(policy, "logits"):
                logits = policy.logits
            if hasattr(policy, "probs"):
                probs = policy.probs
        
        # Consolidate everything else into extras to keep the contract clean
        extras = dict(getattr(output, "extras", None) or {})
        
        # Capture optional network outputs that aren't core behavior
        for attr in ["recurrent_state", "reward", "to_play"]:
            if hasattr(output, attr):
                val = getattr(output, attr)
                if val is not None:
                    extras[attr] = val

        return cls(
            value=value,
            q_values=q_values,
            logits=logits,
            probs=probs,
            action=action,
            extras=extras,
        )
