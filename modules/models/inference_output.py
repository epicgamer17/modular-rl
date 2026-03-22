from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List
import torch
from torch import Tensor
from torch.distributions import Distribution


def batch_recurrent_state(states: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Strictly batches a list of flat {str: Tensor} dictionaries.
    
    Rule: All states in the list must be purely flat Dict[str, Tensor].
    Rule: states[0] defines the keys. All dictionaries must have identical keys.
    Rule: All tensors must have batch size at dimension 0.
    """
    if not states or not states[0]:
        return {}
    
    return {k: torch.cat([s[k] for s in states], dim=0) for k in states[0].keys()}


def unbatch_recurrent_state(state: Dict[str, Tensor]) -> List[Dict[str, Tensor]]:
    """Strictly unbatches a flat {str: Tensor} dictionary into a list of dicts."""
    if not state:
        return []
    
    batch_size = next(iter(state.values())).shape[0]
    return [{k: v[i:i+1] for k, v in state.items()} for i in range(batch_size)]


@dataclass
class WorldModelOutput:
    """
    Represents the Agent's Hypothesis (Predictions) for a single step.
    Output of recurrent_inference and afterstate_recurrent_inference.
    """
    features: torch.Tensor
    reward: Optional[torch.Tensor] = None
    to_play: Optional[torch.Tensor] = None  # Actor-facing: argmax player index (B,)
    to_play_logits: Optional[torch.Tensor] = None  # Learner-facing: pre-softmax logits (B, P)
    q_values: Optional[torch.Tensor] = None

    # Opaque container for all state owned by the world model (dynamics + env heads)
    next_state: Dict[str, Tensor] = field(default_factory=dict)
    instant_reward: Optional[torch.Tensor] = None

    # Environment predictions
    continuation: Optional[torch.Tensor] = None
    continuation_logits: Optional[torch.Tensor] = None

    # Stochastic MuZero specific
    afterstate_features: Optional[torch.Tensor] = None
    chance: Optional[torch.Tensor] = None  # Chance logits
    chance_dist: Optional[Distribution] = None


@dataclass
class InferenceOutput:
    """Strict contract for data yielded to MCTS/Actors."""
    recurrent_state: Dict[str, Tensor] = field(default_factory=dict)
    
    # Behavior
    value: Optional[Tensor] = None
    policy: Optional[Distribution] = None
    action: Optional[Tensor] = None
    q_values: Optional[Tensor] = None
    
    # Environment (World Model)
    reward: Optional[Tensor] = None
    to_play: Optional[Tensor] = None
    chance: Optional[Distribution] = None
    
    # Anything truly bespoke
    extras: Dict[str, Tensor] = field(default_factory=dict)
