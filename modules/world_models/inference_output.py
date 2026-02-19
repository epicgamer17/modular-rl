from typing import NamedTuple, Optional, Any
import torch
from torch.distributions import Distribution


class WorldModelOutput(NamedTuple):
    """
    Represents the Agent's Hypothesis (Predictions) for a single step.
    Output of recurrent_inference and afterstate_recurrent_inference.
    """

    features: torch.Tensor
    reward: Optional[torch.Tensor] = None
    to_play: Optional[torch.Tensor] = None

    # Opaque state (hidden_state, reward_hidden, etc.) passed to next step
    head_state: Any = None
    instant_reward: Optional[torch.Tensor] = None

    # Stochastic MuZero specific
    afterstate_features: Optional[torch.Tensor] = None  # Raw dynamics output
    chance: Optional[torch.Tensor] = None  # Chance logits


class PhysicsOutput(NamedTuple):
    """
    Raw output from unroll_physics (WorldModel).
    Contains STACKED tensors for the entire unrolled sequence.
    """

    latents: torch.Tensor  # [B, T+1, ...]
    rewards: torch.Tensor  # [B, T, ...]
    to_plays: torch.Tensor  # [B, T+1, ...]

    # Stochastic optional fields
    latents_afterstates: Optional[torch.Tensor] = None
    chance_logits: Optional[torch.Tensor] = None
    afterstate_backbone_features: Optional[torch.Tensor] = None
    encoder_softmaxes: Optional[torch.Tensor] = None
    encoder_onehots: Optional[torch.Tensor] = None


class InferenceOutput(NamedTuple):
    """
    The strict contract for data yielded to MCTS/Actor (Single Step).
    Contains semantic, interpreted values (Expected Value, Distributions).
    """

    network_state: Any = None  # Opaque state (hidden_state, reward_hidden, etc.)
    value: float | torch.Tensor = 0.0  # Expected Value (Scalar) V(s)
    q_values: Optional[torch.Tensor] = None  # Action Values Q(s, a)
    policy_logits: Optional[torch.Tensor] = None  # Raw Policy Output (Logits/Params)
    policy: Optional[Distribution | Any] = None  # Action Distribution
    reward: Optional[float | torch.Tensor] = None  # Expected Reward (Scalar)
    chance: Optional[Distribution] = None  # Chance Distribution (for Stochastic MuZero)
    to_play: Optional[int | torch.Tensor] = None  # To Play (Scalar/Class Index)
    extras: Optional[dict] = None  # Opaque extras


class UnrollOutput(NamedTuple):
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
    extras: Optional[dict] = None  # dictionary for any leftovers (e.g. encoder stats)
