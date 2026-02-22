from typing import NamedTuple, Optional, Any
import torch
from torch import Tensor
from torch.distributions import Distribution


class MuZeroNetworkState(NamedTuple):
    """
    Opaque token passed between MuZero's inference calls and the MCTS.

    The search tree stores and forwards this without inspecting it.
    Only ``MuZeroNetwork`` ever unpacks the fields.

    Attributes:
        dynamics: The current latent hidden state (from representation or dynamics).
        wm_memory: Opaque world-model recurrent state (e.g. LSTM hidden for
                   ValuePrefix). ``None`` when no recurrent state is used.
    """

    dynamics: Tensor
    wm_memory: Any = None


class WorldModelOutput(NamedTuple):
    """
    Represents the Agent's Hypothesis (Predictions) for a single step.
    Output of recurrent_inference and afterstate_recurrent_inference.
    """

    features: torch.Tensor
    reward: Optional[torch.Tensor] = None
    to_play: Optional[torch.Tensor] = None  # Actor-facing: argmax player index (B,)
    to_play_logits: Optional[torch.Tensor] = (
        None  # Learner-facing: pre-softmax logits (B, P)
    )
    q_values: Optional[torch.Tensor] = None

    # Opaque state (hidden_state, reward_hidden, etc.) passed to next step
    # The World Model packs all its internal recurrent states into this field.
    # The AgentNetwork treats this as a black box.
    head_state: Any = None
    instant_reward: Optional[torch.Tensor] = None

    # Stochastic MuZero specific
    afterstate_features: Optional[torch.Tensor] = None  # Raw dynamics output
    chance: Optional[torch.Tensor] = None  # Chance logits


class PhysicsOutput(NamedTuple):
    """
    Raw output from unroll_physics (WorldModel).
    Contains STACKED tensors for the entire unrolled sequence.
    All tensors have shape [B, T+1, ...] where T is the number of unroll steps.
    Index 0 for transition-based fields (rewards, chance) is a dummy/padding step.
    """

    latents: torch.Tensor  # [B, T+1, ...]
    rewards: torch.Tensor  # [B, T+1, ...]
    to_plays: torch.Tensor  # [B, T+1, ...]

    # Stochastic optional fields
    latents_afterstates: Optional[torch.Tensor] = None  # [B, T+1, ...]
    chance_logits: Optional[torch.Tensor] = None  # [B, T+1, ...]
    afterstate_backbone_features: Optional[torch.Tensor] = None  # [B, T+1, ...]
    chance_encoder_softmaxes: Optional[torch.Tensor] = None  # [B, T+1, ...]
    encoder_onehots: Optional[torch.Tensor] = None  # [B, T+1, ...]
    target_latents: Optional[torch.Tensor] = None  # [B, T+1, ...]


class InferenceOutput(NamedTuple):
    """
    The strict contract for data yielded to MCTS/Actor (Single Step).
    Contains semantic, interpreted values (Expected Value, Distributions).
    Note: Actor does NOT receive raw logits anymore.
    """

    network_state: Any = None  # Opaque state (hidden_state, reward_hidden, etc.)
    value: float | torch.Tensor = 0.0  # Expected Value (Scalar) V(s)
    q_values: Optional[torch.Tensor] = None  # Action Values Q(s, a)
    policy: Optional[Distribution | Any] = None  # Action Distribution
    reward: Optional[float | torch.Tensor] = None  # Expected Reward (Scalar)
    chance: Optional[Distribution] = None  # Chance Distribution (for Stochastic MuZero)
    to_play: Optional[int | torch.Tensor] = None  # To Play (Scalar/Class Index)
    extras: Optional[dict] = None  # Opaque extras

    # Removed policy_logits as Actor uses Distribution directly.


class LearningOutput(NamedTuple):
    """
    The strict contract for data yielded to the Learner.
    Contains raw logits for mathematically stable loss computation.
    """

    values: torch.Tensor  # [B, T+1, ...] Logits or Values
    policies: Optional[torch.Tensor] = None  # [B, T+1, ...] Logits (PPO/MuZero)
    q_values: Optional[torch.Tensor] = None  # [B, T+1, num_actions]  (Rainbow)
    q_logits: Optional[torch.Tensor] = None  # [B, T+1, num_actions, num_atoms]
    rewards: Optional[torch.Tensor] = None  # [B, T+1, ...] Logits
    to_plays: Optional[torch.Tensor] = None  # [B, T+1, ...] Logits

    # Optional components for specialized agents (Dreamer, Stochastic MuZero)
    latents: Optional[torch.Tensor] = None
    latents_afterstates: Optional[torch.Tensor] = None
    chance_logits: Optional[torch.Tensor] = None
    chance_values: Optional[torch.Tensor] = None
    target_latents: Optional[torch.Tensor] = None  # [B, T+1, ...]
    chance_encoder_softmaxes: Optional[torch.Tensor] = (
        None  # [B, T+1, num_chance] (Stochastic MuZero)
    )
