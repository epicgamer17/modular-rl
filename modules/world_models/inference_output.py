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

    @classmethod
    def batch(cls, states: list["MuZeroNetworkState"]) -> "MuZeroNetworkState":
        """Batches a list of single-item states into one batched state."""
        dynamics = torch.cat([s.dynamics for s in states], dim=0)

        # Handle wm_memory which can be None, Tuple (LSTM), or Tensor
        wm_mem_list = [s.wm_memory for s in states]
        if wm_mem_list[0] is None:
            wm_memory = None
        elif isinstance(wm_mem_list[0], tuple):
            # E.g. LSTM states: (h_n, c_n), each shape [num_layers, batch, hidden]
            batched_lstm = []
            for i in range(len(wm_mem_list[0])):
                tensors = [mem[i] for mem in wm_mem_list]
                if tensors[0].dim() == 3 and tensors[0].shape[1] == 1:
                    batched_lstm.append(torch.cat(tensors, dim=1))
                else:
                    batched_lstm.append(torch.cat(tensors, dim=0))
            wm_memory = tuple(batched_lstm)
        elif isinstance(wm_mem_list[0], torch.Tensor):
            wm_memory = torch.cat(wm_mem_list, dim=0)
        elif isinstance(wm_mem_list[0], dict):
            batched_dict = {}
            for k in wm_mem_list[0].keys():
                tensors = [mem[k] for mem in wm_mem_list]
                if tensors[0] is None:
                    batched_dict[k] = None
                elif isinstance(tensors[0], torch.Tensor):
                    if tensors[0].dim() == 3 and tensors[0].shape[1] == 1:
                        batched_dict[k] = torch.cat(tensors, dim=1)
                    else:
                        batched_dict[k] = torch.cat(tensors, dim=0)
                else:
                    raise ValueError(
                        f"Unknown wm_memory dict value type: {type(tensors[0])}"
                    )
            wm_memory = batched_dict
        else:
            raise ValueError(f"Unknown wm_memory type: {type(wm_mem_list[0])}")

        return cls(dynamics=dynamics, wm_memory=wm_memory)

    def unbatch(self) -> list["MuZeroNetworkState"]:
        """Unbatches this state into a list of single-batch states."""
        batch_size = self.dynamics.shape[0]
        unbatched_dynamics = [self.dynamics[i : i + 1] for i in range(batch_size)]

        if self.wm_memory is None:
            unbatched_memory = [None for _ in range(batch_size)]
        elif isinstance(self.wm_memory, tuple):
            unbatched_memory = []
            for j in range(batch_size):
                mem_j = []
                for t in self.wm_memory:
                    if t.dim() == 3:
                        mem_j.append(t[:, j : j + 1])
                    else:
                        mem_j.append(t[j : j + 1])
                unbatched_memory.append(tuple(mem_j))
        elif isinstance(self.wm_memory, torch.Tensor):
            if self.wm_memory.dim() == 0:
                unbatched_memory = [self.wm_memory] * batch_size  # Broadcast
            else:
                s_batch = self.wm_memory.shape[0]
                if s_batch == 1 and batch_size > 1:
                    unbatched_memory = [self.wm_memory] * batch_size  # Broadcast
                else:
                    unbatched_memory = [
                        self.wm_memory[i : i + 1] for i in range(batch_size)
                    ]
        elif isinstance(self.wm_memory, dict):
            unbatched_memory = [{} for _ in range(batch_size)]
            for k, v in self.wm_memory.items():
                if v is None:
                    for j in range(batch_size):
                        unbatched_memory[j][k] = None
                elif isinstance(v, torch.Tensor):
                    if v.dim() == 0:
                        for j in range(batch_size):
                            unbatched_memory[j][k] = v  # Broadcast
                    else:
                        s_batch = v.shape[0] if v.dim() != 3 else v.shape[1]
                        if s_batch == 1 and batch_size > 1:
                            for j in range(batch_size):
                                unbatched_memory[j][k] = v  # Broadcast
                        else:
                            for j in range(batch_size):
                                if v.dim() == 3:
                                    unbatched_memory[j][k] = v[:, j : j + 1]
                                else:
                                    unbatched_memory[j][k] = v[j : j + 1]
                else:
                    for j in range(batch_size):
                        unbatched_memory[j][k] = v
        else:
            raise ValueError(f"Unknown wm_memory type: {type(self.wm_memory)}")

        return [
            MuZeroNetworkState(
                dynamics=unbatched_dynamics[i], wm_memory=unbatched_memory[i]
            )
            for i in range(batch_size)
        ]


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
    afterstate_features: Optional[torch.Tensor] = None
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
    chance_encoder_embeddings: Optional[torch.Tensor] = None  # [B, T+1, ...]
    chance_encoder_onehots: Optional[torch.Tensor] = None  # [B, T+1, ...]
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

    values: Optional[torch.Tensor] = None  # [B, T+1, ...] Logits or Values
    policies: Optional[torch.Tensor] = None  # [B, T+1, ...] Logits (PPO/MuZero)
    q_values: Optional[torch.Tensor] = (
        None  # [B, T+1, num_actions] (Rainbow/DQN Online)
    )
    q_logits: Optional[torch.Tensor] = (
        None  # [B, T+1, num_actions, num_atoms] (Rainbow Online)
    )
    rewards: Optional[torch.Tensor] = None  # [B, T+1, ...] Logits
    to_plays: Optional[torch.Tensor] = None  # [B, T+1, ...] Logits

    # Optional components for specialized agents (Dreamer, Stochastic MuZero)
    latents: Optional[torch.Tensor] = None
    latents_afterstates: Optional[torch.Tensor] = None
    chance_logits: Optional[torch.Tensor] = None
    chance_values: Optional[torch.Tensor] = None
    target_latents: Optional[torch.Tensor] = None  # [B, T+1, ...]
    chance_encoder_embeddings: Optional[torch.Tensor] = (
        None  # [B, T+1, num_chance] (Stochastic MuZero)
    )
