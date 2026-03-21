from dataclasses import dataclass
from typing import NamedTuple, Optional, Any, Dict, List
import torch
from torch import Tensor
from torch.distributions import Distribution


def batch_recurrent_state(states: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Batches a list of single-agent state dictionaries into one batched dictionary.
    
    Handles:
    - Tensors: Concatenates across batch dimension (Dim 0, or Dim 1 for dim=3).
    - Tuples: Recursively batches elements (e.g. LSTM (h, c)).
    - Dicts: Recursively batches keys.
    - None/Primitives: Preserved from first element.
    """
    if not states:
        return {}

    first = states[0]
    batched = {}

    for k in first.keys():
        vals = [s[k] for s in states]

        if vals[0] is None:
            batched[k] = None
        elif isinstance(vals[0], torch.Tensor):
            # Special case for RNN states: [num_layers, batch, hidden]
            if vals[0].dim() == 3 and vals[0].shape[1] == 1:
                batched[k] = torch.cat(vals, dim=1)
            else:
                batched[k] = torch.cat(vals, dim=0)
        elif isinstance(vals[0], tuple):
            batched_elements = []
            for i in range(len(vals[0])):
                element_list = [v[i] for v in vals]
                if isinstance(element_list[0], torch.Tensor):
                    if element_list[0].dim() == 3 and element_list[0].shape[1] == 1:
                        batched_elements.append(torch.cat(element_list, dim=1))
                    else:
                        batched_elements.append(torch.cat(element_list, dim=0))
                else:
                    batched_elements.append(element_list[0])
            batched[k] = tuple(batched_elements)
        elif isinstance(vals[0], dict):
            batched[k] = batch_recurrent_state(vals)
        else:
            batched[k] = vals[0]

    return batched


def unbatch_recurrent_state(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Unbatches a state dictionary into a list of single-agent state dictionaries.
    """
    if not state:
        return []

    # 1. Determine batch size from the first tensor found
    batch_size = 0
    for v in state.values():
        if isinstance(v, torch.Tensor):
            if v.dim() == 3:  # [L, B, H]
                batch_size = v.shape[1]
            else:  # [B, ...]
                batch_size = v.shape[0]
            break
        elif isinstance(v, dict):
            # Recurse into dict to find a tensor
            sub_unbatched = unbatch_recurrent_state(v)
            if sub_unbatched:
                batch_size = len(sub_unbatched)
                break
    
    if batch_size == 0:
        return [state]

    unbatched = [{} for _ in range(batch_size)]

    for k, v in state.items():
        if v is None:
            for i in range(batch_size):
                unbatched[i][k] = None
        elif isinstance(v, torch.Tensor):
            if v.dim() == 3:
                for i in range(batch_size):
                    unbatched[i][k] = v[:, i : i + 1]
            else:
                if v.dim() == 0 or (v.dim() > 0 and v.shape[0] == 1 and batch_size > 1):
                    for i in range(batch_size):
                        unbatched[i][k] = v  # Broadcast
                else:
                    for i in range(batch_size):
                        unbatched[i][k] = v[i : i + 1]
        elif isinstance(v, tuple):
            tuple_elements_unbatched = []
            for element in v:
                if isinstance(element, torch.Tensor):
                    if element.dim() == 3:
                        tuple_elements_unbatched.append(
                            [element[:, i : i + 1] for i in range(batch_size)]
                        )
                    else:
                        tuple_elements_unbatched.append(
                            [element[i : i + 1] for i in range(batch_size)]
                        )
                else:
                    tuple_elements_unbatched.append([element] * batch_size)
            
            for i in range(batch_size):
                unbatched[i][k] = tuple(
                    elements[i] for elements in tuple_elements_unbatched
                )
        elif isinstance(v, dict):
            sub_unbatched = unbatch_recurrent_state(v)
            for i in range(batch_size):
                unbatched[i][k] = sub_unbatched[i]
        else:
            for i in range(batch_size):
                unbatched[i][k] = v

    return unbatched


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

    # Opaque recurrent state passed to next step
    head_state: Any = None
    instant_reward: Optional[torch.Tensor] = None

    # Stochastic MuZero specific
    afterstate_features: Optional[torch.Tensor] = None
    chance: Optional[torch.Tensor] = None  # Chance logits


class PhysicsOutput(NamedTuple):
    """
    Raw output from unroll_physics (WorldModel).
    Contains STACKED tensors for the entire unrolled sequence.
    """

    latents: torch.Tensor  # [B, T+1, ...]
    rewards: torch.Tensor  # [B, T+1, ...]
    to_plays: torch.Tensor  # [B, T+1, ...]

    # Stochastic optional fields
    latents_afterstates: Optional[torch.Tensor] = None
    chance_logits: Optional[torch.Tensor] = None
    afterstate_backbone_features: Optional[torch.Tensor] = None
    chance_encoder_embeddings: Optional[torch.Tensor] = None
    chance_encoder_onehots: Optional[torch.Tensor] = None
    target_latents: Optional[torch.Tensor] = None


class InferenceOutput(NamedTuple):
    """
    The strict contract for data yielded to MCTS/Actor (Single Step).
    """

    network_state: Any = None  # Generic dictionary (RecurrentState)
    value: float | torch.Tensor = 0.0
    q_values: Optional[torch.Tensor] = None
    policy: Optional[Distribution | Any] = None
    reward: Optional[float | torch.Tensor] = None
    chance: Optional[Distribution] = None
    to_play: Optional[int | torch.Tensor] = None
    extras: Optional[dict] = None
