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

    # 1. Find the first non-None state to use as a template for keys
    first = None
    for s in states:
        if s is not None:
            first = s
            break
    
    if first is None:
        return None

    batched = {}

    for k in first.keys():
        # Safely extract values for this key across all states
        vals = [s[k] if (s is not None and k in s) else None for s in states]
        
        # Find the first non-None value to determine type
        first_val = next((v for v in vals if v is not None), None)

        if first_val is None:
            batched[k] = None
        elif isinstance(first_val, torch.Tensor):
            # Special case for RNN states: [num_layers, batch, hidden]
            # If some values are None, pad them with zeros of the same shape/device
            padded_vals = []
            for v in vals:
                if v is not None:
                    padded_vals.append(v)
                else:
                    padded_vals.append(torch.zeros_like(first_val))
            
            if first_val.dim() == 3 and first_val.shape[1] == 1:
                batched[k] = torch.cat(padded_vals, dim=1)
            else:
                batched[k] = torch.cat(padded_vals, dim=0)
        elif isinstance(first_val, tuple):
            batched_elements = []
            for i in range(len(first_val)):
                # Recursively batch elements, handling None tuples
                element_list = []
                for v in vals:
                    if v is not None:
                        element_list.append(v[i])
                    else:
                        element_list.append(None)
                
                # Check if elements are tensors or need further recursion
                first_elem = next((e for e in element_list if e is not None), None)
                if isinstance(first_elem, torch.Tensor):
                    padded_elems = [e if e is not None else torch.zeros_like(first_elem) for e in element_list]
                    if first_elem.dim() == 3 and first_elem.shape[1] == 1:
                        batched_elements.append(torch.cat(padded_elems, dim=1))
                    else:
                        batched_elements.append(torch.cat(padded_elems, dim=0))
                elif isinstance(first_elem, (dict, list, tuple)):
                    # Further generic recursion if needed, but usually just dicts
                    if isinstance(first_elem, dict):
                        batched_elements.append(batch_recurrent_state(element_list))
                    else:
                        batched_elements.append(first_elem)
                else:
                    batched_elements.append(first_elem)
            batched[k] = tuple(batched_elements)
        elif isinstance(first_val, dict):
            batched[k] = batch_recurrent_state(vals)
        else:
            batched[k] = first_val

    return batched

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
            if len(sub_unbatched) == 1 and batch_size > 1:
                # Broadcast sub-dict if it has no tensors or batch size 1
                for i in range(batch_size):
                    unbatched[i][k] = sub_unbatched[0]
            else:
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

    # Environment predictions
    continuation: Optional[torch.Tensor] = None
    continuation_logits: Optional[torch.Tensor] = None

    # Stochastic MuZero specific
    afterstate_features: Optional[torch.Tensor] = None
    chance: Optional[torch.Tensor] = None  # Chance logits




class InferenceOutput(NamedTuple):
    """
    The strict contract for data yielded to MCTS/Actor (Single Step).
    """

    recurrent_state: Any = None  # Generic dictionary (Opaque Token)
    value: float | torch.Tensor = 0.0
    q_values: Optional[torch.Tensor] = None
    policy: Optional[Distribution | Any] = None
    action: Optional[torch.Tensor] = None  # Greedy or sampled action
    extras: Optional[Dict[str, Any]] = None

    # Search-specific fields (Kept for compatibility unless explicitly asked to move to extras)
    reward: Optional[float | torch.Tensor] = None
    chance: Optional[Distribution] = None
    to_play: Optional[int | torch.Tensor] = None
