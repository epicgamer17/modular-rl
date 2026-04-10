import torch
import numpy as np
from typing import Any, Dict
from core import PipelineComponent, Blackboard


def mask_actions(
    values: torch.Tensor,
    info: Dict[str, Any],
    mask_value: float = -float("inf"),
) -> torch.Tensor:
    """Applies legal-move masking to action values/logits."""
    # Extract mask from various potential keys
    legal_moves = info.get("legal_moves")
    if legal_moves is None:
        mask = info.get("action_mask", info.get("legal_moves_mask"))
        if mask is not None:
            if isinstance(mask, np.ndarray):
                if mask.dtype == bool or (mask.ndim == 1 and np.all((mask == 0) | (mask == 1))):
                    legal_moves = np.where(mask)[0].tolist()
                else:
                    legal_moves = mask.tolist()
            elif torch.is_tensor(mask):
                if mask.dtype == torch.bool:
                    legal_moves = torch.where(mask)[0].cpu().tolist()
                else:
                    legal_moves = mask.cpu().tolist()
            elif isinstance(mask, (list, tuple)) and len(mask) > 0:
                if isinstance(mask[0], (bool, np.bool_)):
                    legal_moves = [i for i, m in enumerate(mask) if m]
                else:
                    legal_moves = mask

    if legal_moves is None:
        return values

    mask_tensor = torch.zeros_like(values, dtype=torch.bool)
    device = values.device
    mask_tensor = mask_tensor.to(device)

    if values.dim() == 1:
        mask_tensor[legal_moves] = True
    elif values.dim() == 2:
        is_batched = True
        if torch.is_tensor(legal_moves) and legal_moves.dim() == 1:
            is_batched = False
        elif isinstance(legal_moves, (list, tuple)) and len(legal_moves) > 0:
            if not isinstance(legal_moves[0], (list, tuple, torch.Tensor, np.ndarray)):
                is_batched = False

        if not is_batched and values.shape[0] == 1:
            mask_tensor[0, legal_moves] = True
        else:
            for i, legal in enumerate(legal_moves):
                if legal is not None and i < mask_tensor.shape[0]:
                    mask_tensor[i, legal] = True

    return torch.where(mask_tensor, values, torch.tensor(mask_value, device=device))


def write_to_blackboard(
    blackboard: Blackboard, action: torch.Tensor, metadata: Dict[str, Any]
) -> None:
    """Writes action selection results to the blackboard."""
    # Squeeze for single-actor consistency
    if action.dim() > 0 and action.shape[0] == 1:
        raw_action = action.item()
        sq_action = action.squeeze(0)
    else:
        raw_action = action.item() if action.numel() == 1 else action.cpu().numpy()
        sq_action = action

    for k, v in metadata.items():
        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == 1:
            metadata[k] = v.squeeze(0)

    blackboard.meta["action"] = raw_action
    blackboard.meta["action_tensor"] = sq_action
    blackboard.meta["action_metadata"] = metadata


class CategoricalSelectorComponent(PipelineComponent):
    """Selects action by sampling from a Categorical distribution."""

    def __init__(self, exploration: bool = True):
        self.exploration = exploration

    def execute(self, blackboard: Blackboard) -> None:
        if blackboard.data.get("done", False):
            write_to_blackboard(blackboard, torch.tensor([0]), {"action_is_none": True})
            blackboard.meta["action"] = None
            return

        info = blackboard.meta.get("info", blackboard.data.get("info", {}))

        from torch.distributions import Categorical

        logits = blackboard.predictions.get("logits")
        probs = blackboard.predictions.get("probs")
        value = blackboard.predictions.get("value")

        if logits is not None:
            logits = mask_actions(logits, info)
            dist = Categorical(logits=logits)
        elif probs is not None:
            dummy_masked = mask_actions(torch.ones_like(probs), info, mask_value=0.0)
            mask_tensor = (dummy_masked == 1.0)

            probs = probs * mask_tensor.float()
            probs_sum = probs.sum(dim=-1, keepdim=True)

            if torch.isnan(probs).any() or (probs_sum < 1e-9).any():
                valid_mask = mask_tensor.float()
                if valid_mask.sum() < 1e-9:
                    valid_mask = torch.ones_like(probs)
                probs = valid_mask / valid_mask.sum(dim=-1, keepdim=True)
            else:
                probs = probs / probs_sum

            dist = Categorical(probs=probs)
        else:
            blackboard.meta["action"] = None
            return

        if self.exploration:
            action = dist.sample()
        else:
            action = torch.argmax(dist.probs, dim=-1)

        metadata = {
            "policy_dist": dist,
            "policy": dist.probs.detach().cpu(),
            "value": value.detach().cpu() if value is not None else None,
        }

        # Handle search metadata if present
        extra = blackboard.predictions.get("extra_metadata")
        if extra:
            metadata.update(extra)

        write_to_blackboard(blackboard, action, metadata)


class ArgmaxSelectorComponent(PipelineComponent):
    """Selects the action with the highest value/logit (Greedy)."""

    def execute(self, blackboard: Blackboard) -> None:
        if blackboard.data.get("done", False):
            write_to_blackboard(blackboard, torch.tensor([0]), {"action_is_none": True})
            blackboard.meta["action"] = None
            return

        info = blackboard.meta.get("info", blackboard.data.get("info", {}))

        q_values = blackboard.predictions.get("q_values")
        logits = blackboard.predictions.get("logits")
        probs = blackboard.predictions.get("probs")
        value = blackboard.predictions.get("value")

        values = q_values if q_values is not None else (logits if logits is not None else probs)
        
        if values is None:
            blackboard.meta["action"] = None
            return

        values = mask_actions(values, info)
        action = torch.argmax(values, dim=-1)

        metadata = {
            "value": value.detach().cpu() if value is not None else None,
            "policy": probs.detach().cpu() if probs is not None else None,
        }
        
        extra = blackboard.predictions.get("extra_metadata")
        if extra:
            metadata.update(extra)

        write_to_blackboard(blackboard, action, metadata)


class EpsilonGreedySelectorComponent(PipelineComponent):
    """Selects action using epsilon-greedy strategy."""

    def __init__(self, epsilon: float = 0.05):
        self.epsilon = epsilon

    def execute(self, blackboard: Blackboard) -> None:
        if blackboard.data.get("done", False):
            write_to_blackboard(blackboard, torch.tensor([0]), {"action_is_none": True})
            blackboard.meta["action"] = None
            return

        info = blackboard.meta.get("info", blackboard.data.get("info", {}))

        q_values = blackboard.predictions.get("q_values")
        probs = blackboard.predictions.get("probs")
        
        if q_values is None:
            blackboard.meta["action"] = None
            return

        if q_values.dim() == 2:
            batch_size = q_values.shape[0]
        else:
            batch_size = 1

        q_values = mask_actions(q_values, info)

        epsilon = blackboard.meta.get("epsilon", self.epsilon)

        r = torch.rand(batch_size, device=q_values.device)
        explore_mask = r < epsilon

        if batch_size == 1:
            if explore_mask.item():
                legal_moves_dummy = mask_actions(torch.zeros_like(q_values), info, mask_value=1.0)
                legal_indices = torch.where(legal_moves_dummy == 1.0)[1] if q_values.dim() == 2 else torch.where(legal_moves_dummy == 1.0)[0]

                if len(legal_indices) > 0:
                    idx = torch.randint(0, len(legal_indices), (1,), device=q_values.device)
                    action = legal_indices[idx]
                else:
                    action = torch.argmax(q_values, dim=-1)
            else:
                action = torch.argmax(q_values, dim=-1)
        else:
            greedy_actions = torch.argmax(q_values, dim=-1)
            action = greedy_actions

        metadata = {
            "value": q_values.max(dim=-1)[0].detach().cpu(),
            "policy": probs.detach().cpu() if probs is not None else None,
            "epsilon": epsilon,
        }
        
        extra = blackboard.predictions.get("extra_metadata")
        if extra:
            metadata.update(extra)

        write_to_blackboard(blackboard, action, metadata)


class NFSPSelectorComponent(PipelineComponent):
    """
    Manages selection between Best Response and Average Strategy for NFSP.
    Expects two sets of inference results in the blackboard.
    """

    def __init__(
        self,
        br_prefix: str = "br_",
        avg_prefix: str = "avg_",
        eta: float = 0.1,
    ):
        self.br_prefix = br_prefix
        self.avg_prefix = avg_prefix
        self.eta = eta
        self.br_selector = ArgmaxSelectorComponent()
        self.avg_selector = ArgmaxSelectorComponent()

    def execute(self, blackboard: Blackboard) -> None:
        import random

        eta = blackboard.meta.get("eta", self.eta)

        if random.random() < eta:
            blackboard.meta["policy_used"] = "best_response"
            # Swap prefixed keys to standard keys for the child selector
            original_preds = blackboard.predictions.copy()
            for key in ["logits", "probs", "q_values", "value", "extra_metadata"]:
                prefixed_key = self.br_prefix + key
                if prefixed_key in original_preds:
                    blackboard.predictions[key] = original_preds[prefixed_key]
            
            self.br_selector.execute(blackboard)
            blackboard.predictions = original_preds
        else:
            blackboard.meta["policy_used"] = "average_strategy"
            original_preds = blackboard.predictions.copy()
            for key in ["logits", "probs", "q_values", "value", "extra_metadata"]:
                prefixed_key = self.avg_prefix + key
                if prefixed_key in original_preds:
                    blackboard.predictions[key] = original_preds[prefixed_key]
            
            self.avg_selector.execute(blackboard)
            blackboard.predictions = original_preds
