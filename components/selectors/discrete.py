import torch
import numpy as np
from typing import Any, Dict
from core import PipelineComponent, Blackboard


class BaseSelectorComponent(PipelineComponent):
    """
    Base class for selection components that read from 'inference_result'
    and write to 'meta["action"]', 'meta["action_metadata"]'.
    """

    def mask_actions(
        self,
        values: torch.Tensor,
        info: Dict[str, Any],
        mask_value: float = -float("inf"),
    ) -> torch.Tensor:
        # Extract mask from various potential keys
        legal_moves = info.get("legal_moves")
        if legal_moves is None:
            mask = info.get("action_mask", info.get("legal_moves_mask"))
            if mask is not None:
                if isinstance(mask, np.ndarray):
                    if mask.dtype == bool or (mask.ndim == 1 and np.all((mask == 0) | (mask == 1))):
                        # Convert boolean/binary mask to indices
                        legal_moves = np.where(mask)[0].tolist()
                    else:
                        legal_moves = mask.tolist()
                elif torch.is_tensor(mask):
                    if mask.dtype == torch.bool:
                        legal_moves = torch.where(mask)[0].cpu().tolist()
                    else:
                        legal_moves = mask.cpu().tolist()
                elif isinstance(mask, (list, tuple)) and len(mask) > 0:
                     # If it's already a list of indices or bools, use as is or convert
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
            # detect if legal_moves is a single mask for a batch-of-1,
            # or a batch of masks.
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
        self, blackboard: Blackboard, action: torch.Tensor, metadata: Dict[str, Any]
    ) -> None:
        # Use .get() to avoid KeyError if inference was skipped
        result = blackboard.predictions.get("inference_result")

        # Merge extra metadata from search/network
        if result is not None and result.extra_metadata:
            for k, v in result.extra_metadata.items():
                if k not in metadata:
                    metadata[k] = v

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


class CategoricalSelectorComponent(BaseSelectorComponent):
    """
    Selects action by sampling from a Categorical distribution.
    """

    def __init__(self, exploration: bool = True):
        self.exploration = exploration

    def execute(self, blackboard: Blackboard) -> None:
        # Skip if agent is already done
        if blackboard.data.get("done", False):
            self.write_to_blackboard(blackboard, torch.tensor([0]), {"action_is_none": True}) # Dummy, step handles None
            blackboard.meta["action"] = None
            return

        result = blackboard.predictions.get("inference_result")
        info = blackboard.meta.get("info", blackboard.data.get("info", {}))

        from torch.distributions import Categorical

        if result is not None:
            if result.logits is not None:
                logits = result.logits
                logits = self.mask_actions(logits, info)
                dist = Categorical(logits=logits)
            else:
                # Manual re-masking for probs
                probs = result.probs
                # Extract mask via base helper
                # We reuse mask_actions logic by creating a dummy one
                # If no legal moves indices were found, we want mask_tensor to be ALL TRUE
                dummy_masked = self.mask_actions(torch.ones_like(probs), info, mask_value=0.0)
                mask_tensor = (dummy_masked == 1.0)

                probs = probs * mask_tensor.float()
                probs_sum = probs.sum(dim=-1, keepdim=True)

                # Check for NaNs or all-zeros
                if torch.isnan(probs).any() or (probs_sum < 1e-9).any():
                    # Fallback to uniform over legal moves
                    valid_mask = mask_tensor.float()
                    if valid_mask.sum() < 1e-9:
                         valid_mask = torch.ones_like(probs) # Ultimate fallback
                    probs = valid_mask / valid_mask.sum(dim=-1, keepdim=True)
                else:
                    probs = probs / probs_sum

                dist = Categorical(probs=probs)

            value = result.value
        else:
            # Try direct keys
            logits = blackboard.predictions.get("logits")
            value = blackboard.predictions.get("value")
            probs = blackboard.predictions.get("probs")

            if logits is not None:
                logits = self.mask_actions(logits, info)
                dist = Categorical(logits=logits)
            elif probs is not None:
                dist = Categorical(probs=probs)
            else:
                # Fallback to None if no policy info at all
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

        self.write_to_blackboard(blackboard, action, metadata)


class ArgmaxSelectorComponent(BaseSelectorComponent):
    """
    Selects the action with the highest value/logit (Greedy).
    """

    def execute(self, blackboard: Blackboard) -> None:
        # Skip if agent is already done
        if blackboard.data.get("done", False):
            self.write_to_blackboard(blackboard, torch.tensor([0]), {"action_is_none": True})
            blackboard.meta["action"] = None
            return

        result = blackboard.predictions.get("inference_result")
        info = blackboard.meta.get("info", blackboard.data.get("info", {}))

        if result is not None:
            values = (
                result.q_values
                if result.q_values is not None
                else (result.logits if result.logits is not None else result.probs)
            )
            value = result.value
            probs = result.probs
        else:
            # Try direct keys
            values = blackboard.predictions.get("logits", blackboard.predictions.get("probs"))
            value = blackboard.predictions.get("value")
            probs = blackboard.predictions.get("probs")

        values = self.mask_actions(values, info)

        action = torch.argmax(values, dim=-1)

        metadata = {
            "value": value.detach().cpu() if value is not None else None,
            "policy": probs.detach().cpu() if probs is not None else None,
        }

        self.write_to_blackboard(blackboard, action, metadata)


class EpsilonGreedySelectorComponent(BaseSelectorComponent):
    """
    Selects action using epsilon-greedy strategy.
    """

    def __init__(self, epsilon: float = 0.05):
        self.epsilon = epsilon

    def execute(self, blackboard: Blackboard) -> None:
        # Skip if agent is already done
        if blackboard.data.get("done", False):
            self.write_to_blackboard(blackboard, torch.tensor([0]), {"action_is_none": True})
            blackboard.meta["action"] = None
            return

        result = blackboard.predictions.get("inference_result")
        info = blackboard.meta.get("info", blackboard.data.get("info", {}))

        if result is not None:
            q_values = result.q_values
            probs = result.probs
        else:
            q_values = blackboard.predictions.get("q_values")
            probs = blackboard.predictions.get("probs")

        if q_values.dim() == 2:
            batch_size = q_values.shape[0]
        else:
            batch_size = 1

        q_values = self.mask_actions(q_values, info)

        # Update epsilon from metadata if available
        epsilon = blackboard.meta.get("epsilon", self.epsilon)

        r = torch.rand(batch_size, device=q_values.device)
        explore_mask = r < epsilon

        if batch_size == 1:
            if explore_mask.item():
                # Random action from legal moves
                legal_moves_dummy = self.mask_actions(torch.zeros_like(q_values), info, mask_value=1.0)
                legal_indices = torch.where(legal_moves_dummy == 1.0)[1] if q_values.dim() == 2 else torch.where(legal_moves_dummy == 1.0)[0]

                if len(legal_indices) > 0:
                    idx = torch.randint(0, len(legal_indices), (1,), device=q_values.device)
                    action = legal_indices[idx]
                else:
                    action = torch.argmax(q_values, dim=-1) # Fallback
            else:
                action = torch.argmax(q_values, dim=-1)
        else:
            # Batched version (simplified)
            greedy_actions = torch.argmax(q_values, dim=-1)
            # For simplicity, we just use greedy for now in batch > 1 if not easily vectorized
            # or we do it properly
            actions = greedy_actions # TODO: implement batched random actions if needed
            action = actions

        metadata = {
            "value": q_values.max(dim=-1)[0].detach().cpu(),
            "policy": probs.detach().cpu() if probs is not None else None,
            "epsilon": epsilon,
        }

        self.write_to_blackboard(blackboard, action, metadata)


class NFSPSelectorComponent(BaseSelectorComponent):
    """
    Manages selection between Best Response and Average Strategy for NFSP.
    Expects two inference results in the blackboard.
    """

    def __init__(
        self,
        br_key: str = "br_inference",
        avg_key: str = "avg_inference",
        eta: float = 0.1,
    ):
        self.br_key = br_key
        self.avg_key = avg_key
        self.eta = eta
        self.br_selector = ArgmaxSelectorComponent()
        self.avg_selector = ArgmaxSelectorComponent()

    def execute(self, blackboard: Blackboard) -> None:
        import random

        eta = blackboard.meta.get("eta", self.eta)

        if random.random() < eta:
            blackboard.meta["policy_used"] = "best_response"
            # Swap inference_result to BR for the selector
            old_result = blackboard.predictions.get("inference_result")
            blackboard.predictions["inference_result"] = blackboard.predictions[
                self.br_key
            ]
            self.br_selector.execute(blackboard)
            blackboard.predictions["inference_result"] = old_result
        else:
            blackboard.meta["policy_used"] = "average_strategy"
            # Swap inference_result to AVG for the selector
            old_result = blackboard.predictions.get("inference_result")
            blackboard.predictions["inference_result"] = blackboard.predictions[
                self.avg_key
            ]
            self.avg_selector.execute(blackboard)
            blackboard.predictions["inference_result"] = old_result
