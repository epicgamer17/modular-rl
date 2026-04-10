from typing import Any, Dict, Optional, Tuple
import torch
from actors.action_selectors.selectors import BaseActionSelector
from torch.distributions import Categorical
from utils.schedule import create_schedule, Schedule, ScheduleConfig


class PPODecorator(BaseActionSelector):
    """
    Decorator that injects PPO-specific metadata (log_prob, value)
    into the selection result.
    """

    def __init__(self, inner_selector: BaseActionSelector):
        self.inner_selector = inner_selector

    def select_action(
        self,
        predictions: Dict[str, torch.Tensor],
        info: Dict[str, Any],
        exploration: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        # 1. Delegate selection to the inner selector
        action, metadata = self.inner_selector.select_action(
            predictions,
            info,
            exploration=exploration,
            **kwargs,
        )

        # 2. Inject PPO metadata
        dist = metadata.get("policy_dist")
        if dist is not None:
            metadata["log_prob"] = dist.log_prob(action).cpu()

        value = predictions.get("value")
        if value is not None:
            metadata["value"] = value.cpu()

        return action, metadata

    def mask_actions(
        self,
        values: torch.Tensor,
        legal_moves: Any,
        mask_value: float = -float("inf"),
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        return self.inner_selector.mask_actions(values, legal_moves, mask_value, device)

    def update_parameters(self, params_dict: Dict[str, Any]) -> None:
        """
        Pass parameter updates down to the inner selector.
        """
        self.inner_selector.update_parameters(params_dict)


