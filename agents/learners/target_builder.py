from abc import ABC, abstractmethod
from typing import Dict
import torch.nn as nn
from torch import Tensor


class BaseTargetBuilder(ABC):
    """
    Abstract base class for calculating Reinforcement Learning targets.

    The TargetBuilder decouples the logic for computing targets (e.g., Bellman targets,
    MuZero unrolled targets, TD-lambda targets) from the Learner and the Losses.
    """

    @abstractmethod
    def build_targets(
        self,
        batch: Dict[str, Tensor],
        predictions: Dict[str, Tensor],
        network: nn.Module,
    ) -> Dict[str, Tensor]:
        """
        Build target tensors for the loss calculation.

        Args:
            batch: A dictionary of tensors containing experience data from the replay buffer.
            predictions: A dictionary of tensors containing the network's current predictions.
            network: The neural network module (may be used for computing targets like target networks).

        Returns:
            A dictionary of target tensors (e.g., {"values": target_values, "policies": target_policies}).
        """
        pass  # pragma: no cover
