from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List, Tuple
import torch
import time
import numpy as np

from modules.agent_nets.base import BaseAgentNetwork
from modules.world_models.inference_output import InferenceOutput


class BasePolicySource(ABC):
    """
    Abstract base class for providing inference predictions to ActionSelectors.
    Encapsulates the difference between raw network inference and search-based (MCTS) policy.
    """

    @abstractmethod
    def get_inference(
        self, obs: torch.Tensor, info: Dict[str, Any], **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Computes or retrieves predictions for the given observation.
        """
        pass
