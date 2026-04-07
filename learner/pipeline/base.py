from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from learner.core import Blackboard

import torch

class PipelineComponent(ABC):
    """
    Base interface for all state and data flow components in the Learner pipeline.
    Components read from and write to the shared 'Blackboard'.
    """
    
    @abstractmethod
    def execute(self, blackboard: 'Blackboard') -> None:
        """
        Execute this component's logic, mutating the blackboard in place.
        """
        pass

class DeviceTransferComponent(PipelineComponent):
    """
    Stage 2 Preprocessing: Moves all tensors in the blackboard.batch to the target device.
    Ensures that subsequent 'Neural' components (Stage 3) receive tensors on the correct GPU.
    """
    def __init__(self, device: torch.device):
        self.device = device

    def execute(self, blackboard: 'Blackboard') -> None:
        for k, v in blackboard.batch.items():
            if torch.is_tensor(v):
                blackboard.batch[k] = v.to(self.device, non_blocking=True)
            elif isinstance(v, dict):
                # Handle nested dicts (e.g. for world model memory)
                for sub_k, sub_v in v.items():
                    if torch.is_tensor(sub_v):
                        v[sub_k] = sub_v.to(self.device, non_blocking=True)
