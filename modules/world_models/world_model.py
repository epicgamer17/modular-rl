# modules/world_model_interface.py (Revised)
from abc import ABC, abstractmethod
from attr import dataclass
from torch import Tensor
from typing import Tuple, Dict, Any
from torch import nn
import torch
from modules.world_models.inference_output import WorldModelOutput


class WorldModelInterface(ABC):
    """
    Abstract Interface for any model/simulator used within the MuZero training loop.
    All implementations (MuZero, Dreamer, PerfectSim) must adhere to these methods.
    """

    @abstractmethod
    def initial_inference(self, observation: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Calculates the initial hidden state from an observation.

        Returns: (hidden_state)
        """
        pass  # pragma: no cover

    @abstractmethod
    def recurrent_inference(
        self, hidden_state: Tensor, action: Tensor, recurrent_state: Any = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates the next hidden state, immediate reward
        from a hidden state and an action.

        Returns: (next_hidden_state, reward, to_play, next_recurrent_state)
        """
        pass  # pragma: no cover

    @abstractmethod
    def unroll_physics(
        self,
        actions,
    ) -> "PhysicsOutput":
        """
        Unrolls a sequence of actions from the current hidden state. Returns all network output seqeunces from this unrolling.
        """
        pass  # pragma: no cover

    @abstractmethod
    def get_networks(self) -> Dict[str, nn.Module]:
        """
        Returns a dictionary of all trainable PyTorch networks within this model.
        Used by the main training loop for optimization and checkpointing.
        """
        pass  # pragma: no cover
