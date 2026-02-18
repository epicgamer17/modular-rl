# modules/world_model_interface.py (Revised)
from abc import ABC, abstractmethod
from attr import dataclass
from torch import Tensor
from typing import Tuple, Dict, Any
from torch import nn
import torch


@dataclass
class WorldModelOutput:
    """
    Represents the Agent's Hypothesis (Predictions).
    Shape: (B, Unroll+1, ...)
    """

    features: torch.Tensor = None
    reward: torch.Tensor = None
    to_play: torch.Tensor = None

    head_state: Any = None
    instant_reward: torch.Tensor = None

    afterstate_features: torch.Tensor = None
    chance: torch.Tensor = None


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
        pass

    @abstractmethod
    def recurrent_inference(
        self, hidden_state: Tensor, action: Tensor, recurrent_state: Any = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates the next hidden state, immediate reward
        from a hidden state and an action.

        Returns: (next_hidden_state, reward, to_play, next_recurrent_state)
        """
        pass

    @abstractmethod
    def unroll_sequence(
        self,
        actions,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Unrolls a sequence of actions from the current hidden state. Returns all network output seqeunces from this unrolling.
        """
        pass

    @abstractmethod
    def get_networks(self) -> Dict[str, nn.Module]:
        """
        Returns a dictionary of all trainable PyTorch networks within this model.
        Used by the main training loop for optimization and checkpointing.
        """
        pass
