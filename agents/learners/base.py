from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np
from torch.optim.sgd import SGD
from torch.optim.adam import Adam
import time
from torch.nn.utils import clip_grad_norm_
from replay_buffers.buffer_factories import create_muzero_buffer
from losses.losses import create_muzero_loss_pipeline
from modules.utils import get_lr_scheduler
from replay_buffers.utils import update_per_beta
from modules.world_models.inference_output import LearningOutput
from abc import ABC, abstractmethod


class BaseLearner(ABC):
    """
    BaseLearner handles the training logic, including buffer management,
    optimizer stepping, and loss computation.
    """

    def __init__(
        self,
        config,
        model,
        device,
        num_actions,
        observation_dimensions,
        observation_dtype,
    ):
        self.config = config
        self.model = model
        self.device = device
        self.num_actions = num_actions
        self.observation_dimensions = observation_dimensions
        self.observation_dtype = observation_dtype

    def _preprocess_observation(self, states: Any) -> torch.Tensor:
        """
        Converts states to torch tensors on the correct device.
        Adds batch dimension if input is a single observation.
        """
        if torch.is_tensor(states):
            if states.device == self.device and states.dtype == torch.float32:
                prepared_state = states
            else:
                prepared_state = states.to(self.device, dtype=torch.float32)
        else:
            np_states = np.array(states, copy=False)
            prepared_state = torch.tensor(
                np_states, dtype=torch.float32, device=self.device
            )

        if prepared_state.ndim == 0:
            prepared_state = prepared_state.unsqueeze(0)

        return prepared_state

    @abstractmethod
    def step(self, stats: StatTracker = None) -> Dict[str, Any]:
        """
        Performs a single training step.
        Returns a dictionary of loss statistics or None if buffer is too small.
        """
        raise NotImplementedError
