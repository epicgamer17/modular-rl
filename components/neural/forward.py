import torch
from typing import Optional, TYPE_CHECKING
from core import PipelineComponent
from core import Blackboard

if TYPE_CHECKING:
    from modules.agent_nets.base import BaseAgentNetwork
    from components.losses.infrastructure import ShapeValidator


class ForwardPassComponent(PipelineComponent):
    """
    Executes the main neural network forward pass.
    Reads inputs from Blackboard batch and writes outputs to Blackboard predictions.
    """
    def __init__(self, agent_network: 'BaseAgentNetwork', shape_validator: Optional['ShapeValidator'] = None):
        self.agent_network = agent_network
        self.shape_validator = shape_validator

    @property
    def reads(self) -> set[str]:
        return {"data.observations"}

    @property
    def writes(self) -> set[str]:
        # Declares standard predictions. 
        # ModularAgentNetwork merges its output dict into blackboard.predictions.
        return {"predictions.values", "predictions.policies"}

    def execute(self, blackboard: Blackboard) -> None:
        """
        Runs learner_inference on the data dictionary.
        Optimizes memory layout for throughput before the pass.
        """
        # OPTIMIZATION: Convert convolutional observations to channels_last for Tensor Cores
        # Only if the device is CUDA and it's a 4D tensor.
        for k, v in blackboard.data.items():
            if torch.is_tensor(v) and v.ndim == 4 and v.device.type == "cuda":
                blackboard.data[k] = v.to(memory_format=torch.channels_last)

        predictions = self.agent_network.learner_inference(
            blackboard.data, shape_validator=self.shape_validator
        )

        # Merge predictions safely into the blackboard
        blackboard.predictions.update(predictions)
