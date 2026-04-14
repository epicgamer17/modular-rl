import torch
from core import PipelineComponent
from core import Blackboard
from core.contracts import Key, Observation, ValueEstimate, PolicyLogits, Reward, ToPlay
from typing import TYPE_CHECKING, Optional, Set

if TYPE_CHECKING:
    from modules.agent_nets.base import BaseAgentNetwork
    from components.losses.infrastructure import ShapeValidator


class ForwardPassComponent(PipelineComponent):
    """
    Component for the neural network forward pass.
    """

    def __init__(self, agent_network: "BaseAgentNetwork", shape_validator: Optional['ShapeValidator'] = None):
        self.agent_network = agent_network
        self.shape_validator = shape_validator

    @property
    def requires(self) -> Set[Key]:
        return {Key("data.observations", Observation)}

    @property
    def provides(self) -> Set[Key]:
        return {
            Key("predictions.values", ValueEstimate),
            Key("predictions.policies", PolicyLogits),
            Key("predictions.rewards", Reward),
            Key("predictions.to_plays", ToPlay),
        }

    def validate(self, blackboard: Blackboard) -> None:
        obs = blackboard.data.get("observations")
        assert obs is not None, "ForwardPassComponent requires 'observations' in blackboard.data"
        assert obs.ndim >= 2, f"Observation must have at least [B, T] or [B, *] dimensions, got {obs.shape}"

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
