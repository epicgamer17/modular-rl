import torch
from core import PipelineComponent
from core import Blackboard
from core.contracts import (
    Key,
    ShapeContract,
    Observation,
    ValueEstimate,
    Policy,
    Reward,
    ToPlay,
    Metric,
    Scalar,
    Logits,
    Probs,
    LogProbs,
    Categorical,
    Quantile,
)
from typing import TYPE_CHECKING, Optional, Set, Dict, Any, Union

if TYPE_CHECKING:
    from modules.agent_nets.base import BaseAgentNetwork


class ForwardPassComponent(PipelineComponent):
    """
    Component for the neural network forward pass.
    """

    def __init__(
        self,
        agent_network: "BaseAgentNetwork",
        obs_key: str = "observations",
    ):
        self.agent_network = agent_network
        self._obs_key = obs_key

        # Deterministic contracts computed at initialization via network introspection
        self._requires = {Key(f"data.{self._obs_key}", Observation)}
        self._provides = {}

        contract = self.agent_network.get_learner_contract()
        for key, sem_type in contract.items():
            self._provides[
                Key(
                    f"predictions.{key}",
                    sem_type,
                    shape=ShapeContract(
                        has_time=True, symbolic=("B", "T"), dtype="float32", ndim=2
                    ),
                )
            ] = "new"

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures observations exist and have at least 2 dims."""
        from core.validation import assert_in_blackboard, assert_is_tensor

        assert_in_blackboard(blackboard, f"data.{self._obs_key}")
        obs = blackboard.data.get(self._obs_key)
        assert_is_tensor(obs, msg=f"for ForwardPassComponent ({self._obs_key})")
        assert (
            obs.ndim >= 2
        ), f"Observation must have at least [B, T] or [B, *] dimensions, got {obs.shape}"

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        """
        Runs learner_inference on the data dictionary.
        Optimizes memory layout for throughput before the pass.
        """
        updates = {}
        inference_data = dict(blackboard.data)

        # OPTIMIZATION: Convert convolutional observations to channels_last for Tensor Cores
        # Only if the device is CUDA and it's a 4D tensor.
        for k, v in blackboard.data.items():
            if torch.is_tensor(v) and v.ndim == 4 and v.device.type == "cuda":
                # Returns optimized tensor for central application
                optimized_v = v.to(memory_format=torch.channels_last)
                updates[f"data.{k}"] = optimized_v
                inference_data[k] = optimized_v

        predictions_dict = self.agent_network.learner_inference(inference_data)

        for k, v in predictions_dict.items():
            updates[f"predictions.{k}"] = v

        return updates
