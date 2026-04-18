import torch
import time
from typing import Any, Optional, Tuple, TYPE_CHECKING, Set, Dict

from core import PipelineComponent, Blackboard
from core.contracts import (
    Key,
    Observation,
    ValueEstimate,
    Policy,
    Reward,
    ToPlay,
    SemanticType,
    Metric,
)

if TYPE_CHECKING:
    from modules.agent_nets.base import BaseAgentNetwork


class NetworkInferenceComponent(PipelineComponent):
    """
    Performs neural network inference for an actor.
    Reads 'obs' from data, writes results to predictions.
    """

    def __init__(
        self,
        agent_network: "BaseAgentNetwork",
        input_shape: Tuple[int, ...],
    ):
        self.agent_network = agent_network
        self.input_shape = input_shape

        self._requires = {Key("data.obs", Observation)}
        self._provides = {
            Key("predictions.q_values", ValueEstimate): "optional",
            Key("predictions.logits", Policy): "optional",
            Key("predictions.probs", Policy): "optional",
            Key("predictions.value", ValueEstimate): "optional",
            Key("predictions.reward", Reward): "optional",
            Key("predictions.to_play", ToPlay): "optional",
            Key("predictions.extra_metadata", SemanticType): "optional",
        }

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures observation tensor exists."""
        from core.validation import assert_is_tensor

        obs = blackboard.data.get("obs")
        if obs is not None and not blackboard.data.get("dones", False):
            assert_is_tensor(obs, msg="for NetworkInferenceComponent")

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        obs = blackboard.data["obs"]
        done = blackboard.data.get("dones", False)
        if obs is None or done:
            return {}

        # Ensure batch dimension [1, ...] if single observation
        if obs.dim() == len(self.input_shape):
            obs = obs.unsqueeze(0)

        updates = {}
        with torch.inference_mode():
            output = self.agent_network.obs_inference(obs)

            # Route results through the updates dictionary
            q_values = getattr(output, "q_values", None)
            if q_values is not None:
                updates["predictions.q_values"] = q_values

            policy = getattr(output, "policy", None)
            if policy is not None:
                logits = getattr(policy, "logits", None)
                if logits is not None:
                    updates["predictions.logits"] = logits
                else:
                    probs = getattr(policy, "probs", None)
                    if probs is not None:
                        updates["predictions.probs"] = probs

            value = getattr(output, "value", None)
            if value is not None:
                if not isinstance(value, torch.Tensor):
                    value = torch.as_tensor([value], device=obs.device)
                updates["predictions.value"] = value

            reward = getattr(output, "reward", None)
            if reward is not None:
                if not isinstance(reward, torch.Tensor):
                    reward = torch.as_tensor([reward], device=obs.device)
                updates["predictions.reward"] = reward

            to_play = getattr(output, "to_play", None)
            if to_play is not None:
                if not isinstance(to_play, torch.Tensor):
                    to_play = torch.as_tensor(
                        [to_play], device=obs.device, dtype=torch.long
                    )
                updates["predictions.to_play"] = to_play

            extras = getattr(output, "extras", None) or {}
            if extras:
                updates["predictions.extra_metadata"] = extras

        return updates
