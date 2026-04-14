import torch
import time
from typing import Any, Optional, Tuple, TYPE_CHECKING, Set
from core import PipelineComponent, Blackboard
from core.contracts import Key, Observation, ValueEstimate, PolicyLogits, Reward, ToPlay, SemanticType

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

    @property
    def requires(self) -> Set[Key]:
        return {Key("data.obs", Observation)}

    @property
    def provides(self) -> Set[Key]:
        return {
            Key("predictions.q_values", ValueEstimate),
            Key("predictions.logits", PolicyLogits),
            Key("predictions.value", ValueEstimate),
            Key("predictions.reward", Reward),
            Key("predictions.to_play", ToPlay),
            Key("predictions.extra_metadata", SemanticType),
        }

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> None:
        obs = blackboard.data["obs"]
        done = blackboard.data.get("done", False)
        if obs is None or done:
            return

        # Ensure batch dimension [1, ...] if single observation
        if obs.dim() == len(self.input_shape):
            obs = obs.unsqueeze(0)

        with torch.inference_mode():
            output = self.agent_network.obs_inference(obs)

            # Write results directly to blackboard
            q_values = getattr(output, "q_values", None)
            if q_values is not None:
                blackboard.predictions["q_values"] = q_values

            policy = getattr(output, "policy", None)
            if policy is not None:
                logits = getattr(policy, "logits", None)
                if logits is not None:
                    blackboard.predictions["logits"] = logits
                else:
                    probs = getattr(policy, "probs", None)
                    if probs is not None:
                        blackboard.predictions["probs"] = probs

            value = getattr(output, "value", None)
            if value is not None:
                if not isinstance(value, torch.Tensor):
                    value = torch.as_tensor([value], device=obs.device)
                blackboard.predictions["value"] = value

            reward = getattr(output, "reward", None)
            if reward is not None:
                if not isinstance(reward, torch.Tensor):
                    reward = torch.as_tensor([reward], device=obs.device)
                blackboard.predictions["reward"] = reward

            to_play = getattr(output, "to_play", None)
            if to_play is not None:
                if not isinstance(to_play, torch.Tensor):
                    to_play = torch.as_tensor(
                        [to_play], device=obs.device, dtype=torch.long
                    )
                blackboard.predictions["to_play"] = to_play

            extras = getattr(output, "extras", None) or {}
            if extras:
                blackboard.predictions["extra_metadata"] = extras
