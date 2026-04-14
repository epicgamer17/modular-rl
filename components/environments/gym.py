from typing import Set, Dict, Any
import torch
from core import PipelineComponent, Blackboard
from core.contracts import Key, Observation, SemanticType, Done, Reward, Action, Metric


class GymObservationComponent(PipelineComponent):
    """
    Reads the environment state and writes it to the Blackboard.
    Handles resets automatically when an episode is done.
    """

    def __init__(self, env):
        self.env = env
        # Initial reset
        result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            self.state, self.info = result
        else:
            self.state, self.info = result, {}
        self.terminated = False
        self.truncated = False
        self.done = False

    @property
    def requires(self) -> Set[Key]:
        return set()

    @property
    def provides(self) -> Dict[Key, str]:
        return {
            Key("data.obs", Observation): "new",
            Key("data.info", SemanticType): "new",
            Key("data.terminated", Done): "new",
            Key("data.truncated", Done): "new",
            Key("data.done", Done): "new",
        }

    def validate(self, blackboard: Blackboard) -> None:
        assert self.env is not None, (
            "GymObservationComponent: env is None"
        )

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        if self.state is None or self.done:
            result = self.env.reset()
            if isinstance(result, tuple) and len(result) == 2:
                self.state, self.info = result
            else:
                self.state, self.info = result, {}
            self.done = False
            self.terminated = False
            self.truncated = False

        # Convert to tensor and ensure batch dimension [1, ...]
        obs_tensor = torch.as_tensor(self.state, dtype=torch.float32)
        if obs_tensor.dim() > 0:  # Only if not a scalar
            # We don't have input_shape here easily, but we can assume we want [1, ...]
            # for a single environment loop.
            obs_tensor = obs_tensor.unsqueeze(0)

        return {
            "data.obs": obs_tensor,
            "data.info": self.info,
            "data.terminated": self.terminated,
            "data.truncated": self.truncated,
            "data.done": self.terminated or self.truncated
        }


class GymStepComponent(PipelineComponent):
    """
    Steps the environment with the selected action and updates transition data.
    Directly modifies the GymObservationComponent to signal resets.
    """

    def __init__(self, env, obs_component: GymObservationComponent):
        self.env = env
        self.obs_component = obs_component

    @property
    def requires(self) -> Set[Key]:
        return {Key("meta.action", Action)}

    @property
    def provides(self) -> Dict[Key, str]:
        return {
            Key("data.reward", Reward): "new",
            Key("data.done", Done): "overwrite",
            Key("data.next_obs", Observation): "new",
            Key("data.terminated", Done): "overwrite",
            Key("data.truncated", Done): "overwrite",
            Key("meta.reward", Metric): "new",
            Key("meta.done", Metric): "new",
            Key("meta.terminated", Metric): "new",
            Key("meta.truncated", Metric): "new",
            Key("meta.info", SemanticType): "new",
        }

    def validate(self, blackboard: Blackboard) -> None:
        assert "action" in blackboard.meta or "actions" in blackboard.predictions, (
            "GymStepComponent: no action found in blackboard.meta['action'] "
            "or blackboard.predictions['actions']"
        )

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        # Pull action from blackboard
        if "action" in blackboard.meta:
            action = blackboard.meta["action"]
        elif "actions" in blackboard.predictions:
            action = blackboard.predictions["actions"].item()
        else:
            raise KeyError(
                "No action found in blackboard.meta['action'] or blackboard.predictions['actions']"
            )

        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # Update observation component for next tick
        self.obs_component.state = next_obs
        self.obs_component.info = info
        self.obs_component.done = done
        self.obs_component.terminated = terminated
        self.obs_component.truncated = truncated

        # Write transition data to blackboard via return
        return {
            "data.reward": float(reward),
            "data.done": done,
            "data.terminated": terminated,
            "data.truncated": truncated,
            "data.next_obs": next_obs,
            "meta.reward": float(reward),
            "meta.done": done,
            "meta.terminated": terminated,
            "meta.truncated": truncated,
            "meta.info": info
        }
