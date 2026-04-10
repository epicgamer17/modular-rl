import torch
from core import PipelineComponent
from core import Blackboard


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

    def execute(self, blackboard: Blackboard) -> None:
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

        blackboard.data["obs"] = obs_tensor
        blackboard.data["info"] = self.info
        blackboard.data["terminated"] = self.terminated
        blackboard.data["truncated"] = self.truncated
        blackboard.data["done"] = self.terminated or self.truncated


class GymStepComponent(PipelineComponent):
    """
    Steps the environment with the selected action and updates transition data.
    Directly modifies the GymObservationComponent to signal resets.
    """

    def __init__(self, env, obs_component: GymObservationComponent):
        self.env = env
        self.obs_component = obs_component

    def execute(self, blackboard: Blackboard) -> None:
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

        # Write transition data to blackboard
        blackboard.data["reward"] = float(reward)
        blackboard.data["done"] = done
        blackboard.data["terminated"] = terminated
        blackboard.data["truncated"] = truncated
        blackboard.data["next_obs"] = next_obs

        # Also write to meta for TelemetryComponent and tests
        blackboard.meta["reward"] = float(reward)
        blackboard.meta["done"] = done
        blackboard.meta["terminated"] = terminated
        blackboard.meta["truncated"] = truncated
        blackboard.meta["info"] = info

        # Update observation component for next tick
        self.obs_component.state = next_obs
        self.obs_component.info = info
        self.obs_component.done = done
        self.obs_component.terminated = terminated
        self.obs_component.truncated = truncated
