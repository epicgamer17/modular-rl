import torch
import numpy as np
from core import PipelineComponent
from core import Blackboard


class PettingZooObservationComponent(PipelineComponent):
    """
    Reads the environment state for the currently active agent in a PettingZoo AEC environment.
    Writes to 'obs', 'info', and 'player_id' on the Blackboard.
    """

    def __init__(self, env):
        self.env = env
        self._initialized = False
        self.done = False

    def execute(self, blackboard: Blackboard) -> None:
        if not self._initialized or not self.env.agents:
            self.env.reset()
            self._initialized = True
            self.done = False

        try:
            agent = self.env.agent_selection
            obs, reward, termination, truncation, info = self.env.last()
        except (KeyError, AttributeError, ValueError) as e:
            # AEC environment error on terminal states
            agent = getattr(self.env, "agent_selection", "unknown")
            obs, reward, termination, truncation, info = None, 0.0, True, False, {}

        # Determine player index
        try:
            player_idx = self.env.possible_agents.index(agent)
        except (ValueError, AttributeError):
            player_idx = 0

        # Handle terminal state where obs might be None
        if obs is None:
            try:
                sp = self.env.observation_space(agent)
                if isinstance(sp, dict) and "observation" in sp:
                    shape = sp["observation"].shape
                else:
                    shape = sp.shape
                obs = np.zeros(shape, dtype=np.float32)
            except Exception:
                # Ultimate fallback - we might have to rely on previous obs shape
                pass

        # Convert to tensor and ensure batch dimension [1, ...]
        if obs is not None:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            if obs_tensor.dim() > 0:
                obs_tensor = obs_tensor.unsqueeze(0)
            blackboard.data["obs"] = obs_tensor
        else:
            blackboard.data["obs"] = None

        blackboard.data["info"] = info
        blackboard.data["terminated"] = termination
        blackboard.data["truncated"] = truncation
        blackboard.data["done"] = termination or truncation
        blackboard.data["reward"] = reward
        blackboard.data["player_id"] = player_idx
        blackboard.data["agent"] = agent


class PettingZooStepComponent(PipelineComponent):
    """
    Steps the PettingZoo environment with the selected action.
    """

    def __init__(self, env, obs_component: PettingZooObservationComponent):
        self.env = env
        self.obs_component = obs_component

    def execute(self, blackboard: Blackboard) -> None:
        action = blackboard.meta["action"]
        agent = blackboard.data["agent"]
        self.env.step(action)

        # In AEC, rewards are accumulated for all agents.
        # The reward for the agent that just acted is in self.env.rewards[agent]
        reward = self.env.rewards.get(agent, 0.0)

        # After step(), we can get the next state for the NEXT agent,
        # but for MuZero/single-actor transitions we often want the current agent's next state.
        # However, AEC is turn-based. We'll provide the reward and terminal state.
        try:
            obs, _, term, trunc, info = self.env.last()
        except (KeyError, AttributeError, ValueError):
            obs, term, trunc, info = None, True, False, {}

        self.obs_component.done = term or trunc

        blackboard.data["reward"] = float(reward)
        blackboard.data["done"] = self.obs_component.done
        blackboard.data["terminated"] = term
        blackboard.data["truncated"] = trunc
        blackboard.data["next_obs"] = obs
        blackboard.meta["next_info"] = info
