"""Shared test doubles for agent tests.

Provides unified mock classes for networks, environments, search,
action selectors, and buffers used across agent test files.
"""

import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace
from modules.models.inference_output import InferenceOutput


# ---------------------------------------------------------------------------
# Mock Networks
# ---------------------------------------------------------------------------


class MockQValueNetwork(torch.nn.Module):
    """Network that returns q_values via obs_inference.

    Used by tester tests (idling, strategies) and action selector tests.
    Deterministic: action 1 always has higher Q-value.
    """

    def __init__(self, num_actions: int = 2):
        super().__init__()
        self.num_actions = num_actions
        self.param = torch.nn.Parameter(torch.zeros(1))

    def obs_inference(self, obs: torch.Tensor):
        """Returns an output with q_values where action 1 is best."""
        batch_size = obs.shape[0]
        q_values = torch.zeros((batch_size, self.num_actions))
        q_values[:, 1] = 1.0  # action 1 is better
        return SimpleNamespace(q_values=q_values)


class MockInferenceNetwork(nn.Module):
    """Network returning InferenceOutput from obs_inference.

    Used by action selector behavior tests.
    """

    def obs_inference(self, obs):
        return InferenceOutput(
            value=torch.tensor([0.0]),
            policy=None,
            reward=None,
            to_play=None,
            network_state=None,
            q_values=None,
        )


class MockConfigurableNetwork(torch.nn.Module):
    """Network that returns a pre-configured output value.

    Used by action selector unit tests.
    """

    def __init__(self, output_val=None):
        super().__init__()
        self.output_val = output_val
        self.input_shape = (4,)

    def obs_inference(self, obs):
        return self.output_val


class MockMCTSNetwork(torch.nn.Module):
    """Minimal network for MCTS metadata/integration tests.

    Returns None from obs_inference (search overrides inference).
    """

    def __init__(self, num_actions: int = 2):
        super().__init__()
        self.input_shape = (4,)
        self.num_actions = num_actions

    def obs_inference(self, obs):
        return None


# ---------------------------------------------------------------------------
# Mock Environments
# ---------------------------------------------------------------------------


class MockGymEnv:
    """Simple gym-compatible environment for single-agent tests.

    Configurable max_steps and obs shape.
    """

    def __init__(self, max_steps: int = 3, obs_shape: tuple = (1,)):
        self.step_count = 0
        self.max_steps = max_steps
        self.obs_shape = obs_shape

    def reset(self, **kwargs):
        self.step_count = 0
        obs = [0.0] * self.obs_shape[0] if len(self.obs_shape) == 1 else np.zeros(self.obs_shape)
        return obs, {"legal_moves": [[0, 1]]}

    def step(self, action):
        self.step_count += 1
        obs = [0.0] * self.obs_shape[0] if len(self.obs_shape) == 1 else np.zeros(self.obs_shape)
        return (
            obs,
            1.0,
            self.step_count >= self.max_steps,
            False,
            {"legal_moves": [[0, 1]]},
        )

    def close(self):
        pass


class MockMultiAgentEnv:
    """Multi-agent environment for tester strategy tests.

    Supports both single-player and multiplayer modes with PettingZoo-style API.
    """

    def __init__(self, is_multiplayer: bool = False, max_steps: int = 5):
        self.is_multiplayer = is_multiplayer
        self.possible_agents = (
            ["player_0", "player_1"] if is_multiplayer else ["player_0"]
        )
        self.agents = self.possible_agents
        self.agent_selection = self.possible_agents[0]
        self.rewards = {a: 0.0 for a in self.possible_agents}
        self.step_count = 0
        self.max_steps = max_steps

    def reset(self, **kwargs):
        self.step_count = 0
        self.agent_selection = self.possible_agents[0]
        return np.zeros((4,)), {"legal_moves": [[0, 1]]}

    def last(self):
        terminated = self.step_count >= self.max_steps
        return np.zeros((4,)), 0.0, terminated, False, {"legal_moves": [[0, 1]]}

    def agent_iter(self):
        while True:
            yield self.agent_selection
            if self.step_count >= self.max_steps:
                break

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= self.max_steps
        if self.is_multiplayer:
            idx = (self.possible_agents.index(self.agent_selection) + 1) % len(
                self.possible_agents
            )
            self.agent_selection = self.possible_agents[idx]
            self.rewards = {a: 1.0 if done else 0.0 for a in self.possible_agents}
        return np.zeros((4,)), 1.0, done, False, {"legal_moves": [[0, 1]]}

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Mock Search
# ---------------------------------------------------------------------------


class MockSearch:
    """Mock MCTS search for action selector and metadata tests.

    Returns deterministic search results.
    """

    def __init__(self, config=None):
        self.config = config

    def run(self, obs, info, agent_network, trajectory_action=None, exploration=True):
        return (
            0.5,
            torch.tensor([0.1, 0.9]),
            torch.tensor([0.0, 1.0]),
            1,
            {"mcts_simulations": 10},
        )

    def run_vectorized(self, obs, infos, agent_network, trajectory_actions=None):
        B = obs.shape[0]
        return (
            [0.5] * B,
            [torch.tensor([0.1, 0.9])] * B,
            [torch.tensor([0.0, 1.0])] * B,
            [1] * B,
            [{}] * B,
        )


class MockMetadataSearch:
    """Mock search that returns specific metadata values for regression tests."""

    def run(self, obs, info, agent_network, trajectory_action=None, exploration=True):
        return (
            0.75,
            torch.tensor([0.2, 0.8]),
            torch.tensor([0.1, 0.9]),
            1,
            {"mcts_simulations": 123, "mcts_search_time": 0.456},
        )


# ---------------------------------------------------------------------------
# Mock Policy Distribution
# ---------------------------------------------------------------------------


class MockPolicyDist:
    """Simple policy distribution stub for action selector tests."""

    def __init__(self, probs: torch.Tensor):
        self.probs = probs

    def sample(self):
        return torch.multinomial(self.probs, 1).squeeze(-1)

    def log_prob(self, action):
        return torch.log(self.probs.gather(-1, action.unsqueeze(-1)).squeeze(-1))


class MockInferenceOutput:
    """Lightweight inference output stub with optional fields."""

    def __init__(self, value=None, policy=None, logits=None):
        self.value = value
        self.policy = policy
        self.logits = logits
        self.network_state = None


# ---------------------------------------------------------------------------
# Mock Buffer
# ---------------------------------------------------------------------------


class MockBuffer:
    """Simple replay buffer mock that records stored sequences."""

    def __init__(self):
        self.stored_sequences = []

    def store_aggregate(self, sequence):
        self.stored_sequences.append(sequence)
