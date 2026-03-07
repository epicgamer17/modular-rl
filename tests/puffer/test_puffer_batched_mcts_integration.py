import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

import torch
import numpy as np
from agents.workers.puffer_actor import GymPufferActor
from agents.action_selectors.decorators import MCTSDecorator
from agents.action_selectors.selectors import ArgmaxSelector
from replay_buffers.sequence import Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


from types import SimpleNamespace


@dataclass
class MockSearchConfig:
    num_simulations: int = 4
    max_search_depth: int = 2
    max_nodes: int = 10
    pb_c_init: float = 1.25
    pb_c_base: float = 19652.0
    discount_factor: float = 0.99
    use_dirichlet: bool = False
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    backprop_method: str = "average"
    policy_extraction: str = "visit_count"
    gumbel_cvisit: float = 50.0
    gumbel_cscale: float = 0.1
    num_codes: int = 1
    search_batch_size: int = 1
    virtual_loss: float = 1.0
    use_virtual_mean: bool = False
    bootstrap_method: str = "network_value"
    scoring_method: str = "ucb"
    use_value_prefix: bool = False
    internal_decision_modifier: str = "none"
    internal_chance_modifier: str = "none"
    num_puffer_workers: int = 1
    num_envs_per_worker: int = 2
    compilation: Any = field(default_factory=lambda: SimpleNamespace(enabled=False))
    game: Any = field(
        default_factory=lambda: SimpleNamespace(num_players=1, num_actions=2)
    )
    temperature_schedule: Any = field(
        default_factory=lambda: SimpleNamespace(
            type="stepwise", steps=[5], values=[1.0, 0.0], with_training_steps=False
        )
    )
    training_steps: int = 0
    known_bounds: Optional[Any] = None
    min_max_epsilon: float = 1e-6


class MockModularSearch:
    def __init__(self, config):
        self.config = config

    def run_vectorized(self, obs, info_list, to_play, agent_network):
        B = obs.shape[0]
        A = 2
        root_values = [0.5] * B
        exploratory_policies = [torch.ones(A) / A for _ in range(B)]
        target_policies = [torch.ones(A) / A for _ in range(B)]
        best_actions = [0] * B
        search_meta = [
            {"mcts_simulations": 4, "mcts_search_time": 0.01} for _ in range(B)
        ]
        return (
            root_values,
            exploratory_policies,
            target_policies,
            best_actions,
            search_meta,
        )


class MockNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = (4,)

    def obs_inference(self, obs):
        from types import SimpleNamespace

        B = obs.shape[0] if obs.dim() > 1 else 1
        return SimpleNamespace(
            policy=type("obj", (object,), {"logits": torch.zeros((B, 2))})(),
            value=torch.zeros(B),
        )


class MockBuffer:
    def store_aggregate(self, sequence):
        pass


def make_mock_env():
    import gymnasium as gym

    class TinyEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(0, 1, (4,))
            self.action_space = gym.spaces.Discrete(2)

        def reset(self, seed=None, options=None):
            return np.zeros(4, dtype=np.float32), {"legal_moves": [0, 1]}

        def step(self, action):
            return (
                np.zeros(4, dtype=np.float32),
                1.0,
                False,
                False,
                {"legal_moves": [0, 1]},
            )

    return TinyEnv()


def test_batched_mcts_puffer():
    config = MockSearchConfig()
    net = MockNetwork()
    search = MockModularSearch(config)
    inner_sel = ArgmaxSelector()
    mcts_sel = MCTSDecorator(inner_sel, search, config)
    buf = MockBuffer()

    actor = GymPufferActor(
        env_factory=make_mock_env,
        agent_network=net,
        action_selector=mcts_sel,
        replay_buffer=buf,
        num_players=1,
        config=config,
    )

    print("Verifying batched MCTS path in GymPufferActor...")
    # This triggers play_sequence -> select_action (batched) -> run_vectorized
    results = actor.play_sequence()
    print("Results:", results)
    assert results["mcts_simulations"] > 0
    print("Batched MCTS path verification SUCCESS.")


if __name__ == "__main__":
    test_batched_mcts_puffer()
