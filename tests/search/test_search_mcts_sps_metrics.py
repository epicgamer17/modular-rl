import pytest

pytestmark = pytest.mark.integration

import time
import torch
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Optional
from agents.action_selectors.selectors import CategoricalSelector
from agents.action_selectors.decorators import TemperatureSelector
from agents.action_selectors.policy_sources import SearchPolicySource
from agents.workers.actors import BaseActor
from replay_buffers.sequence import Sequence
from utils.schedule import ScheduleConfig
import numpy as np


class MockModularSearch:
    def __init__(self, num_sims):
        self.num_sims = num_sims
        self.config = SimpleNamespace(num_simulations=num_sims)

    def run(self, obs, info, agent_network, trajectory_action=None, exploration=True):
        time.sleep(0.01)
        policy = torch.ones(9) / 9
        return (
            0.0,
            policy,
            policy,
            0,
            {"mcts_simulations": self.num_sims, "mcts_search_time": 0.01},
        )

    def run_vectorized(self, obs, info, agent_network, trajectory_actions=None):
        time.sleep(0.01)
        B = obs.shape[0]
        sm_list = [
            {"mcts_simulations": self.num_sims, "mcts_search_time": 0.01}
            for _ in range(B)
        ]
        return (
            [0.0] * B,
            [torch.ones(9) / 9] * B,
            [torch.ones(9) / 9] * B,
            [0] * B,
            sm_list,
        )


class MockConfig:
    def __init__(self, num_sims):
        self.num_simulations = num_sims
        self.temperature_schedule = ScheduleConfig.stepwise(
            steps=[5], values=[1.0, 0.0]
        )
        self.compilation = SimpleNamespace(enabled=False, fullgraph=False)


class MockActor(BaseActor):
    def _detect_num_players(self) -> int:
        return 1

    def _get_player_id(self) -> Optional[str]:
        return None

    def _finalize_episode_info(self, sequence: Sequence) -> None:
        pass

    def _get_score(self, sequence: Sequence) -> float:
        return 0.0

    def _reset_env(self) -> Tuple[Any, Dict[str, Any]]:
        return torch.zeros(3, 3, 3), {}

    def _step_env(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        return torch.zeros(3, 3, 3), 0.0, True, False, {}


def test_sps_metrics_propagation():
    """Verifies that SearchPolicySource provides simulations and search time via actor."""
    torch.manual_seed(42)
    np.random.seed(42)

    num_sims = 100
    config = MockConfig(num_sims)
    search = MockModularSearch(num_sims)

    agent_net = torch.nn.Module()
    agent_net.input_shape = (3, 3, 3)
    agent_net.num_actions = 9
    agent_net.obs_inference = lambda x: None

    # 1. Test SearchPolicySource directly returns search_metadata
    policy_source = SearchPolicySource(search_engine=search, agent_network=agent_net, config=config)
    obs = torch.zeros(1, 3, 3, 3)
    result = policy_source.get_inference(obs, {}, to_play=0)
    assert "search_metadata" in result.extra_metadata
    assert result.extra_metadata["search_metadata"]["mcts_simulations"] == num_sims
    assert result.extra_metadata["search_metadata"]["mcts_search_time"] > 0

    # 2. Test actor aggregates search_metadata into episode stats
    inner = CategoricalSelector()
    selector = TemperatureSelector(inner_selector=inner, config=config)
    mock_buffer = torch.nn.Module()
    mock_buffer.store_aggregate = lambda x: None

    actor = MockActor(
        lambda: None,
        agent_net,
        selector,
        mock_buffer,
        num_players=1,
        config=config,
        name="test",
        policy_source=policy_source,
    )

    ep_stats = actor.play_sequence()
    assert ep_stats["mcts_simulations"] == num_sims
    assert ep_stats["mcts_search_time"] > 0
