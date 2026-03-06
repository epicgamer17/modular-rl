import pytest
pytestmark = pytest.mark.integration

import time
import torch
from typing import Any, Dict, List, Tuple, Type, Optional
from agents.action_selectors.decorators import MCTSDecorator
from agents.action_selectors.selectors import CategoricalSelector
from agents.workers.actors import BaseActor
from replay_buffers.sequence import Sequence
from utils.schedule import ScheduleConfig


class MockSearch:
    def __init__(self, num_sims):
        self.num_sims = num_sims

    def run(self, obs, info, to_play, agent_network, exploration=True):
        # Simulate some search time
        time.sleep(0.01)
        # Return dummy values - uniform policy
        policy = torch.ones(9) / 9
        return 0.0, policy, policy, 0, {"sims": self.num_sims}

    def run_vectorized(self, obs, info, to_play, agent_network):
        # Simulate some search time
        time.sleep(0.01)
        B = obs.shape[0]
        policy = torch.ones(B, 9) / 9
        # In the real system, sm['sims'] is per-tree, but MCTSDecorator
        # overwrites it with total_sims.
        sm_list = [{"sims": self.num_sims} for _ in range(B)]
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
    """Verifies that MCTSDecorator provides simulations and search time."""
    num_sims = 100
    config = MockConfig(num_sims)
    search = MockSearch(num_sims)
    inner = CategoricalSelector()
    decorator = MCTSDecorator(inner, search, config)

    agent_net = torch.nn.Module()
    agent_net.input_shape = (3, 3, 3)
    agent_net.obs_inference = lambda x: None

    obs = torch.zeros(1, 3, 3, 3)

    # 1. Test Single Action Path
    _, metadata = decorator.select_action(agent_net, obs)
    assert "search_metadata" in metadata
    assert metadata["search_metadata"]["mcts_simulations"] == num_sims
    assert metadata["search_metadata"]["mcts_search_time"] > 0

    # 2. Test Vectorized Path
    obs_vec = torch.zeros(4, 3, 3, 3)
    info_vec = {"legal_moves": [list(range(9))] * 4, "player": [0] * 4}
    _, metadata_vec = decorator.select_action(agent_net, obs_vec, info=info_vec)
    # 4 trees * 100 sims = 400
    assert metadata_vec["search_metadata"]["mcts_simulations"] == 400
    assert metadata_vec["search_metadata"]["mcts_search_time"] > 0

    # 3. Test Actor aggregation
    mock_buffer = torch.nn.Module()
    mock_buffer.store_aggregate = lambda x: None
    actor = MockActor(
        lambda: None,
        agent_net,
        decorator,
        mock_buffer,
        num_players=1,
        config=config,
        name="test",
    )

    ep_stats = actor.play_sequence()
    assert ep_stats["mcts_simulations"] == num_sims
    assert ep_stats["mcts_search_time"] > 0
