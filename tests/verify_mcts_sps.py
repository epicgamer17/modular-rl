import time
import torch
import unittest
from typing import Any, Dict, List, Tuple, Type, Optional
from agents.action_selectors.decorators import MCTSDecorator
from agents.action_selectors.selectors import CategoricalSelector
from agents.workers.actors import BaseActor
from replay_buffers.sequence import Sequence
from utils.schedule import ScheduleConfig


class MockSearch:
    def __init__(self, num_sims):
        self.num_sims = num_sims

    def run(self, obs, info, to_play, agent_network):
        # Simulate some search time
        time.sleep(0.01)
        # Return dummy values - uniform policy
        policy = torch.ones(9) / 9
        return 0.0, policy, policy, 0, {"sims": self.num_sims}


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


class TestMCTSSPS(unittest.TestCase):
    def test_sps_calculation_and_aggregation(self):
        num_sims = 100
        config = MockConfig(num_sims)
        search = MockSearch(num_sims)
        inner = CategoricalSelector()
        decorator = MCTSDecorator(inner, search, config)

        agent_net = torch.nn.Module()
        agent_net.input_shape = (3, 3, 3)
        # Device needs to be set for the selector
        agent_net.obs_inference = lambda x: None  # Minimal mock

        obs = torch.zeros(1, 3, 3, 3)

        # Test Decorator
        action, metadata = decorator.select_action(agent_net, obs)
        self.assertIn("search_metadata", metadata)
        self.assertIn("mcts_sps", metadata["search_metadata"])
        sps = metadata["search_metadata"]["mcts_sps"]
        print(f"MCTS SPS in metadata: {sps:.2f}")
        # Expect ~100 / 0.01 = 10000
        self.assertGreater(sps, 0)

        # Test Actor aggregation
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

        # 1. play_sequence
        seq = actor.play_sequence()
        self.assertIn("mcts_sps", seq)
        print(f"MCTS SPS in Sequence stats: {seq['mcts_sps']:.2f}")

        # 2. collect_transitions (Removed)


if __name__ == "__main__":
    unittest.main()
