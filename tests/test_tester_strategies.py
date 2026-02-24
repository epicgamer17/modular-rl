import unittest
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional

from agents.workers.tester import (
    Tester,
    StandardGymTest,
    SelfPlayTest,
    VsAgentTest,
    NetworkAgent,
)
from agents.action_selectors.selectors import ArgmaxSelector
from configs.base import Config
from configs.games.game import GameConfig


class MockNetwork(torch.nn.Module):
    def __init__(self, num_actions=2):
        super().__init__()
        self.num_actions = num_actions
        self.param = torch.nn.Parameter(torch.zeros(1))

    def obs_inference(self, obs: torch.Tensor):
        class Output:
            def __init__(self, num_actions, batch_size):
                self.q_values = torch.zeros((batch_size, num_actions))
                self.q_values[:, 1] = 1.0  # action 1 is better

        return Output(self.num_actions, obs.shape[0])


class MockEnv:
    def __init__(self, is_multiplayer=False):
        self.is_multiplayer = is_multiplayer
        self.possible_agents = (
            ["player_0", "player_1"] if is_multiplayer else ["player_0"]
        )
        self.agents = self.possible_agents
        self.agent_selection = self.possible_agents[0]
        self.rewards = {a: 0.0 for a in self.possible_agents}
        self.step_count = 0
        self.max_steps = 5

    def reset(self, **kwargs):
        self.step_count = 0
        self.agent_selection = self.possible_agents[0]
        return np.zeros((4,)), {"legal_moves": [[0, 1]]}

    def last(self):
        return np.zeros((4,)), 0.0, False, False, {"legal_moves": [[0, 1]]}

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


class TestTesterStrategies(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.network = MockNetwork(num_actions=2).to(self.device).eval()
        self.selector = ArgmaxSelector()

        # Setup dummy config
        game_config = GameConfig(
            max_score=1.0,
            min_score=0.0,
            is_discrete=True,
            is_image=False,
            is_deterministic=True,
            has_legal_moves=True,
            perfect_information=True,
            multi_agent=False,
            num_players=1,
            num_actions=2,
            make_env=lambda: MockEnv(False),
        )
        self.config = Config(config_dict={"test_trials": 2}, game_config=game_config)

    def test_standard_gym_test(self):
        env_factory = lambda: MockEnv(is_multiplayer=False)
        tester = Tester(
            env_factory=env_factory,
            agent_network=self.network,
            action_selector=self.selector,
            num_players=1,
            config=self.config,
            device=self.device,
            name="test_tester",
        )
        strategy = StandardGymTest("std", num_trials=2)
        results = strategy.run(tester, tester.env)

        self.assertIn("score", results)
        self.assertEqual(results["score"], 5.0)

    def test_self_play_test(self):
        env_factory = lambda: MockEnv(is_multiplayer=True)
        self.config.game.multi_agent = True
        self.config.game.num_players = 2

        tester = Tester(
            env_factory=env_factory,
            agent_network=self.network,
            action_selector=self.selector,
            num_players=2,
            config=self.config,
            device=self.device,
            name="test_tester",
        )
        strategy = SelfPlayTest("self", num_trials=2)
        results = strategy.run(tester, tester.env)

        self.assertIn("score", results)
        self.assertEqual(results["p0_score"], 1.0)
        self.assertEqual(results["p1_score"], 1.0)

    def test_vs_agent_test(self):
        env_factory = lambda: MockEnv(is_multiplayer=True)

        class ConstantAgent:
            def predict(self, obs, info, **kwargs):
                return obs, info

            def select_actions(self, pred, info, **kwargs):
                return 0

        tester = Tester(
            env_factory=env_factory,
            agent_network=self.network,
            action_selector=self.selector,
            num_players=2,
            config=self.config,
            device=self.device,
            name="test_tester",
        )

        strategy = VsAgentTest(
            "vs", num_trials=2, opponent=ConstantAgent(), player_idx=0
        )
        results = strategy.run(tester, tester.env)

        self.assertIn("score", results)
        self.assertEqual(results["score"], 1.0)

    def test_network_agent(self):
        opp_network = MockNetwork(num_actions=2).to(self.device).eval()
        opp_selector = ArgmaxSelector()
        opponent = NetworkAgent("opp", opp_network, opp_selector, self.device)

        env_factory = lambda: MockEnv(is_multiplayer=True)
        tester = Tester(
            env_factory=env_factory,
            agent_network=self.network,
            action_selector=self.selector,
            num_players=2,
            config=self.config,
            device=self.device,
            name="test_tester",
        )

        strategy = VsAgentTest("vs_net", num_trials=1, opponent=opponent, player_idx=1)
        results = strategy.run(tester, tester.env)
        self.assertEqual(results["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
