import pytest

pytestmark = pytest.mark.integration

import numpy as np

from agents.workers.tester import SelfPlayTest


class DummyTester:
    def select_test_action(self, state, info, env):
        return 0


class NamedPlayersEnv:
    """Minimal multi-agent env with non-canonical player ids."""

    def __init__(self):
        self.possible_agents = ["alice", "bob"]
        self.rewards = {"alice": 0.0, "bob": 0.0}
        self._done = False
        self._agent_idx = 0

    def reset(self):
        self._done = False
        self._agent_idx = 0
        self.rewards = {"alice": 0.0, "bob": 0.0}
        return np.zeros((4,)), {}

    def agent_iter(self):
        # Two plies is enough for SelfPlayTest.run
        for _ in range(2):
            yield self.possible_agents[self._agent_idx]
            self._agent_idx = (self._agent_idx + 1) % 2

    def last(self):
        return np.zeros((4,)), 0.0, self._done, False, {"legal_moves": [[0, 1]]}

    def step(self, action):
        # End immediately after one full round and assign asymmetric rewards.
        self._done = True
        self.rewards = {"alice": 1.0, "bob": -1.0}


def test_self_play_uses_possible_agents_order_for_player_scores():
    tester = DummyTester()
    env = NamedPlayersEnv()

    results = SelfPlayTest("self", num_trials=1).run(tester, env)

    assert "p0_score" in results and "p1_score" in results
    assert results["p0_score"] == 1.0
    assert results["p1_score"] == -1.0
