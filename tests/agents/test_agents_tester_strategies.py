import pytest
pytestmark = pytest.mark.integration

import copy
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
from tests.agents.conftest import (
    MockQValueNetwork as MockNetwork,
    MockMultiAgentEnv as MockEnv,
)


def _setup_tester_context(rainbow_cartpole_replay_config, make_cartpole_config):
    device = torch.device("cpu")
    network = MockNetwork(num_actions=2).to(device).eval()
    selector = ArgmaxSelector()

    game_config = make_cartpole_config(
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
    config = copy.deepcopy(rainbow_cartpole_replay_config)
    config.game = game_config
    config.test_trials = 2
    return device, network, selector, config


def test_standard_gym_test(rainbow_cartpole_replay_config, make_cartpole_config):
    device, network, selector, config = _setup_tester_context(
        rainbow_cartpole_replay_config, make_cartpole_config
    )
    env_factory = lambda: MockEnv(is_multiplayer=False)
    tester = Tester(
        env_factory=env_factory,
        agent_network=network,
        action_selector=selector,
        replay_buffer=None,
        num_players=1,
        config=config,
        device=device,
        name="test_tester",
    )
    strategy = StandardGymTest("std", num_trials=2)
    results = strategy.run(tester, tester.env)

    assert "score" in results
    assert results["score"] == 5.0


def test_self_play_test(rainbow_cartpole_replay_config, make_cartpole_config):
    device, network, selector, config = _setup_tester_context(
        rainbow_cartpole_replay_config, make_cartpole_config
    )
    env_factory = lambda: MockEnv(is_multiplayer=True)
    config.game.multi_agent = True
    config.game.num_players = 2

    tester = Tester(
        env_factory=env_factory,
        agent_network=network,
        action_selector=selector,
        replay_buffer=None,
        num_players=2,
        config=config,
        device=device,
        name="test_tester",
    )
    strategy = SelfPlayTest("self", num_trials=2)
    results = strategy.run(tester, tester.env)

    assert "score" in results
    assert results["p0_score"] == 1.0
    assert results["p1_score"] == 1.0


def test_vs_agent_test(rainbow_cartpole_replay_config, make_cartpole_config):
    device, network, selector, config = _setup_tester_context(
        rainbow_cartpole_replay_config, make_cartpole_config
    )
    env_factory = lambda: MockEnv(is_multiplayer=True)

    class ConstantAgent:
        def predict(self, obs, info, **kwargs):
            return obs, info

        def select_actions(self, pred, info, **kwargs):
            return 0

    tester = Tester(
        env_factory=env_factory,
        agent_network=network,
        action_selector=selector,
        replay_buffer=None,
        num_players=2,
        config=config,
        device=device,
        name="test_tester",
    )

    strategy = VsAgentTest("vs", num_trials=2, opponent=ConstantAgent(), player_idx=0)
    results = strategy.run(tester, tester.env)

    assert "score" in results
    assert results["score"] == 1.0


def test_network_agent(rainbow_cartpole_replay_config, make_cartpole_config):
    device, network, selector, config = _setup_tester_context(
        rainbow_cartpole_replay_config, make_cartpole_config
    )
    opp_network = MockNetwork(num_actions=2).to(device).eval()
    opp_selector = ArgmaxSelector()
    opponent = NetworkAgent("opp", opp_network, opp_selector, device)

    env_factory = lambda: MockEnv(is_multiplayer=True)
    tester = Tester(
        env_factory=env_factory,
        agent_network=network,
        action_selector=selector,
        replay_buffer=None,
        num_players=2,
        config=config,
        device=device,
        name="test_tester",
    )

    strategy = VsAgentTest("vs_net", num_trials=1, opponent=opponent, player_idx=1)
    results = strategy.run(tester, tester.env)
    assert results["score"] == 1.0
