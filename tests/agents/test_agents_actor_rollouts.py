import pytest
import torch
import numpy as np
import os

# Disable gymnasium plugins that cause circular imports in 0.29.1
os.environ["GYM_DISABLE_PLUGINS"] = "1"
os.environ["ALE_PY_DISABLE_REGISTRATION"] = "1"

import random
import gymnasium as gym

from agents.workers.actors import GymActor
from agents.workers.tester import Tester as TesterWorker, StandardGymTest
from modules.agent_nets.agent_network import AgentNetwork
from agents.action_selectors.factory import SelectorFactory
from replay_buffers.buffer_factories import create_dqn_buffer

pytestmark = pytest.mark.integration


def test_gym_actor_rollout(make_cartpole_config, make_rainbow_config_dict):
    """
    Test real GymActor rollout for a few steps.
    """
    # 1. Setup deterministic seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 2. Setup real, lightweight configuration
    # We use num_envs=1 to keep it simple and avoid PufferActor complexity for this specific test
    # unless also specifically requested, but the prompt mentioned PufferActor or base Actor.
    # Base GymActor is easier to verify for step() logic.
    config = make_cartpole_config(num_envs_per_worker=1)

    # 3. Instantiate real network and selector
    # For Rainbow CartPole, we need a RainbowConfig structure
    from configs.agents.rainbow_dqn import RainbowConfig

    rainbow_config_dict = make_rainbow_config_dict(
        action_selector={
            "base": {"type": "epsilon_greedy", "kwargs": {"epsilon": 0.05}}
        },
        executor_type="local",
        n_step=3,
        discount_factor=0.9,
        per_alpha=0.6,
        per_beta_schedule={"type": "constant", "initial": 0.4},
        per_epsilon=1e-6,
        per_use_batch_weights=True,
        per_use_initial_max_priority=True,
        minibatch_size=32,
        replay_buffer_size=100,
        min_replay_buffer_size=1,
    )
    rainbow_config = RainbowConfig(rainbow_config_dict, config)

    cartpole_obs_shape = (4,)  # CartPole observation: [pos, vel, angle, angular_vel]
    network = AgentNetwork(
        config=rainbow_config,
        input_shape=cartpole_obs_shape,
        num_actions=config.num_actions,
    )
    selector = SelectorFactory.create(rainbow_config.action_selector)

    # 4. Create a dummy buffer (GymActor needs it for play_sequence, but we will call step() directly)
    buffer = create_dqn_buffer(
        observation_dimensions=cartpole_obs_shape,
        max_size=10,
        num_actions=config.num_actions,
        batch_size=1,
    )

    # 5. Instantiate Actor
    device = torch.device("cpu")
    actor = GymActor(
        env_factory=config.make_env,
        agent_network=network,
        action_selector=selector,
        replay_buffer=buffer,
        config=config,
        device=device,
    )

    # 6. Call step() for a few environment steps
    actor.reset()
    for _ in range(3):
        transition = actor.step()

        # 7. Assertions
        assert "state" in transition
        assert "action" in transition
        assert "reward" in transition
        assert "next_state" in transition
        assert "done" in transition

        # Check shapes
        assert transition["state"].shape == cartpole_obs_shape
        assert isinstance(transition["action"], (int, np.integer))
        assert isinstance(transition["reward"], float)
        assert transition["next_state"].shape == cartpole_obs_shape
        assert isinstance(transition["done"], bool)


def test_tester_worker_rollout(make_cartpole_config, make_rainbow_config_dict):
    """
    Test real Tester worker with StandardGymTest.
    """
    # 1. Setup deterministic seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 2. Setup real, lightweight configuration
    config = make_cartpole_config(num_envs_per_worker=1, test_trials=2)

    from configs.agents.rainbow_dqn import RainbowConfig

    rainbow_config_dict = make_rainbow_config_dict(
        action_selector={
            "base": {"type": "epsilon_greedy", "kwargs": {"epsilon": 0.0}}
        },  # Greedy for testing
        executor_type="local",
        n_step=3,
        discount_factor=0.9,
        per_alpha=0.6,
        per_beta_schedule={"type": "constant", "initial": 0.4},
        per_epsilon=1e-6,
        per_use_batch_weights=True,
        per_use_initial_max_priority=True,
        minibatch_size=32,
        replay_buffer_size=100,
        min_replay_buffer_size=1,
    )
    rainbow_config = RainbowConfig(rainbow_config_dict, config)

    network = AgentNetwork(
        config=rainbow_config,
        input_shape=(4,),  # CartPole observation: [pos, vel, angle, angular_vel]
        num_actions=config.num_actions,
    )
    selector = SelectorFactory.create(rainbow_config.action_selector)

    # 3. Instantiate Tester
    device = torch.device("cpu")
    test_types = [StandardGymTest("standard_test", num_trials=2)]

    tester = TesterWorker(
        env_factory=config.make_env,
        agent_network=network,
        action_selector=selector,
        replay_buffer=None,
        num_players=1,
        config=config,
        device=device,
        name="test_tester",
        test_types=test_types,
    )

    # 4. Run tests (which performs rollouts internally)
    results = tester.run_tests()

    # 5. Assertions
    assert "standard_test" in results
    test_res = results["standard_test"]
    assert "score" in test_res
    assert "min_score" in test_res
    assert "max_score" in test_res
    assert "duration" in test_res

    # Verify it actually ran trials
    assert isinstance(test_res["score"], float)
