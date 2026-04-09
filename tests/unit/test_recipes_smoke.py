import torch
import pytest
import gymnasium as gym
from envs.factories.tictactoe import tictactoe_factory

from registries.muzero import (
    make_muzero_network,
    make_muzero_search_engine,
    make_muzero_replay_buffer,
    make_muzero_learner,
)
from registries.ppo import (
    make_ppo_network,
    make_ppo_replay_buffer,
    make_ppo_learner,
)
from registries.rainbow import (
    make_rainbow_network,
    make_rainbow_replay_buffer,
    make_rainbow_learner,
)
from registries.dqn import (
    make_dqn_network,
    make_dqn_replay_buffer,
    make_dqn_learner,
)

pytestmark = pytest.mark.unit

def test_muzero_recipe_instantiation():
    """Verify that the MuZero recipe builds all components correctly."""
    DEVICE = torch.device("cpu")
    env = tictactoe_factory()
    num_actions = env.action_space("player_1").n
    obs_dim = env.observation_space("player_1").shape

    # 1. Network
    agent_network = make_muzero_network(
        obs_dim=obs_dim,
        num_actions=num_actions,
        device=DEVICE
    )
    assert agent_network is not None
    assert "world_model" in agent_network.components

    # 2. Search
    search = make_muzero_search_engine(
        num_actions=num_actions,
        device=DEVICE
    )
    assert search is not None

    # 3. Buffer
    buffer = make_muzero_replay_buffer(
        obs_dim=obs_dim,
        num_actions=num_actions
    )
    assert buffer is not None

    # 4. Learner
    optimizer = torch.optim.Adam(agent_network.parameters(), lr=1e-3)
    learner = make_muzero_learner(
        agent_network=agent_network,
        optimizer=optimizer,
        batch_size=8,
        unroll_steps=5,
        num_actions=num_actions,
        device=DEVICE
    )
    assert learner is not None
    env.close()

def test_ppo_recipe_instantiation():
    """Verify that the PPO recipe builds all components correctly."""
    DEVICE = torch.device("cpu")
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape
    num_actions = env.action_space.n

    # 1. Network
    agent_network = make_ppo_network(
        obs_dim=obs_dim,
        num_actions=num_actions,
        device=DEVICE
    )
    assert agent_network is not None

    # 2. Buffer
    buffer = make_ppo_replay_buffer(
        obs_dim=obs_dim,
        num_actions=num_actions
    )
    assert buffer is not None

    # 3. Learner
    optimizer = torch.optim.Adam(agent_network.parameters(), lr=2.5e-4)
    learner = make_ppo_learner(
        agent_network=agent_network,
        optimizer=optimizer,
        minibatch_size=128,
        num_actions=num_actions,
        device=DEVICE
    )
    assert learner is not None
    env.close()

def test_rainbow_recipe_instantiation():
    """Verify that the Rainbow recipe builds all components correctly."""
    DEVICE = torch.device("cpu")
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape
    num_actions = env.action_space.n

    # 1. Network
    agent_network = make_rainbow_network(
        obs_dim=obs_dim,
        num_actions=num_actions,
        device=DEVICE
    )
    target_network = make_rainbow_network(
        obs_dim=obs_dim,
        num_actions=num_actions,
        device=DEVICE
    )

    # 2. Buffer
    buffer = make_rainbow_replay_buffer(
        obs_dim=obs_dim,
        num_actions=num_actions
    )
    assert buffer is not None

    # 3. Learner
    optimizer = torch.optim.Adam(agent_network.parameters(), lr=1e-3)
    learner = make_rainbow_learner(
        agent_network=agent_network,
        target_network=target_network,
        optimizer=optimizer,
        replay_buffer=buffer,
        device=DEVICE
    )
    assert learner is not None
    env.close()

def test_dqn_recipe_instantiation():
    """Verify that the DQN recipe builds all components correctly."""
    DEVICE = torch.device("cpu")
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape
    num_actions = env.action_space.n

    # 1. Network
    agent_network = make_dqn_network(
        obs_dim=obs_dim,
        num_actions=num_actions,
        device=DEVICE
    )
    target_network = make_dqn_network(
        obs_dim=obs_dim,
        num_actions=num_actions,
        device=DEVICE
    )

    # 2. Buffer
    buffer = make_dqn_replay_buffer(
        obs_dim=obs_dim,
        num_actions=num_actions
    )
    assert buffer is not None

    # 3. Learner
    optimizer = torch.optim.Adam(agent_network.parameters(), lr=1e-3)
    learner = make_dqn_learner(
        agent_network=agent_network,
        target_network=target_network,
        optimizer=optimizer,
        device=DEVICE
    )
    assert learner is not None
    env.close()
