import copy

import numpy as np
import pytest

pytestmark = pytest.mark.integration

import gymnasium as gym
import torch

from configs.agents.muzero import MuZeroConfig
from modules.agent_nets.modular import ModularAgentNetwork
from modules.heads.reward import ValuePrefixRewardHead
from modules.world_models.inference_output import MuZeroNetworkState
from modules.world_models.muzero_world_model import MuzeroWorldModel


class MockEnv:
    def __init__(self):
        self.action_space = gym.spaces.Discrete(5)

    def close(self):
        pass


def _build_value_prefix_config(
    rainbow_cartpole_replay_config, make_cartpole_config, **overrides
):
    game = make_cartpole_config(
        max_score=100,
        min_score=0,
        is_discrete=True,
        is_image=False,
        is_deterministic=False,
        has_legal_moves=False,
        perfect_information=True,
        multi_agent=False,
        num_players=1,
        num_actions=5,
        make_env=MockEnv,
    )

    config_dict = copy.deepcopy(rainbow_cartpole_replay_config.config_dict)
    config_dict.update(
        {
            "world_model_cls": MuzeroWorldModel,
            "stochastic": True,
            "num_chance": 10,
            "use_value_prefix": True,
            "lstm_hidden_size": 16,
            "lstm_horizon_len": 5,
            "prediction_backbone": {"type": "identity"},
            "representation_backbone": {"type": "identity"},
            "dynamics_backbone": {"type": "identity"},
            "afterstate_dynamics_backbone": {"type": "identity"},
            "chance_encoder_backbone": {"type": "identity"},
            "value_head": {"output_strategy": {"type": "scalar"}},
            "reward_head": {"output_strategy": {"type": "scalar"}},
        }
    )
    config_dict.update(overrides)
    return MuZeroConfig(config_dict, game)


def test_use_value_prefix_network(
    rainbow_cartpole_replay_config, make_cartpole_config
):
    torch.manual_seed(42)
    np.random.seed(42)

    print("Testing ValuePrefixRewardHead integration...")

    print("Initializing MuZero Config with use_value_prefix=True...")
    config = _build_value_prefix_config(
        rainbow_cartpole_replay_config, make_cartpole_config
    )

    print("Initializing Network...")
    input_shape = (4,)
    net = ModularAgentNetwork(config, input_shape, config.game.num_actions)

    reward_head = net.components["world_model"].reward_head
    print(f"Reward Head Type: {type(reward_head)}")
    assert isinstance(
        reward_head, ValuePrefixRewardHead
    ), "Reward head must be ValuePrefixRewardHead"

    assert hasattr(reward_head, "lstm"), "ValuePrefixRewardHead must have LSTM"
    assert (
        reward_head.lstm.hidden_size == 16
    ), f"LSTM hidden size should be 16, got {reward_head.lstm.hidden_size}"

    print("Testing Hidden State Inference...")
    batch_size = 2
    hidden_state = torch.randn(batch_size, 4)
    h_0 = torch.zeros(1, batch_size, 16)
    c_0 = torch.zeros(1, batch_size, 16)

    network_state = MuZeroNetworkState(
        dynamics=hidden_state, wm_memory={"reward_hidden": (h_0, c_0)}
    )

    action_idx = torch.randint(0, config.num_chance, (batch_size,))
    action = torch.nn.functional.one_hot(
        action_idx, num_classes=config.num_chance
    ).float()

    output = net.hidden_state_inference(network_state, action)

    reward = output.reward
    next_network_state = output.network_state

    h_1, c_1 = next_network_state.wm_memory["reward_hidden"]

    print("Step 1 done.")
    print(f"Reward shape: {reward.shape}")
    print(f"Reward Hidden h_1 shape: {h_1.shape}")

    expected_reward_step_1 = torch.tensor(
        [
            [-0.011193251237273216, 0.019075289368629456],
            [-0.011193251237273216, 0.019075289368629456],
        ]
    )
    expected_h1_prefix = torch.tensor(
        [
            0.09417809545993805,
            0.003162681357935071,
            -0.03786328434944153,
            -0.018035821616649628,
            -0.06740731000900269,
            0.04291834682226181,
            0.010197310708463192,
            0.010774987749755383,
        ]
    )
    assert torch.allclose(reward, expected_reward_step_1, atol=1e-6, rtol=1e-5)
    assert torch.allclose(h_1[0, 0, :8], expected_h1_prefix, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(h_1, h_0), "LSTM state should update"

    output_2 = net.hidden_state_inference(next_network_state, action)

    reward_2 = output_2.reward
    h_2, _ = output_2.network_state.wm_memory["reward_hidden"]

    print("Step 2 done.")
    expected_reward_step_2 = torch.tensor([0.012203490361571312, 0.027373969554901123])
    expected_h2_prefix = torch.tensor(
        [
            0.16868829727172852,
            0.04990871623158455,
            -0.0451614186167717,
            -0.017601564526557922,
            -0.1609778255224228,
            0.0750245600938797,
            -0.0019532593432813883,
            0.038320910185575485,
        ]
    )
    assert torch.allclose(reward_2, expected_reward_step_2, atol=1e-6, rtol=1e-5)
    assert torch.allclose(h_2[0, 0, :8], expected_h2_prefix, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(h_2, h_1), "LSTM state should update again"

    print("Success! ValuePrefixRewardHead is integrated and functioning.")


def test_use_value_prefix_network_invalid_action_shape_raises(
    rainbow_cartpole_replay_config, make_cartpole_config
):
    torch.manual_seed(42)
    np.random.seed(42)

    config = _build_value_prefix_config(
        rainbow_cartpole_replay_config, make_cartpole_config
    )
    net = ModularAgentNetwork(config, (4,), config.game.num_actions)

    batch_size = 2
    network_state = MuZeroNetworkState(
        dynamics=torch.randn(batch_size, 4),
        wm_memory={
            "reward_hidden": (
                torch.zeros(1, batch_size, config.lstm_hidden_size),
                torch.zeros(1, batch_size, config.lstm_hidden_size),
            )
        },
    )
    # Stochastic dynamics expects one-hot vectors with width `config.num_chance`.
    invalid_action = torch.randn(batch_size, config.num_chance - 1)

    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        net.hidden_state_inference(network_state, invalid_action)
