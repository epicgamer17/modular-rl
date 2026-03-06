import copy

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

    assert reward.shape[0] == batch_size, f"Shape mismatch: {reward.shape}"
    assert not torch.allclose(h_1, h_0), "LSTM state should update"

    output_2 = net.hidden_state_inference(next_network_state, action)

    h_2, _ = output_2.network_state.wm_memory["reward_hidden"]

    print("Step 2 done.")
    assert not torch.allclose(h_2, h_1), "LSTM state should update again"

    print("Success! ValuePrefixRewardHead is integrated and functioning.")
