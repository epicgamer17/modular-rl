import torch
import gymnasium as gym
from configs.agents.muzero import MuZeroConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel
from modules.agent_nets.muzero import AgentNetwork
from modules.heads.reward import ValuePrefixRewardHead
from configs.games.game import GameConfig


class MockEnv:
    def __init__(self):
        self.action_space = gym.spaces.Discrete(5)

    def close(self):
        pass


def test_value_prefix_network():
    print("Testing ValuePrefixRewardHead integration...")
    game = GameConfig(
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
        make_env=lambda: MockEnv(),
    )

    config_dict = {
        "world_model_cls": MuzeroWorldModel,
        "stochastic": True,
        "num_chance": 10,
        "value_prefix": True,
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

    print("Initializing MuZero Config with value_prefix=True...")
    config = MuZeroConfig(config_dict, game)

    print("Initializing Network...")
    input_shape = (4,)  # Flat input
    net = Network(config, config.game.num_actions, input_shape)

    # Verify Reward Head Type
    reward_head = net.world_model.dynamics.reward_head
    print(f"Reward Head Type: {type(reward_head)}")
    assert isinstance(
        reward_head, ValuePrefixRewardHead
    ), "Reward head must be ValuePrefixRewardHead"

    # Check LSTM existence
    assert hasattr(reward_head, "lstm"), "ValuePrefixRewardHead must have LSTM"
    assert (
        reward_head.lstm.hidden_size == 16
    ), f"LSTM hidden size should be 16, got {reward_head.lstm.hidden_size}"

    # Test Recurrent Inference with State
    print("Testing Recurrent Inference...")
    batch_size = 2
    hidden_state = torch.randn(batch_size, 4)
    # Since stochastic=True, recurrent_inference expects chance codes (one-hot)
    action_idx = torch.randint(0, 10, (batch_size,))  # num_chance=10
    action = torch.nn.functional.one_hot(action_idx, num_classes=10).float()

    # Initial Reward Hidden State
    # ValuePrefixRewardHead.get_initial_state returns dict {"reward_hidden": (h, c)}
    # But MuzeroWorldModel.recurrent_inference expects separate tensors (reward_h_states, reward_c_states)
    # The agent/trainer usually handles this unpacking/packing.
    # Let's manually create them for this test.
    bn = 1  # bidirectional? No, usually 1.
    h_0 = torch.zeros(1, batch_size, 16)
    c_0 = torch.zeros(1, batch_size, 16)

    # Run step 1
    reward, next_hidden, value, policy, to_play, (h_1, c_1) = net.recurrent_inference(
        hidden_state, action, h_0, c_0
    )

    print("Step 1 done.")
    print(f"Reward shape: {reward.shape}")
    print(f"Next Hidden State shape: {next_hidden.shape}")
    print(f"Reward Hidden h_1 shape: {h_1.shape}")

    assert reward.shape == (batch_size,)  # Scalar reward (B,)
    assert not torch.allclose(h_1, h_0), "LSTM state should update"

    # Run step 2
    reward_2, next_hidden_2, value_2, policy_2, to_play_2, (h_2, c_2) = (
        net.recurrent_inference(next_hidden, action, h_1, c_1)
    )
    print("Step 2 done.")
    assert not torch.allclose(h_2, h_1), "LSTM state should update again"

    print("Success! ValuePrefixRewardHead is integrated and functioning.")


if __name__ == "__main__":
    test_value_prefix_network()
