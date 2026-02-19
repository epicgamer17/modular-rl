import torch
import gymnasium as gym
from configs.agents.muzero import MuZeroConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel
from modules.agent_nets.muzero import MuZeroNetwork
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
        "support_range": 5,
        "action_selector": {"base": {"type": "categorical", "kwargs": {}}},
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
    print("Initializing Network...")
    input_shape = (4,)  # Flat input
    net = MuZeroNetwork(config, config.game.num_actions, input_shape)

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
    print("Testing Hidden State Inference...")
    batch_size = 2
    # Create opaque network state
    hidden_state = torch.randn(batch_size, 4)
    bn = 1
    h_0 = torch.zeros(1, batch_size, 16)
    c_0 = torch.zeros(1, batch_size, 16)

    network_state = {
        "dynamics": hidden_state,
        "wm_memory": {"reward_hidden": (h_0, c_0)},
    }

    # Since stochastic=True, inference expects chance codes (one-hot)
    # But wait, hidden_state_inference in MuZero (if stochastic) might expect action to be different?
    # No, action is standard action.
    # But stochastic muzero usually samples chance code in world model if not provided?
    # Standard hidden_state_inference calls wm.recurrent_inference.

    action_idx = torch.randint(0, 5, (batch_size,))  # Actions
    action = torch.nn.functional.one_hot(action_idx, num_classes=5).float()

    # Run step 1
    output = net.hidden_state_inference(network_state, action)

    reward = output.reward
    next_network_state = output.network_state

    # Extract LSTM state from opaque state
    h_1, c_1 = next_network_state["wm_memory"]["reward_hidden"]

    print("Step 1 done.")
    print(f"Reward shape: {reward.shape}")
    print(f"Reward Hidden h_1 shape: {h_1.shape}")

    assert reward.shape == (batch_size, 1) or reward.shape == (
        batch_size,
    ), f"Shape mismatch: {reward.shape}"
    assert not torch.allclose(h_1, h_0), "LSTM state should update"

    # Run step 2
    output_2 = net.hidden_state_inference(next_network_state, action)

    h_2, c_2 = output_2.network_state["wm_memory"]["reward_hidden"]

    print("Step 2 done.")
    assert not torch.allclose(h_2, h_1), "LSTM state should update again"

    print("Success! ValuePrefixRewardHead is integrated and functioning.")


if __name__ == "__main__":
    test_value_prefix_network()
