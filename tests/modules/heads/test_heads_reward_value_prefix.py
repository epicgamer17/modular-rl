import pytest
import numpy as np
import torch

from modules.heads.reward import ValuePrefixRewardHead
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.heads.reward import ValuePrefixRewardHeadConfig
from modules.heads.strategies import Categorical
from modules.agent_nets.modular import ModularAgentNetwork
from modules.world_models.inference_output import MuZeroNetworkState
from configs.agents.muzero import MuZeroConfig

# Globally mark file for integration due to ModularAgentNetwork dependencies
pytestmark = pytest.mark.integration

# --- Unit Tests ---


def test_value_prefix_reward_head_horizon_reset():
    """Verifies that the LSTM hidden state and cumulative rewards reset at the horizon."""
    torch.manual_seed(42)

    arch_config = ArchitectureConfig({"noisy_sigma": 0.0})
    config = ValuePrefixRewardHeadConfig(
        {"lstm_hidden_size": 16, "lstm_horizon_len": 5}
    )
    strategy = Categorical(num_classes=5)

    head = ValuePrefixRewardHead(
        arch_config=arch_config, input_shape=(8,), strategy=strategy, config=config
    )

    x = torch.randn(2, 8)
    state = {
        "step_count": torch.tensor([[5.0], [3.0]]),
        "cumulative_reward": torch.tensor([[10.0], [4.0]]),
        "reward_hidden": (torch.ones(1, 2, 16), torch.ones(1, 2, 16)),
    }

    logits, new_state, instant_reward = head(x, state)

    assert logits.shape == (2, 5)
    assert instant_reward.shape == (2,)

    # Step count at horizon (5) should reset to 0 before adding 1.
    # Step count not at horizon (3) should simply add 1.
    assert torch.allclose(new_state["step_count"], torch.tensor([[1.0], [4.0]]))

    h_n, c_n = new_state["reward_hidden"]
    assert not torch.allclose(h_n[:, 0, :], h_n[:, 1, :])


# --- Integration Tests ---


def _build_value_prefix_config(make_muzero_config_dict, cartpole_game_config):
    """Helper to create a deterministic MuZero config with Value Prefix enabled."""
    config_dict = make_muzero_config_dict(
        stochastic=True,
        num_chance=10,
        use_value_prefix=True,
        lstm_hidden_size=16,
        lstm_horizon_len=5,
    )
    # Ensure heads output scalars to match the test structure
    config_dict["value_head"]["output_strategy"] = {"type": "scalar"}
    config_dict["reward_head"]["output_strategy"] = {"type": "scalar"}

    return MuZeroConfig(config_dict, cartpole_game_config)


def test_use_value_prefix_network_integration(
    make_muzero_config_dict, cartpole_game_config
):
    """Verifies full ModularAgentNetwork integration with ValuePrefixRewardHead."""
    torch.manual_seed(42)
    np.random.seed(42)

    config = _build_value_prefix_config(make_muzero_config_dict, cartpole_game_config)
    net = ModularAgentNetwork(config, (4,), config.game.num_actions)

    reward_head = net.components["world_model"].reward_head
    assert isinstance(
        reward_head, ValuePrefixRewardHead
    ), "Reward head must be ValuePrefixRewardHead"
    assert hasattr(reward_head, "lstm"), "ValuePrefixRewardHead must have LSTM"
    assert (
        reward_head.lstm.hidden_size == 16
    ), f"LSTM hidden size should be 16, got {reward_head.lstm.hidden_size}"

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

    # --- Step 1 ---
    output = net.hidden_state_inference(network_state, action)
    reward = output.reward
    next_network_state = output.network_state
    h_1, c_1 = next_network_state.wm_memory["reward_hidden"]

    assert reward.shape == (batch_size,)
    assert h_1.shape == (1, batch_size, 16)
    assert not torch.allclose(h_1, h_0), "LSTM state should update"

    # --- Step 2 ---
    output_2 = net.hidden_state_inference(next_network_state, action)
    reward_2 = output_2.reward
    h_2, _ = output_2.network_state.wm_memory["reward_hidden"]

    assert reward_2.shape == (batch_size,)
    assert h_2.shape == (1, batch_size, 16)
    assert not torch.allclose(h_2, h_1), "LSTM state should update again"


def test_use_value_prefix_network_invalid_action_shape_raises(
    make_muzero_config_dict, cartpole_game_config
):
    """Verifies the network safely catches shape mismatches during inference."""
    torch.manual_seed(42)
    np.random.seed(42)

    config = _build_value_prefix_config(make_muzero_config_dict, cartpole_game_config)
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

    # Note: Using regex pattern to catch various PyTorch shape mismatch strings
    with pytest.raises(RuntimeError, match="shape|size mismatch"):
        net.hidden_state_inference(network_state, invalid_action)
