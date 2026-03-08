import pytest
import torch
from modules.heads.reward import ValuePrefixRewardHead
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.heads.reward import ValuePrefixRewardHeadConfig
from modules.heads.strategies import ScalarStrategy

pytestmark = pytest.mark.unit


def test_value_prefix_instant_reward_calculation():
    """
    Verifies that ValuePrefixRewardHead correctly calculates instant_reward
    as (current_cumulative - effective_parent_cumulative).
    """
    torch.manual_seed(42)

    arch_config = ArchitectureConfig({"noisy_sigma": 0.0})
    config = ValuePrefixRewardHeadConfig(
        {"lstm_hidden_size": 8, "lstm_horizon_len": 10}
    )
    strategy = ScalarStrategy()  # Simple scalar output

    head = ValuePrefixRewardHead(
        arch_config=arch_config, input_shape=(4,), strategy=strategy, config=config
    )

    # Mock some input
    x = torch.randn(1, 4)

    # CASE 1: First step (step_count=0, cumulative=0)
    state = {
        "step_count": torch.tensor([[0.0]]),
        "cumulative_reward": torch.tensor([[0.0]]),
    }

    logits, new_state, instant_reward = head(x, state)

    # expected_cumulative = strategy.to_expected_value(logits)
    # instant_reward = expected_cumulative - 0.0
    assert torch.allclose(instant_reward, new_state["cumulative_reward"].squeeze())

    # CASE 2: Middle step (step_count=1, cumulative=10.0)
    prev_cumulative = torch.tensor([[10.0]])
    state = {
        "step_count": torch.tensor([[1.0]]),
        "cumulative_reward": prev_cumulative,
    }

    logits, new_state, instant_reward = head(x, state)

    # instant_reward should be current_cumulative - prev_cumulative
    expected_instant = new_state["cumulative_reward"] - prev_cumulative
    assert torch.allclose(instant_reward, expected_instant.squeeze())


def test_value_prefix_horizon_subtraction_reset():
    """
    Verifies that when the horizon is reached, the cumulative baseline used for
    subtraction resets to 0, even if the state's cumulative_reward is non-zero.
    """
    torch.manual_seed(42)

    arch_config = ArchitectureConfig({"noisy_sigma": 0.0})
    config = ValuePrefixRewardHeadConfig({"lstm_hidden_size": 8, "lstm_horizon_len": 5})
    strategy = ScalarStrategy()

    head = ValuePrefixRewardHead(
        arch_config=arch_config, input_shape=(4,), strategy=strategy, config=config
    )

    x = torch.randn(1, 4)

    # step_count=5 means we hit the horizon.
    # The head should reset internal LSTM state AND treat parent_cumulative as 0 for subtraction.
    state = {
        "step_count": torch.tensor([[5.0]]),
        "cumulative_reward": torch.tensor(
            [[100.0]]
        ),  # Huge value to detect if it's subtracted
    }

    logits, new_state, instant_reward = head(x, state)

    # If it didn't reset, instant_reward would be roughly (current_cum - 100) -> very negative.
    # If it DID reset, instant_reward should be roughly equal to new_state["cumulative_reward"].
    assert torch.allclose(
        instant_reward, new_state["cumulative_reward"].squeeze(), atol=1e-5
    )
    assert new_state["step_count"] == 1.0
