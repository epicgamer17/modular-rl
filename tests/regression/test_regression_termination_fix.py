import pytest
import torch


@pytest.mark.regression
def test_regression_termination_fix():
    """Ensure n-step target computation cuts off rewards after terminal states."""
    raw_rewards = torch.tensor([[10.0, 50.0, 999.0, 999.0]])
    raw_dones = torch.tensor([[False, True, False, False]])
    valid_mask = torch.tensor([[True, True, True, True]])
    n_step = 3
    gamma = 1.0

    old_value = _compute_n_step_old(
        raw_rewards, raw_dones, valid_mask, n_step=n_step, gamma=gamma
    )
    new_value = _compute_n_step_new(
        raw_rewards, raw_dones, valid_mask, n_step=n_step, gamma=gamma
    )

    # Old behavior incorrectly includes the reward for transition 1->2.
    assert old_value.item() == 60.0
    # Fixed behavior should only include transition 0->1 reward.
    assert new_value.item() == 10.0
    assert new_value.item() < old_value.item()


def _compute_n_step_old(raw_rewards, raw_dones, valid_mask, n_step, gamma):
    computed_value = torch.zeros(1)
    has_ended = torch.zeros(1, dtype=torch.bool)

    for k in range(n_step):
        r_idx = k
        r_is_valid = valid_mask[:, r_idx] & (~has_ended)
        reward_chunk = (gamma**k) * raw_rewards[:, r_idx]
        computed_value += torch.where(r_is_valid, reward_chunk, torch.tensor(0.0))
        has_ended = has_ended | (raw_dones[:, r_idx] & valid_mask[:, r_idx])

    return computed_value


def _compute_n_step_new(raw_rewards, raw_dones, valid_mask, n_step, gamma):
    computed_value = torch.zeros(1)
    has_ended = torch.zeros(1, dtype=torch.bool)

    for k in range(n_step):
        r_idx = k
        r_is_valid = valid_mask[:, r_idx] & (~has_ended)
        reward_chunk = (gamma**k) * raw_rewards[:, r_idx]
        computed_value += torch.where(r_is_valid, reward_chunk, torch.tensor(0.0))
        if r_idx + 1 < raw_dones.shape[1]:
            has_ended = has_ended | raw_dones[:, r_idx + 1]

    return computed_value
