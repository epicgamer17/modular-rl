import pytest
import torch
from atomic_rl.td.updates import (
    gtd0_update_,
    semi_gradient_td_update_,
    tdc_update_,
    true_online_td_update_,
)

pytestmark = pytest.mark.unit


def test_semi_gradient_td_update():
    weights = torch.tensor([1.0, 2.0])
    error = 0.5
    alpha = 0.1
    update_vector = torch.tensor([0.5, 0.5])

    updated = semi_gradient_td_update_(error, weights.clone(), alpha, update_vector)
    # expected: weights + alpha * error * update_vector = [1.0, 2.0] + 0.05 * [0.5, 0.5] = [1.025, 2.025]
    expected = torch.tensor([1.025, 2.025])
    torch.testing.assert_close(updated, expected)


def test_true_online_td_update():
    weights = torch.tensor([1.0, 2.0])
    error = 0.5
    v_curr = 1.5
    v_old = 1.0
    features = torch.tensor([0.5, 0.5])
    trace = torch.tensor([0.8, 0.8])
    alpha = 0.1

    updated = true_online_td_update_(
        error, v_curr, v_old, features, weights.clone(), alpha, trace
    )
    assert updated.shape == (2,)


def test_gtd0_update():
    weights = torch.tensor([1.0, 2.0])
    u = torch.tensor([0.1, 0.2])
    error = 0.5
    features = torch.tensor([1.0, 0.0])
    next_features = torch.tensor([0.0, 1.0])
    gamma = 0.9
    alpha = 0.1
    beta = 0.01
    terminated = False

    w_out, u_out = gtd0_update_(
        error, features, next_features, gamma, weights.clone(), u.clone(), alpha, beta, terminated
    )
    assert w_out.shape == (2,)
    assert u_out.shape == (2,)


def test_tdc_update():
    weights = torch.tensor([1.0, 2.0])
    w = torch.tensor([0.1, 0.2])
    error = 0.5
    features = torch.tensor([1.0, 0.0])
    next_features = torch.tensor([0.0, 1.0])
    gamma = 0.9
    alpha = 0.1
    beta = 0.01
    terminated = False

    w_out, aux_out = tdc_update_(
        error, features, next_features, gamma, weights.clone(), w.clone(), alpha, beta, terminated
    )
    assert w_out.shape == (2,)
    assert aux_out.shape == (2,)
