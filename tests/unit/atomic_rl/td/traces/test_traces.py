import pytest
import torch
from atomic_rl.td.traces import (
    compute_accumulating_traces,
    compute_replacing_traces,
    compute_true_online_traces,
)

pytestmark = pytest.mark.unit


def test_update_accumulating_traces():
    traces = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    gradients = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
    terminated = torch.tensor([0.0, 1.0])
    gamma = 0.9
    lam = 0.9

    expected = torch.tensor([
        [1.31, 0.81, 0.81],
        [0.0, 0.5, 0.0]
    ])

    res = compute_accumulating_traces(traces, gradients, gamma, lam, terminated)
    torch.testing.assert_close(res, expected)


def test_update_replacing_traces():
    traces = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    features = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    terminated = torch.tensor([0.0, 1.0])
    gamma = 0.9
    lam = 0.9

    expected = torch.tensor([
        [0.81, 1.0, 0.81],
        [0.0, 1.0, 0.0]
    ])

    res = compute_replacing_traces(traces, features, gamma, lam, terminated)
    torch.testing.assert_close(res, expected)


def test_compute_true_online_traces():
    traces = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    features = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    alpha = 0.1
    gamma = 0.9
    lam = 0.9
    terminated = torch.tensor([0.0, 1.0])

    res = compute_true_online_traces(traces, features, alpha, gamma, lam, terminated)
    assert res.shape == (2, 2)
    assert res[1, 0].item() == pytest.approx(0.4595)



def test_traces_assertions():
    with pytest.raises(AssertionError, match="Trace and gradient shapes must match"):
        compute_accumulating_traces(torch.randn(2, 3), torch.randn(2, 2), 0.9, 0.9, torch.randn(2))
