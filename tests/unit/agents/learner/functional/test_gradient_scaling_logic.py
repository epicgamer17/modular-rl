import torch
import torch.nn as nn
import pytest
from modules.utils import scale_gradient
from agents.learner.losses.loss_pipeline import LossPipeline
from modules.models.world_model import WorldModel
from modules.models.agent_network import AgentNetwork
from agents.learner.target_builders import SequenceInfrastructureBuilder
from typing import Tuple, Dict, Any

pytestmark = pytest.mark.unit


def test_gradient_scaling_logic():
    """
    Tier 1 Unit Test: Gradient Scaling Logic Verification
    Checks:
    1. Loss from each head is scaled by 1/K (via gradient_scales).
    2. Gradient entering dynamics is scaled by 1/2.
    """
    K = 5
    B = 2
    hidden_dim = 8

    # 1. Test Loss Scaling (1/K)
    # --------------------------
    # Simulation: elementwise_loss [B, T=K+1]
    elementwise_loss = torch.ones((B, K + 1), requires_grad=True).clone()
    gradient_scales = torch.tensor([[1.0] + [1.0 / K] * K]).reshape(1, -1)

    # Instrumentation: Register hook on the input tensor
    loss_grads = []

    def loss_hook(grad):
        loss_grads.append(grad.clone())

    elementwise_loss.register_hook(loss_hook)

    # Simulate LossPipeline.run logic:
    scaled_loss = scale_gradient(elementwise_loss, gradient_scales)

    # Trigger backward
    loss_sum = scaled_loss.sum()
    loss_sum.backward()

    # Expected:
    # grad[0] = 1.0
    # grad[1:] = 1/K = 0.2
    assert len(loss_grads) == 1
    captured_grad = loss_grads[0]

    expected_loss_grad = torch.zeros((B, K + 1))
    expected_loss_grad[:, 0] = 1.0
    expected_loss_grad[:, 1:] = 1.0 / K

    torch.testing.assert_close(captured_grad, expected_loss_grad)
    print(f"Verified 1/K ({1/K}) Loss Scaling via backward hooks")

    # 2. Test Dynamics Scaling (1/2)
    # -----------------------------
    # Simulation: WorldModel.unroll_physics logic
    current_latent = torch.ones((B, hidden_dim), requires_grad=True).clone()

    dyn_grads = []

    def dyn_hook(grad):
        dyn_grads.append(grad.clone())

    current_latent.register_hook(dyn_hook)

    # Step 1: scale_gradient(current_latent, 0.5)
    after_step_latent = scale_gradient(current_latent, 0.5)

    # Simulate downstream usage (e.g., next dynamics step)
    loss_dyn = (after_step_latent * 10.0).sum()
    loss_dyn.backward()

    # Expected: grad = 10.0 * 0.5 = 5.0
    assert len(dyn_grads) == 1
    captured_dyn_grad = dyn_grads[0]
    expected_dyn_grad = torch.full((B, hidden_dim), 5.0)

    torch.testing.assert_close(captured_dyn_grad, expected_dyn_grad)
    print("Verified 1/2 Dynamics Gradient Scaling via backward hooks")
