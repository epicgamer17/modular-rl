import pytest
import torch
import torch.nn.functional as F
from unittest.mock import MagicMock
from losses.basic_losses import (
    PPOPolicyLoss,
    PPOValueLoss,
    StandardDQNLossModule,
    C51LossModule,
    ImitationLoss,
)
from modules.world_models.inference_output import LearningOutput

pytestmark = pytest.mark.unit


@pytest.fixture
def base_config():
    config = MagicMock()
    config.minibatch_size = 4
    config.policy_loss_function = F.cross_entropy
    config.value_loss_function = F.mse_loss
    config.loss_function = F.mse_loss
    config.clip_param = 0.2
    config.entropy_coefficient = 0.01
    config.critic_coefficient = 0.5
    config.atom_size = 1
    config.v_min = -10
    config.v_max = 10
    config.support = torch.linspace(-10, 10, 21)
    config.bootstrap_on_truncated = False
    config.discount_factor = 0.99
    config.n_step = 1
    return config


def test_ppo_policy_loss(base_config):
    device = torch.device("cpu")
    loss_module = PPOPolicyLoss(
        config=base_config, device=device, clip_param=0.2, entropy_coefficient=0.01
    )

    predictions = {"policies": torch.randn((4, 10), device=device)}
    targets = {
        "actions": torch.randint(0, 10, (4,), device=device),
        "old_log_probs": torch.randn((4,), device=device),
        "advantages": torch.randn((4,), device=device),
    }
    context = {}

    loss = loss_module.compute_loss(predictions, targets, context)
    assert loss.shape == (4,)


def test_ppo_value_loss(base_config):
    device = torch.device("cpu")
    loss_module = PPOValueLoss(
        config=base_config, device=device, critic_coefficient=0.5, atom_size=1
    )

    predictions = {"values": torch.randn((4, 1), device=device)}
    targets = {"returns": torch.randn((4,), device=device)}
    context = {}

    loss = loss_module.compute_loss(predictions, targets, context)
    assert loss.shape == (4,)

    # Test atom case
    loss_module.atom_size = 21
    loss_module.v_min = -10
    loss_module.v_max = 10
    predictions = {"values": torch.randn((4, 21), device=device)}
    loss = loss_module.compute_loss(predictions, targets, context)
    assert loss.shape == (4,)


def test_standard_dqn_loss_module(base_config):
    device = torch.device("cpu")
    action_selector = MagicMock()
    loss_module = StandardDQNLossModule(base_config, device, action_selector)

    predictions = {
        "q_values": torch.randn((4, 10), device=device),
        "next_q_values": torch.randn((4, 10), device=device),
    }
    targets = {
        "q_values": torch.randn((4,), device=device),
        "actions": torch.randint(0, 10, (4,), device=device),
        "rewards": torch.randn((4,), device=device),
        "dones": torch.zeros((4,), device=device).bool(),
    }
    context = {"agent_network": MagicMock()}

    loss = loss_module.compute_loss(predictions, targets, context)
    assert loss.shape == (4,)


def test_c51_loss_module(base_config):
    device = torch.device("cpu")
    action_selector = MagicMock()
    action_selector.select_action.return_value = (
        torch.randint(0, 10, (4,), device=device),
        {},
    )

    base_config.atom_size = 21
    loss_module = C51LossModule(base_config, device, action_selector)

    predictions = {
        "q_logits": torch.randn((4, 10, 21), device=device),
        "next_q_logits": torch.randn((4, 10, 21), device=device),
    }
    targets = {
        "q_logits": torch.randn((4, 21), device=device).softmax(dim=-1),
        "actions": torch.randint(0, 10, (4,), device=device),
        "rewards": torch.randn((4,), device=device),
        "dones": torch.zeros((4,), device=device).bool(),
    }
    context = {"agent_network": MagicMock()}

    loss = loss_module.compute_loss(predictions, targets, context)
    assert loss.shape == (4,)


def test_imitation_loss(base_config):
    device = torch.device("cpu")
    base_config.loss_function = torch.nn.CrossEntropyLoss(reduction="none")
    loss_module = ImitationLoss(base_config, device, num_actions=10)

    predictions = {"policies": torch.randn((4, 10), device=device)}
    targets = {"target_policies": torch.randint(0, 10, (4,), device=device)}

    loss = loss_module.compute_loss(predictions, targets, {})
    assert loss.shape == (4,)

    # Test distribution targets
    targets = {"target_policies": torch.randn((4, 10), device=device).softmax(dim=-1)}
    loss = loss_module.compute_loss(predictions, targets, {})
    assert loss.shape == (4,)
