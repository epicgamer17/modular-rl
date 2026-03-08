import pytest
import torch
import torch.nn.functional as F
from unittest.mock import MagicMock
from losses.losses import (
    ValueLoss,
    PolicyLoss,
    RewardLoss,
    ToPlayLoss,
    RelativeToPlayLoss,
    ConsistencyLoss,
    ChanceQLoss,
    SigmaLoss,
    VQVAECommitmentLoss,
    StandardDQNLoss,
    C51Loss,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def base_config():
    config = MagicMock()
    config.minibatch_size = 4
    config.support_range = None
    config.value_loss_function = F.mse_loss
    config.value_loss_factor = 1.0
    config.policy_loss_function = F.cross_entropy
    config.reward_loss_function = F.mse_loss
    config.to_play_loss_function = F.cross_entropy
    config.to_play_loss_factor = 1.0
    config.consistency_loss_factor = 1.0
    config.stochastic = True
    config.use_true_chance_codes = False
    config.sigma_loss = F.cross_entropy
    config.vqvae_commitment_cost_factor = 1.0
    config.loss_function = F.mse_loss
    config.mask_absorbing = True
    config.game.num_players = 2
    return config


def test_value_loss(base_config):
    device = torch.device("cpu")
    loss_module = ValueLoss(base_config, device)

    predictions = {"values": torch.randn((4, 1), device=device)}
    targets = {"values": torch.randn((4,), device=device)}
    context = {"is_same_game": torch.ones((4, 1), device=device).bool()}

    # Test scalar case
    loss = loss_module.compute_loss(predictions, targets, context, k=0)
    assert loss.shape == (4,)

    # Test support case
    base_config.support_range = 10
    predictions = {"values": torch.randn((4, 21), device=device)}
    targets = {"values": torch.randn((4,), device=device)}
    loss = loss_module.compute_loss(predictions, targets, context, k=0)
    assert loss.shape == (4,)


def test_policy_loss(base_config):
    device = torch.device("cpu")
    loss_module = PolicyLoss(base_config, device)

    predictions = {"policies": torch.randn((4, 10), device=device)}
    targets = {"policies": torch.randn((4, 10), device=device).softmax(dim=-1)}
    context = {"has_valid_action_mask": torch.ones((4, 1), device=device).bool()}

    # Test cross-entropy case
    loss = loss_module.compute_loss(predictions, targets, context, k=0)
    assert loss.shape == (4,)

    # Test KL div case
    base_config.policy_loss_function = F.kl_div
    loss = loss_module.compute_loss(predictions, targets, context, k=0)
    assert loss.shape == (4,)


def test_reward_loss(base_config):
    device = torch.device("cpu")
    loss_module = RewardLoss(base_config, device)

    predictions = {"rewards": torch.randn((4, 1), device=device)}
    targets = {"rewards": torch.randn((4,), device=device)}
    context = {"is_same_game": torch.ones((4, 1), device=device).bool()}

    # Test scalar case
    loss = loss_module.compute_loss(predictions, targets, context, k=0)
    assert loss.shape == (4,)

    # Test support case
    base_config.support_range = 10
    predictions = {"rewards": torch.randn((4, 21), device=device)}
    targets = {"rewards": torch.randn((4,), device=device)}
    loss = loss_module.compute_loss(predictions, targets, context, k=0)
    assert loss.shape == (4,)


def test_to_play_loss(base_config):
    device = torch.device("cpu")
    loss_module = ToPlayLoss(base_config, device)

    predictions = {"to_plays": torch.randn((4, 2), device=device)}
    targets = {"to_plays": torch.randint(0, 2, (4,), device=device)}
    context = {}

    loss = loss_module.compute_loss(predictions, targets, context, k=1)
    assert loss.shape == (4,)

    # Test should_compute
    assert loss_module.should_compute(k=1, context={}) == True
    base_config.game.num_players = 1
    assert loss_module.should_compute(k=1, context={}) == False


def test_relative_to_play_loss(base_config):
    device = torch.device("cpu")
    loss_module = RelativeToPlayLoss(base_config, device)

    predictions = {"to_plays": torch.randn((4, 2), device=device)}
    targets = {"to_plays": torch.randint(0, 2, (4,), device=device)}
    context = {"full_targets": {"to_plays": torch.randint(0, 2, (4, 2), device=device)}}

    loss = loss_module.compute_loss(predictions, targets, context, k=1)
    assert loss.shape == (4,)

    # Test should_compute
    assert loss_module.should_compute(k=1, context=context) == True
    base_config.game.num_players = 1
    assert loss_module.should_compute(k=1, context=context) == False


def test_consistency_loss(base_config):
    device = torch.device("cpu")
    agent_network = MagicMock()
    agent_network.project.return_value = torch.randn((4, 64), device=device)

    loss_module = ConsistencyLoss(base_config, device, agent_network)

    predictions = {"latent_states": torch.randn((4, 64), device=device)}
    targets = {"consistency_targets": torch.randn((4, 64), device=device)}
    context = {}

    loss = loss_module.compute_loss(predictions, targets, context, k=1)
    assert loss.shape == (4,)


def test_chance_q_loss(base_config):
    device = torch.device("cpu")
    loss_module = ChanceQLoss(base_config, device)

    predictions = {"chance_values": torch.randn((4, 1), device=device)}
    targets = {"values": torch.randn((4, 2), device=device)}
    context = {
        "target_values_next": torch.randn((4,), device=device),
        "is_same_game": torch.ones((4, 2), device=device).bool(),
    }

    # Test scalar
    loss = loss_module.compute_loss(predictions, targets, context, k=1)
    assert loss.shape == (4,)

    # Test support
    base_config.support_range = 10
    predictions = {"chance_values": torch.randn((4, 21), device=device)}
    loss = loss_module.compute_loss(predictions, targets, context, k=1)
    assert loss.shape == (4,)


def test_sigma_loss(base_config):
    device = torch.device("cpu")
    loss_module = SigmaLoss(base_config, device)

    predictions = {"chance_codes": torch.randn((4, 8), device=device)}
    targets = {"chance_codes": torch.randint(0, 8, (4, 1), device=device)}
    context = {}

    loss = loss_module.compute_loss(predictions, targets, context, k=1)
    assert loss.shape == (4,)


def test_vqvae_commitment_loss(base_config):
    device = torch.device("cpu")
    loss_module = VQVAECommitmentLoss(base_config, device)

    predictions = {"chance_encoder_embeddings": torch.randn((4, 8), device=device)}
    targets = {"chance_codes": torch.randint(0, 8, (4, 1), device=device)}
    context = {}

    loss = loss_module.compute_loss(predictions, targets, context, k=1)
    assert loss.shape == (4,)


def test_standard_dqn_loss(base_config):
    device = torch.device("cpu")
    loss_module = StandardDQNLoss(base_config, device)

    predictions = {"online_q_values": torch.randn((4,), device=device)}
    targets = {"target_q_values": torch.randn((4,), device=device)}
    context = {}

    loss = loss_module.compute_loss(predictions, targets, context, k=0)
    assert loss.shape == (4,)


def test_c51_loss(base_config):
    device = torch.device("cpu")
    loss_module = C51Loss(base_config, device)

    predictions = {"online_dist": torch.randn((4, 21), device=device)}
    targets = {"target_dist": torch.randn((4, 21), device=device).softmax(dim=-1)}
    context = {}

    loss = loss_module.compute_loss(predictions, targets, context, k=0)
    assert loss.shape == (4,)
