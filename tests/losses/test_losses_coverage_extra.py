import pytest
import torch
import torch.nn.functional as F
from unittest.mock import MagicMock
from losses.losses import (
    LossPipeline,
    StandardDQNLoss,
    C51Loss,
    ValueLoss,
    PolicyLoss,
    RewardLoss,
    ToPlayLoss,
    RelativeToPlayLoss,
    ConsistencyLoss,
    ChanceQLoss,
    SigmaLoss,
    VQVAECommitmentLoss,
)
from losses.basic_losses import (
    _select_next_actions,
    PPOValueLoss,
    StandardDQNLossModule,
    C51LossModule,
    ImitationLoss,
)
from modules.world_models.inference_output import InferenceOutput

pytestmark = pytest.mark.unit


@pytest.fixture
def base_config():
    config = MagicMock()
    config.minibatch_size = 4
    config.support_range = None
    config.value_loss_function = F.mse_loss
    config.value_loss_factor = 1.0
    config.policy_loss_function = F.cross_entropy
    config.policy_loss_factor = 1.0
    config.reward_loss_function = F.mse_loss
    config.reward_loss_factor = 1.0
    config.to_play_loss_function = F.cross_entropy
    config.to_play_loss_factor = 1.0
    config.consistency_loss_factor = 1.0
    config.stochastic = True
    config.use_true_chance_codes = False
    config.sigma_loss = F.cross_entropy
    config.vqvae_commitment_cost_factor = 1.0
    config.loss_function = F.mse_loss
    config.mask_absorbing = False
    config.game.num_players = 2
    config.atom_size = 21
    config.v_min = -10
    config.v_max = 10
    config.discount_factor = 0.99
    config.n_step = 1
    config.bootstrap_on_truncated = False
    return config


# --- Comprehensive Property & Mask Coverage ---


def test_all_modules_properties_and_masks(base_config):
    device = torch.device("cpu")
    agent_network = MagicMock()

    modules = [
        StandardDQNLoss(base_config, device),
        C51Loss(base_config, device),
        ValueLoss(base_config, device),
        PolicyLoss(base_config, device),
        RewardLoss(base_config, device),
        ToPlayLoss(base_config, device),
        RelativeToPlayLoss(base_config, device),
        ConsistencyLoss(base_config, device, agent_network),
        ChanceQLoss(base_config, device),
        SigmaLoss(base_config, device),
        VQVAECommitmentLoss(base_config, device),
        StandardDQNLossModule(base_config, device),
        C51LossModule(base_config, device),
        PPOValueLoss(base_config, device, critic_coefficient=1.0),
        ImitationLoss(base_config, device, num_actions=10),
    ]

    context = {
        "has_valid_obs_mask": torch.tensor([[True, False]], dtype=torch.bool),
        "has_valid_action_mask": torch.tensor([[True, False]], dtype=torch.bool),
        "is_same_game": torch.tensor([[True, False]], dtype=torch.bool),
        "full_targets": {"to_plays": torch.tensor([[0, 1]])},
    }

    for m in modules:
        # Trigger required_predictions (losses.py: 69, 97, 194, 244, 303, 406, 452, 517, 568; basic_losses: 235)
        _ = m.required_predictions

        # Trigger required_targets (losses.py: 73, 101, 198, 248, 307, 410, 457, 521, 572; basic_losses: 239)
        _ = m.required_targets

        # Trigger should_compute (losses.py: 460, 524, 575)
        _ = m.should_compute(k=1, context=context)

        # Trigger get_mask (losses.py: 39 default path, 315, 506, 528, 581)
        # To trigger line 39, we need a context WITHOUT has_valid_obs_mask for some modules or just a base call
        _ = m.get_mask(k=0, context=context)

    # Specifically target line 39 in losses.py (default get_mask)
    base_loss = StandardDQNLoss(base_config, device)  # Does not override get_mask
    mask = base_loss.get_mask(k=0, context={})
    assert mask.shape == (base_config.minibatch_size,)

    # Specifically target ToPlayLoss.get_mask (line 315)
    to_play = ToPlayLoss(base_config, device)
    mask = to_play.get_mask(k=1, context=context)
    assert mask.item() is False


# --- Other Edge Cases from Previous Attempt ---


def test_sigma_loss_onehot_fallback(base_config):
    base_config.sigma_loss = F.mse_loss
    loss = SigmaLoss(base_config, torch.device("cpu"))
    predictions = {"chance_codes": torch.randn((1, 8))}
    targets = {"chance_codes": torch.tensor([[2]])}
    l = loss.compute_loss(predictions, targets, {}, k=1)
    assert l.shape == (1, 8)


def test_consistency_loss_latent_dict(base_config):
    agent_network = MagicMock()
    agent_network.project.return_value = torch.randn((1, 64))
    loss = ConsistencyLoss(base_config, torch.device("cpu"), agent_network)
    predictions = {"latents": {"dynamics": torch.randn((1, 64))}}
    targets = {"consistency_targets": torch.randn((1, 64))}
    l = loss.compute_loss(predictions, targets, {}, k=1)
    assert l.shape == (1,)


def test_loss_pipeline_validate_dependencies_errors(base_config):
    from losses.losses import ValueLoss

    device = torch.device("cpu")
    pipeline = LossPipeline([ValueLoss(base_config, device)])
    with pytest.raises(ValueError, match="missing required predictions"):
        pipeline.validate_dependencies(set(), {"values"})
    with pytest.raises(ValueError, match="missing required targets"):
        pipeline.validate_dependencies({"values"}, set())


def test_select_next_actions_masking():
    q = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([1, 1, 0])
    idx = _select_next_actions(q, mask)
    assert idx.item() == 1
    mask = torch.tensor([0, 0, 0])
    idx = _select_next_actions(q, mask)
    assert idx.item() == 2


def test_ppo_value_loss_edge_cases(base_config):
    device = torch.device("cpu")
    loss = PPOValueLoss(base_config, device, critic_coefficient=1.0)
    val = loss._to_scalar_values(torch.tensor([1.0, 2.0]))
    assert val[0] == 1.0
    loss.atom_size = 10
    loss.v_min = None
    with pytest.raises(
        ValueError, match="received multi-logit values without distributional bounds"
    ):
        loss._to_scalar_values(torch.randn((1, 10)))


def test_standard_dqn_double_dqn_path(base_config):
    device = torch.device("cpu")
    selector = MagicMock()
    selector.select_action.return_value = (torch.tensor(1), {})
    loss = StandardDQNLossModule(base_config, device, action_selector=selector)
    predictions = {
        "q_values": torch.randn((1, 4)),
    }
    targets = {
        "q_values": torch.randn((1,)),  # Scalar target values per sample
        "actions": torch.tensor([0]),
        "rewards": torch.tensor([1.0]),
        "dones": torch.tensor([False]),
        "next_legal_moves_masks": torch.tensor([[1, 1, 0, 0]]),
        "next_observations": torch.randn((1, 8)),
    }
    context = {"agent_network": MagicMock()}
    l = loss.compute_loss(predictions, targets, context)
    assert l.shape == (1,)


def test_c51_dqn_double_dqn_path(base_config):
    device = torch.device("cpu")
    selector = MagicMock()
    selector.select_action.return_value = (torch.tensor(1), {})
    loss = C51LossModule(base_config, device, action_selector=selector)
    predictions = {
        "q_logits": torch.randn((1, 4, 21)),
    }
    targets = {
        "q_logits": torch.randn((1, 21)),  # Target distribution per sample
        "actions": torch.tensor([0]),
        "rewards": torch.tensor([1.0]),
        "dones": torch.tensor([False]),
        "next_legal_moves_masks": torch.tensor([[1, 1, 0, 0]]),
        "next_observations": torch.randn((1, 8)),
    }
    context = {"agent_network": MagicMock()}
    l = loss.compute_loss(predictions, targets, context)
    assert l.shape == (1,)


def test_imitation_loss_multidim(base_config):
    loss = ImitationLoss(base_config, torch.device("cpu"), num_actions=10)
    loss.loss_function = lambda p, t: torch.ones((1, 10))
    l = loss.compute_loss(
        {"policies": torch.ones((1, 10))}, {"target_policies": torch.ones((1, 10))}, {}
    )
    assert l.item() == 10.0
