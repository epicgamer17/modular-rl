import pytest
import torch
import torch.nn.functional as F
import numpy as np

from agents.learners.ppo_learner import PPOLearner
from modules.agent_nets.modular import ModularAgentNetwork
from configs.agents.ppo import PPOConfig
from configs.agents.rainbow_dqn import RainbowConfig

pytestmark = pytest.mark.unit


def test_kl_divergence_normalization():
    """
    Test the logic of KL divergence calculation with manual normalization.
    """
    torch.manual_seed(42)

    # Unnormalized distributions
    p = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    q = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float32)

    # Manual normalization
    p_norm = p / p.sum(dim=-1, keepdim=True)
    q_norm = q / q.sum(dim=-1, keepdim=True)

    log_p = torch.log(p_norm + 1e-10)
    kl_val = F.kl_div(log_p, q_norm, reduction="sum")

    assert kl_val >= 0, f"KL divergence should be non-negative, got {kl_val}"

    # Test identical distributions -> KL = 0
    log_q = torch.log(q_norm + 1e-10)
    kl_identical = F.kl_div(log_q, q_norm, reduction="sum")
    assert torch.isclose(kl_identical, torch.tensor(0.0), atol=1e-6)


def test_ppo_learner_math(make_ppo_config_dict, make_cartpole_config):
    """
    Test PPOLearner loss calculation.
    """
    torch.manual_seed(42)
    device = torch.device("cpu")
    num_actions = 2
    obs_dim = (4,)

    # 1. Use factory from conftest.py
    config_dict = make_ppo_config_dict(
        train_policy_iterations=1,
        train_value_iterations=1,
    )
    game_config = make_cartpole_config()
    config = PPOConfig(config_dict, game_config)

    # 2. Instantiate network and learner
    network = ModularAgentNetwork(config, obs_dim, num_actions)
    learner = PPOLearner(config, network, device, num_actions, obs_dim, torch.float32)

    # 3. Create tiny fake batch (batch size 2)
    batch = {
        "observations": torch.randn(2, 4),
        "actions": torch.tensor([0, 1]),
        "log_probabilities": torch.tensor([-0.5, -0.6]),
        "advantages": torch.tensor([1.0, -1.0]),
        "returns": torch.tensor([10.0, 5.0]),
    }

    # 4. Directly call internal loss math
    losses = learner.compute_loss(
        batch=batch,
        actions=batch["actions"],
        old_log_probs=batch["log_probabilities"],
        advantages=batch["advantages"],
        returns=batch["returns"],
    )

    # 5. Assert valid losses
    assert "policy_loss" in losses
    assert "value_loss" in losses
    assert not torch.isnan(losses["policy_loss"])
    assert not torch.isnan(losses["value_loss"])
