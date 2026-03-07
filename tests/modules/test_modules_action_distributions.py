import pytest
import torch

from modules.heads.strategies import Categorical, GaussianStrategy
from modules.heads.policy import PolicyHead
from modules.distributions import TanhBijector, SampleDist, OneHotDist, Deterministic
from configs.agents.ppo import PPOConfig
from configs.games.cartpole import CartPoleConfig

pytestmark = pytest.mark.unit


class TestDirectDistributions:
    """Part 1: Direct distribution instantiation tests."""

    def test_categorical_direct_output_shape(self):
        torch.manual_seed(42)
        batch_size, num_actions = 4, 5
        logits = torch.randn(batch_size, num_actions)
        strategy = Categorical(num_classes=num_actions)
        dist = strategy.get_distribution(logits)

        sample = dist.sample()
        assert sample.shape == (batch_size,)
        log_prob = dist.log_prob(sample)
        assert log_prob.shape == (batch_size,)
        assert not torch.isnan(log_prob).any()

    def test_gaussian_direct_output_shape(self):
        torch.manual_seed(42)
        batch_size, action_dim = 4, 3
        network_output = torch.randn(batch_size, action_dim * 2)
        strategy = GaussianStrategy(action_dim=action_dim)
        dist = strategy.get_distribution(network_output)

        sample = dist.sample()
        assert sample.shape == (batch_size, action_dim)
        log_prob = dist.log_prob(sample)
        assert log_prob.shape == (batch_size, action_dim)
        assert not torch.isnan(log_prob).any()

    def test_categorical_all_illegal_mask(self):
        torch.manual_seed(42)
        batch_size, num_actions = 4, 5
        logits = torch.randn(batch_size, num_actions)

        masked_logits = logits.clone()
        masked_logits[:, :] = -float("inf")

        dist = torch.distributions.Categorical(logits=masked_logits, validate_args=False)
        probs = dist.probs
        assert torch.isnan(probs).all() or (probs == 0).all()

    def test_categorical_all_legal_mask(self):
        torch.manual_seed(42)
        batch_size, num_actions = 4, 5
        logits = torch.randn(batch_size, num_actions)
        mask = torch.ones(batch_size, num_actions, dtype=torch.bool)

        masked_logits = logits.clone()
        masked_logits[~mask] = -float("inf")

        dist = torch.distributions.Categorical(logits=masked_logits)
        sample = dist.sample()
        log_prob = dist.log_prob(sample)
        assert not torch.isnan(log_prob).any()
        assert not torch.isinf(log_prob).any()


class TestDistributionClasses:
    """Part 2: Test distribution utility classes from modules/distributions.py."""

    def test_tanh_bijector_numerical_stability(self):
        torch.manual_seed(42)
        bijector = TanhBijector()
        x = torch.randn(4, 10)
        y = bijector(x)
        log_det = bijector.log_abs_det_jacobian(x, y)
        assert not torch.isnan(log_det).any()
        assert not torch.isinf(log_det).any()

    def test_one_hot_dist_mode(self):
        torch.manual_seed(42)
        batch_size, num_classes = 4, 5
        logits = torch.randn(batch_size, num_classes)
        dist = OneHotDist(logits=logits)
        mode = dist.mode()
        assert mode.shape == (batch_size, num_classes)
        assert torch.allclose(mode.sum(dim=-1), torch.ones(batch_size))

    def test_deterministic_distribution(self):
        torch.manual_seed(42)
        value = torch.randn(4, 3)
        dist = Deterministic(value)
        assert dist.mean.shape == (4, 3)
        assert dist.mode.shape == (4, 3)
        sample = dist.sample()
        assert sample.shape == (4, 3)

    def test_sample_dist_with_normal(self):
        torch.manual_seed(42)
        batch_size, feature_dim = 4, 8
        mean = torch.randn(batch_size, feature_dim)
        std = torch.ones(batch_size, feature_dim)
        base_dist = torch.distributions.Normal(mean, std)
        dist = SampleDist(base_dist, samples=10)
        sample = dist.sample()
        assert sample.shape == (batch_size, feature_dim)
        assert not torch.isnan(sample).any()


class TestPolicyHeadFromConfig:
    """Part 3: Policy Head integration with real config."""

    def test_policy_head_discrete_direct(self):
        torch.manual_seed(42)
        from configs.modules.architecture_config import ArchitectureConfig
        from modules.heads.policy import PolicyHead

        arch_config = ArchitectureConfig(
            {"type": "identity", "kwargs": {"hidden_widths": [32]}}
        )
        strategy = Categorical(num_classes=2)
        head = PolicyHead(
            arch_config=arch_config,
            input_shape=(64,),
            strategy=strategy,
        )

        latent = torch.randn(4, 64)
        logits, state, dist = head(latent)

        assert dist is not None
        sample = dist.sample()
        assert sample.shape == (4,)

    def test_policy_head_continuous_direct(self):
        torch.manual_seed(42)
        from configs.modules.architecture_config import ArchitectureConfig
        from modules.heads.policy import PolicyHead

        arch_config = ArchitectureConfig(
            {"type": "identity", "kwargs": {"hidden_widths": [32]}}
        )
        strategy = GaussianStrategy(action_dim=2)
        head = PolicyHead(
            arch_config=arch_config,
            input_shape=(64,),
            strategy=strategy,
        )

        latent = torch.randn(4, 64)
        logits, state, dist = head(latent)

        assert dist is not None
        sample = dist.sample()
        assert sample.shape == (4, 2)
        log_prob = dist.log_prob(sample)
        assert log_prob.shape == (4, 2)
        assert not torch.isnan(log_prob).any()
