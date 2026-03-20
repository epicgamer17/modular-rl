import pytest
import torch
from torch import distributions
from modules.heads.policy import PolicyHead
from modules.heads.value import ValueHead
from modules.heads.reward import RewardHead
from modules.heads.q import QHead
from agents.learner.losses.representations import (
    ClassificationRepresentation,
    CategoricalRepresentation,
    GaussianRepresentation,
    ScalarRepresentation,
    IdentityRepresentation,
)
from configs.modules.architecture_config import ArchitectureConfig

pytestmark = pytest.mark.unit

def test_policy_head_categorical_inference():
    arch_config = ArchitectureConfig({"noisy_sigma": 0.0})
    repr = ClassificationRepresentation(num_classes=5)
    head = PolicyHead(arch_config, input_shape=(16,), representation=repr)
    
    x = torch.randn(2, 16)
    _, _, inference = head(x)
    
    assert isinstance(inference, distributions.Categorical)
    assert inference.logits.shape == (2, 5)

def test_policy_head_gaussian_inference():
    arch_config = ArchitectureConfig({"noisy_sigma": 0.0})
    repr = GaussianRepresentation(action_dim=3)
    head = PolicyHead(arch_config, input_shape=(16,), representation=repr)
    
    x = torch.randn(2, 16)
    _, _, inference = head(x)
    
    assert isinstance(inference, distributions.Normal)
    assert inference.mean.shape == (2, 3)
    assert inference.stddev.shape == (2, 3)

def test_value_head_scalar_inference():
    arch_config = ArchitectureConfig({"noisy_sigma": 0.0})
    repr = ScalarRepresentation()
    head = ValueHead(arch_config, input_shape=(16,), representation=repr)
    
    x = torch.randn(2, 16)
    _, _, inference = head(x)
    
    assert torch.is_tensor(inference)
    assert inference.shape == (2,)

def test_reward_head_categorical_inference():
    arch_config = ArchitectureConfig({"noisy_sigma": 0.0})
    # C51-style reward
    repr = CategoricalRepresentation(vmin=-10, vmax=10, bins=21)
    head = RewardHead(arch_config, input_shape=(16,), representation=repr)
    
    x = torch.randn(2, 16)
    _, _, inference = head(x)
    
    assert torch.is_tensor(inference)
    assert inference.shape == (2,)
    # Should be the expected value
    assert not torch.isnan(inference).any()

def test_q_head_categorical_inference():
    arch_config = ArchitectureConfig({"noisy_sigma": 0.0})
    repr = CategoricalRepresentation(vmin=-10, vmax=10, bins=21)
    head = QHead(arch_config, input_shape=(16,), representation=repr, hidden_widths=[8], num_actions=4)
    
    x = torch.randn(2, 16)
    _, _, inference = head(x)
    
    # QHead returns a Distribution for search
    assert isinstance(inference, distributions.Categorical)
    assert inference.logits.shape == (2, 4, 21)

def test_identity_representation_inference():
    repr = IdentityRepresentation(num_features=128)
    logits = torch.randn(2, 128)
    inference = repr.to_inference(logits)
    
    assert torch.is_tensor(inference)
    assert inference.shape == (2, 128)
    assert torch.allclose(inference, logits)
