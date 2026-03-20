import pytest
import torch
import torch.nn.functional as F
import numpy as np
from agents.learner.losses import (
    LossPipeline, ValueLoss, PolicyLoss, RewardLoss, 
    QBootstrappingLoss, ConsistencyLoss,
    ToPlayLoss, RelativeToPlayLoss, ClippedSurrogateLoss
)
from agents.learner.losses.representations import (
    ScalarRepresentation, ClassificationRepresentation, 
    IdentityRepresentation, DiscreteSupportRepresentation, C51Representation
)

pytestmark = pytest.mark.unit

class MockConfig:
    def __init__(self, **kwargs):
        self.minibatch_size = kwargs.get('minibatch_size', 4)
        self.unroll_steps = kwargs.get('unroll_steps', 0)
        class Game:
            def __init__(self, num_actions): self.num_actions = num_actions
        self.game = Game(kwargs.get('num_actions', 4))
        self.atom_size = kwargs.get('atom_size', 1)
        self.policy_loss_factor = kwargs.get('policy_loss_factor', 1.0)
        self.reward_loss_factor = kwargs.get('reward_loss_factor', 1.0)
        self.value_loss_factor = kwargs.get('value_loss_factor', 1.0)
        self.consistency_loss_factor = kwargs.get('consistency_loss_factor', 1.0)
        self.policy_loss_function = kwargs.get('policy_loss_function', F.cross_entropy)
        self.loss_factor = kwargs.get('loss_factor', 1.0)
        self.cosine_similarity = kwargs.get('cosine_similarity', True)
        self.ppo_clip_param = kwargs.get('ppo_clip_param', 0.2)
        
        for k, v in kwargs.items():
            setattr(self, k, v)

@pytest.fixture
def mock_agent_network():
    class MockNetwork(torch.nn.Module):
        def project(self, x, grad=True):
            return x.clone()
        
        def learner_inference(self, batch):
            return {"q_values": torch.zeros(1, 1, 4), "q_logits": torch.zeros(1, 1, 4, 21)}

    return MockNetwork()

def test_value_loss_vectorization():
    config = MockConfig()
    repr = ScalarRepresentation()
    loss_module = ValueLoss(config, torch.device("cpu"), representation=repr)

    B, T = 4, 1
    predictions = {"values": torch.randn(B, T, 1)}
    targets = {"values": torch.randn(B, T), "value_mask": torch.ones(B, T, dtype=torch.bool)}

    loss = loss_module.compute_loss(predictions, targets, {})
    assert loss.shape == (B, T)
    assert not torch.isnan(loss).any()

def test_q_bootstrapping_loss_vectorization():
    config = MockConfig(atom_size=21)
    repr = DiscreteSupportRepresentation(vmin=0, vmax=10, bins=21)
    loss_module = QBootstrappingLoss(config, torch.device("cpu"), representation=repr)

    B, T = 4, 1
    num_actions = 4
    predictions = {"q_logits": torch.randn(B, T, num_actions, 21)}
    
    targets = {
        "actions": torch.randint(0, num_actions, (B, T)),
        "q_logits": torch.softmax(torch.randn(B, T, 21), dim=-1),
        "value_mask": torch.ones(B, T, dtype=torch.bool)
    }

    loss = loss_module.compute_loss(predictions, targets, {})
    assert loss.shape == (B, T)
    assert not torch.isnan(loss).any()

def test_policy_loss_vectorization():
    config = MockConfig()
    repr = ClassificationRepresentation(num_classes=4)
    loss_module = PolicyLoss(config, torch.device("cpu"), representation=repr)

    B, T = 4, 1
    # PolicyLoss expects pred_key="policies"
    predictions = {"policies": torch.randn(B, T, 4)}
    targets = {
        "policies": torch.softmax(torch.randn(B, T, 4), dim=-1), 
        "policy_mask": torch.ones(B, T, dtype=torch.bool)
    }

    loss = loss_module.compute_loss(predictions, targets, {})
    assert loss.shape == (B, T)
    assert not torch.isnan(loss).any()

def test_consistency_loss_vectorization(mock_agent_network):
    config = MockConfig()
    loss_module = ConsistencyLoss(
        config, torch.device("cpu"), representation=None, agent_network=mock_agent_network
    )

    B, T = 4, 1
    predictions = {
        "consistency_logits": torch.randn(B, T, 128),
        "predictions_latent": torch.randn(B, T, 128)
    }
    targets = {
        "consistency_targets": torch.randn(B, T, 128),
        "targets_latent": torch.randn(B, T, 128),
        "consistency_mask": torch.ones(B, T, dtype=torch.bool)
    }

    loss = loss_module.compute_loss(predictions, targets, {"network": mock_agent_network})
    assert loss.shape == (B, T)
    assert not torch.isnan(loss).any()

def test_loss_pipeline_vectorization():
    config = MockConfig()

    repr_val = ScalarRepresentation()
    repr_pol = ClassificationRepresentation(num_classes=4)
    
    losses = [
        ValueLoss(config, torch.device("cpu"), representation=repr_val),
        PolicyLoss(config, torch.device("cpu"), representation=repr_pol)
    ]
    pipeline = LossPipeline(losses)

    B, T = 4, 1
    predictions = {
        "values": torch.randn(B, T, 1),
        "policies": torch.randn(B, T, 4)
    }
    targets = {
        "values": torch.randn(B, T),
        "policies": torch.softmax(torch.randn(B, T, 4), dim=-1),
        "value_mask": torch.ones(B, T, dtype=torch.bool),
        "policy_mask": torch.ones(B, T, dtype=torch.bool)
    }

    total_loss_dict, metrics, priorities = pipeline.run(predictions, targets, {})
    
    # Loss pipeline run returns total_loss_dict (per-optimizer)
    # Each scalar loss is in metrics
    assert "ValueLoss" in metrics
    assert "PolicyLoss" in metrics

def test_clipped_surrogate_loss():
    config = MockConfig()
    # ClippedSurrogateLoss(config, device, representation, clip_param, entropy_coefficient)
    loss_module = ClippedSurrogateLoss(
        config, torch.device("cpu"), representation=ClassificationRepresentation(4),
        clip_param=0.2, entropy_coefficient=0.01
    )

    B, T = 4, 1
    num_actions = 4
    
    predictions = {
        "policies": torch.randn(B, T, num_actions)
    }
    targets = {
        "actions": torch.randint(0, num_actions, (B, T)),
        "old_log_probs": torch.randn(B, T),
        "advantages": torch.randn(B, T),
        "policy_mask": torch.ones(B, T, dtype=torch.bool)
    }

    loss = loss_module.compute_loss(predictions, targets, {})
    assert loss.shape == (B, T)
