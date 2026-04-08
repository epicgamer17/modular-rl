import torch
import torch.nn as nn
import pytest
from learner.losses.aggregator import LossAggregator
from learner.core import Blackboard
from learner.losses.shape_validator import ShapeValidator

pytestmark = pytest.mark.unit

class MockLossModule(nn.Module):
    def __init__(self, name, value, device=torch.device("cpu")):
        super().__init__()
        self.name = name
        self.value = value
        self.device = device
        self.optimizer_name = "default"
    
    def compute_loss(self, predictions, targets):
        # Return [B, T] tensor of losses
        # Targets are [B, T, ...]
        key = next(iter(targets.keys()))
        B, T = targets[key].shape[:2]
        return torch.ones(B, T, device=self.device) * self.value, {}

    def get_mask(self, targets):
        # Return [B, T] boolean mask
        key = next(iter(targets.keys()))
        B, T = targets[key].shape[:2]
        if "masks" in targets:
            return targets["masks"]
        return torch.ones(B, T, device=self.device, dtype=torch.bool)

class MockPriorityComputer:
    def compute(self, all_elementwise_losses, predictions, targets):
        return torch.tensor([1.0, 2.0])

def test_loss_aggregator_aggregation():
    """Verify LossAggregator aggregates multiple losses and writes to meta."""
    m1 = MockLossModule("Loss1", 1.0)
    m2 = MockLossModule("Loss2", 2.0)
    
    # 1. Setup with valid args (B=2, K=1 -> T=2)
    agg = LossAggregator(modules=[m1, m2], minibatch_size=2, unroll_steps=1)
    
    bb = Blackboard(batch={})
    # T=2, B=2
    bb.targets["dummy"] = torch.zeros(2, 2)
    bb.targets["masks"] = torch.ones(2, 2, dtype=torch.bool)
    # Mock infrastructure (in meta as agreed)
    bb.meta["weights"] = torch.ones(2)
    bb.meta["gradient_scales"] = torch.ones((1, 2))
    
    agg.execute(bb)
    
    # Each loss returns 1.0 and 2.0 per (B,T) element.
    # Aggregator means over B and T.
    # Mean(Loss1) = 1.0, Mean(Loss2) = 2.0. Total = 3.0
    assert "Loss1" in bb.meta
    assert "Loss2" in bb.meta
    assert "loss" in bb.meta
    assert bb.meta["loss"] == 3.0

def test_loss_aggregator_priorities():
    """Verify LossAggregator computes and writes priorities to meta."""
    m1 = MockLossModule("Loss1", 1.0)
    pc = MockPriorityComputer()
    
    agg = LossAggregator(modules=[m1], priority_computer=pc, minibatch_size=2, unroll_steps=1)
    
    bb = Blackboard(batch={})
    bb.targets["dummy"] = torch.zeros(2, 2)
    bb.targets["masks"] = torch.ones(2, 2, dtype=torch.bool)
    # Mock infrastructure
    bb.meta["weights"] = torch.ones(2)
    bb.meta["gradient_scales"] = torch.ones((1, 2))
    
    agg.execute(bb)
    
    assert "priorities" in bb.meta
    torch.testing.assert_close(bb.meta["priorities"], torch.tensor([1.0, 2.0]))
