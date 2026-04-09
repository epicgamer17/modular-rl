import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock
from components.targets import TDTargetComponent, DistributionalTargetComponent, PassThroughTargetComponent, UniversalInfrastructureComponent
from core import Blackboard

pytestmark = pytest.mark.unit

class MockNetwork(nn.Module):
    def __init__(self, val=1.0):
        super().__init__()
        self.val = val
    def learner_inference(self, batch):
        # Return [B, T=1, num_actions] Q-values
        B = batch["observations"].shape[0]
        return {"q_values": torch.ones(B, 1, 2) * self.val}

def test_td_target_component_math():
    """Verify TDTargetComponent performs correct Bellman math."""
    target_net = MockNetwork(val=10.0)
    gamma = 0.9
    n_step = 1
    component = TDTargetComponent(target_net, gamma=gamma, n_step=n_step)
    
    # B=2 transitions
    batch = {
        "rewards": torch.tensor([1.0, 2.0]),
        "dones": torch.tensor([False, True]),
        "next_observations": torch.randn(2, 4),
        "actions": torch.tensor([0, 1])
    }
    bb = Blackboard(data=batch)
    
    component.execute(bb)
    
    # Target 0: 1.0 + 0.9 * 10.0 = 10.0
    # Target 1: 2.0 + 0.0 * 10.0 = 2.0
    expected = torch.tensor([[10.0], [2.0]])
    torch.testing.assert_close(bb.targets["values"], expected)

def test_td_target_n_step_logic():
    """Verify TDTargetComponent applies gamma^n discount correctly."""
    target_net = MockNetwork(val=10.0)
    gamma = 0.9
    n_step = 3
    # discount = 0.9^3 = 0.729
    component = TDTargetComponent(target_net, gamma=gamma, n_step=n_step)
    
    batch = {
        "rewards": torch.tensor([1.0]),
        "dones": torch.tensor([False]),
        "next_observations": torch.randn(1, 4),
        "actions": torch.tensor([0])
    }
    bb = Blackboard(data=batch)
    component.execute(bb)
    
    # expected = 1.0 + 0.729 * 10.0 = 8.29
    expected = torch.tensor([[8.29]])
    torch.testing.assert_close(bb.targets["values"], expected)

def test_distributional_target_component_math():
    """Verify DistributionalTargetComponent performs correct Bellman shift and projection."""
    class MockRepresentation:
        def __init__(self):
            self.support = torch.tensor([-1.0, 0.0, 1.0])
        def project_onto_grid(self, shifted_support, probabilities):
            # Just return shifted_support for verification in the test
            return shifted_support

    class MockDistNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.components = {"q_head": MagicMock()}
            self.components["q_head"].representation = MockRepresentation()
        def learner_inference(self, batch):
            B = batch["observations"].shape[0]
            # [B, T=1, num_actions, atom_size]
            return {
                "q_values": torch.zeros(B, 1, 2),
                "q_logits": torch.ones(B, 1, 2, 3) 
            }

    online_net = MockDistNetwork()
    target_net = MockDistNetwork()
    component = DistributionalTargetComponent(
        target_network=target_net,
        online_network=online_net,
        gamma=0.9,
        n_step=1
    )
    
    batch = {
        "rewards": torch.tensor([1.0]),
        "dones": torch.tensor([False]),
        "next_observations": torch.randn(1, 4),
        "actions": torch.tensor([0])
    }
    bb = Blackboard(data=batch)
    
    # We need to mock project_onto_grid to capture its arguments or check outputs
    # Since we set project_onto_grid to return shifted_support:
    # shifted_support = reward + gamma * support
    # = 1.0 + 0.9 * [-1, 0, 1] = [0.1, 1.0, 1.9]
    
    component.execute(bb)
    
    # Implementation produces [B, 1, Atoms] (no actions dimension, indexed by loss)
    expected_support = torch.tensor([[[0.1, 1.0, 1.9]]])
    torch.testing.assert_close(bb.targets["q_logits"], expected_support)

def test_pass_through_target_component():
    """Verify PassThroughTargetComponent filters and reshapes keys correctly."""
    component = PassThroughTargetComponent(keys_to_keep=["keep_me"])
    
    batch = {
        "keep_me": torch.tensor([1, 2, 3]),
        "ignore_me": torch.tensor([4, 5, 6])
    }
    bb = Blackboard(data=batch)
    
    component.execute(bb)
    
    assert "keep_me" in bb.targets
    assert "ignore_me" not in bb.targets
    # Should have added T=1 dimension
    assert bb.targets["keep_me"].shape == (3, 1)

def test_universal_infrastructure_component():
    """Verify UniversalInfrastructureComponent generates default masks and weights."""
    component = UniversalInfrastructureComponent()
    
    # Need some target to identify batch size
    bb = Blackboard(data={})
    bb.targets["some_val"] = torch.zeros(5, 1)
    
    component.execute(bb)
    
    assert "value_mask" in bb.targets
    assert bb.targets["value_mask"].shape == (5, 1)
    assert bb.targets["value_mask"].all()
    
    assert "weights" in bb.meta
    assert bb.meta["weights"].shape == (5,)
    assert bb.meta["weights"].all()
