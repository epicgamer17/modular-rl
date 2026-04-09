import torch
import torch.nn.functional as F
import pytest
from components.math import LossAggregatorComponent
from components.memory import PriorityUpdateComponent
from components.math import ValueLoss
from components.math import QBootstrappingLoss
from core import Blackboard
from core import PipelineComponent

pytestmark = pytest.mark.unit

class MockPriorityComputer:
    def compute(self, elementwise_losses, predictions, targets):
        # Just sum up all elementwise losses across keys and return mean per batch
        total = None
        for v in elementwise_losses.values():
            if total is None:
                total = v
            else:
                total = total + v
        return total.mean(dim=1) # [B]

def test_loss_aggregator_component():
    """Verify LossAggregatorComponent weighted summation."""
    weights = {"loss_a": 1.0, "loss_b": 0.5}
    agg = LossAggregatorComponent(loss_weights=weights)
    bb = Blackboard()
    
    # Mock some existing scalar losses
    bb.losses["loss_a"] = torch.tensor(1.0)
    bb.losses["loss_b"] = torch.tensor(2.0)
    
    agg.execute(bb)
    
    assert "total_loss" in bb.losses
    assert "default" in bb.losses["total_loss"]
    # 1.0 * 1.0 + 0.5 * 2.0 = 2.0
    assert bb.losses["total_loss"]["default"].item() == pytest.approx(2.0)



def test_loss_aggregator_disjoint_optimizers():
    """Verify LossAggregatorComponent handles different optimizer keys."""
    agg_p = LossAggregatorComponent(loss_weights={"policy": 1.0}, optimizer_key="actor")
    agg_v = LossAggregatorComponent(loss_weights={"value": 1.0}, optimizer_key="critic")
    
    bb = Blackboard()
    bb.losses["policy"] = torch.tensor(0.5)
    bb.losses["value"] = torch.tensor(0.8)
    
    agg_p.execute(bb)
    agg_v.execute(bb)
    
    assert bb.losses["total_loss"]["actor"].item() == pytest.approx(0.5)
    assert bb.losses["total_loss"]["critic"].item() == pytest.approx(0.8)



def test_value_loss_component_contracts():
    """Verify ValueLoss writes scalar and elementwise losses."""
    loss = ValueLoss(name="my_val_loss")
    bb = Blackboard()
    
    # B=2, T=3
    bb.predictions["values"] = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
    bb.targets["values"] = torch.tensor([[[1.1], [2.1], [3.1]], [[4.1], [5.1], [6.1]]])
    bb.targets["value_mask"] = torch.ones((2, 3), dtype=torch.bool)
    
    loss.execute(bb)
    
    assert "my_val_loss" in bb.losses
    assert "my_val_loss" in bb.meta
    assert "elementwise_losses" in bb.meta
    assert "my_val_loss" in bb.meta["elementwise_losses"]
    
    # Check shape of elementwise loss [B, T]
    assert bb.meta["elementwise_losses"]["my_val_loss"].shape == (2, 3)
    
    # MSE is (0.1)**2 = 0.01
    expected_scalar = 0.01
    assert pytest.approx(bb.losses["my_val_loss"].item(), rel=1e-3) == expected_scalar

def test_priority_update_component():
    """Verify PriorityUpdateComponent calls computer and writes to meta."""
    pc = MockPriorityComputer()
    comp = PriorityUpdateComponent(priority_computer=pc)
    bb = Blackboard()
    
    # B=2, T=2
    bb.meta["elementwise_losses"] = {
        "l1": torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    }
    
    comp.execute(bb)
    
    assert "priorities" in bb.meta
    # Mean of [1, 1] is 1, mean of [2, 2] is 2
    torch.testing.assert_close(bb.meta["priorities"], torch.tensor([1.0, 2.0]))

def test_q_bootstrapping_loss_contracts():
    """Verify QBootstrappingLoss selects actions and computes loss."""
    loss = QBootstrappingLoss(name="q_loss")
    bb = Blackboard()
    
    # B=1, T=2, Actions=2
    bb.predictions["q_values"] = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    bb.targets["actions"] = torch.tensor([[0, 1]])
    bb.targets["values"] = torch.tensor([[0.5, 1.5]])
    bb.targets["value_mask"] = torch.ones((1, 2), dtype=torch.bool)
    
    loss.execute(bb)
    
    assert "q_loss" in bb.losses
    assert bb.losses["q_loss"].item() == 0.25
    assert bb.meta["elementwise_losses"]["q_loss"].shape == (1, 2)

def test_clipped_value_loss_contracts():
    """Verify ClippedValueLoss logic."""
    from components.math import ClippedValueLoss
    loss = ClippedValueLoss(clip_param=0.2, name="clipped_v")
    bb = Blackboard()
    
    # V=1.0, OldV=0.9, Target=1.2. 
    # Unclipped error: (1.0 - 1.2)^2 = 0.04
    # Clipped value: 0.9 + clamp(0.1, -0.2, 0.2) = 1.0. 
    # Clipped error: (1.0 - 1.2)^2 = 0.04
    # Max: 0.04
    
    bb.predictions["values"] = torch.tensor([[1.0]])
    bb.targets["returns"] = torch.tensor([[1.2]])
    bb.targets["values"] = torch.tensor([[0.9]])
    bb.targets["value_mask"] = torch.ones((1, 1), dtype=torch.bool)
    
    loss.execute(bb)
    assert pytest.approx(bb.losses["clipped_v"].item()) == 0.04
    
    # Test clipping: V=1.5, OldV=1.0, Target=2.0 (clip=0.2)
    # Unclipped: (1.5 - 2.0)^2 = 0.25
    # Clipped: 1.0 + clamp(0.5, -0.2, 0.2) = 1.2. 
    # Error: (1.2 - 2.0)^2 = 0.64
    # Max(0.25, 0.64) = 0.64
    bb.predictions["values"] = torch.tensor([[1.5]])
    bb.targets["returns"] = torch.tensor([[2.0]])
    bb.targets["values"] = torch.tensor([[1.0]])
    
    loss.execute(bb)
    assert pytest.approx(bb.losses["clipped_v"].item()) == 0.64

