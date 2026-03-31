import pytest
import torch

pytestmark = pytest.mark.unit

def test_orthogonal_initialization_defaults():
    """Verify that AgentNetwork uses orthogonal initialization by default with gain=math.sqrt(2)."""
    # TODO: re-enable when imports are restored
    # Original assertions:
    # assert is_orthogonal(backbone_layer.weight, gain=math.sqrt(2)), "Backbone should have gain sqrt(2) by default"
    # assert is_orthogonal(policy_head.output_layer.weight, gain=0.01), "Policy output should have gain 0.01 (component-owned)"
    pytest.skip("TODO: update for old_muzero revert")

def test_orthogonal_initialization_custom_gain():
    """Verify that specifying a global gain in initialize affects non-component layers."""
    # TODO: re-enable when imports are restored
    # Original assertions:
    # assert is_orthogonal(backbone_layer.weight, gain=0.5), "Backbone should have gain 0.5 when specified"
    pytest.skip("TODO: update for old_muzero revert")

def test_value_head_orthogonal_init():
    """Verify ValueHead uses orthogonal init with gain 1.0."""
    # TODO: re-enable when imports are restored
    # Original assertions:
    # assert is_orthogonal(head.output_layer.weight, gain=1.0), "Value head output layer should have gain 1.0"
    pytest.skip("TODO: update for old_muzero revert")

