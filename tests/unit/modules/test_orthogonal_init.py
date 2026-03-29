import pytest
import torch
import torch.nn as nn
from modules.models.agent_network import AgentNetwork
from modules.heads.policy import PolicyHead
from modules.heads.value import ValueHead
from modules.backbones.mlp import MLPBackbone

pytestmark = pytest.mark.unit

def is_orthogonal(tensor: torch.Tensor, gain: float = 1.0, tol: float = 1e-5) -> bool:
    """
    Check if a tensor's weights are orthogonal with a specific gain.
    For a matrix W (out, in), W @ W.T should be gain^2 * I (if out <= in).
    """
    if tensor.dim() < 2:
        return False
    
    # Flatten if it's a conv layer
    flat_w = tensor.view(tensor.size(0), -1)
    
    # We check rows or columns depending on which is smaller
    rows, cols = flat_w.shape
    if rows <= cols:
        # W @ W.T = gain^2 * I
        product = flat_w @ flat_w.t()
        expected = torch.eye(rows, device=tensor.device) * (gain ** 2)
    else:
        # W.T @ W = gain^2 * I
        product = flat_w.t() @ flat_w
        expected = torch.eye(cols, device=tensor.device) * (gain ** 2)
        
    return torch.allclose(product, expected, atol=tol)

def test_orthogonal_initialization_defaults():
    """Verify that AgentNetwork uses orthogonal initialization by default with gain=1.0."""
    torch.manual_seed(42)
    
    # Create simple components
    input_shape = (4,)
    num_actions = 2
    from agents.learner.losses.representations import ClassificationRepresentation
    rep = ClassificationRepresentation(num_classes=num_actions)
    
    def backbone_fn(input_shape):
        return MLPBackbone(input_shape=input_shape, widths=[8])
    
    net = AgentNetwork(
        input_shape=input_shape,
        num_actions=num_actions,
        representation_fn=lambda **kwargs: nn.Identity(),
        memory_core_fn=backbone_fn,
        head_fns={"policy": lambda **kwargs: PolicyHead(representation=rep, **kwargs)}
    )
    
    # Defaults to orthogonal with gain=1.0 for the backbone
    net.initialize()
    
    backbone_layer = next(m for m in net.components["memory_core"].modules() if isinstance(m, nn.Linear))
    assert is_orthogonal(backbone_layer.weight, gain=1.0), "Backbone should have gain 1.0 by default"
    
    policy_head = net.components["behavior_heads"]["policy"]
    assert is_orthogonal(policy_head.output_layer.weight, gain=0.01), "Policy output should have gain 0.01 (component-owned)"

def test_orthogonal_initialization_custom_gain():
    """Verify that specifying a global gain in initialize affects non-component layers."""
    torch.manual_seed(42)
    input_shape = (4,)
    backbone_fn = lambda input_shape: MLPBackbone(input_shape=input_shape, widths=[8])
    
    net = AgentNetwork(
        input_shape=input_shape,
        num_actions=2,
        memory_core_fn=backbone_fn,
    )
    
    # Use orthogonal with gain 0.5
    net.initialize(kernel_initializer="orthogonal", gain=0.5)
    
    backbone_layer = next(m for m in net.components["memory_core"].modules() if isinstance(m, nn.Linear))
    assert is_orthogonal(backbone_layer.weight, gain=0.5), "Backbone should have gain 0.5 when specified"

def test_value_head_orthogonal_init():
    """Verify ValueHead uses orthogonal init with gain 1.0."""
    from agents.learner.losses.representations import ScalarRepresentation
    rep = ScalarRepresentation()
    
    head = ValueHead(input_shape=(8,), representation=rep)
    head.init_weights()
    assert is_orthogonal(head.output_layer.weight, gain=1.0), "Value head output layer should have gain 1.0"

