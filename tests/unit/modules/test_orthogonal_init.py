import math
import pytest
import torch
import torch.nn as nn

from modules.agent_nets.modular import ModularAgentNetwork
from modules.heads.policy import PolicyHead
from modules.heads.value import ValueHead
from modules.backbones.mlp import MLPBackbone
from agents.learner.losses.representations import ClassificationRepresentation, ScalarRepresentation

pytestmark = pytest.mark.unit


def _get_weight(module):
    """Extract the raw nn.Linear weight from a LinearBlock or NoisyLinear wrapper."""
    if hasattr(module, "layer") and isinstance(module.layer, nn.Linear):
        return module.layer.weight  # LinearBlock wraps nn.Linear as .layer
    if hasattr(module, "weight"):
        return module.weight
    raise AttributeError(f"Cannot find weight on {type(module).__name__}")


def _is_orthogonal(weight: torch.Tensor, gain: float = 1.0, atol: float = 0.05) -> bool:
    """Check if a weight matrix was initialized with orthogonal init at the given gain."""
    if weight.ndim < 2:
        return True
    w = weight.data
    if w.shape[0] > w.shape[1]:
        product = w.T @ w
    else:
        product = w @ w.T
    expected = (gain ** 2) * torch.eye(product.shape[0])
    return torch.allclose(product, expected, atol=atol)


def test_orthogonal_initialization_defaults():
    """Verify that AgentNetwork uses orthogonal initialization by default with gain=math.sqrt(2)."""
    torch.manual_seed(42)
    input_shape = (4,)
    num_actions = 2

    backbone = MLPBackbone(input_shape=input_shape, widths=[32])
    pol_rep = ClassificationRepresentation(num_actions)
    val_rep = ScalarRepresentation()
    policy_head = PolicyHead(input_shape=backbone.output_shape, representation=pol_rep)
    value_head = ValueHead(input_shape=backbone.output_shape, representation=val_rep)

    network = ModularAgentNetwork(
        input_shape=input_shape,
        num_actions=num_actions,
        components={
            "backbone": backbone,
            "policy_head": policy_head,
            "value_head": value_head,
        },
    )

    # Apply orthogonal init (nn.init.orthogonal_ with default gain=1.0)
    network.initialize("orthogonal")

    # Backbone first layer: LinearStack._layers[0] is Sequential(LinearBlock, norm)
    backbone_linear = backbone.stack._layers[0][0]  # LinearBlock
    backbone_weight = _get_weight(backbone_linear)
    assert _is_orthogonal(backbone_weight, gain=1.0), (
        "Backbone should be orthogonally initialized"
    )

    # Policy head output layer
    policy_weight = _get_weight(policy_head.output_layer)
    assert _is_orthogonal(policy_weight, gain=1.0), (
        "Policy output should be orthogonally initialized"
    )


def test_orthogonal_initialization_custom_gain():
    """Verify that specifying a global gain in initialize affects non-component layers."""
    torch.manual_seed(42)
    input_shape = (4,)
    num_actions = 2

    backbone = MLPBackbone(input_shape=input_shape, widths=[32])
    pol_rep = ClassificationRepresentation(num_actions)
    policy_head = PolicyHead(input_shape=backbone.output_shape, representation=pol_rep)

    network = ModularAgentNetwork(
        input_shape=input_shape,
        num_actions=num_actions,
        components={
            "backbone": backbone,
            "policy_head": policy_head,
        },
    )

    # Current initialize() uses nn.init.orthogonal_ with default gain=1.0
    network.initialize("orthogonal")

    backbone_linear = backbone.stack._layers[0][0]
    backbone_weight = _get_weight(backbone_linear)
    assert _is_orthogonal(backbone_weight, gain=1.0), (
        "Backbone should have orthogonal init with default gain"
    )


def test_value_head_orthogonal_init():
    """Verify ValueHead uses orthogonal init with gain 1.0."""
    torch.manual_seed(42)
    input_shape = (32,)
    val_rep = ScalarRepresentation()

    head = ValueHead(input_shape=input_shape, representation=val_rep)

    # Apply orthogonal init to the head
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    head.apply(init_weights)

    # output_layer is a LinearBlock wrapping nn.Linear
    output_weight = _get_weight(head.output_layer)
    assert _is_orthogonal(output_weight, gain=1.0), (
        "Value head output layer should have gain 1.0"
    )
