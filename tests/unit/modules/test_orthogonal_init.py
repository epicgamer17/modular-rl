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


def test_orthogonal_initialization_ppo_fidelity():
    """
    Verify high-fidelity PPO initialization:
    - Hidden (Backbone/Neck): sqrt(2) gain, 0 bias
    - Policy Output: 0.01 gain, 0 bias
    - Value Output: 1.0 gain, 0 bias
    """
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

    # We want to support a granular initialization that aligns with Baselines
    # Proposed API: dictionary of gains
    network.initialize({
        "backbone": math.sqrt(2),
        "policy_head": 0.01,
        "value_head": 1.0,
    })

    # 1. Backbone (Hidden Layers)
    # MLPBackbone has a LinearStack. LinearStack has Sequential(LinearBlock, norm)
    backbone_linear = backbone.stack._layers[0][0]  # LinearBlock
    backbone_weight = _get_weight(backbone_linear)
    assert _is_orthogonal(backbone_weight, gain=math.sqrt(2)), (
        f"Backbone should have sqrt(2) gain, got approx {_is_orthogonal(backbone_weight, gain=math.sqrt(2))}"
    )
    assert torch.allclose(backbone_linear.layer.bias, torch.zeros_like(backbone_linear.layer.bias)), (
        "Backbone bias should be 0"
    )

    # 2. Policy Head Output
    policy_weight = _get_weight(policy_head.output_layer)
    assert _is_orthogonal(policy_weight, gain=0.01), (
        f"Policy output should have 0.01 gain"
    )
    assert torch.allclose(policy_head.output_layer.layer.bias, torch.zeros_like(policy_head.output_layer.layer.bias)), (
        "Policy output bias should be 0"
    )

    # 3. Value Head Output
    value_weight = _get_weight(value_head.output_layer)
    assert _is_orthogonal(value_weight, gain=1.0), (
        "Value output should have 1.0 gain"
    )
    assert torch.allclose(value_head.output_layer.layer.bias, torch.zeros_like(value_head.output_layer.layer.bias)), (
        "Value output bias should be 0"
    )


if __name__ == "__main__":
    from agents.learner.losses.representations import ClassificationRepresentation, ScalarRepresentation
    from modules.backbones.mlp import MLPBackbone
    from modules.agent_nets.modular import ModularAgentNetwork
    from modules.heads.policy import PolicyHead
    from modules.heads.value import ValueHead
    import math

    test_orthogonal_initialization_ppo_fidelity()
    print("Test passed!")
