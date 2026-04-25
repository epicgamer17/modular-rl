import torch
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.signals import MissingInput
from runtime.registry import OperatorSpec, PortSpec, Scalar, SingleQ

EXPLORATION_SPEC = OperatorSpec.create(
    name="Exploration",
    inputs={
        "q_values": SingleQ,
        "epsilon": PortSpec(spec=Scalar("float32"), required=False),
    },
    outputs={"action": Scalar("int64")},
    pure=False,
    deterministic=False,
    allowed_contexts={"actor"},
    math_category="distribution"
)

def op_epsilon_greedy(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> Any:
    """
    Performs epsilon-greedy exploration.

    Inputs:
        q_values: Tensor of Q-values for each action.

    Parameters:
        epsilon: Probability of choosing a random action.
        act_dim: Number of possible actions.

    Returns:
        The selected action (int).
    """
    q_values = inputs.get("q_values")
    if q_values is None:
        return MissingInput("q_values")

    # Fetch epsilon from port, fallback to node params
    epsilon = inputs.get("epsilon", node.params.get("epsilon", 0.0))
    act_dim = node.params["act_dim"]

    # Ensure context exists and use its RNG for deterministic exploration
    rng = context.rng if context else None

    if rng and rng.random() < epsilon:
        return rng.randint(0, act_dim - 1)

    # Greedy choice
    return torch.argmax(q_values).item()


