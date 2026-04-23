"""
Operators for exploration policies.
Ensures deterministic exploration using ExecutionContext RNG.
"""

from typing import Dict, Any, Optional
import torch
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.values import MissingInput


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

    # TODO: a little messy here with multiple fallbacks, clean this up somehow, maybe a better input system that uses requires and provides.
    # Fetch epsilon from port, fallback to node params
    epsilon = inputs.get("epsilon", node.params.get("epsilon", 0.0))
    act_dim = node.params["act_dim"]

    # Ensure context exists and use its RNG for deterministic exploration
    # If no context is provided, fallback to standard random to avoid crash,
    # but this should be caught in tests.
    rng = context.rng if context else None

    if rng and rng.random() < epsilon:
        return rng.randint(0, act_dim - 1)

    # Greedy choice
    return torch.argmax(q_values).item()
