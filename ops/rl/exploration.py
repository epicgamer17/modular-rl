import torch
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.signals import MissingInput

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

def op_linear_decay(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> float:
    """Linear decay of a value over time."""
    clock = inputs.get("clock", node.params.get("clock", 0))
    start_val = node.params.get("start_val", 1.0)
    end_val = node.params.get("end_val", 0.1)
    total_steps = node.params.get("total_steps", 1000)
    
    # Simple linear decay formula
    if clock >= total_steps:
        return end_val
    return start_val - (start_val - end_val) * (clock / total_steps)
