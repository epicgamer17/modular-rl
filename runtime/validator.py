"""
Runtime validation for operator outputs.
Ensures that all computations produce valid, finite data and follow RL contracts.
"""

from typing import Any
import torch
from core.graph import Node
from runtime.values import Value, NoOp, Skipped, MissingInput


def validate_operator_output(node: Node, output: Any) -> None:
    """
    Performs runtime assertions on operator outputs.

    Checks:
    - No raw None allowed (use NoOp() or Value(None)).
    - Tensors must be finite (no NaN or Inf).
    - Losses must be scalars.
    - Actions must be valid (non-negative if discrete).

    Args:
        node: The graph Node that produced the output.
        output: The result returned by the operator function.

    Raises:
        RuntimeError: If any validation rule is violated.
    """
    if output is None:
        raise RuntimeError(
            f"Operator for node '{node.node_id}' ({node.node_type}) returned raw None. "
            "Operators must return a RuntimeValue (e.g., Value, NoOp, or Skipped)."
        )

    # Unwrap if it's a Value object
    data = output.data if isinstance(output, Value) else output

    # Skip deep checks for control values
    if isinstance(output, (NoOp, Skipped, MissingInput)):
        return

    # Handle batched/schema outputs (dictionaries)
    if isinstance(data, dict):
        for key, val in data.items():
            _check_single_value(node, val, context=key)
    else:
        _check_single_value(node, data)


def _check_single_value(node: Node, value: Any, context: str = "") -> None:
    """Checks a single data item for validity."""
    ctx_msg = f" field '{context}'" if context else ""

    if isinstance(value, torch.Tensor):
        # 1. NaN / Infinite check
        if not torch.isfinite(value).all():
            raise RuntimeError(
                f"Node '{node.node_id}'{ctx_msg} produced non-finite values (NaN/Inf). "
                "This usually indicates exploding gradients or division by zero."
            )

        # 2. Scalar Loss check
        is_loss = (
            "loss" in node.node_type.lower()
            or "loss" in node.tags
            or context.lower() == "loss"
        )
        if is_loss:
            if value.numel() != 1:
                raise RuntimeError(
                    f"Loss node '{node.node_id}'{ctx_msg} produced non-scalar output "
                    f"(shape {list(value.shape)}). Expected a single scalar value."
                )

        # 3. Action bounds check (basic)
        if "action" in context.lower() or "action" in node.node_type.lower():
            if value.dtype in [torch.int32, torch.int64]:
                if (value < 0).any():
                    raise RuntimeError(
                        f"Node '{node.node_id}'{ctx_msg} produced negative discrete action. "
                        "Discrete actions must be non-negative integers."
                    )
