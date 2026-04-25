import torch
import time
from typing import Dict, Any, Callable, Optional, List
from core.graph import Node, NodeId
from runtime.refs import Value
from runtime.signals import NoOp, Skipped, MissingInput
from runtime.registry import register_spec

# Global operator registry mapping node_type -> execution function
# def run(node: Node, inputs: Dict[NodeId, Any]) -> Any
OPERATOR_REGISTRY: Dict[
    str, Callable[[Node, Dict[NodeId, Any], Any], Any]
] = {}


def _validate_output(node: Node, output: Any) -> None:
    """Internal runtime check for node output validity."""
    if output is None:
        raise RuntimeError(
            f"Node '{node.node_id}' of type '{node.node_type}' returned raw None."
        )

    data = output.data if isinstance(output, Value) else output
    if isinstance(output, (NoOp, Skipped, MissingInput)):
        return

    # Check for non-finite values
    if isinstance(data, torch.Tensor):
        if not torch.isfinite(data).all():
            raise RuntimeError(f"Node '{node.node_id}' produced non-finite values.")

        # Loss scalar enforcement
        if "Loss" in node.node_type and data.numel() > 1:
            raise RuntimeError(
                f"Node '{node.node_id}' of type '{node.node_type}' produced non-scalar output for loss."
            )

    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                if not torch.isfinite(v).all():
                    raise RuntimeError(
                        f"Node '{node.node_id}' produced non-finite values in output dictionary."
                    )

                # Action enforcement
                if k == "action" and v.dtype in [
                    torch.int64,
                    torch.int32,
                    torch.int,
                    torch.long,
                ]:
                    if (v < 0).any():
                        raise RuntimeError(
                            f"Node '{node.node_id}' produced negative discrete action."
                        )


def ValidatedOperator(op_func):
    """Decorator that applies runtime assertions to an operator's output."""

    def wrapper(
        node: Node, inputs: Dict[NodeId, Any], context: Any
    ) -> Any:
        output = op_func(node, inputs, context)
        _validate_output(node, output)
        return output

    return wrapper


def register_operator(
    node_type: str,
    func: Callable[[Node, Dict[NodeId, Any], Any], Any],
    spec: Optional[Any] = None,
):
    """Registers an execution function and optional specification for a node type."""
    # Wrap all registered operators with validation logic
    OPERATOR_REGISTRY[node_type] = ValidatedOperator(func)
    if spec:
        register_spec(node_type, spec)
