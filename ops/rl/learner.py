import torch
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.state import GradientRegistry
from runtime.signals import MissingInput


def op_optimizer_step(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
):
    """Generic optimizer step. Resolves the optimizer via handle and steps it on the loss."""
    loss = inputs.get("loss")
    if loss is None:
        return MissingInput("loss")

    opt_handle = node.params.get("optimizer_handle", "main_opt")
    if context is None:
        raise RuntimeError("Optimizer node requires an ExecutionContext.")
    opt_state = context.get_optimizer(opt_handle)
    opt_state.step(loss)
    return loss.item() if hasattr(loss, "item") else float(loss)


def op_backward(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> bool:
    """Computes gradients for the given loss."""
    loss = inputs.get("loss")
    if loss is None:
        return MissingInput("loss")

    model_handle = node.params.get("model_handle")
    opt_handle = node.params.get("optimizer_handle", "main_opt")
    if context and model_handle:
        model = context.get_model(model_handle)
        try:
            opt_state = context.get_optimizer(opt_handle)
        except KeyError:
            opt_state = None

        if opt_state is not None:
            opt_state.zero_grad(
                gradient_registry=context.gradient_registry,
                model_handle=model_handle,
                clear_registry=False,
            )
            opt_state.backward(loss)
        else:
            for param in model.parameters():
                param.grad = None
            context.gradient_registry.clear_current(model_handle)
            loss.backward()

        flat_grads = GradientRegistry.flatten_model_grads(model)
        context.gradient_registry.write(model_handle, flat_grads)
    else:
        loss.backward()

    return True

def op_grad_buffer(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """Retrieves gradients for a specific model."""
    handle = node.params.get("model_handle")
    if not handle:
        return None
    if context:
        return context.get_gradients(handle)
    return None

def op_accumulate_grad(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> Dict[str, Any]:
    """Accumulates the current microbatch gradient into a persistent buffer."""
    handle = node.params.get("model_handle")
    if not handle or context is None:
        return {"grads": None, "count": 0, "ready": False}

    grads = context.gradient_registry.accumulate(handle, None)
    count = context.gradient_registry.count(handle)
    k = max(1, int(node.params.get("k", 1)))
    return {"grads": grads, "count": count, "ready": count > 0 and count % k == 0}

def op_optimizer_step_every(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> Dict[str, Any]:
    """Steps the optimizer only when the accumulated microbatch count reaches k."""
    handle = node.params.get("model_handle")
    optimizer_handle = node.params.get("optimizer_handle", "main_opt")
    k = max(1, int(node.params.get("k", 1)))
    if not handle or context is None:
        return {"stepped": False, "count": 0, "loss": 0.0, "grad_norm": 0.0, "lr": 0.0}

    count = context.gradient_registry.count(handle)
    if count == 0 or count % k != 0:
        lr = 0.0
        try:
            lr = context.get_optimizer(optimizer_handle).optimizer.param_groups[0]["lr"]
        except KeyError:
            pass
        return {
            "stepped": False,
            "count": count,
            "loss": 0.0,
            "grad_norm": 0.0,
            "lr": lr,
        }

    model = context.get_model(handle)
    opt_state = context.get_optimizer(optimizer_handle)
    grads = context.get_gradients(handle)
    step_results = opt_state.step_from_grad_buffer(model, grads)
    context.gradient_registry.clear(handle)
    return {"stepped": True, "count": count, **step_results}
