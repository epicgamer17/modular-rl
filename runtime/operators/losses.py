import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.state import GradientRegistry
from runtime.values import MissingInput
from runtime.executor import register_operator


def op_mse_loss(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    pred = inputs.get("pred")
    target = inputs.get("target")
    if pred is None:
        return MissingInput("pred")
    if target is None:
        return MissingInput("target")
    return nn.functional.mse_loss(pred, target)


def op_weighted_sum(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    # weights is a dict of {input_name: weight}
    weights = node.params.get("weights", {})
    total = torch.tensor(
        0.0, device=next(iter(inputs.values())).device if inputs else None
    )
    for name, weight in weights.items():
        val = inputs.get(name)
        if val is not None:
            total = total + weight * val
    return total


def op_mean(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    val = inputs.get("input")
    if val is None:
        return MissingInput("input")
    return val.mean()


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

        # TODO: what is the reason for these two branches? can we collapse it into one?
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
    # TODO: should we fail fast or is this intended?
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
    # TODO: should we fail fast or is this intended?
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
        # TODO: will this silently skip learning if there is a key error?
        # should we crash instead?
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


def register_loss_operators():
    register_operator("MSELoss", op_mse_loss)
    register_operator("WeightedSum", op_weighted_sum)
    register_operator("Mean", op_mean)
    register_operator("Backward", op_backward)
    register_operator("GradBuffer", op_grad_buffer)
    register_operator("AccumulateGrad", op_accumulate_grad)
    register_operator("OptimizerStepEvery", op_optimizer_step_every)
