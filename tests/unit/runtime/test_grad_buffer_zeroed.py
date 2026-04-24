import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from runtime.context import ExecutionContext
from runtime.state import OptimizerState

pytestmark = pytest.mark.unit


def test_grad_buffer_zeroed():
    model = nn.Linear(4, 1)
    ctx = ExecutionContext()
    ctx.model_registry.register("online_q", model)

    opt_state = OptimizerState(optim.SGD(model.parameters(), lr=0.1))
    ctx.optimizer_registry.register("main_opt", opt_state)

    loss = model(torch.randn(3, 4)).pow(2).mean()
    loss.backward()
    ctx.gradient_registry.write(
        "online_q",
        torch.cat([param.grad.detach().reshape(-1) for param in model.parameters() if param.grad is not None]),
    )

    assert ctx.get_gradients("online_q") is not None
    assert any(param.grad is not None for param in model.parameters())

    opt_state.zero_grad(gradient_registry=ctx.gradient_registry, model_handle="online_q")

    assert ctx.get_gradients("online_q") is None
    assert all(param.grad is None for param in model.parameters())
