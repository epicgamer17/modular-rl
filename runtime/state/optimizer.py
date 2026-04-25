"""Optimization state: optimizer wrapper and gradient registry."""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn


class GradientRegistry:
    """
    A central registry for named gradient buffers.
    Stores detached flattened gradients keyed by model handle so backward,
    clipping, accumulation, and optimizer stepping can be decoupled.
    """

    def __init__(self):
        self._grads: Dict[str, Optional[torch.Tensor]] = {}
        self._accumulated: Dict[str, Optional[torch.Tensor]] = {}
        self._counts: Dict[str, int] = {}

    def register(self, name: str, grad: Optional[torch.Tensor] = None) -> None:
        """Registers a gradient slot under a specific handle."""
        self._grads[name] = self._clone(grad)
        self._accumulated[name] = None
        self._counts[name] = 0

    def get(self, name: str) -> Optional[torch.Tensor]:
        """Retrieves the current gradient buffer for a handle."""
        if name not in self._grads and name not in self._accumulated:
            raise KeyError(f"Gradient handle '{name}' not found in registry.")
        grad = self._accumulated.get(name)
        if grad is None:
            grad = self._grads.get(name)
        return None if grad is None else grad.clone()

    def get_current(self, name: str) -> Optional[torch.Tensor]:
        """Retrieves the most recent microbatch gradient buffer for a handle."""
        if name not in self._grads:
            raise KeyError(f"Gradient handle '{name}' not found in registry.")
        grad = self._grads[name]
        return None if grad is None else grad.clone()

    def write(self, name: str, grad: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Overwrites the current microbatch gradient slot with a detached clone."""
        cloned = self._clone(grad)
        self._grads[name] = cloned
        return None if cloned is None else cloned.clone()

    def accumulate(self, name: str, grad: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Accumulates a detached gradient into an existing slot."""
        incoming = self._clone(grad)
        if incoming is None:
            incoming = self._clone(self._grads.get(name))
        if incoming is None:
            return self.get(name) if name in self._grads or name in self._accumulated else None
        current = self._accumulated.get(name)
        if current is None:
            self._accumulated[name] = incoming
        else:
            self._accumulated[name] = current + incoming
        self._counts[name] = self._counts.get(name, 0) + 1
        return self._accumulated[name].clone()

    def clear(self, name: str) -> None:
        """Clears a gradient slot without removing the handle."""
        self._grads[name] = None
        self._accumulated[name] = None
        self._counts[name] = 0

    def clear_current(self, name: str) -> None:
        """Clears only the current microbatch gradient slot."""
        self._grads[name] = None

    def clip(self, name: str, max_norm: Optional[float]) -> float:
        """
        Clips a stored gradient buffer in-place and returns its pre-clip norm.
        """
        grad = self._accumulated.get(name)
        if grad is None:
            grad = self._grads.get(name)
        if grad is None:
            return 0.0
        grad_norm = torch.linalg.vector_norm(grad).item()
        if max_norm is None:
            return grad_norm
        if grad_norm > max_norm and grad_norm > 0.0:
            scale = max_norm / grad_norm
            if self._accumulated.get(name) is not None:
                self._accumulated[name] = grad * scale
            else:
                self._grads[name] = grad * scale
        return grad_norm

    def reduce(self, name: str, grads: List[torch.Tensor], op: str = "mean") -> Optional[torch.Tensor]:
        """Reduces a collection of gradients into a single stored buffer."""
        reduced_inputs = [g.detach().clone() for g in grads if g is not None]
        if not reduced_inputs:
            self._grads[name] = None
            return None
        stacked = torch.stack(reduced_inputs)
        if op == "sum":
            reduced = stacked.sum(dim=0)
        elif op == "mean":
            reduced = stacked.mean(dim=0)
        else:
            raise ValueError(f"Unsupported gradient reduce op '{op}'.")
        self._accumulated[name] = reduced
        self._counts[name] = len(reduced_inputs)
        return reduced.clone()

    def count(self, name: str) -> int:
        """Returns the number of microbatches accumulated for a handle."""
        return self._counts.get(name, 0)

    @staticmethod
    def flatten_model_grads(model: nn.Module) -> Optional[torch.Tensor]:
        """Flattens all non-None parameter gradients for a model."""
        grads = [p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return None
        return torch.cat(grads)

    @staticmethod
    def _clone(grad: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if grad is None:
            return None
        return grad.detach().clone()


class OptimizerState:
    """
    Stores optimizer-specific state (e.g., moments, step count).
    Guarantees the full optimization lifecycle: zero_grad, backward, clip, and step.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, grad_clip: Optional[float] = None):
        self.optimizer = optimizer
        self.grad_clip = grad_clip

    def get_state(self) -> Dict[str, Any]:
        """Returns the internal state of the optimizer."""
        return self.optimizer.state_dict()

    def set_state(self, state_dict: Dict[str, Any]) -> None:
        """Sets the internal state of the optimizer."""
        self.optimizer.load_state_dict(state_dict)

    def zero_grad(
        self,
        gradient_registry: Optional[GradientRegistry] = None,
        model_handle: Optional[str] = None,
        clear_registry: bool = True,
    ) -> None:
        """Clears the gradients of all optimized parameters."""
        self.optimizer.zero_grad(set_to_none=True)
        if gradient_registry is not None and model_handle is not None and clear_registry:
            gradient_registry.clear(model_handle)

    def backward(self, loss: torch.Tensor) -> None:
        """Computes the gradients of the loss w.r.t. parameters."""
        loss.backward()

    def clip_grad(self) -> float:
        """Clips the gradients and returns the gradient norm."""
        params = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    params.append(p)

        if not params:
            return 0.0

        max_norm = self.grad_clip if self.grad_clip is not None else float("inf")
        return torch.nn.utils.clip_grad_norm_(params, max_norm).item()

    def load_grad_buffer(self, model: nn.Module, grad_buffer: Optional[torch.Tensor]) -> None:
        """Loads a flattened gradient buffer into model parameters."""
        self.optimizer.zero_grad(set_to_none=True)
        if grad_buffer is None:
            return

        flat = grad_buffer.detach()
        offset = 0
        for param in model.parameters():
            numel = param.numel()
            grad_slice = flat[offset : offset + numel]
            if grad_slice.numel() != numel:
                raise ValueError("Gradient buffer shape does not match model parameters.")
            param.grad = grad_slice.view_as(param).to(device=param.device, dtype=param.dtype).clone()
            offset += numel

        if offset != flat.numel():
            raise ValueError("Gradient buffer has extra elements beyond model parameters.")

    def step_from_grad_buffer(
        self, model: nn.Module, grad_buffer: Optional[torch.Tensor]
    ) -> Dict[str, float]:
        """Applies a precomputed gradient buffer to parameters and steps the optimizer."""
        self.load_grad_buffer(model, grad_buffer)
        grad_norm = self.clip_grad()
        self.optimizer.step()
        lr = self.optimizer.param_groups[0]["lr"]
        return {"loss": 0.0, "grad_norm": grad_norm, "lr": lr}

    def step(self, loss: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Performs a single optimization step.
        If loss is provided, it performs the full monolithic cycle (zero_grad, backward, clip, step).
        If loss is NOT provided, it assumes zero_grad, backward, and clip have already been called.

        Returns:
            Dictionary with grad_norm, lr, and loss.
        """
        if loss is not None:
            self.zero_grad()
            loss_value = loss.item()
            self.backward(loss)
            grad_norm = self.clip_grad()
        else:
            loss_value = 0.0
            grad_norm = 0.0  # Should have been returned by clip_grad() call

        self.optimizer.step()

        # Get current learning rate (from first param group)
        lr = self.optimizer.param_groups[0]["lr"]

        return {"loss": loss_value, "grad_norm": grad_norm, "lr": lr}
