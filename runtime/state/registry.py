"""Handle-based registries for runtime objects (models, buffers, optimizers, callables)."""

from typing import Callable, Dict, Any, TYPE_CHECKING
import torch.nn as nn

if TYPE_CHECKING:
    from runtime.state.optimizer import OptimizerState


class ModelRegistry:
    """
    A central registry for named PyTorch modules.
    Allows graph nodes to reference models by name (ModelHandle)
    instead of carrying raw Python objects.
    """

    def __init__(self):
        self._models: Dict[str, nn.Module] = {}

    def register(self, name: str, model: nn.Module) -> None:
        """Registers a model under a specific handle."""
        self._models[name] = model

    def get(self, name: str) -> nn.Module:
        """Retrieves a model by its handle."""
        if name not in self._models:
            raise KeyError(f"Model handle '{name}' not found in registry.")
        return self._models[name]


class BufferRegistry:
    """
    A central registry for named ReplayBuffers.
    Allows graph nodes to reference buffers by string handles.
    """

    def __init__(self):
        self._buffers: Dict[str, Any] = {}

    def register(self, name: str, buffer: Any) -> None:
        """Registers a buffer under a specific handle."""
        self._buffers[name] = buffer

    def get(self, name: str) -> Any:
        """Retrieves a buffer by its handle."""
        if name not in self._buffers:
            raise KeyError(f"Buffer handle '{name}' not found in registry.")
        return self._buffers[name]


class OptimizerRegistry:
    """
    A central registry for named OptimizerStates.
    Allows graph nodes to reference optimizers by handle.
    """

    def __init__(self):
        self._optimizers: Dict[str, "OptimizerState"] = {}

    def register(self, name: str, optimizer: "OptimizerState") -> None:
        """Registers an optimizer under a specific handle."""
        self._optimizers[name] = optimizer

    def get(self, name: str) -> "OptimizerState":
        """Retrieves an optimizer by its handle."""
        if name not in self._optimizers:
            raise KeyError(f"Optimizer handle '{name}' not found in registry.")
        return self._optimizers[name]


class CallableRegistry:
    """
    A central registry for named callables (e.g. expert policies for DAgger).
    Lets graphs reference functions by string handle so node params stay pure.
    """

    def __init__(self):
        self._fns: Dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        """Registers a callable under a specific handle."""
        self._fns[name] = fn

    def get(self, name: str) -> Callable:
        """Retrieves a callable by its handle."""
        if name not in self._fns:
            raise KeyError(f"Callable handle '{name}' not found in registry.")
        return self._fns[name]
