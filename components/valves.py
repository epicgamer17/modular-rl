from typing import Set, Dict, Any
from core import PipelineComponent, Blackboard
from core.contracts import Key, SemanticType


class GradientScaleValve(PipelineComponent):
    """
    Valve Component: Manipulates the autograd graph during backpropagation.
    Registers a hook on a specific prediction tensor to scale its gradient.

    This is commonly used in algorithms like MuZero to downscale the gradient
    flowing into the dynamics model to prevent it from overwhelming the
    representation network.
    """

    def __init__(self, key: str, scale: float):
        self.key = key
        self.scale = scale

        self._requires = {Key(f"predictions.{self.key}", SemanticType)}
        self._provides = {}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures the target prediction tensor exists and is attached to the autograd graph."""
        tensor = blackboard.predictions.get(self.key)
        assert (
            tensor is not None
        ), f"[{self.__class__.__name__}] Target '{self.key}' missing from predictions."
        assert (
            tensor.requires_grad
        ), f"[{self.__class__.__name__}] Tensor '{self.key}' is detached from the autograd graph."

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        """Registers a gradient scaling hook on the target tensor."""
        tensor = blackboard.predictions[self.key]
        scale = self.scale
        tensor.register_hook(lambda grad: grad * scale)
        return {}
