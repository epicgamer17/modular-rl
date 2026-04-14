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

    @property
    def requires(self) -> Set[Key]:
        return {Key(f"predictions.{self.key}", SemanticType)}

    @property
    def provides(self) -> Set[Key]:
        return set()

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        """
        Finds the specified key in predictions and registers the scaling hook.
        
        Raises:
            KeyError: If the target key is missing from predictions.
            RuntimeError: If the tensor is detached from the autograd graph.
        """
        # Fail loud and fast. If this component is in the Recipe, its prerequisites MUST exist.
        tensor = blackboard.predictions.get(self.key)
        
        if tensor is None:
            raise KeyError(f"[{self.__class__.__name__}] Target '{self.key}' missing from predictions.")
        if not tensor.requires_grad:
            raise RuntimeError(f"[{self.__class__.__name__}] Tensor '{self.key}' is detached from the autograd graph.")
            
        # Capture scale and register hook
        scale = self.scale
        tensor.register_hook(lambda grad: grad * scale)
        return {}
