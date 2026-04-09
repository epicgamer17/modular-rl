import torch
from core import PipelineComponent
from core import Blackboard

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

    def execute(self, blackboard: Blackboard) -> None:
        """
        Finds the specified key in predictions and registers the scaling hook.
        If the tensor does not require gradients (e.g. inference passes), it is skipped.
        """
        if self.key in blackboard.predictions:
            tensor = blackboard.predictions[self.key]
            
            # Tensors derived from inference_mode or without requires_grad
            # cannot have hooks registered on them.
            if torch.is_tensor(tensor) and tensor.requires_grad:
                # Capture scale in closure
                scale = self.scale
                # Use a fast inplace lambda to avoid overhead
                tensor.register_hook(lambda grad: grad * scale)
