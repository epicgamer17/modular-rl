from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
import torch
from torch import nn, Tensor

from modules.utils import kernel_initializer_wrapper
from modules.world_models.inference_output import InferenceOutput


class BaseAgentNetwork(nn.Module, ABC):
    """
    Enforces a strict separation between Actor (Environment) and Learner (Training) APIs.
    All Agent Networks must implement these methods to be compatible with ModularSearch and BlackboardEngine.
    """

    def __init__(self):
        super().__init__()
        # Note: input_shape and num_actions are no longer stored here to encourage
        # modularity. Components (Heads, Backbones) should manage their own shapes.
        # Callers (ActionSelectors, Actors) should manage batching.

    def initialize(
        self, initializer: Optional[Union[Callable[[Tensor], None], str, Dict[str, float]]] = None
    ) -> None:
        """
        Unified initialization for all Agent Network components.
        Recursively applies the initializer function to all applicable layers (Conv, Linear).
        If initializer is a dict, it maps component names to gains for orthogonal initialization.
        """
        init_val = kernel_initializer_wrapper(initializer)
        if init_val is None:
            return

        # Handle Dict of gains (Special case for high-fidelity PPO/MuZero)
        if isinstance(init_val, dict):
            # We assume the network is composed of named components (e.g. self.components in ModularAgentNetwork)
            # or just recursive application with a default if not found in dict.
            default_gain = init_val.get("default", 1.0)
            
            # If the object has a 'components' attribute (like ModularAgentNetwork), we use it
            components = getattr(self, "components", {"self": self})

            for comp_name, module in components.items():
                gain = init_val.get(comp_name, default_gain)
                
                def init_comp_weights(m):
                    if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                        if hasattr(m, "weight") and m.weight is not None:
                            nn.init.orthogonal_(m.weight, gain=gain)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                
                module.apply(init_comp_weights)
            return

        # Handle Standard Init Function
        init_fn = init_val
        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                if hasattr(m, "weight") and m.weight is not None:
                    init_fn(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(init_weights)

    @property
    def device(self) -> torch.device:
        """Returns the device this network is currently on."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def reset_noise(self) -> None:
        """Resamples NoisyNet parameters across all layers recursively."""

        def _reset_recursive(m):
            if hasattr(m, "reset_noise") and callable(m.reset_noise):
                m.reset_noise()

        self.apply(_reset_recursive)

    @abstractmethod
    def obs_inference(self, obs: Tensor, **kwargs) -> InferenceOutput:
        """Standard interface for raw observation inputs (Initial Inference)."""
        pass

    @abstractmethod
    def hidden_state_inference(
        self, hidden_state: Any, action: Tensor, **kwargs
    ) -> InferenceOutput:
        """Standard interface for MuZero-style latent rollout (Recurrent Inference)."""
        pass

    def afterstate_inference(
        self, network_state: Any, action: Tensor
    ) -> InferenceOutput:
        """
        Optional interface for Stochastic MuZero (Afterstate Inference).
        If the architecture does not support stochasticity, this will raise NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support afterstate_inference."
        )

    @abstractmethod
    def learner_inference(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """Standard interface for Learner to get raw math tensors across a sequence."""
        pass

    def compile(self, mode: str = "default", fullgraph: bool = False) -> None:
        """Compiles the inference methods for performance gains on supported platforms."""
        # Check if MPS (Mac) is active. MPS currently doesn't support torch.compile.
        # Contract: obs is ALWAYS expected to be batched [B, ...]
        # Validation and unsqueezing now happen in the ActionSelector or Actor.
        if self.device.type == "mps":
            print("Skipping torch.compile on Apple Silicon (MPS).")
            return

        # Compile the specific inference methods.
        self.obs_inference = torch.compile(
            self.obs_inference, mode=mode, fullgraph=fullgraph
        )
        self.hidden_state_inference = torch.compile(
            self.hidden_state_inference, mode=mode, fullgraph=fullgraph
        )
        self.learner_inference = torch.compile(
            self.learner_inference, mode=mode, fullgraph=fullgraph
        )
        # Note: afterstate_inference is optional and might raise error if compiled.
        try:
            self.afterstate_inference = torch.compile(
                self.afterstate_inference, mode=mode, fullgraph=fullgraph
            )
        except (AttributeError, NotImplementedError):
            pass
