from typing import Tuple, Optional, Callable, Dict, Any
import torch
from torch import nn, Tensor
from core.contracts import SemanticType, Structure
from modules.representations import BaseRepresentation
from modules.layers.noisy_linear import build_linear_layer


class BaseHead(nn.Module):
    """
    Base class for all network heads.
    Handles an optional neck (modular backbone) and standard initialization.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        representation: BaseRepresentation,
        neck: Optional[nn.Module] = None,
        noisy_sigma: float = 0.0,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.representation = representation

        # 1. Neck (optional modular backbone associated with the head)
        self.neck = neck if neck is not None else nn.Identity()

        # If neck has output_shape attribute, use it; otherwise infer it
        if hasattr(self.neck, "output_shape"):
            self.output_shape = self.neck.output_shape
        else:
            self.output_shape = input_shape

        self.flat_dim = self._get_flat_dim(self.output_shape)

        # 2. Final Output Layer
        self.output_layer = None
        if self.representation is not None:
            self.output_layer = build_linear_layer(
                in_features=self.flat_dim,
                out_features=self.representation.num_features,
                sigma=noisy_sigma,
            )

    def _get_flat_dim(self, shape: Tuple[int, ...]) -> int:
        flat = 1
        for dim in shape:
            flat *= dim
        return flat

    def reset_noise(self) -> None:
        """Recursive reset_noise for all child modules."""
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "reset_noise"):
                module.reset_noise()

    def forward(
        self, x: Tensor, state: Optional[Dict[str, Any]] = None
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Standard forward pass: neck -> output_layer -> strategy."""
        x = self.process_input(x)
        logits = self.output_layer(x)
        return logits, state if state is not None else {}

    def process_input(self, x: Tensor) -> Tensor:
        """Helper to pass input through neck and flatten it."""
        x = self.neck(x)
        if x.dim() > 2:
            x = x.flatten(1, -1)
        return x

    def get_structure(self) -> Structure:
        """Returns the semantic structure of this head's output."""
        return self.representation.get_structure()

    def get_contracts(self, prefix: str) -> Dict[str, "Key"]:
        """
        Returns a dictionary of semantic Keys provided by this head.
        Used by the AgentNetwork to build the automated learner contract.
        """
        from core.contracts import Key, ShapeContract

        # Default event shape from representation (e.g. [bins] or [2 * action_dim])
        event_shape = (self.representation.num_features,)

        # Determine symbolic names (Heads typically produce [B, T, Feature])
        symbolic = ("B", "T", "F")
        if self.representation.num_features == 1:
            symbolic = ("B", "T", "1")

        main_key = Key(
            path=prefix,
            semantic_type=self.semantic_type,
            metadata=self.representation.get_metadata(),
            shape=ShapeContract(
                ndim=3,
                time_dim=1,
                event_shape=event_shape,
                symbolic=symbolic,
                dtype=torch.float32,
            ),
        )

        return {prefix: main_key}

    @property
    def semantic_type(self) -> Any:
        """The base semantic class for this head's primary output."""
        return SemanticType
