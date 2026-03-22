from typing import Tuple, Optional, Dict, Any
import torch
from torch import Tensor
from .base import BaseHead, HeadOutput
from agents.learner.losses.representations import BaseRepresentation
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig
from modules.backbones.factory import BackboneFactory
from modules.backbones.mlp import build_dense, NoisyLinear


class PolicyHead(BaseHead):
    """
    Predicts the action distribution (Policy).
    Supports both discrete (Categorical) and continuous (Gaussian) actions via Representation.
    Integrates explicit action masking into the compute graph for stable learning.
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        representation: BaseRepresentation,
        neck_config: Optional[BackboneConfig] = None,
        name: Optional[str] = None,
    ):
        super().__init__(arch_config, input_shape, representation, neck_config, name=name)

        # 1. Heads now take ownership of their own architectural blocks
        self.neck = BackboneFactory.create(neck_config, input_shape)
        self.output_shape = self.neck.output_shape
        self.flat_dim = self._get_flat_dim(self.output_shape)

        # 2. Explicitly build the policy projection layer
        self.output_layer = build_dense(
            in_features=self.flat_dim,
            out_features=self.representation.num_features,
            sigma=self.arch_config.noisy_sigma,
        )

    def reset_noise(self) -> None:
        """Propagate noise reset through the neck and output layer."""
        if hasattr(self.neck, "reset_noise"):
            self.neck.reset_noise()
        if isinstance(self.output_layer, NoisyLinear):
            self.output_layer.reset_noise()

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> HeadOutput:
        """Returns HeadOutput containing masked logits and/or distribution object."""
        action_mask = kwargs.get("action_mask")
        # 1. Architectural neck phase -> flatten
        x = self.neck(x)
        if x.dim() > 2:
            x = x.flatten(1, -1)

        # 2. Output Projection Layer
        logits = self.output_layer(x)

        # 3. Apply Masking Logic (Logit-level)
        if action_mask is not None:
            HUGE_NEG = torch.tensor(-1e8, dtype=logits.dtype, device=logits.device)
            logits = torch.where(action_mask.bool(), logits, HUGE_NEG)

            from modules.distributions import MaskedCategorical

            inference = MaskedCategorical(logits=logits, mask=action_mask)
        else:
            inference = self.representation.to_inference(logits)

        return HeadOutput(
            training_tensor=logits,
            inference_tensor=inference,
            state=state if state is not None else {},
        )
