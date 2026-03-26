from typing import Tuple, Optional, Dict, Any, Callable
import torch
from torch import nn, Tensor

from .base import BaseHead, HeadOutput
from configs.modules.backbones.base import BackboneConfig
from agents.learner.losses.representations import BaseRepresentation
from modules.backbones.mlp import build_dense, NoisyLinear


class QHead(BaseHead):
    """
    Modular Q-Network Head.
    Structure: [Neck (Optional)] -> [Hidden Backbone] -> [Output Layer] -> [Strategy (Reshape)]
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        representation: BaseRepresentation,
        hidden_backbone_fn: Callable[[Tuple[int, ...]], nn.Module],
        num_actions: int,
        neck_fn: Optional[Callable[[Tuple[int, ...]], nn.Module]] = None,
        noisy_sigma: float = 0.0,
        name: Optional[str] = None,
        input_source: str = "default",
        **kwargs,
    ):
        super().__init__(
            input_shape,
            representation,
            neck_fn=neck_fn,
            noisy_sigma=noisy_sigma,
            name=name,
            input_source=input_source,
            **kwargs,
        )

        # 1. Heads now build their own feature architecture (neck)
        if self.neck_fn is not None:
            self.neck = self.neck_fn(input_shape=input_shape)
        else:
            self.neck = nn.Identity()
            self.neck.output_shape = input_shape

        self.output_shape = self.neck.output_shape
        self.flat_dim = self._get_flat_dim(self.neck, input_shape)

        self.num_actions = num_actions
        self.noisy = noisy_sigma != 0

        # 2. Hidden Layers are now a Backbone!
        self.hidden_layers = hidden_backbone_fn(input_shape=(self.flat_dim,))

        # 3. Final Output Layer
        self.output_layer = build_dense(
            in_features=self.hidden_layers.output_shape[0],
            out_features=self.representation.num_features * self.num_actions,
            sigma=noisy_sigma,
        )

    def reset_noise(self) -> None:
        """Propagate noise reset through the neck and output layers."""
        if hasattr(self.neck, "reset_noise"):
            self.neck.reset_noise()
        if hasattr(self.hidden_layers, "reset_noise"):
            self.hidden_layers.reset_noise()
        if isinstance(self.output_layer, NoisyLinear):
            self.output_layer.reset_noise()

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
        is_inference: bool = False,
        **kwargs,
    ) -> HeadOutput:
        # 1. Processing neck -> flatten
        x = self.neck(x)
        if x.dim() > 2:
            x = x.flatten(1, -1)

        # 2. Hidden Backbone Phase
        x = self.hidden_layers(x)

        # 3. Output Projection Phase
        logits = self.output_layer(x)
        logits = logits.view(-1, self.num_actions, self.representation.num_features)

        # 4. Standard Return
        new_state = state if state is not None else {}
        inference = None
        if is_inference:
            inference = self.representation.to_expected_value(logits)

        return HeadOutput(
            training_tensor=logits,
            inference_tensor=inference,
            state=new_state,
            metrics=self.compute_metrics(logits, inference),
        )

    def compute_metrics(
        self,
        training_tensor: torch.Tensor,
        inference_tensor: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Calculates Q-specific diagnostics (e.g., mean predicted Q-value)."""
        metrics = {}
        with torch.inference_mode():
            val = (
                inference_tensor
                if inference_tensor is not None
                else self.representation.to_expected_value(training_tensor)
            )
            metrics["mean"] = val.mean().item()
        return metrics


class DuelingQHead(BaseHead):
    """
    Modular Dueling Q-Network Head.
    Structure:
        [Neck] -> Split -> [Value Backbone] -> [Value Output]
                        -> [Advantage Backbone] -> [Advantage Output]
               -> Aggregation (Q = V + A - mean(A))
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        representation: BaseRepresentation,
        value_hidden_backbone_fn: Callable[[Tuple[int, ...]], nn.Module],
        advantage_hidden_backbone_fn: Callable[[Tuple[int, ...]], nn.Module],
        num_actions: int,
        neck_fn: Optional[Callable[[Tuple[int, ...]], nn.Module]] = None,
        noisy_sigma: float = 0.0,
        name: Optional[str] = None,
        input_source: str = "default",
        **kwargs,
    ):
        super().__init__(
            input_shape,
            representation,
            neck_fn=neck_fn,
            noisy_sigma=noisy_sigma,
            name=name,
            input_source=input_source,
            **kwargs,
        )

        # 1. Heads now build their own feature architecture (neck)
        if self.neck_fn is not None:
            self.neck = self.neck_fn(input_shape=input_shape)
        else:
            self.neck = nn.Identity()
            self.neck.output_shape = input_shape

        self.output_shape = self.neck.output_shape
        self.flat_dim = self._get_flat_dim(self.neck, input_shape)

        self.num_actions = num_actions
        self.noisy = noisy_sigma != 0

        # 2. Value Stream (Backbone + Output)
        self.value_hidden = value_hidden_backbone_fn(input_shape=(self.flat_dim,))
        self.value_output = build_dense(
            in_features=self.value_hidden.output_shape[0],
            out_features=self.representation.num_features,
            sigma=noisy_sigma,
        )

        # 3. Advantage Stream (Backbone + Output)
        self.advantage_hidden = advantage_hidden_backbone_fn(input_shape=(self.flat_dim,))
        self.advantage_output = build_dense(
            in_features=self.advantage_hidden.output_shape[0],
            out_features=self.representation.num_features * self.num_actions,
            sigma=noisy_sigma,
        )

    def reset_noise(self) -> None:
        """Propagate noise reset through the neck and stream streams."""
        if hasattr(self.neck, "reset_noise"):
            self.neck.reset_noise()

        for stream in [self.value_hidden, self.advantage_hidden]:
            if hasattr(stream, "reset_noise"):
                stream.reset_noise()

        for out in [self.value_output, self.advantage_output]:
            if isinstance(out, NoisyLinear):
                out.reset_noise()

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
        is_inference: bool = False,
        **kwargs,
    ) -> HeadOutput:
        # 1. Processing neck -> flatten
        x = self.neck(x)
        if x.dim() > 2:
            x = x.flatten(1, -1)

        # 2. Parallel Stream Processing
        v = self.value_hidden(x)
        v = self.value_output(v)  # (B, atoms)
        v = v.view(-1, 1, self.representation.num_features)  # (B, 1, atoms)

        a = self.advantage_hidden(x)
        a = self.advantage_output(a)  # (B, actions * atoms)
        a = a.view(-1, self.num_actions, self.representation.num_features)

        # Aggregation Phase: Q = V + (A - mean(A))
        a_mean = a.mean(dim=1, keepdim=True)
        q = v + a - a_mean

        new_state = state if state is not None else {}
        inference = None
        if is_inference:
            inference = self.representation.to_expected_value(q)

        return HeadOutput(
            training_tensor=q,
            inference_tensor=inference,
            state=new_state,
            metrics=self.compute_metrics(q, inference),
        )

    def compute_metrics(
        self,
        training_tensor: torch.Tensor,
        inference_tensor: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Calculates Q-specific diagnostics (e.g., mean predicted Q-value)."""
        metrics = {}
        with torch.inference_mode():
            val = (
                inference_tensor
                if inference_tensor is not None
                else self.representation.to_expected_value(training_tensor)
            )
            metrics["mean"] = val.mean().item()
        return metrics
