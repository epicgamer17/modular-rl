import torch
from typing import Dict, Any, Optional


class ShapeValidator:
    """
    Validates tensor shapes for predictions and targets in the LossPipeline.
    Ensures that data matches expected minibatch_size, unroll_steps (K), and action dimensions.
    """

    def __init__(
        self,
        minibatch_size: int,
        unroll_steps: int = 0,
        num_actions: int = 0,
        atom_size: int = 1,
        support_range: Optional[int] = None,
    ):
        self.B = minibatch_size
        self.K = unroll_steps
        self.num_actions = num_actions
        self.atom_size = atom_size
        if self.atom_size == 1 and support_range is not None:
            self.atom_size = (support_range * 2) + 1

        self.T = self.K + 1

    def validate(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> None:
        """
        Validate the shapes of predictions and targets.
        """
        self.validate_predictions(predictions)

        for key, tensor in targets.items():
            if torch.is_tensor(tensor):
                self._check_shape_strict(key, tensor, is_prediction=False)

    def validate_predictions(self, predictions: Dict[str, torch.Tensor]) -> None:
        """
        Validate that all predictions follow the universal [B, T, ...] contract.
        """
        for key, tensor in predictions.items():
            if torch.is_tensor(tensor):
                self._check_shape_strict(key, tensor, is_prediction=True)


    def _check_shape_strict(
        self, key: str, tensor: torch.Tensor, is_prediction: bool
    ) -> None:
        """Strict validation for Universal T: always expects [B, T, ...]."""
        # 1. System/Infrastructure tensors that do not conform to Universal T
        if key in ["weights", "gradient_scales", "metrics"]:
            return  # Safely bypass the validator for these specific keys

        shape = list(tensor.shape)
        prefix = f"[{'Prediction' if is_prediction else 'Target'}] '{key}'"

        # 2. Batch Size (Dimension 0)
        assert (
            shape[0] == self.B
        ), f"{prefix} batch size mismatch: expected {self.B}, got {shape[0]} | full shape: {shape}"

        # 2. Sequence Length (Dimension 1)
        assert (
            len(shape) >= 2
        ), f"{prefix} must have at least 2 dimensions [B, T, ...], got {shape} | full shape: {shape}"
        assert (
            shape[1] == self.T
        ), f"{prefix} sequence length mismatch: expected {self.T}, got {shape[1]} | full shape: {shape}"

        # 3. Content specific checks
        if key == "policies":
            assert (
                shape[2] == self.num_actions
            ), f"{prefix} action dim mismatch: expected {self.num_actions}, got {shape[2]} | full shape: {shape}"
        elif key in ["values", "returns"]:
            # Could be scalar [B, T] or distributional [B, T, atoms]
            if len(shape) == 3:
                assert (
                    shape[2] == self.atom_size
                ), f"{prefix} distributional target mismatch: expected {self.atom_size} atoms, got {shape[2]} | full shape: {shape}"
            else:
                assert (
                    len(shape) == 2
                ), f"{prefix} scalar target mismatch: expected 2D [B, T], got {shape} | full shape: {shape}"
        elif key.endswith("_mask") or key == "masks":
            # Semantic masks (value_mask, reward_mask, policy_mask, q_mask) must be exactly [B, T]
            assert (
                len(shape) == 2
            ), f"{prefix} mask must be exactly 2D [B, T], got {shape} | full shape: {shape}"
