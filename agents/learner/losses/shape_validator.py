import torch
from typing import Dict, Any


class ShapeValidator:
    """
    Validates tensor shapes for predictions and targets in the LossPipeline.
    Ensures that data matches expected minibatch_size, unroll_steps (K), and action dimensions.
    """

    def __init__(self, config: Any):
        self.config = config
        self.B = config.minibatch_size
        # Multi-step unrolling (MuZero) uses unroll_steps (K).
        # Sequence length is K+1.
        self.K = getattr(config, "unroll_steps", 0)
        self.T = self.K + 1
        self.num_actions = getattr(config.game, "num_actions", 0)
        
        # Support/Atoms for distributional RL
        self.atom_size = getattr(config, "atom_size", 1)
        if self.atom_size == 1 and hasattr(config, "support_range") and config.support_range is not None:
            # MuZero style support
            self.atom_size = (config.support_range * 2) + 1

    def validate(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> None:
        """
        Validate the shapes of predictions and targets.
        """
        for key, tensor in predictions.items():
            if torch.is_tensor(tensor):
                self._check_shape(key, tensor, is_prediction=True)

        for key, tensor in targets.items():
            if torch.is_tensor(tensor):
                self._check_shape(key, tensor, is_prediction=False)

    def _check_shape(self, key: str, tensor: torch.Tensor, is_prediction: bool) -> None:
        shape = list(tensor.shape)
        prefix = f"[{'Prediction' if is_prediction else 'Target'}] '{key}'"

        # 1. Batch Size (Dimension 0)
        assert (
            shape[0] == self.B
        ), f"{prefix} batch size mismatch: expected {self.B}, got {shape[0]}"

        # 2. Sequence Length and Content (Dimension 1+)
        # PPO: T=1, often omitted. MuZero: T > 1, usually present.
        has_sequence_dim = len(shape) >= 2 and shape[1] in [self.T, self.T - 1]
        is_single_step = (self.T == 1)

        if key == "policies":
            # Expected (B, T, A) or (B, A)
            if has_sequence_dim:
                assert shape[1] == self.T, f"{prefix} sequence length mismatch: expected {self.T}, got {shape[1]}"
                assert shape[2] == self.num_actions, f"{prefix} action dim mismatch: expected {self.num_actions}, got {shape[2]}"
            elif is_single_step:
                assert shape[1] == self.num_actions, f"{prefix} action dim mismatch: expected {self.num_actions}, got {shape[1]}"
            else:
                raise AssertionError(f"{prefix} shape {shape} invalid for T={self.T}, num_actions={self.num_actions}")

        elif key in ["values", "q_values", "q_logits", "chance_values"]:
            # Expected (B, T, Atoms) or (B, T) or (B, Atoms) or (B,)
            if has_sequence_dim:
                # (B, T) or (B, T, Atoms)
                if len(shape) == 3:
                     assert shape[2] in [1, self.atom_size], f"{prefix} atom dim mismatch: expected 1 or {self.atom_size}, got {shape[2]}"
            elif is_single_step:
                # (B,) or (B, Atoms)
                if len(shape) == 2:
                     assert shape[1] in [1, self.num_actions, self.atom_size], f"{prefix} second dim mismatch for single-step {key}"
            else:
                # Targets might include bootstrap step (T+1)
                if not is_prediction and shape[1] == self.T + 1:
                    pass
                else:
                    raise AssertionError(f"{prefix} shape {shape} invalid for T={self.T}")

        elif key == "rewards":
            # Rewards are often (B, K) where K = T-1
            if has_sequence_dim:
                assert shape[1] in [self.T, self.T - 1], f"{prefix} sequence length mismatch: expected {self.T} or {self.T-1}, got {shape[1]}"
            elif is_single_step:
                # (B,) or (B, 1)
                pass
            else:
                 raise AssertionError(f"{prefix} shape {shape} invalid for T={self.T}")

        elif key == "actions":
            # (B, T) or (B, T, 1) or (B, 1) or (B,)
            if has_sequence_dim:
                assert shape[1] in [self.T, self.T - 1], f"{prefix} sequence length mismatch"
            elif is_single_step:
                pass
            else:
                raise AssertionError(f"{prefix} shape {shape} invalid for T={self.T}")

        elif key == "latents" or key == "latent_states":
            # (B, T, D) or (B, D)
            if has_sequence_dim:
                assert shape[1] == self.T, f"{prefix} sequence length mismatch: expected {self.T}, got {shape[1]}"
            elif is_single_step:
                pass
