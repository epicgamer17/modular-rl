from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor
from modules.utils import support_to_scalar, scalar_to_support


class OutputStrategy(nn.Module, ABC):
    """
    Base class for all output processing strategies.
    Handles converting logits to probabilities, scalars, and distributions.
    """

    @abstractmethod
    def logits_to_probs(self, logits: Tensor) -> Tensor:
        """Converts raw logits to a probability distribution."""
        pass

    @abstractmethod
    def logits_to_scalar(self, logits: Tensor) -> Tensor:
        """Converts raw logits from the head to a scalar output."""
        pass

    @abstractmethod
    def scalar_to_target(self, scalar: Tensor) -> Tensor:
        """Converts a ground-truth scalar to a target probability distribution."""
        pass

    @property
    @abstractmethod
    def num_bins(self) -> int:
        """The required output size for the dense layer in the head."""
        pass


class Regression(OutputStrategy):
    """Pure regression: identity transform."""

    def __init__(self, output_size: int = 1):
        super().__init__()
        self._output_size = output_size

    def logits_to_probs(self, logits: Tensor) -> Tensor:
        # Not applicable, but returning identity/sigmoid if needed?
        # For regression, we usually don't have probs.
        return logits

    def logits_to_scalar(self, logits: Tensor) -> Tensor:
        # Assumes logits is already a scalar or has size 1 on last dim
        if logits.shape[-1] == 1:
            return logits.squeeze(-1)
        return logits

    def scalar_to_target(self, scalar: Tensor) -> Tensor:
        return scalar

    @property
    def num_bins(self) -> int:
        return self._output_size


class Categorical(OutputStrategy):
    """Standard categorical distribution (e.g., Policy, To-Play)."""

    def __init__(self, num_classes: int):
        super().__init__()
        self._num_classes = num_classes

    def logits_to_probs(self, logits: Tensor) -> Tensor:
        return torch.softmax(logits, dim=-1)

    def logits_to_scalar(self, logits: Tensor) -> Tensor:
        # Could return argmax or expectation if classes have numerical meaning
        # For most cases, this isn't strictly used as a "scalar value"
        return torch.argmax(logits, dim=-1).float()

    def scalar_to_target(self, scalar: Tensor) -> Tensor:
        # Convert class indices to one-hot
        return torch.nn.functional.one_hot(
            scalar.long(), num_classes=self._num_classes
        ).float()

    @property
    def num_bins(self) -> int:
        return self._num_classes

    @property
    def is_probabilistic(self) -> bool:
        return True


class MuZeroSupport(OutputStrategy):
    """MuZero-style support: Invertible transform + categorical buckets."""

    def __init__(self, support_range: int, eps: float = 0.001):
        super().__init__()
        self.support_range = support_range
        self.eps = eps

    def logits_to_probs(self, logits: Tensor) -> Tensor:
        return torch.softmax(logits, dim=-1)

    def logits_to_scalar(self, logits: Tensor) -> Tensor:
        probs = self.logits_to_probs(logits)
        return support_to_scalar(probs, self.support_range, self.eps)

    def scalar_to_target(self, scalar: Tensor) -> Tensor:
        return scalar_to_support(scalar, self.support_range, self.eps)

    @property
    def num_bins(self) -> int:
        return 2 * self.support_range + 1


class C51Support(OutputStrategy):
    """C51-style support: Fixed atoms on a linear scale."""

    def __init__(self, v_min: float, v_max: float, num_atoms: int):
        super().__init__()
        self.v_min = v_min
        self.v_max = v_max
        self._num_atoms = num_atoms
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

    def logits_to_probs(self, logits: Tensor) -> Tensor:
        return torch.softmax(logits, dim=-1)

    def logits_to_scalar(self, logits: Tensor) -> Tensor:
        probs = self.logits_to_probs(logits)
        return (probs * self.support).sum(dim=-1)

    def scalar_to_target(self, scalar: Tensor) -> Tensor:
        # Standard C51 projection math
        # [B] -> [B, num_atoms]
        target = scalar.clamp(self.v_min, self.v_max)
        b = (target - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        # Handle edge cases where l == u
        l_mask = l == u
        if l_mask.any():
            u[l_mask] += 1  # dummy move to allow weight calc

        res = torch.zeros(
            (*scalar.shape, self._num_atoms), device=scalar.device, dtype=scalar.dtype
        )

        # We need to use scatter or similar to distribute weights
        # For simplicity in this implementation:
        # res[l] += (u - b); res[u] += (b - l)

        # Flatten for easy indexing if batch is multi-dim
        flat_res = res.view(-1, self._num_atoms)
        flat_scalar = b.view(-1)
        flat_l = l.view(-1)
        flat_u = u.view(-1)

        batch_idx = torch.arange(flat_scalar.size(0), device=scalar.device)

        flat_res[batch_idx, flat_l] += flat_u.float() - flat_scalar

        # Mask out-of-bounds u
        valid_u = flat_u < self._num_atoms
        flat_res[batch_idx[valid_u], flat_u[valid_u]] += (
            flat_scalar[valid_u] - flat_l[valid_u].float()
        )

        return res

    @property
    def num_bins(self) -> int:
        return self._num_atoms


class DreamerSupport(OutputStrategy):
    """DreamerV2/V3 support: Symlog transform + categorical buckets."""

    def __init__(self, support_range: int):
        super().__init__()
        self.support_range = support_range
        # Dreamer often uses buckets from -range to range
        self.register_buffer(
            "support",
            torch.linspace(-support_range, support_range, 2 * support_range + 1),
        )

    def symlog(self, x: Tensor) -> Tensor:
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

    def symexp(self, x: Tensor) -> Tensor:
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

    def logits_to_probs(self, logits: Tensor) -> Tensor:
        return torch.softmax(logits, dim=-1)

    def logits_to_scalar(self, logits: Tensor) -> Tensor:
        probs = self.logits_to_probs(logits)
        val = (probs * self.support).sum(dim=-1)
        return self.symexp(val)

    def scalar_to_target(self, scalar: Tensor) -> Tensor:
        val = self.symlog(scalar)
        # Project onto buckets
        val = val.clamp(-self.support_range, self.support_range)
        floor = val.floor()
        prob = val - floor

        res = torch.zeros(
            (*scalar.shape, 2 * self.support_range + 1),
            device=scalar.device,
            dtype=scalar.dtype,
        )

        flat_res = res.view(-1, 2 * self.support_range + 1)
        flat_val = val.view(-1)
        flat_floor = floor.view(-1).long() + self.support_range

        batch_idx = torch.arange(flat_val.size(0), device=scalar.device)
        flat_res[batch_idx, flat_floor] = 1.0 - (flat_val - floor.view(-1))

        valid_u = (flat_floor + 1) < (2 * self.support_range + 1)
        flat_res[batch_idx[valid_u], flat_floor[valid_u] + 1] = (
            flat_val[valid_u] - floor.view(-1)[valid_u]
        )

        return res

    @property
    def num_bins(self) -> int:
        return 2 * self.support_range + 1
