from abc import ABC, abstractmethod
from typing import Optional, Dict
import torch
from torch import Tensor
import torch.nn.functional as F


class BaseRepresentation(ABC):
    """
    Base interface for mathematical representations of values (scalars, distributions, etc.).
    Isolates representation logic from network heads and target builders.
    """

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """The final output dimension for the network head."""
        pass

    @abstractmethod
    def to_scalar(self, logits: Tensor) -> Tensor:
        """Converts network output logits into expected scalar values."""
        pass

    @abstractmethod
    def to_representation(self, scalar_targets: Tensor) -> Tensor:
        """Converts mathematically pure targets into the target distribution expected by the loss metric."""
        pass

    def format_target(self, targets: Dict[str, Tensor]) -> Tensor:
        """
        Converts raw target ingredients from the TargetBuilder into the final target representation.
        Default implementation handles simple scalar values.
        """
        return self.to_representation(targets["values"])


class ScalarRepresentation(BaseRepresentation):
    """
    Identity representation for raw scalar values.
    Used for standard MSE/Regression heads.
    """

    @property
    def num_classes(self) -> int:
        return 1

    def to_scalar(self, logits: Tensor) -> Tensor:
        return logits.reshape(logits.shape[:-1])

    def to_representation(self, scalar_targets: Tensor) -> Tensor:
        return scalar_targets.reshape(*scalar_targets.shape, 1)


class DiscreteValuedRepresentation(BaseRepresentation):
    """
    Base class for value representations that map scalars to a categorical support.
    Provides common logic for expectation (to_scalar) and two-hot projection (to_representation).
    """

    def __init__(self, vmin: float, vmax: float, bins: int):
        assert bins > 1, f"Discrete representations require at least 2 bins, got {bins}"
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.support = torch.linspace(vmin, vmax, bins)
        self.delta_z = (vmax - vmin) / (bins - 1)

    @property
    def num_classes(self) -> int:
        return self.bins

    def to_scalar(self, logits: Tensor) -> Tensor:
        """Expected value over the support: [B, T, bins] -> [B, T]"""
        assert (
            logits.shape[-1] == self.bins
        ), f"Expected last dimension to be {self.bins}, got {logits.shape[-1]}"
        orig_shape = logits.shape
        flat_logits = logits.reshape(-1, self.bins)

        probs = torch.softmax(flat_logits, dim=-1)
        support = self.support.to(device=logits.device, dtype=logits.dtype)
        flat_scalar = (probs * support).sum(dim=-1)

        return flat_scalar.reshape(*orig_shape[:-1])

    def to_representation(self, scalar_targets: Tensor) -> Tensor:
        """Project scalar targets onto discrete support: [B, T] -> [B, T, bins]"""
        device = scalar_targets.device
        dtype = scalar_targets.dtype
        orig_shape = scalar_targets.shape

        # 1. Flatten into [N]
        x = scalar_targets.reshape(-1).clamp(self.vmin, self.vmax)

        # 2. Process
        b = (x - self.vmin) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        p_u = b - l.float()
        p_l = 1.0 - p_u

        flat_l = l.view(-1, 1).clamp(0, self.bins - 1)
        flat_u = u.view(-1, 1).clamp(0, self.bins - 1)
        flat_p_l = p_l.view(-1, 1)
        flat_p_u = p_u.view(-1, 1)

        num_elements = x.shape[0]
        projected = torch.zeros((num_elements, self.bins), device=device, dtype=dtype)

        projected.scatter_add_(1, flat_l, flat_p_l)
        projected.scatter_add_(1, flat_u, flat_p_u)

        # 3. Unflatten back to [B, T, bins]
        return projected.view(*orig_shape, self.bins)

    def format_target(self, targets: Dict[str, Tensor]) -> Tensor:
        """Handle ingredient-based targets or fallback to scalar."""
        if "values" in targets:
            return self.to_representation(targets["values"])
        raise ValueError(
            f"{self.__class__.__name__} received a targets dict without 'values' or algorithm-specific ingredients."
        )


class TwoHotRepresentation(DiscreteValuedRepresentation):
    def __init__(self, vmin: float, vmax: float, bins: int):
        super().__init__(vmin, vmax, bins)


class CategoricalRepresentation(DiscreteValuedRepresentation):
    def __init__(self, vmin: float, vmax: float, bins: int):
        super().__init__(vmin, vmax, bins)

    def format_target(self, targets: Dict[str, Tensor]) -> Tensor:
        """
        The Baker: Implements the C51 Bellman Projection.
        Projects shifted atom probabilities back onto the fixed support grid.
        Supports [B, T] inputs via Flatten-Process-Unflatten.
        """
        if "next_q_logits" not in targets:
            raise ValueError(
                "CategoricalRepresentation requires 'next_q_logits' in targets."
            )

        # 1. Capture original prefix shape (e.g., [B, T])
        raw_next_logits = targets["next_q_logits"]
        assert (
            raw_next_logits.shape[-1] == self.bins
        ), f"CategoricalRepresentation expected {self.bins} atoms, got {raw_next_logits.shape[-1]}"
        orig_prefix = raw_next_logits.shape[:-2]  # Everything before [Actions, Atoms]
        num_actions = raw_next_logits.shape[-2]
        atoms = raw_next_logits.shape[-1]

        # 2. Flatten all inputs into [N, ...]
        next_q_logits = raw_next_logits.reshape(-1, num_actions, atoms)
        next_actions = targets["next_actions"].reshape(-1).long()
        rewards = targets["rewards"].reshape(-1)
        dones = targets["dones"].reshape(-1)
        gamma = targets.get("gamma", 0.99)
        n_step = targets.get("n_step", 1)

        device = rewards.device
        N = rewards.shape[0]
        discount = gamma**n_step

        # 3. Process distributional Bellman math on flattened 2D buffer
        chosen_logits = next_q_logits[torch.arange(N, device=device), next_actions]
        next_probs = torch.softmax(chosen_logits, dim=-1)
        support = self.support.to(device)

        # Shifted support: Tz = r + discount * (1 - done) * z
        Tz = rewards.view(-1, 1) + discount * (1.0 - dones).view(-1, 1) * support.view(
            1, -1
        )
        Tz = Tz.clamp(self.vmin, self.vmax)

        # True mathematical bounds
        b = (Tz - self.vmin) / self.delta_z
        l = b.floor().long()
        u = l + 1

        # Conserve mass: (u - b) + (b - l) always equals 1
        p_l = next_probs * (u.float() - b)
        p_u = next_probs * (b - l.float())

        # Safe indexing for scattering
        # (if b = bins-1, u=bins, so we clamp it back to bins-1 for the array index)
        l_idx = l.clamp(0, self.bins - 1)
        u_idx = u.clamp(0, self.bins - 1)

        projected_dist_flat = torch.zeros((N, self.bins), device=device)
        projected_dist_flat.scatter_add_(1, l_idx, p_l)
        projected_dist_flat.scatter_add_(1, u_idx, p_u)

        # 4. Unflatten back to [B, T, bins]
        return projected_dist_flat.reshape(*orig_prefix, self.bins)


class ExponentialBucketsRepresentation(BaseRepresentation):
    def __init__(self, vmin: float, vmax: float, bins: int):
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.log_vmin = self._log_transform(torch.tensor(vmin)).item()
        self.log_vmax = self._log_transform(torch.tensor(vmax)).item()
        self._inner = DiscreteValuedRepresentation(self.log_vmin, self.log_vmax, bins)

    def _log_transform(self, x: Tensor) -> Tensor:
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def _exp_inverse(self, y: Tensor) -> Tensor:
        return torch.sign(y) * (torch.exp(torch.abs(y)) - 1.0)

    @property
    def num_classes(self) -> int:
        return self.bins

    def to_scalar(self, logits: Tensor) -> Tensor:
        log_scalar = self._inner.to_scalar(logits)
        return self._exp_inverse(log_scalar)

    def to_representation(self, scalar_targets: Tensor) -> Tensor:
        log_targets = self._log_transform(scalar_targets)
        return self._inner.to_representation(log_targets)


class ClassificationRepresentation(BaseRepresentation):
    def __init__(self, num_classes: int):
        self._num_classes = num_classes

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def to_scalar(self, logits: Tensor) -> Tensor:
        assert (
            logits.shape[-1] == self._num_classes
        ), f"Expected last dimension to be {self._num_classes}, got {logits.shape[-1]}"
        orig_shape = logits.shape
        flat_logits = logits.reshape(-1, self._num_classes)
        flat_scalar = torch.argmax(flat_logits, dim=-1).float()
        return flat_scalar.reshape(*orig_shape[:-1])

    def to_representation(self, scalar_targets: Tensor) -> Tensor:
        orig_shape = scalar_targets.shape
        flat_targets = scalar_targets.reshape(-1).long()
        flat_one_hot = F.one_hot(flat_targets, num_classes=self._num_classes).float()
        return flat_one_hot.reshape(*orig_shape, self._num_classes)

    def format_target(
        self, targets: Dict[str, Tensor], target_key: str = "values"
    ) -> Tensor:
        """If the target is already a distribution, return it directly."""
        target = targets[target_key]
        if target.ndim > 1 and target.shape[-1] == self._num_classes:
            return target
        return self.to_representation(target)


def get_representation(
    num_classes: int = 1,
    support_range: Optional[int] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    bins: Optional[int] = None,
    mode: str = "linear",
) -> BaseRepresentation:
    if vmin is not None and vmax is not None and bins is not None:
        if mode == "exponential":
            print("Using ExponentialBucketsRepresentation")
            return ExponentialBucketsRepresentation(vmin, vmax, bins)
        elif mode == "categorical":
            print("Using CategoricalRepresentation")
            return CategoricalRepresentation(vmin, vmax, bins)
        print("Using TwoHotRepresentation")
        return TwoHotRepresentation(vmin, vmax, bins)

    if support_range is not None:
        return TwoHotRepresentation(
            float(-support_range), float(support_range), 2 * support_range + 1
        )

    if num_classes > 1:
        return ClassificationRepresentation(num_classes)

    return ScalarRepresentation()
