from abc import ABC, abstractmethod
from typing import Optional
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


class ScalarRepresentation(BaseRepresentation):
    """
    Identity representation for raw scalar values.
    Used for standard MSE/Regression heads.
    """

    @property
    def num_classes(self) -> int:
        return 1

    def to_scalar(self, logits: Tensor) -> Tensor:
        return logits.squeeze(-1)

    def to_representation(self, scalar_targets: Tensor) -> Tensor:
        return scalar_targets


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
        """Expected value over the support."""
        probs = torch.softmax(logits, dim=-1)
        support = self.support.to(device=logits.device, dtype=logits.dtype)
        return (probs * support).sum(dim=-1)

    def to_representation(self, scalar_targets: Tensor) -> Tensor:
        """Project scalar targets onto discrete support using two-hot encoding."""
        device = scalar_targets.device
        dtype = scalar_targets.dtype

        x = scalar_targets.clamp(self.vmin, self.vmax)
        b = (x - self.vmin) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        p_u = b - l.float()
        p_l = 1.0 - p_u

        batch_shape = x.shape
        flat_l = l.view(-1, 1)
        flat_u = u.view(-1, 1)
        flat_p_l = p_l.view(-1, 1)
        flat_p_u = p_u.view(-1, 1)

        num_elements = flat_l.shape[0]
        projected = torch.zeros((num_elements, self.bins), device=device, dtype=dtype)

        projected.scatter_add_(1, flat_l, flat_p_l)
        projected.scatter_add_(1, flat_u, flat_p_u)

        return projected.view(*batch_shape, self.bins)


class TwoHotRepresentation(DiscreteValuedRepresentation):
    def __init__(self, vmin: float, vmax: float, bins: int):
        super().__init__(vmin, vmax, bins)


class CategoricalRepresentation(DiscreteValuedRepresentation):
    def __init__(self, vmin: float, vmax: float, bins: int):
        super().__init__(vmin, vmax, bins)


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
        return torch.argmax(logits, dim=-1).float()

    def to_representation(self, scalar_targets: Tensor) -> Tensor:
        return F.one_hot(scalar_targets.long(), num_classes=self._num_classes).float()


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
            return ExponentialBucketsRepresentation(vmin, vmax, bins)
        return TwoHotRepresentation(vmin, vmax, bins)

    if support_range is not None:
        return TwoHotRepresentation(
            float(-support_range), float(support_range), 2 * support_range + 1
        )

    if num_classes > 1:
        return ClassificationRepresentation(num_classes)

    return ScalarRepresentation()
