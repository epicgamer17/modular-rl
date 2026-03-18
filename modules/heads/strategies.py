from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from modules.utils import support_to_scalar, scalar_to_support
from modules.distributions import Deterministic
from losses.representations import (
    BaseRepresentation,
    ScalarRepresentation,
    TwoHotRepresentation,
    CategoricalRepresentation,
    ClassificationRepresentation,
    ExponentialBucketsRepresentation,
)


class OutputStrategy(nn.Module, ABC):
    """
    Base class for all output processing strategies.
    Handles converting logits to probabilities, scalars, and distributions.
    """

    @property
    @abstractmethod
    def num_bins(self) -> int:
        """The required output size for the dense layer in the head."""
        pass  # pragma: no cover

    @abstractmethod
    def compute_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Computes the loss between network predictions and targets."""
        pass  # pragma: no cover

    @abstractmethod
    def get_distribution(
        self, network_output: Tensor
    ) -> torch.distributions.Distribution | Deterministic:
        """Used by Policy Heads for action selection."""
        pass  # pragma: no cover

    @abstractmethod
    def to_expected_value(self, network_output: Tensor) -> Tensor:
        """Used by Value/Reward Heads to get the actual scalar number."""
        pass  # pragma: no cover

    @property
    @abstractmethod
    def representation(self) -> BaseRepresentation:
        """Returns the corresponding BaseRepresentation for this strategy."""
        pass  # pragma: no cover

    @abstractmethod
    def scalar_to_target(self, scalar: Tensor) -> Tensor:
        """Converts a ground-truth scalar to a target format (distribution/scalar)."""
        pass  # pragma: no cover


class ScalarStrategy(OutputStrategy):

    def __init__(self, output_size: int = 1):
        super().__init__()
        self._output_size = output_size

    @property
    def num_bins(self) -> int:
        return self._output_size

    @property
    def representation(self) -> BaseRepresentation:
        return ScalarRepresentation()

    def compute_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return F.mse_loss(predictions, targets, reduction="none")

    def to_expected_value(self, network_output: Tensor) -> Tensor:
        # For pure regression, the network output IS the expected value!
        if network_output.shape[-1] == 1:
            return network_output.squeeze(-1)
        return network_output

    def get_distribution(self, network_output: Tensor):
        """Wraps the deterministic output so Action Selectors can still call .sample()"""
        return Deterministic(network_output)

    def scalar_to_target(self, scalar: Tensor) -> Tensor:
        return scalar


class Categorical(OutputStrategy):
    """Standard categorical distribution (e.g., Policy, To-Play)."""

    def __init__(self, num_classes: int):
        super().__init__()
        self._num_classes = num_classes

    @property
    def num_bins(self) -> int:
        return self._num_classes

    @property
    def representation(self) -> BaseRepresentation:
        return ClassificationRepresentation(self._num_classes)

    def compute_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # Targets are expected to be one-hot or indices?
        # If targets are indices:
        if targets.dim() == predictions.dim() - 1:
            targets = targets.long()
        # If targets are one-hot (float):
        if targets.is_floating_point():
            # cross_entropy expects class indices usually, but can take probs with recent pytorch
            # or use KLDivLoss. But standard is CrossEntropyLoss which supports probabilities in newer versions.
            # For robustness, we assume targets might be one-hot probability distributions (like from MCTS)
            return F.cross_entropy(predictions, targets, reduction="none")
        else:
            return F.cross_entropy(predictions, targets, reduction="none")

    def get_distribution(
        self, network_output: Tensor
    ) -> torch.distributions.Distribution:
        return torch.distributions.Categorical(logits=network_output)

    def to_expected_value(self, network_output: Tensor) -> Tensor:
        # Could return argmax or expectation if classes have numerical meaning
        # For most cases (like policy), we don't strictly use this as a value
        # But if used for value, maybe return max prob index?
        return torch.argmax(network_output, dim=-1).float()

    def scalar_to_target(self, scalar: Tensor) -> Tensor:
        # Convert class indices to one-hot
        return torch.nn.functional.one_hot(
            scalar.long(), num_classes=self._num_classes
        ).float()


class MuZeroSupport(OutputStrategy):
    """MuZero-style support: Invertible transform + categorical buckets."""

    def __init__(self, support_range: int, eps: float = 0.001):
        super().__init__()
        self.support_range = support_range
        self.eps = eps

    @property
    def num_bins(self) -> int:
        return 2 * self.support_range + 1

    @property
    def representation(self) -> BaseRepresentation:
        return TwoHotRepresentation(
            vmin=float(-self.support_range),
            vmax=float(self.support_range),
            bins=2 * self.support_range + 1,
        )

    def compute_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # MuZero uses cross entropy between projected support and logits
        return F.cross_entropy(predictions, targets, reduction="none")

    def get_distribution(
        self, network_output: Tensor
    ) -> torch.distributions.Distribution:
        return torch.distributions.Categorical(logits=network_output)

    def to_expected_value(self, network_output: Tensor) -> Tensor:
        probs = torch.softmax(network_output, dim=-1)
        return support_to_scalar(probs, self.support_range, self.eps)

    def scalar_to_target(self, scalar: Tensor) -> Tensor:
        return scalar_to_support(scalar, self.support_range, self.eps)


class C51Support(OutputStrategy):
    """C51-style support: Fixed atoms on a linear scale."""

    def __init__(self, v_min: float, v_max: float, num_atoms: int):
        super().__init__()
        self.v_min = v_min
        self.v_max = v_max
        self._num_atoms = num_atoms
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

    @property
    def num_bins(self) -> int:
        return self._num_atoms

    @property
    def representation(self) -> BaseRepresentation:
        return CategoricalRepresentation(
            vmin=self.v_min, vmax=self.v_max, bins=self._num_atoms
        )

    def compute_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # C51 typically uses KL Divergence, but CrossEntropy is mathematically equivalent
        # when targets are fixed distributions (ignoring entropy constant)
        return F.cross_entropy(predictions, targets, reduction="none")

    def get_distribution(
        self, network_output: Tensor
    ) -> torch.distributions.Distribution:
        return torch.distributions.Categorical(logits=network_output)

    def to_expected_value(self, network_output: Tensor) -> Tensor:
        probs = torch.softmax(network_output, dim=-1)
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


class GaussianStrategy(OutputStrategy):
    """
    Gaussian distribution for continuous actions.
    Network outputs mean and log_std for each action dimension.
    """

    def __init__(
        self, action_dim: int, min_log_std: float = -20.0, max_log_std: float = 2.0
    ):
        super().__init__()
        self.action_dim = action_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    @property
    def num_bins(self) -> int:
        return 2 * self.action_dim

    def compute_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # predictions are (mean, log_std)
        # targets are continuous values
        # Negative Log Likelihood
        dist = self.get_distribution(predictions)
        return -dist.log_prob(targets).sum(dim=-1)

    def get_distribution(
        self, network_output: Tensor
    ) -> torch.distributions.Distribution:
        mu, log_std = network_output.chunk(2, dim=-1)
        mu = torch.tanh(mu)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = log_std.exp()
        return torch.distributions.Normal(mu, std)

    def to_expected_value(self, network_output: Tensor) -> Tensor:
        # Return deterministic mean (squashed)
        mu, _ = network_output.chunk(2, dim=-1)
        return torch.tanh(mu)

    def scalar_to_target(self, scalar: Tensor) -> Tensor:
        # Target is already the continuous action
        return scalar
