from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
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
    def num_features(self) -> int:
        """The required output dimension for the network head (e.g., bins or classes)."""
        pass

    @abstractmethod
    def to_inference(self, logits: Tensor) -> Any:
        """
        Converts raw logits into a format suitable for the Actor or MCTS.
        Returns:
            A torch.distributions.Distribution object OR a Tensor (expected value).
        """
        pass

    @abstractmethod
    def to_expected_value(self, logits: Tensor) -> Tensor:
        """Converts network output logits strictly into expected scalar values."""
        pass

    @abstractmethod
    def to_representation(self, scalar_targets: Tensor) -> Tensor:
        """Converts mathematically pure targets into the target distribution expected by the loss engine."""
        pass

    def format_target(
        self, targets: Dict[str, Tensor], target_key: str = "values"
    ) -> Tensor:
        """
        Converts raw target ingredients from the TargetBuilder into the final target representation.
        Default implementation handles simple scalar values.
        """
        return self.to_representation(targets[target_key])


class ScalarRepresentation(BaseRepresentation):
    """
    Identity representation for raw scalar values.
    Used for standard MSE/Regression heads.
    """

    @property
    def num_features(self) -> int:
        return 1

    def to_inference(self, logits: Tensor) -> Tensor:
        """For regression, the prediction is the scalar value."""
        return self.to_expected_value(logits)

    def to_expected_value(self, logits: Tensor) -> Tensor:
        return logits.reshape(logits.shape[:-1])

    def to_representation(self, scalar_targets: Tensor) -> Tensor:
        return scalar_targets.reshape(*scalar_targets.shape, 1)


class TwoHotRepresentation(BaseRepresentation):
    """
    Representation that maps scalars to a categorical support via two-hot projection.
    Provides common logic for expectation (to_expected_value) and two-hot projection (to_representation).
    Used primarily for MuZero-style values and rewards.
    """

    def __init__(self, vmin: float, vmax: float, bins: int):
        assert bins > 1, f"Discrete representations require at least 2 bins, got {bins}"
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.support = torch.linspace(vmin, vmax, bins)
        self.delta_z = (vmax - vmin) / (bins - 1)

    @property
    def num_features(self) -> int:
        return self.bins

    def to_inference(self, logits: Tensor) -> torch.distributions.Categorical:
        """Returns the categorical distribution over support bins."""
        return torch.distributions.Categorical(logits=logits)

    def to_expected_value(self, logits: Tensor) -> Tensor:
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

    def format_target(
        self, targets: Dict[str, Tensor], target_key: str = "values"
    ) -> Tensor:
        """Handle ingredient-based targets or fallback to scalar."""
        if target_key in targets:
            return self.to_representation(targets[target_key])
        raise ValueError(
            f"{self.__class__.__name__} received a targets dict without '{target_key}' or algorithm-specific ingredients."
        )


class C51Representation(TwoHotRepresentation):
    def __init__(self, vmin: float, vmax: float, bins: int):
        super().__init__(vmin, vmax, bins)

    def format_target(
        self, targets: Dict[str, Tensor], target_key: str = "next_q_logits"
    ) -> Tensor:
        """
        The Baker: Implements the C51 Bellman Projection.
        Projects shifted atom probabilities back onto the fixed support grid.
        Supports [B, T] inputs via Flatten-Process-Unflatten.
        """
        if "next_q_logits" not in targets:
            raise ValueError(
                "C51Representation requires 'next_q_logits' in targets."
            )

        # 1. Capture original prefix shape (e.g., [B, T])
        raw_next_logits = targets["next_q_logits"]
        assert (
            raw_next_logits.shape[-1] == self.bins
        ), f"C51Representation expected {self.bins} atoms, got {raw_next_logits.shape[-1]}"
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
    def num_features(self) -> int:
        return self.bins

    def to_inference(self, logits: Tensor) -> torch.distributions.Categorical:
        return self._inner.to_inference(logits)

    def to_expected_value(self, logits: Tensor) -> Tensor:
        log_scalar = self._inner.to_expected_value(logits)
        return self._exp_inverse(log_scalar)

    def to_representation(self, scalar_targets: Tensor) -> Tensor:
        log_targets = self._log_transform(scalar_targets)
        return self._inner.to_representation(log_targets)


class ClassificationRepresentation(BaseRepresentation):
    def __init__(self, num_classes: int):
        self._num_classes = num_classes

    @property
    def num_features(self) -> int:
        return self._num_classes

    def to_inference(self, logits: Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=logits)

    def to_expected_value(self, logits: Tensor) -> Tensor:
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


class IdentityRepresentation(BaseRepresentation):
    """
    Identity representation for vectors or scalars where no conversion is needed.
    """

    def __init__(self, num_features: int = 1):
        self._num_features = num_features

    @property
    def num_features(self) -> int:
        return self._num_features

    def to_inference(self, logits: Tensor) -> Tensor:
        return logits

    def to_expected_value(self, logits: Tensor) -> Tensor:
        return logits

    def to_representation(self, scalar_targets: Tensor) -> Tensor:
        return scalar_targets

    def format_target(
        self, targets: Dict[str, Tensor], target_key: str = "values"
    ) -> Tensor:
        return targets.get(target_key, targets.get("values"))


class GaussianRepresentation(BaseRepresentation):
    """
    Representation for continuous values using a Gaussian distribution.
    Expects logits to be [..., 2 * action_dim] where first half is mean, second is log_std.
    """

    def __init__(
        self,
        action_dim: int,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
    ):
        self.action_dim = action_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    @property
    def num_features(self) -> int:
        return 2 * self.action_dim

    def to_inference(self, logits: Tensor) -> torch.distributions.Normal:
        """Returns a Normal distribution from [mean, log_std] logits."""
        mean, log_std = torch.chunk(logits, 2, dim=-1)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mean, std)

    def to_expected_value(self, logits: Tensor) -> Tensor:
        """Returns the mean of the Gaussian."""
        mean, _ = torch.chunk(logits, 2, dim=-1)
        return mean

    def to_representation(self, scalar_targets: Tensor) -> Tensor:
        """For continuous targets, the representation is often just the scalar target."""
        return scalar_targets


def get_representation(
    config: Optional[Union[Dict[str, Any], int, Any]] = None,
    **kwargs,
) -> BaseRepresentation:
    """
    Factory to create a representation strategy.
    Can take a configuration dictionary (e.g. from output_strategy),
    a ConfigBase object, or direct arguments.
    """
    if config is None:
        config = kwargs
    elif isinstance(config, int):
        config = {"num_classes": config, **kwargs}
    elif isinstance(config, dict):
        config = {**config, **kwargs}
    elif hasattr(config, "config_dict"):
        config = {**config.config_dict, **kwargs}
    else:
        # Fallback for unexpected types
        config = kwargs

    mode = config.get("type", config.get("mode", "scalar"))
    num_classes = config.get(
        "num_classes", config.get("num_features", config.get("num_atoms", 1))
    )
    bins = config.get("bins", num_classes)
    vmin = config.get("vmin", config.get("v_min"))
    vmax = config.get("vmax", config.get("v_max"))
    support_range = config.get("support_range")
    identity = config.get("identity", False)

    if identity:
        return IdentityRepresentation()

    # Two-Hot / Categorical support logic
    if vmin is not None and vmax is not None:
        if mode == "exponential":
            return ExponentialBucketsRepresentation(vmin, vmax, bins)
        elif mode == "categorical" or mode == "c51":
            return C51Representation(vmin, vmax, bins)
        return TwoHotRepresentation(vmin, vmax, bins)

    # MuZero style support_range
    if support_range is not None or mode == "muzero":
        if support_range is None:
            # Fallback/Default for muzero if not provided?
            # Usually it's 300 or similar, but better to require it.
            support_range = 300.0

        support_range = float(support_range)
        return TwoHotRepresentation(
            -support_range, support_range, int(2 * support_range + 1)
        )

    # Classification logic
    if num_classes > 1 or mode == "classification" or mode == "categorical":
        return ClassificationRepresentation(num_classes)

    # Gaussian / Continuous logic
    if mode == "gaussian" or mode == "continuous":
        action_dim = config.get("action_dim", 1)
        min_log_std = config.get("min_log_std", -20.0)
        max_log_std = config.get("max_log_std", 2.0)
        return GaussianRepresentation(action_dim, min_log_std, max_log_std)

    return ScalarRepresentation()
