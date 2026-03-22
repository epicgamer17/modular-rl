import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


class TanhBijector(td.Transform):
    def __init__(self, validate_args=False):
        super().__init__(cache_size=1)

    @property
    def bijector(self):
        return self

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        # Clipping to avoid NaN in atanh
        y = torch.clamp(y, -0.99999997, 0.99999997)
        return torch.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # Formula: 2 * (log(2) - x - softplus(-2x))
        # This matches the TF implementation for numerical stability
        return 2.0 * (
            torch.log(torch.tensor(2.0, device=x.device)) - x - F.softplus(-2.0 * x)
        )


class MaskedCategorical(td.Categorical):
    """
    A categorical distribution that applies an action mask to the logits.
    The mask must be a boolean tensor where True indicates a valid action.
    """

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        # We use a large negative value to effectively set invalid action probabilities to 0.
        # -1e8 is used instead of -inf for better numerical stability during gradients.
        HUGE_NEG = torch.tensor(-1e8, dtype=logits.dtype, device=logits.device)
        masked_logits = torch.where(mask.bool(), logits, HUGE_NEG)
        super().__init__(logits=masked_logits)


class Deterministic(td.Distribution):
    """
    A deterministic 'distribution' that always returns the same value.
    Useful for wrapping deterministic outputs in an interface that expects distributions.
    """

    def __init__(self, value: torch.Tensor, validate_args=None):
        self.value = value
        batch_shape = value.shape[:-1]
        event_shape = value.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        return self.value.expand(sample_shape + self.value.shape)

    def log_prob(self, value):
        raise NotImplementedError

    @property
    def mode(self):
        return self.value

    @property
    def mean(self):
        return self.value
