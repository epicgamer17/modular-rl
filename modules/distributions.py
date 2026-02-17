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
            torch.log(torch.tensor(2.0).to(x.device)) - x - F.softplus(-2.0 * x)
        )


class SampleDist:
    """
    Dist that allows backprop through sampling (Reparameterization Trick).
    """

    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def sample(self, sample_shape=torch.Size()):
        return self._dist.rsample(sample_shape)

    def mode(self):
        # Approximate mode for complex distributions
        # For Normal, mode == mean, but for TanhNormal it differs slightly
        sample = self._dist.rsample((self._samples,))
        logprob = self._dist.log_prob(sample)
        # Sum logprob over feature dimensions if independent
        if logprob.dim() > sample.dim() - 1:
            logprob = logprob.sum(-1)
        best_idx = torch.argmax(logprob, dim=0)
        # Gather the best sample
        # (Assuming sample is [S, B, D] and indices are [B])
        # This logic simplifies to just returning mean if simple dist
        if isinstance(self._dist, td.Independent) and isinstance(
            self._dist.base_dist, td.Normal
        ):
            return self._dist.mean
        return self._dist.mean  # Fallback

    def entropy(self):
        return self._dist.entropy()


class OneHotDist(td.OneHotCategorical):
    """
    Wrapper for OneHotCategorical that adds Straight-Through Estimator in sample.
    """

    def __init__(self, logits=None, probs=None):
        super().__init__(logits=logits, probs=probs)

    def mode(self):
        probs = self.probs
        mode = torch.argmax(probs, dim=-1)
        return F.one_hot(mode, num_classes=probs.shape[-1]).float()

    def sample(self, sample_shape=torch.Size()):
        # Straight-through estimator logic
        # 1. Sample hard indices
        if sample_shape:
            # TODO: Handle sample_shape > 1 for straight through if needed
            raise NotImplementedError("Sample shape not supported for ST-OneHot yet")

        # Hard sample
        indices = super().sample(sample_shape).argmax(dim=-1)
        one_hot = F.one_hot(indices, self.param_shape[-1]).float()

        # Soft sample for gradients (probabilities)
        probs = self.probs
        # y = hard + (soft - soft.detach())
        return one_hot + probs - probs.detach()


class Deterministic:
    """A mock PyTorch Distribution for deterministic continuous policies."""

    def __init__(self, value: torch.Tensor):
        self.value = value

    def sample(self, sample_shape=torch.Size()):
        # Sampling a deterministic distribution just returns the exact value
        # Must handle sample_shape if provided (though internal logic usually handles it)
        if sample_shape:
            return self.value.expand(sample_shape + self.value.shape)
        return self.value

    def rsample(self, sample_shape=torch.Size()):
        # Enables the reparameterization trick for gradients
        if sample_shape:
            return self.value.expand(sample_shape + self.value.shape)
        return self.value

    def log_prob(self, action: torch.Tensor):
        # The log probability of the exact action is 0 (log(1)).
        # Anything else is negative infinity.
        # We use a small epsilon match or strict equality?
        # Typically for deterministic policies in RL (like DDPG), log_prob isn't used
        # or is considered undefined/Dirac.
        # But to be safe:
        is_match = (action == self.value).all(dim=-1, keepdim=True)
        return torch.where(is_match, torch.zeros_like(self.value), -torch.inf)
