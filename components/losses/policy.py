import torch
import torch.nn.functional as F
from typing import Any
from core import PipelineComponent
from core import Blackboard
from .infrastructure import apply_infrastructure


class PolicyLoss(PipelineComponent):
    """
    Policy prediction loss module.
    Handles both soft-target (distribution) and hard-target (class index) policies.
    When target shapes differ from prediction shapes, targets are cast to long (class indices).
    Set log_kl=True to compute and log approximate KL divergence (useful for MuZero-style training).
    """

    def __init__(
        self,
        loss_fn: Any,
        loss_factor: float = 1.0,
        mask_key: str = "policy_mask",
        target_key: str = "policies",
        log_kl: bool = True,
        name: str = "policy_loss",
    ):
        self.loss_fn = loss_fn
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.target_key = target_key
        self.log_kl = log_kl
        self.name = name

    def execute(self, blackboard: Blackboard) -> None:
        preds = blackboard.predictions["policies"]
        targets = blackboard.targets[self.target_key]

        B, T = preds.shape[:2]

        # Handle shape mismatch: if targets aren't the same shape as preds,
        # treat them as class indices (e.g. imitation learning / behavioral cloning)
        if targets.shape == preds.shape:
            flat_targets = targets.flatten(0, 1)
        else:
            flat_targets = targets.flatten(0, 1).long()

        raw_loss = self.loss_fn(
            preds.flatten(0, 1), flat_targets, reduction="none"
        )
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)
        elementwise_loss = raw_loss.reshape(B, T) * self.loss_factor

        # Pass through infrastructure
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        # KL divergence logging (only meaningful for soft distribution targets)
        if self.log_kl and targets.shape == preds.shape:
            with torch.no_grad():
                log_q = F.log_softmax(preds, dim=-1)
                log_p = torch.log(targets + 1e-10)
                kl = (targets * (log_p - log_q)).sum(dim=-1).mean()
                blackboard.meta["approx_kl"] = kl.item()

        # Write out
        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()


class ClippedSurrogateLoss(PipelineComponent):
    """PPO Clipped Surrogate Policy Loss."""

    def __init__(
        self,
        clip_param: float,
        entropy_coefficient: float,
        mask_key: str = "policy_mask",
        name: str = "policy_loss",
    ):
        self.clip_param = clip_param
        self.entropy_coefficient = entropy_coefficient
        self.mask_key = mask_key
        self.name = name

    def execute(self, blackboard: Blackboard) -> None:
        policy_logits = blackboard.predictions["policies"]
        actions = blackboard.targets["actions"]
        old_log_probs = blackboard.targets["old_log_probs"]
        advantages = blackboard.targets["advantages"]

        dist = torch.distributions.Categorical(logits=policy_logits)
        log_probs = dist.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * advantages
        )

        entropy = dist.entropy()
        elementwise_loss = -torch.min(surr1, surr2) - self.entropy_coefficient * entropy

        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        with torch.no_grad():
            approx_kl = (old_log_probs - log_probs).mean()
            blackboard.meta["approx_kl"] = approx_kl.item()

        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()
