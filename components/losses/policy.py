import torch
import torch.nn.functional as F
from typing import Any
from core import PipelineComponent
from core import Blackboard
from core.path_resolver import resolve_blackboard_path
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

    @property
    def reads(self) -> set[str]:
        return {"predictions.policies", self.target_key}

    @property
    def writes(self) -> set[str]:
        return {f"losses.{self.name}", f"meta.{self.name}"}

    def execute(self, blackboard: Blackboard) -> None:
        preds = blackboard.predictions["policies"]
        targets = resolve_blackboard_path(blackboard, self.target_key)
        # TODO: enforce always time dimension some how.

        # 2. Align targets to predictions time dimension if missing (e.g. from flat buffer)
        if targets.ndim == preds.ndim - 1:
            targets = targets.unsqueeze(1)

        # 3. Enforce shape matching (strictly require distributions)
        assert targets.shape == preds.shape, (
            f"PolicyLoss Contract Violation: targets {targets.shape} must match "
            f"predictions {preds.shape}. For index-based targets (BC), use "
            f"OneHotPolicyTargetComponent first."
        )
        flat_targets = targets.flatten(0, 1)

        raw_loss = self.loss_fn(preds.flatten(0, 1), flat_targets, reduction="none")
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)
        B, T = preds.shape[:2]
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
        actions_key: str = "actions",
        old_log_probs_key: str = "old_log_probs",
        advantages_key: str = "advantages",
        name: str = "policy_loss",
    ):
        self.clip_param = clip_param
        self.entropy_coefficient = entropy_coefficient
        self.mask_key = mask_key
        self.actions_key = actions_key
        self.old_log_probs_key = old_log_probs_key
        self.advantages_key = advantages_key
        self.name = name

    @property
    def reads(self) -> set[str]:
        return {
            "predictions.policies",
            self.actions_key,
            self.old_log_probs_key,
            self.advantages_key,
        }

    @property
    def writes(self) -> set[str]:
        return {f"losses.{self.name}", f"meta.{self.name}"}

    def execute(self, blackboard: Blackboard) -> None:
        policy_logits = blackboard.predictions["policies"]
        actions = resolve_blackboard_path(blackboard, self.actions_key)
        old_log_probs = resolve_blackboard_path(blackboard, self.old_log_probs_key)
        advantages = resolve_blackboard_path(blackboard, self.advantages_key)

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
