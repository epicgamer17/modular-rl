import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional
from learner.pipeline.base import PipelineComponent
from learner.core import Blackboard
from learner.losses.base import apply_infrastructure


class PolicyLoss(PipelineComponent):
    """Policy prediction loss module."""
    def __init__(
        self,
        loss_fn: Any,
        loss_factor: float = 1.0,
        mask_key: str = "policy_mask",
        name: str = "policy_loss",
    ):
        self.loss_fn = loss_fn
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.name = name

    def execute(self, blackboard: Blackboard) -> None:
        preds = blackboard.predictions["policies"]
        targets = blackboard.targets["policies"]
        
        B, T = preds.shape[:2]
        
        # Flatten B, T
        raw_loss = self.loss_fn(preds.flatten(0, 1), targets.flatten(0, 1), reduction="none")
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)
        elementwise_loss = raw_loss.reshape(B, T) * self.loss_factor

        # Pass through infrastructure
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        # Logging
        with torch.no_grad():
            log_q = F.log_softmax(preds, dim=-1)
            log_p = torch.log(targets + 1e-10)
            kl = (targets * (log_p - log_q)).sum(dim=-1).mean()
            blackboard.meta["approx_kl"] = kl.item()

        # Write out
        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()


class ClippedSurrogateLoss(PipelineComponent):
    """PPO Clipped Surrogate Policy Loss.

    Reconstructs the action distribution internally from raw policy logits
    to comply with the 'No PyTorch Object Passing in Loss Functions' rule.
    Loss functions operate on raw tensors, not Distribution objects.
    """
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
        """Compute PPO clipped surrogate loss from raw policy logits.

        Reads raw logits from ``blackboard.predictions["policies"]`` and
        builds a ``Categorical`` distribution locally so that the pipeline
        never transports Distribution objects.
        """
        policy_logits = blackboard.predictions["policies"]
        actions = blackboard.targets["actions"]
        old_log_probs = blackboard.targets["old_log_probs"]
        advantages = blackboard.targets["advantages"]

        # Build distribution from raw logits inside the loss  [B, T, num_actions] -> Categorical
        dist = torch.distributions.Categorical(logits=policy_logits)

        # [B, T]
        log_probs = dist.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages

        entropy = dist.entropy()
        elementwise_loss = -torch.min(surr1, surr2) - self.entropy_coefficient * entropy

        # Pass through infrastructure
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        with torch.no_grad():
            approx_kl = (old_log_probs - log_probs).mean()
            blackboard.meta["approx_kl"] = approx_kl.item()

        # Write out
        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()


class ImitationLoss(PipelineComponent):
    """Imitation learning loss (Behavioral Cloning)."""
    def __init__(
        self,
        loss_fn: Any,
        loss_factor: float = 1.0,
        mask_key: str = "policy_mask",
        target_key: str = "policies",
        name: str = "policy_loss",
    ):
        self.loss_fn = loss_fn
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.target_key = target_key
        self.name = name

    def execute(self, blackboard: Blackboard) -> None:
        pred = blackboard.predictions["policies"]
        target = blackboard.targets[self.target_key]
        
        B, T = pred.shape[:2]
        
        # Determine if target is class indices or probabilities
        if target.shape == pred.shape:
            flat_target = target.flatten(0, 1)
        else:
            flat_target = target.flatten(0, 1).long()

        raw_loss = self.loss_fn(pred.flatten(0, 1), flat_target, reduction="none")
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)
        elementwise_loss = raw_loss.reshape(B, T) * self.loss_factor

        # Pass through infrastructure
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        # Write out
        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()
