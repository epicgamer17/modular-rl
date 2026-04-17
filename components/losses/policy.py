import torch
import torch.nn.functional as F
from typing import Any, Set, Dict
from core import PipelineComponent
from core import Blackboard
from core.path_resolver import resolve_blackboard_path
from core.contracts import (
    Key,
    ShapeContract,
    Policy,
    Action,
    Advantage,
    LogProb,
    LossScalar,
    Metric,
    Logits,
    Probs,
    LogProbs,
    Scalar,
)
from .infrastructure import apply_infrastructure
from core.validation import assert_same_batch, assert_time_val


class PolicyLoss(PipelineComponent):
    """
    Policy prediction loss module.
    Handles both soft-target (distribution) and hard-target (class index) policies.
    When target shapes differ from prediction shapes, targets are cast to long (class indices).
    Set log_kl=True to compute and log approximate KL divergence (useful for MuZero-style training).
    """

    def __init__(
        self,
        target_key: str = "policies",
        mask_key: str = "policy_mask",
        loss_fn: Any = F.cross_entropy,
        loss_factor: float = 1.0,
        name: str = "policy_loss",
        log_kl: bool = True,
    ):
        self.target_key = target_key
        self.mask_key = mask_key
        self.loss_fn = loss_fn
        self.loss_factor = loss_factor
        self.name = name
        self.log_kl = log_kl

        # Deterministic contracts computed at initialization
        # TODO: shape contracts? how do we handle time dimension from unrolls or from LSTM memory vs single step DQN, A2C, PPO, etc?
        self._requires = {
            Key(
                "predictions.policies",
                Policy[Logits],
                shape=ShapeContract(
                    semantic_shape=("B", "T", "A")
                ),
            ),
            Key(
                self.target_key,
                Policy[Probs],
                shape=ShapeContract(
                    semantic_shape=("B", "T", "A")
                ),
            ),
        }
        self._provides = {
            Key(f"losses.{self.name}", LossScalar): "new",
            Key(f"meta.{self.name}", Metric): "new",
            Key(f"meta.approx_kl", Metric): "optional",
        }

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    @property
    def constraints(self) -> list[str]:
        return [
            f"same_batch(predictions.policies, {self.target_key})",
            f"time_aligned(predictions.policies, {self.target_key})",
            f"feature_dim_match(predictions.policies, {self.target_key})",
        ]

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures both prediction and target exist, are tensors, and aligned."""
        from core.validation import (
            assert_in_blackboard,
            assert_is_tensor,
            assert_same_batch,
            assert_shape_sanity,
        )

        assert_in_blackboard(blackboard, "predictions.policies")
        assert_in_blackboard(blackboard, self.target_key)

        preds = blackboard.predictions["policies"]
        targets = resolve_blackboard_path(blackboard, self.target_key)

        assert_is_tensor(preds, msg=f"in {self.name} (predictions)")
        assert_is_tensor(targets, msg=f"in {self.name} (targets)")

        assert_same_batch(preds, targets, msg=f"in {self.name}")

        # Rigorous shape checks
        assert_shape_sanity(
            preds, min_rank=3, msg=f"Prediction {self.name} must have [B, T, K]"
        )

        # Hypothetical alignment for validation purposes
        # TODO: enforce always time dimension some how.

        check_targets = targets
        if check_targets.ndim == preds.ndim - 1:
            check_targets = check_targets.unsqueeze(1)

        assert check_targets.shape == preds.shape, (
            f"PolicyLoss Contract Violation: targets {targets.shape} (locally aligned to {check_targets.shape}) "
            f"must match predictions {preds.shape}. Expected [B, T, K]."
        )

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        """Compute policy loss and write to losses."""
        preds = blackboard.predictions["policies"]
        targets = resolve_blackboard_path(blackboard, self.target_key)

        # 1. Robust Time Alignment: Handle [B, K] -> [B, 1, K]
        # This occurs when learner_inference adds T=1 but buffer remains flat.
        if targets.ndim == preds.ndim - 1:
            targets = targets.unsqueeze(1)

        # 2. Flatten leads dimensions [B, T] into [N] for standard PyTorch loss functions
        B, T = preds.shape[:2]
        flat_preds = preds.flatten(0, 1)
        flat_targets = targets.flatten(0, 1)

        raw_loss = self.loss_fn(flat_preds, flat_targets, reduction="none")

        # Sanity check: ensure loss is elementwise-compatible [B*T]
        expected_shape = flat_preds.shape if raw_loss.ndim == flat_preds.ndim else (B * T,)
        if raw_loss.ndim > 1 and raw_loss.shape == flat_preds.shape:
             # Cross entropy with soft targets returns [B*T, K]
             pass
        elif raw_loss.shape != (B * T,):
             raise AssertionError(f"{self.name}: loss shape {raw_loss.shape} != expected {(B*T,)}")

        # If loss_fn returned per-class loss (for distribution targets), sum it up
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)

        elementwise_loss = raw_loss.reshape(B, T) * self.loss_factor

        # Pass through infrastructure
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        outputs = {
            f"losses.{self.name}": scalar_loss,
            f"meta.{self.name}": scalar_loss.item(),
        }

        # KL divergence logging (only meaningful for soft distribution targets)
        if self.log_kl and targets.shape == preds.shape:
            with torch.no_grad():
                log_q = F.log_softmax(preds, dim=-1)
                log_p = torch.log(targets + 1e-10)
                kl = (targets * (log_p - log_q)).sum(dim=-1).mean()
                outputs["meta.approx_kl"] = kl.item()

        return outputs


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

        # Deterministic contracts computed at initialization
        self._requires = {
            Key(
                "predictions.policies",
                Policy,
                shape=ShapeContract(
                    semantic_shape=("B", "T", "A")
                ),
            ),
            Key(
                self.actions_key,
                Action,
                shape=ShapeContract(semantic_shape=("B", "T")),
            ),
            Key(
                self.old_log_probs_key,
                LogProb[LogProbs],
                shape=ShapeContract(semantic_shape=("B", "T")),
            ),
            Key(
                self.advantages_key,
                Advantage[Scalar],
                shape=ShapeContract(semantic_shape=("B", "T")),
            ),
        }
        self._provides = {
            Key(f"losses.{self.name}", LossScalar): "new",
            Key(f"meta.{self.name}", Metric): "new",
            Key(f"meta.approx_kl", Metric): "new",
        }

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures all required tensors for PPO surrogate loss exist and match."""
        from core.validation import (
            assert_in_blackboard,
            assert_is_tensor,
            assert_same_batch,
            assert_time_val,
        )

        # Existence checks
        assert_in_blackboard(blackboard, "predictions.policies")
        assert_in_blackboard(blackboard, self.actions_key)
        assert_in_blackboard(blackboard, self.old_log_probs_key)
        assert_in_blackboard(blackboard, self.advantages_key)

        # Extraction for refined checks
        logits = blackboard.predictions["policies"]
        actions = resolve_blackboard_path(blackboard, self.actions_key)
        old_log_probs = resolve_blackboard_path(blackboard, self.old_log_probs_key)
        advantages = resolve_blackboard_path(blackboard, self.advantages_key)

        # Type and Shape alignment
        assert_is_tensor(logits, msg=f"in {self.name} (logits)")
        assert_is_tensor(actions, msg=f"in {self.name} (actions)")
        assert_is_tensor(old_log_probs, msg=f"in {self.name} (old_log_probs)")
        assert_is_tensor(advantages, msg=f"in {self.name} (advantages)")

        B, T = logits.shape[:2]
        assert_same_batch(logits, actions, msg=f"in {self.name}")
        assert_time_val(actions, T, msg=f"in {self.name}")
        assert_time_val(advantages, T, msg=f"in {self.name}")
        assert_same_batch(old_log_probs, advantages, msg=f"in {self.name}")

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        """Compute PPO surrogate loss."""
        # 1. Extract inputs
        policy_logits = blackboard.predictions["policies"]
        actions = resolve_blackboard_path(blackboard, self.actions_key)
        old_log_probs = resolve_blackboard_path(blackboard, self.old_log_probs_key)
        advantages = resolve_blackboard_path(blackboard, self.advantages_key)

        # 2. Robust Shape Alignment: Ensure buffer data matches [B, T] structure of predictions
        # ModularAgentNetwork.learner_inference returns [B, 1, K] or [B, T, K]
        # Buffer usually provides [B] or [B, T]
        B, T = policy_logits.shape[:2]

        if actions.shape != (B, T):
            actions = actions.reshape(B, T)
        if old_log_probs.shape != (B, T):
            old_log_probs = old_log_probs.reshape(B, T)
        if advantages.shape != (B, T):
            advantages = advantages.reshape(B, T)

        # Sanity check: [B, T] alignment for surrogate computation
        expected_bt = (B, T)
        assert actions.shape == expected_bt, f"{self.name}: actions shape {actions.shape} != {expected_bt}"
        assert (
            old_log_probs.shape == expected_bt
        ), f"{self.name}: old_log_probs shape {old_log_probs.shape} != {expected_bt}"
        assert (
            advantages.shape == expected_bt
        ), f"{self.name}: advantages shape {advantages.shape} != {expected_bt}"

        # 3. Compute log probabilities
        dist = torch.distributions.Categorical(logits=policy_logits)
        log_probs = dist.log_prob(actions)

        # 4. Compute surrogate objective
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * advantages
        )

        policy_loss = -torch.min(surr1, surr2)

        # 5. Add Entropy Regularization
        entropy = dist.entropy()
        elementwise_loss = (policy_loss - self.entropy_coefficient * entropy).reshape(B, T)
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        outputs = {
            f"losses.{self.name}": scalar_loss,
            f"meta.{self.name}": scalar_loss.item(),
        }

        with torch.no_grad():
            approx_kl = (old_log_probs - log_probs).mean()
            outputs["meta.approx_kl"] = approx_kl.item()

        return outputs
