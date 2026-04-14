import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Set
from core import PipelineComponent, Blackboard
from core.path_resolver import resolve_blackboard_path
from core.contracts import Key, Reward, ToPlay, PolicyLogits, ValueTarget, LossScalar, SemanticType, Observation
from .infrastructure import apply_infrastructure
from core.validation import assert_same_batch, assert_compatible_value


class RewardLoss(PipelineComponent):
    """Reward prediction loss module."""

    def __init__(
        self,
        loss_fn: Any,
        loss_factor: float = 1.0,
        mask_key: str = "reward_mask",
        target_key: str = "rewards",
        name: str = "reward_loss",
    ):
        self.loss_fn = loss_fn
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.target_key = target_key
        self.name = name
        
        # Deterministic contracts computed at initialization
        self._requires = {
            Key("predictions.rewards", Reward),
            Key(self.target_key, Reward)
        }
        self._provides = {Key(f"losses.{self.name}", LossScalar)}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Set[Key]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        preds = blackboard.predictions["rewards"]
        targets = resolve_blackboard_path(blackboard, self.target_key)
        assert_same_batch(preds, targets, msg=f"in {self.name}")
        assert_compatible_value(preds, targets, msg=f"in {self.name}")

    def execute(self, blackboard: Blackboard) -> None:
        preds = blackboard.predictions["rewards"]
        targets = resolve_blackboard_path(blackboard, self.target_key)

        B, T = preds.shape[:2]

        # Handle scalar vs distributional heads
        # If preds is [B, T, 1] and targets is [B, T], we must align
        # TODO: handle this better, enforce a contract
        if preds.ndim == 3 and targets.ndim == 2:
            targets = targets.unsqueeze(-1)

        raw_loss = self.loss_fn(
            preds.flatten(0, 1), targets.flatten(0, 1), reduction="none"
        )

        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)
        elementwise_loss = raw_loss.reshape(B, T) * self.loss_factor

        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()


class ToPlayLoss(PipelineComponent):
    """To-play prediction loss module (classification)."""

    def __init__(
        self,
        loss_fn: Any = F.cross_entropy,
        loss_factor: float = 1.0,
        mask_key: str = "to_play_mask",
        target_key: str = "to_plays",
        name: str = "to_play_loss",
    ):
        self.loss_fn = loss_fn
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.target_key = target_key
        self.name = name
        
        # Deterministic contracts computed at initialization
        self._requires = {
            Key("predictions.to_plays", ToPlay),
            Key(self.target_key, ToPlay)
        }
        self._provides = {Key(f"losses.{self.name}", LossScalar)}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Set[Key]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        preds = blackboard.predictions.get("to_plays")
        targets = resolve_blackboard_path(blackboard, self.target_key)
        if preds is not None:
            assert_same_batch(preds, targets, msg=f"in {self.name}")

    def execute(self, blackboard: Blackboard) -> None:
        preds = blackboard.predictions.get("to_plays")
        try:
            targets = resolve_blackboard_path(blackboard, self.target_key)
        except KeyError:
            targets = None

        if preds is None or targets is None:
            return

        B, T = preds.shape[:2]

        # F.cross_entropy expects [N, C] and [N, C] (soft targets)
        # or [N, C] and [N] (class indices).
        # targets is already [B, T, C] (projections or one-hots)
        raw_loss = self.loss_fn(
            preds.flatten(0, 1), targets.flatten(0, 1), reduction="none"
        )
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)
        elementwise_loss = raw_loss.reshape(B, T) * self.loss_factor

        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()


class ChanceQLoss(PipelineComponent):
    """Loss for stochastic muzero chance Q heads."""

    def __init__(
        self,
        loss_factor: float = 1.0,
        mask_key: str = "afterstate_value_mask",
        target_key: str = "chance_values_next",
        name: str = "chance_q_loss",
    ):
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.target_key = target_key
        self.name = name
        
        # Deterministic contracts computed at initialization
        self._requires = {
            Key("predictions.chance_q_logits", PolicyLogits),
            Key(self.target_key, ValueTarget)
        }
        self._provides = {Key(f"losses.{self.name}", LossScalar)}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Set[Key]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> None:
        try:
            formatted_target = resolve_blackboard_path(blackboard, self.target_key)
        except KeyError:
            formatted_target = None

        if formatted_target is None:
            raise KeyError(
                f"ChanceQLoss requires '{self.target_key}' in targets or data."
            )

        pred = blackboard.predictions["chance_q_logits"]
        B, T = pred.shape[:2]

        flat_pred = pred.reshape(B * T, -1)
        flat_target = formatted_target.reshape(B * T, -1)

        raw_loss = F.cross_entropy(flat_pred, flat_target, reduction="none")
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)

        elementwise_loss = raw_loss.reshape(B, T) * self.loss_factor

        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()


class ConsistencyLoss(PipelineComponent):
    """Latent consistency loss (EfficientZero style)."""

    def __init__(
        self,
        loss_factor: float = 1.0,
        mask_key: str = "masks",
        name: str = "consistency_loss",
    ):
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.name = name
        
        # Deterministic contracts computed at initialization
        self._requires = {
            Key("predictions.projected_latents", SemanticType),
            Key("targets.consistency_targets", SemanticType)
        }
        self._provides = {Key(f"losses.{self.name}", LossScalar)}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Set[Key]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> None:
        preds = blackboard.predictions.get("projected_latents")
        targets = blackboard.targets.get("consistency_targets")

        if preds is None or targets is None:
            return

        B, T = preds.shape[:2]

        preds_norm = F.normalize(preds, p=2.0, dim=-1)
        targets_norm = F.normalize(targets, p=2.0, dim=-1)

        elementwise_loss = -(preds_norm * targets_norm).sum(dim=-1) * self.loss_factor
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()


class SigmaLoss(PipelineComponent):
    """Loss for sigma prediction (stochastic MuZero)."""

    def __init__(
        self,
        loss_factor: float = 1.0,
        mask_key: str = "masks",
        target_key: str = "sigmas",
        name: str = "sigma_loss",
    ):
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.target_key = target_key
        self.name = name
        
        # Deterministic contracts computed at initialization
        self._requires = {
            Key("predictions.sigma_logits", PolicyLogits),
            Key(self.target_key, SemanticType)
        }
        self._provides = {Key(f"losses.{self.name}", LossScalar)}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Set[Key]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> None:
        preds = blackboard.predictions.get("sigma_logits")
        try:
            targets = resolve_blackboard_path(blackboard, self.target_key)
        except KeyError:
            targets = None

        if preds is None or targets is None:
            return

        B, T = preds.shape[:2]
        raw_loss = F.cross_entropy(
            preds.flatten(0, 1), targets.flatten(0, 1), reduction="none"
        )
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)

        elementwise_loss = raw_loss.reshape(B, T) * self.loss_factor
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()


class CommitmentLoss(PipelineComponent):
    """VQ-VAE commitment cost for encoder (stochastic MuZero)."""

    def __init__(
        self,
        loss_factor: float = 1.0,
        mask_key: str = "masks",
        name: str = "commitment_loss",
    ):
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.name = name
        
        # Deterministic contracts computed at initialization
        self._requires = {Key("predictions.commitment_loss", LossScalar)}
        self._provides = {Key(f"losses.{self.name}", LossScalar)}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Set[Key]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> None:
        elementwise_loss = blackboard.predictions.get("commitment_loss")
        if elementwise_loss is None:
            return

        elementwise_loss = elementwise_loss * self.loss_factor
        scalar_loss = apply_infrastructure(elementwise_loss, blackboard, self.mask_key)

        blackboard.losses[self.name] = scalar_loss
        blackboard.meta[self.name] = scalar_loss.item()


class LatentConsistencyComponent(PipelineComponent):
    """EfficientZero consistency loss target builder."""

    def __init__(self, agent_network: nn.Module):
        self.agent_network = agent_network

    @property
    def requires(self) -> Set[Key]:
        return {Key("data.unroll_observations", Observation)}

    @property
    def provides(self) -> Set[Key]:
        return {Key("targets.consistency_targets", SemanticType)}

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> None:
        real_obs = blackboard.data["unroll_observations"].float()
        batch_size, unroll_len = real_obs.shape[:2]
        flat_obs = real_obs.flatten(0, 1)

        with torch.no_grad():
            initial_out = self.agent_network.obs_inference(flat_obs)
            real_latents = initial_out.network_state.dynamics
            proj_targets = self.agent_network.project(real_latents, grad=False)
            normalized_targets = torch.nn.functional.normalize(
                proj_targets, p=2.0, dim=-1, eps=1e-5
            )

        blackboard.targets["consistency_targets"] = normalized_targets.reshape(
            batch_size, unroll_len, -1
        ).detach()
