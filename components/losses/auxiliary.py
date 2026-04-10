import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from core import PipelineComponent
from core import Blackboard
from .infrastructure import apply_infrastructure


class RewardLoss(PipelineComponent):
    """Reward prediction loss module."""

    def __init__(
        self,
        loss_fn: Any,
        loss_factor: float = 1.0,
        mask_key: str = "reward_mask",
        name: str = "reward_loss",
    ):
        self.loss_fn = loss_fn
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.name = name

    def execute(self, blackboard: Blackboard) -> None:
        preds = blackboard.predictions["rewards"]
        targets = blackboard.targets["rewards"]

        B, T = preds.shape[:2]

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
    """To-play prediction loss module."""

    def __init__(
        self,
        loss_fn: Any = F.binary_cross_entropy_with_logits,
        loss_factor: float = 1.0,
        mask_key: str = "to_play_mask",
        name: str = "to_play_loss",
    ):
        self.loss_fn = loss_fn
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.name = name

    def execute(self, blackboard: Blackboard) -> None:
        preds = blackboard.predictions.get("to_plays")
        targets = blackboard.targets.get("to_plays")

        if preds is None or targets is None:
            return

        B, T = preds.shape[:2]
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
        name: str = "chance_q_loss",
    ):
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.name = name

    def execute(self, blackboard: Blackboard) -> None:
        formatted_target = blackboard.targets.get("chance_values_next")
        if formatted_target is None:
            raise KeyError("ChanceQLoss requires 'chance_values_next' in targets.")

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
        name: str = "sigma_loss",
    ):
        self.loss_factor = loss_factor
        self.mask_key = mask_key
        self.name = name

    def execute(self, blackboard: Blackboard) -> None:
        preds = blackboard.predictions.get("sigma_logits")
        targets = blackboard.targets.get("sigmas")

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
