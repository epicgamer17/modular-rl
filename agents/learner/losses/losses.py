import torch
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
from torch import Tensor
from modules.world_models.inference_output import InferenceOutput
from utils.telemetry import append_metric


class BaseLoss(ABC):
    """
    Unified base class and execution engine for all loss modules.
    Handles data extraction, representation bridging, and masking.
    """

    def __init__(
        self,
        config: Any,
        device: torch.device,
        pred_key: str,
        target_key: str,
        mask_key: str,
        representation: Any, # Mandatory
        loss_fn: Optional[Any] = None,
        optimizer_name: str = "default",
        loss_factor: float = 1.0,
    ):
        self.config = config
        self.device = device
        self.pred_key = pred_key
        self.target_key = target_key
        self.mask_key = mask_key
        self.representation = representation
        self.loss_fn = loss_fn
        self.optimizer_name = optimizer_name
        self.loss_factor = loss_factor
        self.name = self.__class__.__name__

    @property
    def required_predictions(self) -> set[str]:
        return {self.pred_key}

    @property
    def required_targets(self) -> set[str]:
        return {self.target_key, self.mask_key}

    def should_compute(self, predictions: dict, targets: dict, context: dict) -> bool:
        """Determine if this loss should be computed for the batch."""
        return True

    def get_mask(self, targets: dict) -> torch.Tensor:
        """Get the mask to apply for this loss (B, T)."""
        mask = targets.get(self.mask_key)
        if mask is None:
            raise KeyError(
                f"Missing required mask '{self.mask_key}' for {self.__class__.__name__}"
            )
        return mask

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> torch.Tensor:
        """
        Pure Vectorized Execution Engine.
        Returns:
            elementwise_tensor of shape (B, T)
        """
        # 1. Extract [B, T, ...] inputs
        pred = predictions[self.pred_key]
        target_ingredients = targets # Representation will pull what it needs
        
        # 2. Format targets through the Representation bridge
        # We pass target_key to help representations that handle multiple inputs
        if hasattr(self.representation, "format_target") and callable(self.representation.format_target):
            formatted_target = self.representation.format_target(target_ingredients, target_key=self.target_key)
        else:
            formatted_target = targets[self.target_key]

        # 3. Apply raw PyTorch loss function
        assert pred.shape == formatted_target.shape, (
            f"{self.__class__.__name__}: shape mismatch between pred {pred.shape} "
            f"and formatted_target {formatted_target.shape}"
        )
        
        # Determine B, T robustly
        B, T = 1, 1
        for key, tensor in targets.items():
            if key == "gradient_scales":
                continue
            if torch.is_tensor(tensor) and tensor.ndim >= 2:
                B, T = tensor.shape[:2]
                break
        
        # Flatten B, T to handle universal T contract correctly
        orig_shape = pred.shape
        flat_pred = pred.reshape(B * T, *orig_shape[2:])
        flat_target = formatted_target.reshape(B * T, *orig_shape[2:])
        
        raw_loss = self.loss_fn(flat_pred, flat_target, reduction="none")
        
        # 4. Collapse and Reshape to [B, T] result
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.sum(dim=-1)
            
        return self.loss_factor * raw_loss.reshape(B, T)

    def compute_priority(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> Optional[torch.Tensor]:
        return None


# ============================================================================
# OLD DQN-STYLE LOSSES (Updated to work with unified interface)
# ============================================================================


class StandardDQNLoss(BaseLoss):
    def __init__(
        self,
        config,
        device,
        representation: Any,
        optimizer_name: str = "default",
        mask_key: str = "value_mask",
    ):
        super().__init__(
            config=config, 
            device=device,
            pred_key="q_values",
            target_key="q_values",
            mask_key=mask_key,
            representation=representation,
            loss_fn=config.loss_function,
            optimizer_name=optimizer_name
        )


    def compute_loss(
        self, predictions: dict, targets: dict, context: dict
    ) -> torch.Tensor:
        q_values = predictions["q_values"]
        actions = targets["actions"].long()

        # 1. Capture and Validate Shapes
        assert q_values.ndim == 3, f"StandardDQNLoss requires [B, T, Actions] predictions, got {q_values.shape}"
        assert actions.ndim == 2, f"StandardDQNLoss requires [B, T] action targets, got {actions.shape}"
        B, T, num_actions = q_values.shape

        # 2. Flatten for vectorized indexing
        flat_q = q_values.reshape(-1, num_actions)
        flat_actions = actions.reshape(-1)

        # [B * T]
        selected_q = flat_q[torch.arange(B * T, device=self.device), flat_actions]
        targets_val = targets["q_values"].reshape(-1)

        # 3. Compute and Unflatten
        assert selected_q.shape == targets_val.shape, f"StandardDQNLoss: shape mismatch between selected_q {selected_q.shape} and targets_val {targets_val.shape}"
        loss = self.config.loss_function(selected_q, targets_val, reduction="none")
        return loss.reshape(B, T)

    def compute_priority(self, predictions: dict, targets: dict, context: dict) -> torch.Tensor:
        q_values = predictions["q_values"]
        actions = targets["actions"].long()
        target_q = targets["q_values"]
        B, T = actions.shape[:2]

        # Q-value of the chosen action [B, T]
        flat_q = q_values.reshape(-1, q_values.shape[-1])
        pred_q = flat_q[torch.arange(B * T, device=self.device), actions.reshape(-1)].reshape(B, T)

        # Return max TD-error over the sequence for each batch element [B]
        assert target_q.shape == pred_q.shape, f"StandardDQNLoss priority: shape mismatch between target_q {target_q.shape} and pred_q {pred_q.shape}"
        td_error = torch.abs(target_q - pred_q).detach()
        return td_error.max(dim=1).values


class C51Loss(BaseLoss):
    def __init__(
        self,
        config,
        device,
        representation: Any,
        action_selector: Optional[object] = None,
        optimizer_name: str = "default",
        mask_key: str = "value_mask",
    ):
        super().__init__(
            config=config,
            device=device,
            pred_key="q_logits",
            target_key="q_logits",
            mask_key=mask_key,
            representation=representation,
            loss_fn=None, # Custom internal math
            optimizer_name=optimizer_name
        )
        self.action_selector = action_selector
        self.support = torch.linspace(
            self.config.v_min,
            self.config.v_max,
            self.config.atom_size,
            device=self.device,
        )

    @property
    def required_predictions(self) -> set[str]:
        return {"q_logits"}

    @property
    def required_targets(self) -> set[str]:
        return {"q_logits", "actions"}

    def _project_target_distribution(
        self,
        rewards: torch.Tensor,
        terminal_mask: torch.Tensor,
        next_probs: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = rewards.shape[0]
        discount = self.config.discount_factor**self.config.n_step
        delta_z = (self.config.v_max - self.config.v_min) / (self.config.atom_size - 1)

        # Compute the projected support: Tz = r + gamma * z
        tz = (
            rewards.view(-1, 1)
            + discount * (~terminal_mask.bool()).view(-1, 1) * self.support.view(1, -1)
        ).clamp(self.config.v_min, self.config.v_max)

        # Map back to index space: b = (Tz - v_min) / delta_z
        b = (tz - self.config.v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        dist_l = u.float() - b
        dist_u = b - l.float()

        mask_equal = l == u
        dist_l[mask_equal] = 1.0
        dist_u[mask_equal] = 0.0

        projected = torch.zeros((batch_size, self.config.atom_size), device=self.device)
        projected.scatter_add_(1, l, next_probs * dist_l)
        projected.scatter_add_(1, u, next_probs * dist_u)

        return projected

    def compute_loss(
        self, predictions: dict, targets: dict, context: dict
    ) -> torch.Tensor:
        online_q_logits = predictions["q_logits"]
        actions = targets["actions"].long()
        target_q_logits = targets["q_logits"]

        # 1. Capture and Validate
        assert online_q_logits.ndim == 4, f"C51Loss requires [B, T, Actions, Atoms], got {online_q_logits.shape}"
        B, T, num_actions, atoms = online_q_logits.shape

        # 2. Flatten and select
        flat_logits = online_q_logits.reshape(-1, num_actions, atoms)
        flat_actions = actions.reshape(-1)
        
        # [B * T, Atoms]
        chosen_logits = flat_logits[torch.arange(B * T, device=self.device), flat_actions]
        log_probs = F.log_softmax(chosen_logits, dim=-1)
        
        # [B * T, Atoms]
        flat_target_q_logits = target_q_logits.reshape(-1, atoms)
        
        # [B * T]
        assert flat_target_q_logits.shape == log_probs.shape, f"C51Loss: shape mismatch between flat_target_q_logits {flat_target_q_logits.shape} and log_probs {log_probs.shape}"
        loss = -(flat_target_q_logits * log_probs).sum(dim=-1)
        return loss.reshape(B, T)

    def compute_priority(self, predictions: dict, targets: dict, context: dict) -> torch.Tensor:
        # Predict Expected Q
        probs = torch.softmax(predictions["q_logits"], dim=-1)
        support = self.support.to(device=probs.device, dtype=probs.dtype)
        q_values = (probs * support).sum(dim=-1) # [B, T, Actions]
        
        actions = targets["actions"].long()
        B, T = actions.shape[:2]
        
        flat_q = q_values.reshape(-1, q_values.shape[-1])
        pred_q = flat_q[torch.arange(B * T, device=self.device), actions.reshape(-1)].reshape(B, T)

        # Target Expected Q [B, T]
        target_q = (targets["q_logits"] * support).sum(dim=-1)
        assert target_q.shape == pred_q.shape, f"C51Loss priority: shape mismatch between target_q {target_q.shape} and pred_q {pred_q.shape}"

        return torch.abs(target_q - pred_q).detach().max(dim=1).values


# ============================================================================
# NEW MUZERO-STYLE LOSSES (Updated to work with unified interface)
# ============================================================================


class ValueLoss(BaseLoss):
    """Value prediction loss module."""

    def __init__(
        self,
        config,
        device,
        representation: Any,
        optimizer_name: str = "default",
        mask_key: str = "value_mask",
    ):
        super().__init__(
            config=config,
            device=device,
            pred_key="values",
            target_key="values",
            mask_key=mask_key,
            representation=representation,
            loss_fn=config.value_loss_function,
            optimizer_name=optimizer_name,
            loss_factor=config.value_loss_factor,
        )



class PolicyLoss(BaseLoss):
    """Policy prediction loss module."""

    def __init__(
        self,
        config,
        device,
        representation: Any,
        optimizer_name: str = "default",
        mask_key: str = "policy_mask",
    ):
        super().__init__(
            config=config,
            device=device,
            pred_key="policies",
            target_key="policies",
            mask_key=mask_key,
            representation=representation,
            loss_fn=config.policy_loss_function,
            optimizer_name=optimizer_name,
        )

    pass


class RewardLoss(BaseLoss):
    """Reward prediction loss module."""

    def __init__(
        self,
        config,
        device,
        representation: Any,
        optimizer_name: str = "default",
        mask_key: str = "reward_mask",
    ):
        super().__init__(
            config=config,
            device=device,
            pred_key="rewards",
            target_key="rewards",
            mask_key=mask_key,
            representation=representation,
            loss_fn=config.reward_loss_function,
            optimizer_name=optimizer_name,
        )

    pass


class ToPlayLoss(BaseLoss):
    """To-play (turn indicator) prediction loss module."""

    def __init__(
        self,
        config,
        device,
        representation: Any,
        optimizer_name: str = "default",
        mask_key: str = "to_play_mask",
    ):
        super().__init__(
            config=config,
            device=device,
            pred_key="to_plays",
            target_key="to_plays",
            mask_key=mask_key,
            representation=representation,
            loss_fn=config.to_play_loss_function,
            optimizer_name=optimizer_name,
            loss_factor=config.to_play_loss_factor,
        )

    def should_compute(self, predictions: dict, targets: dict, context: dict) -> bool:
        # Only compute for multi-player games
        return self.config.game.num_players != 1



class RelativeToPlayLoss(BaseLoss):
    """
    To-play loss for relative turn shifts (ΔP).
    Calculates ΔP targets from the sequence of absolute player indices:
    ΔP_k = (P_k - P_{k-1}) mod num_players.
    """

    def __init__(
        self,
        config,
        device,
        representation: Any,
        optimizer_name: str = "default",
        mask_key: str = "to_play_mask",
    ):
        super().__init__(
            config=config,
            device=device,
            pred_key="to_plays", # predictions["to_plays"] contains ΔP logits
            target_key="to_plays", # targets["to_plays"] contains full_to_plays
            mask_key=mask_key,
            representation=representation,
            optimizer_name=optimizer_name
        )

    def should_compute(self, predictions: dict, targets: dict, context: dict) -> bool:
        return self.config.game.num_players > 1

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B, T)"""
        # predictions["to_plays"] contains ΔP logits for seq (B, T, num_players)
        # Note: In MuZero, ΔP is usually predicted for k=1..T-1.
        delta_p_logits = predictions["to_plays"]
        full_to_plays = targets["to_plays"]
        
        # 1. Capture and Validate
        assert delta_p_logits.ndim == 3, f"RelativeToPlayLoss requires [B, T, num_players], got {delta_p_logits.shape}"
        assert full_to_plays.ndim == 2, f"RelativeToPlayLoss requires [B, T] to_plays, got {full_to_plays.shape}"
        B, T = full_to_plays.shape
        num_players = self.config.game.num_players

        # 2. Calculate ΔP targets: [B, T-1]
        p_prev = full_to_plays[:, :-1]
        p_curr = full_to_plays[:, 1:]
        target_delta_p = (p_curr - p_prev) % num_players
        
        # 3. Align and Compute Vectorized Cross Entropy
        # ΔP for k=1 corresponds to predictions index k=1 (if root is k=0)
        # Actually MuZero predicts p_k from s_k. s_k is produced by step(s_{k-1}, a_{k-1}).
        # So delta_p_logits[:, 1] is the prediction for ΔP_1 = (P1 - P0).
        
        # Zero-pad or slice to match T?
        # Standard approach: only compute for indices where we have a prev player.
        # [B, T-1]
        preds_p = delta_p_logits[:, 1:].reshape(-1, num_players)
        targets_p = target_delta_p.reshape(-1)
        assert preds_p.shape[0] == targets_p.shape[0], f"RelativeToPlayLoss: count mismatch between predictions {preds_p.shape[0]} and targets {targets_p.shape[0]}"
        loss_flat = self.config.to_play_loss_function(
            preds_p,
            targets_p.long(),
            reduction="none"
        )
        
        # Reshape and pad first index (k=0) with 0.0 to maintain [B, T] shape
        loss_seq = loss_flat.reshape(B, T-1)
        zero_pad = torch.zeros((B, 1), device=self.device)
        total_loss = torch.cat([zero_pad, loss_seq], dim=1)
        
        return self.config.to_play_loss_factor * total_loss

        # Loss calculation
        loss = self.config.to_play_loss_factor * self.config.to_play_loss_function(
            delta_p_logits_k, target_delta_p_k, reduction="none"
        )

        return loss


class ConsistencyLoss(BaseLoss):
    """Consistency loss module (EfficientZero style)."""

    def __init__(
        self,
        config,
        device,
        representation: Any,
        agent_network,
        optimizer_name: str = "default",
        mask_key: str = "policy_mask"
    ):
        super().__init__(
            config=config,
            device=device,
            pred_key="latents",
            target_key="consistency_targets",
            mask_key=mask_key,
            representation=representation,
            loss_fn=None, # Custom internal logic
            optimizer_name=optimizer_name,
            loss_factor=config.consistency_loss_factor,
        )
        self.agent_network = agent_network

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B, T)"""
        latent_states = predictions["latents"]
        target_features = targets["consistency_targets"]
        
        # 1. Capture and Validate
        assert latent_states.ndim >= 3, f"ConsistencyLoss requires [B, T, D], got {latent_states.shape}"
        assert target_features.ndim >= 2, f"ConsistencyLoss requires [B, T, D], got {target_features.shape}"
        B, T = target_features.shape[:2]

        if isinstance(latent_states, dict):
            latent_states = latent_states["dynamics"]

        # 2. Process Vectorized projection on flattened batch-time
        flat_latents = latent_states.reshape(-1, latent_states.shape[-1])
        proj_preds = self.agent_network.project(flat_latents, grad=True)
        f2 = F.normalize(proj_preds, p=2.0, dim=-1, eps=1e-5)

        # 3. Compute Vectorized Similarity Loss [B, T]
        flat_targets = target_features.reshape(-1, target_features.shape[-1])
        assert flat_targets.shape == f2.shape, f"ConsistencyLoss: shape mismatch between targets {flat_targets.shape} and predictions {f2.shape}"
        flat_consistency = -(flat_targets * f2).sum(dim=1)
        
        return self.config.consistency_loss_factor * flat_consistency.reshape(B, T)


class ChanceQLoss(BaseLoss):
    """Q-value loss for chance nodes (stochastic MuZero)."""

    def __init__(
        self,
        config,
        device,
        representation: Any,
        optimizer_name: str = "default"
    ):
        super().__init__(
            config=config,
            device=device,
            pred_key="chance_values",
            target_key="values",
            mask_key="value_mask",
            representation=representation,
            loss_fn=config.value_loss_function,
            optimizer_name=optimizer_name,
            loss_factor=getattr(config, "chance_loss_factor", 1.0),
        )

    def should_compute(self, predictions: dict, targets: dict, context: dict) -> bool:
        return self.config.stochastic



class SigmaLoss(BaseLoss):
    """Sigma (chance code prediction) loss for stochastic MuZero."""

    def __init__(
        self,
        config,
        device,
        representation: Any,
        optimizer_name: str = "default"
    ):
        super().__init__(
            config=config,
            device=device,
            pred_key="chance_logits",
            target_key="chance_codes",
            mask_key="value_mask",
            representation=representation,
            loss_fn=config.sigma_loss,
            optimizer_name=optimizer_name,
            loss_factor=getattr(config, "sigma_loss_factor", 1.0),
        )

    def should_compute(self, predictions: dict, targets: dict, context: dict) -> bool:
        return self.config.stochastic



class VQVAECommitmentLoss(BaseLoss):
    """VQ-VAE commitment cost for encoder (stochastic MuZero)."""

    def __init__(
        self,
        config,
        device,
        representation: Any,
        optimizer_name: str = "default"
    ):
        super().__init__(
            config=config,
            device=device,
            pred_key="chance_encoder_embeddings",
            target_key="chance_codes",
            mask_key="value_mask",
            representation=representation,
            loss_fn=F.mse_loss, # Standard commitment cost
            optimizer_name=optimizer_name,
            loss_factor=config.vqvae_commitment_cost_factor,
        )

    def should_compute(self, predictions: dict, targets: dict, context: dict) -> bool:
        return self.config.stochastic and not self.config.use_true_chance_codes



class PPOPolicyLoss(BaseLoss):
    def __init__(
        self,
        config,
        device,
        representation: Any,
        clip_param: float,
        entropy_coefficient: float,
        policy_strategy: Optional[object] = None,
        optimizer_name: str = "default",
    ):
        super().__init__(
            config=config, 
            device=device,
            pred_key="policies",
            target_key="actions",
            mask_key="policy_mask", 
            representation=representation,
            optimizer_name=optimizer_name
        )
        self.clip_param = clip_param
        self.entropy_coefficient = entropy_coefficient
        self.policy_strategy = policy_strategy

    @property
    def required_targets(self) -> set[str]:
        return {"actions", "old_log_probs", "advantages"}

    def should_compute(self, predictions: dict, targets: dict, context: dict) -> bool:
        return True

    def compute_loss(self, predictions: dict, targets: dict, context: dict) -> torch.Tensor:
        """PPO Policy Loss: returns [B, T]"""
        policy_logits = predictions["policies"]
        actions = targets["actions"]
        old_log_probs = targets["old_log_probs"]
        advantages = targets["advantages"]
        
        # 1. Capture and Validate
        assert policy_logits.ndim == 3, f"PPOPolicyLoss requires [B, T, A], got {policy_logits.shape}"
        B, T = old_log_probs.shape[:2]

        # 2. Vectorized Distribution math
        if self.policy_strategy is not None:
             # Strategy should handle [B, T, A] -> Dist with [B, T] sample shapes
             dist = self.policy_strategy.get_distribution(policy_logits)
        else:
             dist = torch.distributions.Categorical(logits=policy_logits)
        
        # [B, T]
        log_probs = dist.log_prob(actions)
        assert log_probs.shape == old_log_probs.shape, f"PPOPolicyLoss: shape mismatch between log_probs {log_probs.shape} and old_log_probs {old_log_probs.shape}"
        ratio = torch.exp(log_probs - old_log_probs)
        
        assert ratio.shape == advantages.shape, f"PPOPolicyLoss ratio vs advantages: {ratio.shape} vs {advantages.shape}"
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        
        entropy = dist.entropy()
        
        loss = -torch.min(surr1, surr2) - self.entropy_coefficient * entropy
        
        # 3. Stats for full sequence
        with torch.no_grad():
             approx_kl = (old_log_probs - log_probs).mean()
             if "approx_kl" not in context:
                  context["approx_kl"] = []
             context["approx_kl"].append(approx_kl.item())
             
        return loss
        # Note: mask is applied in LossPipeline.run
        # but for PPO it's usually 1s.


class PPOValueLoss(BaseLoss):
    def __init__(
        self,
        config,
        device,
        representation: Any,
        critic_coefficient: float,
        atom_size: int = 1,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        value_strategy: Optional[object] = None,
        optimizer_name: str = "default",
    ):
        super().__init__(
            config=config,
            device=device,
            pred_key="values",
            target_key="returns",
            mask_key="value_mask",
            representation=representation,
            optimizer_name=optimizer_name
        )
        self.critic_coefficient = critic_coefficient
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        self.value_strategy = value_strategy

    def _to_scalar_values(self, value_logits: torch.Tensor) -> torch.Tensor:
        if self.value_strategy is not None:
            return self.value_strategy.to_expected_value(value_logits)

        if value_logits.ndim == 1:
            return value_logits

        if value_logits.shape[-1] == 1:
            return value_logits.squeeze(-1)

        if self.atom_size > 1 and self.v_min is not None and self.v_max is not None:
            support = torch.linspace(
                self.v_min,
                self.v_max,
                value_logits.shape[-1],
                device=value_logits.device,
                dtype=value_logits.dtype,
            )
            probs = torch.softmax(value_logits, dim=-1)
            return (probs * support).sum(dim=-1)

        raise ValueError(
            "PPOValueLoss received multi-logit values without distributional bounds "
            "(v_min/v_max)."
        )

    def compute_loss(
        self, predictions: dict, targets: dict, context: dict
    ) -> torch.Tensor:
        """PPO Value Loss: returns [B, T]"""
        value_logits = predictions["values"]
        returns = targets["returns"]
        
        # 1. Capture and Validate
        assert value_logits.ndim >= 2, f"PPOValueLoss requires [B, T, ...], got {value_logits.shape}"
        B, T = returns.shape[:2]
        
        # 2. Vectorized Conversion
        # Flatten to use existing _to_scalar_values or wrap it
        flat_logits = value_logits.reshape(-1, value_logits.shape[-1])
        flat_values = self._to_scalar_values(flat_logits)
        values = flat_values.reshape(B, T)
        
        # 3. Compute Vectorized MSE [B, T]
        assert returns.shape == values.shape, f"PPOValueLoss: shape mismatch between returns {returns.shape} and values {values.shape}"
        return self.critic_coefficient * ((returns - values) ** 2)


class ImitationLoss(BaseLoss):
    def __init__(
        self, 
        config, 
        device, 
        representation: Any,
        optimizer_name: str = "default",
        mask_key: str = "policy_mask",
    ):
        super().__init__(
            config=config,
            device=device,
            pred_key="policies",
            target_key="policies",
            mask_key=mask_key,
            representation=representation,
            loss_fn=config.policy_loss_function,
            optimizer_name=optimizer_name
        )


# ============================================================================
# UNIFIED LOSS PIPELINE
# ============================================================================


class LossPipeline:
    """
    Unified pipeline that handles both single-step (DQN) and sequence (MuZero) losses.
    Validated at initialization to ensure all required keys are present.
    """

    def __init__(self, modules: list[BaseLoss]):
        self.modules = modules

    def validate_dependencies(
        self, network_output_keys: set[str], target_keys: set[str]
    ) -> None:
        """
        Verify that the provided keys satisfy all module requirements.
        Raises ValueError with detailed error message on failure.
        """
        for module in self.modules:
            missing_preds = module.required_predictions - network_output_keys
            missing_targets = module.required_targets - target_keys

            if missing_preds:
                raise ValueError(
                    f"Module {module.name} missing required predictions: {missing_preds}. "
                    f"Available: {network_output_keys}"
                )
            if missing_targets:
                raise ValueError(
                    f"Module {module.name} missing required targets: {missing_targets}. "
                    f"Available: {target_keys}"
                )

    def run(
        self,
        predictions: dict,
        targets: dict,
        context: dict = {},
        weights: Optional[torch.Tensor] = None,
        gradient_scales: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], torch.Tensor]:
        """
        Run the loss pipeline in a single vectorized pass across all sequence steps.
        """
        from modules.utils import support_to_scalar, scale_gradient
        from agents.learner.losses.shape_validator import ShapeValidator
        import numpy as np

        # 1. Config and Shape Validation
        import time
        start_time = time.perf_counter()
        config = self.modules[0].config
        device = self.modules[0].device
        ShapeValidator(config).validate(predictions, targets)

        # 2. Key/Format Normalization
        if not isinstance(predictions, dict):
            predictions = getattr(predictions, "_asdict", lambda: vars(predictions))()
        if not isinstance(targets, dict):
            targets = targets if isinstance(targets, dict) else vars(targets)

        # Determine dimensions B and T robustly
        B, T = 1, 1
        for key, tensor in predictions.items():
            if key == "gradient_scales":
                continue
            if torch.is_tensor(tensor) and tensor.ndim >= 2:
                B, T = tensor.shape[:2]
                break

        # 3. Defaults and Scaling
        if weights is None:
            weights = torch.ones(B, device=device)
        if gradient_scales is None:
            gradient_scales = torch.ones((1, T), device=device)

        context["full_targets"] = targets
        # Vectorized ChanceQ targets: shift values by 1
        if "values" in targets:
            v = targets["values"]
            v_next = torch.zeros_like(v)
            v_next[:, :-1] = v[:, 1:]
            context["target_values_next"] = v_next

        total_loss_dict = {
            m.optimizer_name: torch.tensor(0.0, device=device) for m in self.modules
        }
        loss_dict = {}
        priorities = torch.zeros(B, device=device)

        # 4. Single-Pass Vectorized Execution
        for module in self.modules:
            if not module.should_compute(predictions, targets, context):
                continue

            # Compute [B, T] elementwise loss
            elementwise_loss = module.compute_loss(predictions, targets, context)
            mask = module.get_mask(targets)
            
            # 1. Scale by Gradient Scales [1, T] and PER Weights [B, 1]
            # scale_gradient handles the [1, T] broadcasting internally
            scaled_loss = scale_gradient(elementwise_loss, gradient_scales)
            weighted_loss = scaled_loss * weights.reshape(B, 1)

            # 2. Mask and Reduce (Sum-over-Mask)
            # We sum across BOTH Batch and Time to get the total transition-weighted loss
            masked_weighted_loss = (weighted_loss * mask.float()).sum()
            # Normalize by the total number of valid mathematical transitions
            valid_transition_count = mask.float().sum().clamp(min=1.0)

            # 3. Final Transition-Averaged Loss [Scalar]
            total_scalar_loss = masked_weighted_loss / valid_transition_count

            total_loss_dict[module.optimizer_name] += total_scalar_loss
            loss_dict[module.name] = total_scalar_loss.item()

            # PER Priorities [B]
            if hasattr(module, "compute_priority"):
                p = module.compute_priority(predictions, targets, context)
                if p is not None:
                    priorities = torch.max(priorities, p)

        # 5. Extract Auxiliary metrics from context (approx_kl, etc.)
        for key, value in context.items():
            if key in ["full_targets", "target_values_next", "has_valid_action_mask", "is_same_game"]:
                continue
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
                loss_dict[key] = float(np.mean(value))

        loss_dict["loss_pipeline_latency_ms"] = (time.perf_counter() - start_time) * 1000
        return total_loss_dict, loss_dict, priorities
