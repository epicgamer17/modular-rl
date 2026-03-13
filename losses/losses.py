import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union
from modules.world_models.inference_output import InferenceOutput


class LossModule(ABC):
    """
    Unified base class for all loss modules.
    Works for both single-step (DQN, C51) and sequence (MuZero) losses.
    """

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.name = self.__class__.__name__

    @property
    @abstractmethod
    def required_predictions(self) -> set[str]:
        """Set of keys required in the predictions dict."""
        pass  # pragma: no cover

    @property
    @abstractmethod
    def required_targets(self) -> set[str]:
        """Set of keys required in the targets dict."""
        pass  # pragma: no cover

    def should_compute(self, k: int, context: dict) -> bool:
        """Determine if this loss should be computed at step k."""
        return True

    def get_mask(self, k: int, context: dict) -> torch.Tensor:
        """Get the mask to apply for this loss at step k."""
        if "has_valid_obs_mask" in context:
            return context["has_valid_obs_mask"][:, k]
        return torch.ones(self.config.minibatch_size, device=self.device)

    @abstractmethod
    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
        k: int = 0,
    ) -> torch.Tensor:
        """
        Compute elementwise loss for a single step k.

        Returns:
            elementwise_tensor of shape (B,) or (B, atoms)
        """
        pass  # pragma: no cover


# ============================================================================
# OLD DQN-STYLE LOSSES (Updated to work with unified interface)
# ============================================================================


class StandardDQNLoss(LossModule):
    def __init__(self, config, device):
        super().__init__(config, device)

    @property
    def required_predictions(self) -> set[str]:
        return {"online_q_values"}

    @property
    def required_targets(self) -> set[str]:
        return {"target_q_values"}

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
        k: int = 0,
    ) -> torch.Tensor:
        """Returns elementwise_loss of shape (B,)"""
        preds = predictions["online_q_values"]
        target_vals = targets["target_q_values"]

        # Calculate Elementwise (MSE or Huber)
        elementwise = self.config.loss_function(preds, target_vals, reduction="none")
        return elementwise


class C51Loss(LossModule):
    def __init__(self, config, device):
        super().__init__(config, device)

    @property
    def required_predictions(self) -> set[str]:
        return {"online_dist"}

    @property
    def required_targets(self) -> set[str]:
        return {"target_dist"}

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
        k: int = 0,
    ) -> torch.Tensor:
        """C51-style: Returns elementwise_loss of shape (B,)"""
        logit_preds = predictions["online_dist"]  # Contains Logits
        target_vals = targets["target_dist"]  # Contains Probs (m)

        # Cross Entropy: -Sum(target * log_softmax(pred))
        log_probs = F.log_softmax(logit_preds, dim=1)
        elementwise = -torch.sum(target_vals * log_probs, dim=1)

        return elementwise


# ============================================================================
# NEW MUZERO-STYLE LOSSES (Updated to work with unified interface)
# ============================================================================


class ValueLoss(LossModule):
    """Value prediction loss module."""

    def __init__(self, config, device):
        super().__init__(config, device)

    @property
    def required_predictions(self) -> set[str]:
        return {"values"}

    @property
    def required_targets(self) -> set[str]:
        return {"values"}

    def should_compute(self, k: int, context: dict) -> bool:
        return True  # Compute at all steps

    def get_mask(self, k: int, context: dict) -> torch.Tensor:
        # Value loss is computed for all steps in the same game, even after terminal
        return context["is_same_game"][:, k]

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
        k: int = 0,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B,)"""
        values_k = predictions["values"]
        target_values_k = targets["values"]

        # Convert to support if needed
        if self.config.support_range is not None:
            from modules.utils import scalar_to_support

            target_values_k = scalar_to_support(
                target_values_k, self.config.support_range
            ).to(self.device)
            predicted_values_k = values_k
        else:
            # Squeeze to match target shape
            predicted_values_k = values_k.squeeze(-1)  # Convert (B, 1) -> (B,)

        assert (
            predicted_values_k.shape == target_values_k.shape
        ), f"{predicted_values_k.shape} = {target_values_k.shape}"

        # Value Loss: (B,)
        value_loss_k = self.config.value_loss_function(
            predicted_values_k, target_values_k, reduction="none"
        )
        if value_loss_k.ndim > 1:
            value_loss_k = value_loss_k.sum(dim=-1)

        value_loss = self.config.value_loss_factor * value_loss_k

        return value_loss


class PolicyLoss(LossModule):
    """Policy prediction loss module."""

    def __init__(self, config, device):
        super().__init__(config, device)

    @property
    def required_predictions(self) -> set[str]:
        return {"policies"}

    @property
    def required_targets(self) -> set[str]:
        return {"policies"}

    def should_compute(self, k: int, context: dict) -> bool:
        return True  # Compute at all steps

    def get_mask(self, k: int, context: dict) -> torch.Tensor:
        # IMPORTANT: Policy Loss uses Policy Mask (excludes terminal)
        return context["has_valid_action_mask"][:, k]

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
        k: int = 0,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B,)"""
        policies_k = predictions["policies"]
        target_policies_k = targets["policies"]

        if self.config.policy_loss_function == F.kl_div:
            # KL Div expects log-probabilities as input, but the network outputs logits
            # Also, kl_div without 'batchmean' returns [B, A], so we sum over actions
            log_probs = F.log_softmax(policies_k, dim=-1)
            policy_loss = self.config.policy_loss_function(
                log_probs, target_policies_k, reduction="none"
            )
            if policy_loss.ndim > 1:
                policy_loss = policy_loss.sum(dim=-1)
        else:
            # Default cross_entropy handles logits internally and returns [B] natively
            policy_loss = self.config.policy_loss_function(
                policies_k, target_policies_k, reduction="none"
            )

        return policy_loss


class RewardLoss(LossModule):
    """Reward prediction loss module."""

    def __init__(self, config, device):
        super().__init__(config, device)

    @property
    def required_predictions(self) -> set[str]:
        return {"rewards"}

    @property
    def required_targets(self) -> set[str]:
        return {"rewards"}

    def should_compute(self, k: int, context: dict) -> bool:
        return k > 0  # Only for k > 0

    def get_mask(self, k: int, context: dict) -> torch.Tensor:
        # Reward loss is computed for all steps in the same game, even after terminal
        return context["is_same_game"][:, k]

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
        k: int = 0,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B,)"""
        rewards_k = predictions["rewards"]
        target_rewards_k = targets["rewards"]

        # Convert to support if needed
        if self.config.support_range is not None:
            from modules.utils import scalar_to_support

            target_rewards_k = scalar_to_support(
                target_rewards_k, self.config.support_range
            ).to(self.device)
            predicted_rewards_k = rewards_k
        else:
            predicted_rewards_k = rewards_k.squeeze(-1)  # Convert (B, 1) -> (B,)

        assert (
            predicted_rewards_k.shape == target_rewards_k.shape
        ), f"{predicted_rewards_k.shape} = {target_rewards_k.shape}"

        # Reward Loss: (B,)
        reward_loss_k = self.config.reward_loss_function(
            predicted_rewards_k, target_rewards_k, reduction="none"
        )
        if reward_loss_k.ndim > 1:
            reward_loss_k = reward_loss_k.sum(dim=-1)

        reward_loss = reward_loss_k

        return reward_loss


class ToPlayLoss(LossModule):
    """To-play (turn indicator) prediction loss module."""

    def __init__(self, config, device):
        super().__init__(config, device)

    @property
    def required_predictions(self) -> set[str]:
        return {"to_plays"}

    @property
    def required_targets(self) -> set[str]:
        return {"to_plays"}

    def should_compute(self, k: int, context: dict) -> bool:
        # Only compute for multi-player games and k > 0
        return k > 0 and self.config.game.num_players != 1

    def get_mask(self, k: int, context: dict) -> torch.Tensor:
        # To-play exists for the terminal state too
        return context["has_valid_obs_mask"][:, k]

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
        k: int = 0,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B,)"""
        to_plays_k = predictions["to_plays"]
        target_to_plays_k = targets["to_plays"]

        # To-Play Loss: (B,)
        to_play_loss = (
            self.config.to_play_loss_factor
            * self.config.to_play_loss_function(
                to_plays_k, target_to_plays_k, reduction="none"
            )
        )

        return to_play_loss


class RelativeToPlayLoss(LossModule):
    """
    To-play loss for relative turn shifts (ΔP).
    Calculates ΔP targets from the sequence of absolute player indices:
    ΔP_k = (P_k - P_{k-1}) mod num_players.
    """

    def __init__(self, config, device):
        super().__init__(config, device)

    @property
    def required_predictions(self) -> set[str]:
        return {"to_plays"}

    @property
    def required_targets(self) -> set[str]:
        # Needs to_plays to calculate delta
        return {"to_plays"}

    def should_compute(self, k: int, context: dict) -> bool:
        # Only compute for multi-player games and k > 0 (needs k-1)
        return (
            k > 0
            and self.config.game.num_players > 1
            and "full_targets" in context
            and "to_plays" in context["full_targets"]
        )

    def get_mask(self, k: int, context: dict) -> torch.Tensor:
        return context["has_valid_obs_mask"][:, k]

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
        k: int = 0,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B,)"""
        # predictions["to_plays"] contains ΔP logits for step k (shape (B, num_players))
        delta_p_logits_k = predictions["to_plays"]

        # Calculate target ΔP_k = (P_k - P_{k-1}) mod N
        full_targets = context["full_targets"]
        p_k = full_targets["to_plays"][:, k]
        p_prev = full_targets["to_plays"][:, k - 1]
        num_players = self.config.game.num_players

        target_delta_p_k = (p_k - p_prev) % num_players

        # Loss calculation
        loss = self.config.to_play_loss_factor * self.config.to_play_loss_function(
            delta_p_logits_k, target_delta_p_k, reduction="none"
        )

        return loss


class ConsistencyLoss(LossModule):
    """Consistency loss module (EfficientZero style)."""

    def __init__(self, config, device, agent_network):
        super().__init__(config, device)
        self.agent_network = agent_network

    @property
    def required_predictions(self) -> set[str]:
        return {"latents"}

    @property
    def required_targets(self) -> set[str]:
        return {"consistency_targets"}

    def should_compute(self, k: int, context: dict) -> bool:
        return k > 0  # Only for k > 0

    def get_mask(self, k: int, context: dict) -> torch.Tensor:
        # Consistency valid if policy is valid (step is not terminal)
        return context["has_valid_action_mask"][:, k]

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
        k: int = 0,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B,)"""
        latent_states_k = predictions["latents"]
        target_features_k = targets["consistency_targets"]
        if isinstance(latent_states_k, dict):
            latent_states_k = latent_states_k["dynamics"]

        # Process the predicted latent (Prediction)
        # We project, then predict (SimSiam style predictor head)
        proj_preds = self.agent_network.project(latent_states_k, grad=True)
        f2 = F.normalize(proj_preds, p=2.0, dim=-1, eps=1e-5)

        # Compare against learner-precomputed target features.
        current_consistency = -(target_features_k * f2).sum(dim=1)
        consistency_loss = self.config.consistency_loss_factor * current_consistency

        return consistency_loss


class ChanceQLoss(LossModule):
    """Q-value loss for chance nodes (stochastic MuZero)."""

    def __init__(self, config, device):
        super().__init__(config, device)

    @property
    def required_predictions(self) -> set[str]:
        return {"chance_values"}

    @property
    def required_targets(self) -> set[str]:
        # Uses target_values_next which is targets["values"][:, k]
        return {"values"}

    def should_compute(self, k: int, context: dict) -> bool:
        return self.config.stochastic and k > 0

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
        k: int = 0,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B,)"""
        # Note: chance values are indexed at k-1 in the stochastic arrays
        chance_values_k = predictions["chance_values"]
        # Target is derived from replay values at k+1, not a separate learner target head.
        target_chance_values_k = context.get("target_values_next")

        # Convert to support if needed
        if self.config.support_range is not None:
            from modules.utils import scalar_to_support

            target_chance_values_k = scalar_to_support(
                target_chance_values_k, self.config.support_range
            ).to(self.device)
            predicted_chance_values_k = chance_values_k
        else:
            predicted_chance_values_k = chance_values_k.squeeze(
                -1
            )  # Convert (B, 1) -> (B,)

        assert (
            predicted_chance_values_k.shape == target_chance_values_k.shape
        ), f"{predicted_chance_values_k.shape} = {target_chance_values_k.shape}"

        q_loss_k = self.config.value_loss_function(
            predicted_chance_values_k,
            target_chance_values_k,
            reduction="none",
        )
        if q_loss_k.ndim > 1:
            q_loss_k = q_loss_k.sum(dim=-1)

        q_loss = self.config.value_loss_factor * q_loss_k

        return q_loss

    def get_mask(self, k: int, context: dict) -> torch.Tensor:
        # Chance Q target is value from next step. Compute if both are in same game.
        return context["is_same_game"][:, k]


class SigmaLoss(LossModule):
    """Sigma (chance code prediction) loss for stochastic MuZero."""

    def __init__(self, config, device):
        super().__init__(config, device)

    @property
    def required_predictions(self) -> set[str]:
        return {"chance_codes"}

    @property
    def required_targets(self) -> set[str]:
        return {"chance_codes"}

    def should_compute(self, k: int, context: dict) -> bool:
        return self.config.stochastic and k > 0

    def get_mask(self, k: int, context: dict) -> torch.Tensor:
        # no chance nodes from terminal -> absorbing
        return context["has_valid_action_mask"][:, k]

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
        k: int = 0,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B,)"""
        # Note: indexed at k-1 in the stochastic arrays
        latent_code_probabilities_k = predictions["chance_codes"]
        target_codes_k = targets["chance_codes"].squeeze(-1).long()
        # Default config uses cross entropy (logits + class index). Keep one-hot fallback
        # for custom losses expecting distribution targets.
        if self.config.sigma_loss == F.cross_entropy:
            sigma_loss = self.config.sigma_loss(
                latent_code_probabilities_k, target_codes_k, reduction="none"
            )
        else:
            chance_encoder_onehot_k_plus_1 = F.one_hot(
                target_codes_k, num_classes=latent_code_probabilities_k.shape[-1]
            ).float()
            sigma_loss = self.config.sigma_loss(
                latent_code_probabilities_k,
                chance_encoder_onehot_k_plus_1.detach(),
                reduction="none",
            )

        return sigma_loss


class VQVAECommitmentLoss(LossModule):
    """VQ-VAE commitment cost for encoder (stochastic MuZero)."""

    def __init__(self, config, device):
        super().__init__(config, device)

    @property
    def required_predictions(self) -> set[str]:
        return {"chance_encoder_embeddings"}

    @property
    def required_targets(self) -> set[str]:
        return {"chance_codes"}

    def should_compute(self, k: int, context: dict) -> bool:
        return (
            self.config.stochastic and k > 0 and not self.config.use_true_chance_codes
        )

    def get_mask(self, k: int, context: dict) -> torch.Tensor:
        # no chance nodes from terminal -> absorbing
        return context["has_valid_action_mask"][:, k]

    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
        k: int = 0,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B,)"""
        # Note: indexed at k-1 in the stochastic arrays
        chance_encoder_embedding_k_plus_1 = predictions["chance_encoder_embeddings"]
        target_codes_k = targets["chance_codes"].squeeze(-1).long()
        chance_encoder_onehot_k_plus_1 = F.one_hot(
            target_codes_k, num_classes=chance_encoder_embedding_k_plus_1.shape[-1]
        ).float()

        # VQ-VAE commitment cost between c_t+k+1 and (c^e)_t+k+1 ||c_t+k+1 - (c^e)_t+k+1||^2
        diff = (
            chance_encoder_embedding_k_plus_1 - chance_encoder_onehot_k_plus_1.detach()
        )  # TODO: lightzero does not detach here, try both
        vqvae_commitment_cost = self.config.vqvae_commitment_cost_factor * torch.sum(
            diff.pow(2), dim=-1
        )

        return vqvae_commitment_cost


# ============================================================================
# UNIFIED LOSS PIPELINE
# ============================================================================


class LossPipeline:
    """
    Unified pipeline that handles both single-step (DQN) and sequence (MuZero) losses.
    Validated at initialization to ensure all required keys are present.
    """

    def __init__(self, modules: list[LossModule]):
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
    ) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        """
        Run the loss pipeline across all unroll steps.

        Args:
            predictions: Dict of tensors with shape (B, K+1, ...) or (B, ...)
            targets: Dict of tensors with shape (B, K+1, ...) or (B, ...)
            context: Additional context (masks, etc.)
            weights: PER weights of shape (B,)
            gradient_scales: Gradient scales of shape (1, K+1)

        Returns:
            total_loss: Scalar loss for backpropagation
            loss_dict: Dictionary of accumulated losses for logging
            priorities: Priority tensor of shape (B,) for PER
        """
        from modules.utils import support_to_scalar, scale_gradient

        # Parameters from first module
        config = self.modules[0].config
        device = self.modules[0].device

        # Convert NamedTuples/dataclasses to dicts if necessary
        predictions = predictions._asdict()
        targets = vars(targets)
        assert predictions is not None and targets is not None
        if weights is None:
            weights = torch.ones(config.minibatch_size, device=device)

        if gradient_scales is None:
            gradient_scales = torch.ones((1, 1), device=device)

        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {module.name: 0.0 for module in self.modules}
        priorities = torch.zeros(config.minibatch_size, device=device)

        # Determine unroll steps from gradient_scales (1, K+1)
        # For non-sequence (DQN), gradient_scales is usually (1, 1)
        unroll_steps = gradient_scales.shape[1] - 1
        expected_steps = unroll_steps + 1

        context["full_targets"] = targets

        for k in range(expected_steps):
            # Extract predictions and targets for step k
            preds_k = self._extract_step_data(predictions, k, expected_steps)
            targets_k = self._extract_step_data(targets, k, expected_steps)

            # --- 1. Priority Update (Only for k=0) ---
            if k == 0:
                priorities = self._calculate_priorities(
                    preds_k, targets_k, config, device
                )

            # --- 2. Compute losses for this step ---
            step_loss = torch.zeros(config.minibatch_size, device=device)

            # Special case for ChanceQLoss which needs the next value
            if (
                "values" in targets
                and torch.is_tensor(targets["values"])
                and targets["values"].ndim > 2
                and k + 1 < targets["values"].shape[1]
            ):
                context["target_values_next"] = targets["values"][:, k + 1]

            for module in self.modules:
                if not module.should_compute(k, context):
                    continue

                # Compute elementwise loss: (B,)
                loss_k = module.compute_loss(
                    predictions=preds_k, targets=targets_k, context=context, k=k
                )

                # Apply mask if any
                if getattr(config, "mask_absorbing", False):
                    mask_k = module.get_mask(k, context)
                    loss_k = loss_k * mask_k

                # Accumulate for this step
                step_loss = step_loss + loss_k

                # Accumulate for logging (unweighted)
                loss_dict[module.name] += loss_k.sum().item()

            # --- 3. Apply gradient scaling and PER weights ---
            scale_k = gradient_scales[:, k].item()
            scaled_loss_k = scale_gradient(step_loss, scale_k)
            weighted_scaled_loss_k = scaled_loss_k * weights

            # Accumulate total loss (scalar)
            total_loss += weighted_scaled_loss_k.sum()

        # Average the total loss by batch size
        loss_mean = total_loss / config.minibatch_size

        # Average accumulated losses for logging
        for key in loss_dict:
            loss_dict[key] /= config.minibatch_size

        return loss_mean, loss_dict, priorities

    def _calculate_priorities(
        self, preds_k: dict, targets_k: dict, config, device
    ) -> torch.Tensor:
        """Calculate PER priorities for the current batch (k=0)."""
        from modules.utils import support_to_scalar

        # Standard MuZero/DQN approach: Value/Q error
        values_k = preds_k.get("values")
        if values_k is None:
            values_k = preds_k.get("online_q_values")

        target_values_k = targets_k.get("values")
        if target_values_k is None:
            target_values_k = targets_k.get("target_q_values")

        if values_k is None or target_values_k is None:
            # Fallback for other loss types if values aren't present
            return torch.zeros(
                config.minibatch_size, device=device, dtype=torch.float32
            )

        if config.support_range is not None and values_k.ndim > 1:
            # Support-based values (C51 or MuZero)
            pred_scalar = support_to_scalar(values_k, config.support_range)
            # If target is already scalar, keep it. If support, convert.
            if target_values_k.ndim > 1:
                target_scalar = support_to_scalar(target_values_k, config.support_range)
            else:
                target_scalar = target_values_k
            priority = torch.abs(target_scalar - pred_scalar)
        else:
            # Scalar values (Standard DQN)
            priority = torch.abs(target_values_k - values_k.squeeze(-1))

        return priority.detach()

    def _extract_step_data(
        self, tensor_dict: dict, k: int, expected_steps: int
    ) -> dict:
        """
        Extract data for unroll step `k`.
        Supports (B, K+1, ...) and (B, ...) shapes.
        """
        step_data = {}
        for key, tensor in tensor_dict.items():
            if tensor is None or not torch.is_tensor(tensor):
                continue

            if tensor.ndim > 1 and tensor.shape[1] == expected_steps:
                # Sequence data: (B, K+1, ...)
                step_data[key] = tensor[:, k]
            elif tensor.ndim > 1 and tensor.shape[1] == expected_steps - 1:
                # Transition-aligned data (e.g., rewards in some cases)
                if k > 0:
                    step_data[key] = tensor[:, k - 1]
            else:
                # Non-sequence data: (B, ...)
                step_data[key] = tensor

        return step_data
