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

    def __init__(self, config, device, optimizer_name: str = "default"):
        self.config = config
        self.device = device
        self.optimizer_name = optimizer_name
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

    def compute_priority(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
        k: int = 0,
    ) -> Optional[torch.Tensor]:
        """
        Calculates PER priorities.
        Returns None if this specific loss module does not drive priorities.
        """
        return None  # pragma: no cover


# ============================================================================
# OLD DQN-STYLE LOSSES (Updated to work with unified interface)
# ============================================================================


class StandardDQNLoss(LossModule):
    def __init__(self, config, device, action_selector: Optional[object] = None):
        super().__init__(config, device)
        self.action_selector = action_selector

    @property
    def required_predictions(self) -> set[str]:
        return {"q_values"}

    @property
    def required_targets(self) -> set[str]:
        return {"q_values", "actions"}

    def compute_loss(
        self, predictions: dict, targets: dict, context: dict, k: int = 0
    ) -> torch.Tensor:
        actions = targets["actions"].long()
        batch_size = actions.shape[0]
        selected_q = predictions["q_values"][
            torch.arange(batch_size, device=self.device), actions
        ]

        targets_val = targets["q_values"]
        # Return elementwise loss (B,)
        return self.config.loss_function(selected_q, targets_val, reduction="none")

    def compute_priority(self, predictions, targets, context, k=0):
        q_values = predictions["q_values"]
        actions = targets["actions"].long()

        # Q-value of the chosen action
        pred_q = q_values[torch.arange(q_values.shape[0], device=self.device), actions]
        target_q = targets["q_values"]

        return torch.abs(target_q - pred_q).detach()


class C51Loss(LossModule):
    def __init__(self, config, device, action_selector: Optional[object] = None):
        super().__init__(config, device)
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
        self, predictions: dict, targets: dict, context: dict, k: int = 0
    ) -> torch.Tensor:
        online_q_logits = predictions["q_logits"]

        actions = targets["actions"].to(self.device).long()
        batch_size = actions.shape[0]
        target_q_logits = targets["q_logits"]
        chosen_logits = online_q_logits[
            torch.arange(batch_size, device=self.device), actions
        ]
        log_probs = F.log_softmax(chosen_logits, dim=-1)
        # Return elementwise loss (B,)
        return -(target_q_logits * log_probs).sum(dim=-1)

    def compute_priority(self, predictions, targets, context, k=0):
        # Predict Expected Q
        probs = torch.softmax(predictions["q_logits"], dim=-1)
        q_values = (probs * self.support).sum(dim=-1)
        actions = targets["actions"].long()
        pred_q = q_values[torch.arange(q_values.shape[0], device=self.device), actions]

        # Target Expected Q
        target_q = (targets["q_logits"] * self.support).sum(dim=-1)

        return torch.abs(target_q - pred_q).detach()


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

    def compute_priority(self, predictions, targets, context, k=0):
        from modules.utils import support_to_scalar

        values_k = predictions["values"]
        target_values_k = targets["values"]

        if self.config.support_range is not None:
            # Predictions are 21-atom distributions; convert to scalar for comparison.
            # Targets are stored as raw scalars in the replay buffer (not distributions).
            pred_scalar = support_to_scalar(values_k, self.config.support_range)
            target_scalar = (
                target_values_k.squeeze(-1)
                if target_values_k.ndim > 1
                else target_values_k
            )
        else:
            pred_scalar = values_k.squeeze(-1)
            target_scalar = (
                target_values_k.squeeze(-1)
                if target_values_k.ndim > 1
                else target_values_k
            )

        return torch.abs(target_scalar - pred_scalar).detach()


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
        return {"chance_logits"}

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
        # predictions["chance_logits"] is the sigma head output (logits over chance codes)
        latent_code_probabilities_k = predictions["chance_logits"]
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


class PPOPolicyLoss(LossModule):
    def __init__(
        self,
        config,
        device,
        clip_param: float,
        entropy_coefficient: float,
        policy_strategy: Optional[object] = None,
    ):
        super().__init__(config, device)
        self.clip_param = clip_param
        self.entropy_coefficient = entropy_coefficient
        self.policy_strategy = policy_strategy

    @property
    def required_predictions(self) -> set[str]:
        return {"policies"}

    @property
    def required_targets(self) -> set[str]:
        return {"actions", "old_log_probs", "advantages"}

    def compute_loss(
        self, predictions: dict, targets: dict, context: dict, k: int = 0
    ) -> torch.Tensor:
        policy_logits = predictions["policies"]
        actions = targets["actions"]
        old_log_probs = targets["old_log_probs"]
        advantages = targets["advantages"]

        if self.policy_strategy is not None:
            dist = self.policy_strategy.get_distribution(policy_logits)
        else:
            dist = torch.distributions.Categorical(logits=policy_logits)
        log_probs = dist.log_prob(actions)

        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * advantages
        )

        entropy = dist.entropy()
        # Return elementwise loss (B,)
        loss = -torch.min(surr1, surr2) - self.entropy_coefficient * entropy

        with torch.no_grad():
            approx_kl = (old_log_probs - log_probs).mean()
            if "approx_kl" not in context:
                context["approx_kl"] = []
            context["approx_kl"].append(approx_kl.item())

        return loss


class PPOValueLoss(LossModule):
    def __init__(
        self,
        config,
        device,
        critic_coefficient: float,
        atom_size: int = 1,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        value_strategy: Optional[object] = None,
    ):
        super().__init__(config, device)
        self.critic_coefficient = critic_coefficient
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        self.value_strategy = value_strategy

    @property
    def required_predictions(self) -> set[str]:
        return {"values"}

    @property
    def required_targets(self) -> set[str]:
        return {"returns"}

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
        self, predictions: dict, targets: dict, context: dict, k: int = 0
    ) -> torch.Tensor:
        value_logits = predictions["values"]
        returns = targets["returns"]
        values = self._to_scalar_values(value_logits)
        # Return elementwise loss (B,)
        return self.critic_coefficient * ((returns - values) ** 2)


class ImitationLoss(LossModule):
    def __init__(self, config, device, num_actions: int):
        super().__init__(config, device)
        self.num_actions = num_actions
        self.loss_function = getattr(
            config, "loss_function", torch.nn.CrossEntropyLoss(reduction="none")
        )

    @property
    def required_predictions(self) -> set[str]:
        return {"policies"}

    @property
    def required_targets(self) -> set[str]:
        return {"target_policies"}

    def compute_loss(
        self, predictions: dict, targets: dict, context: dict, k: int = 0
    ) -> torch.Tensor:
        policy_logits = predictions["policies"]
        target_policies = targets["target_policies"]

        if target_policies.dim() == 1:
            # Handle class indices
            targets_onehot = torch.zeros(
                target_policies.shape[0], self.num_actions, device=self.device
            )
            targets_onehot.scatter_(1, target_policies.unsqueeze(1).long(), 1.0)
            target_policies = targets_onehot

        # Return elementwise loss (B,)
        loss = self.loss_function(policy_logits, target_policies)
        if loss.dim() > 1:
            loss = loss.sum(dim=-1)
        return loss


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
    ) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], Dict[str, float], torch.Tensor]:
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
        if isinstance(predictions, dict):
            pass
        elif hasattr(predictions, "_asdict"):
            predictions = predictions._asdict()
        else:
            predictions = vars(predictions)
        targets = targets if isinstance(targets, dict) else vars(targets)
        assert predictions is not None and targets is not None
        if weights is None:
            weights = torch.ones(config.minibatch_size, device=device)

        if gradient_scales is None:
            gradient_scales = torch.ones((1, 1), device=device)

        total_loss_dict = {module.optimizer_name: torch.tensor(0.0, device=device) for module in self.modules}
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
                    preds_k, targets_k, context, config, device
                )

            # --- 2. Compute losses for this step ---
            step_losses = {
                opt_name: torch.zeros(config.minibatch_size, device=device)
                for opt_name in total_loss_dict.keys()
            }

            # Special case for ChanceQLoss which needs the next value.
            # targets["values"] is 2D [B, T] for scalar targets, so ndim >= 2.
            if (
                "values" in targets
                and torch.is_tensor(targets["values"])
                and targets["values"].ndim >= 2
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

                # Accumulate for this step for the specific optimizer
                step_losses[module.optimizer_name] += loss_k

                # Accumulate for logging (unweighted)
                loss_dict[module.name] += loss_k.sum().item()

            # --- 3. Apply gradient scaling and PER weights ---
            scale_k = gradient_scales[:, k].item()
            
            for opt_name, step_loss in step_losses.items():
                scaled_loss_k = scale_gradient(step_loss, scale_k)
                weighted_scaled_loss_k = scaled_loss_k * weights

                # Accumulate total loss per optimizer
                total_loss_dict[opt_name] += weighted_scaled_loss_k.sum()

        # Average the total loss by batch size
        for opt_name in total_loss_dict:
            total_loss_dict[opt_name] = total_loss_dict[opt_name] / config.minibatch_size

        # Average accumulated losses for logging
        for key in loss_dict:
            loss_dict[key] /= config.minibatch_size

        if len(total_loss_dict) == 1 and "default" in total_loss_dict:
            return total_loss_dict["default"], loss_dict, priorities
        else:
            return total_loss_dict, loss_dict, priorities

    def _calculate_priorities(
        self,
        preds_k: dict,
        targets_k: dict,
        context: dict,
        config,
        device,
    ) -> torch.Tensor:
        """Calculate PER priorities for the current batch (k=0)."""

        # Delegate priority calculation to the loss modules.
        # The first module that returns a priority tensor defines the PER priorities.
        for module in self.modules:
            priority = module.compute_priority(preds_k, targets_k, context, k=0)
            if priority is not None:
                return priority

        # raise ValueError(
        #     "No active LossModule provided a priority calculation. "
        #     "Ensure the primary module (e.g., ValueLoss, StandardDQNLoss, or C51Loss) "
        #     "implements compute_priority()."
        # )

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
