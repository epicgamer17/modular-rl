import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union
import numpy as np
from modules.world_models.inference_output import InferenceOutput


class LossModule(ABC):
    """
    Unified base class for all loss modules.
    Works for both single-step (DQN, C51) and sequence (MuZero) losses.
    """

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.sequence_loss = False  # Override to True for MuZero-style losses
        self.name = self.__class__.__name__

    # === Old DQN-style Interface ===
    def ensure_predictions(self, agent, context: dict):
        """Check context for predictions; if missing, compute and add them."""
        pass

    def ensure_targets(self, agent, context: dict):
        """Check context for targets; if missing, compute and add them."""
        pass

    # === New MuZero-style Interface ===
    def should_compute(self, k: int, context: dict) -> bool:
        """Determine if this loss should be computed at step k. For sequence losses only."""
        return True

    def get_mask(self, k: int, context: dict) -> torch.Tensor:
        """Get the mask to apply for this loss at step k. For sequence losses only."""
        if "has_valid_obs_mask" in context:
            return context["has_valid_obs_mask"][:, k]
        return torch.ones(self.config.minibatch_size, device=self.device)

    # === Unified Compute Interface ===
    @abstractmethod
    def compute_loss(
        self,
        agent=None,
        context: dict = None,
        k: int = None,
        predictions: dict = None,
        targets: dict = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Compute loss. Supports both interfaces:

        Old DQN-style: compute_loss(agent, context) -> (loss, elementwise)
        New MuZero-style: compute_loss(predictions=..., targets=..., k=..., context=...) -> elementwise

        Returns:
            For non-sequence (DQN): (scalar_loss, elementwise_tensor)
            For sequence (MuZero): elementwise_tensor of shape (B,)
        """
        pass


# ============================================================================
# OLD DQN-STYLE LOSSES (Updated to work with unified interface)
# ============================================================================


class StandardDQNLoss(LossModule):
    def __init__(self, config, device, action_selector=None):
        super().__init__(config, device)
        self.sequence_loss = False
        self.action_selector = action_selector

    def ensure_predictions(self, agent, context: dict):
        # IF EXISTS: Skip
        if "online_q_values" in context:
            return

        # PREPARE
        observations = context["observations"]
        actions = context["actions"].to(self.device).long()

        # COMPUTE
        # (B, num_actions)
        all_q_values = agent.predict(observations)
        # (B) - Select Q-value for specific action taken
        selected_q_values = all_q_values[range(self.config.minibatch_size), actions]

        # POPULATE
        context["online_q_values"] = selected_q_values

    def ensure_targets(self, agent, context: dict):
        # IF EXISTS: Skip
        if "target_q_values" in context:
            return

        # PREPARE
        with torch.no_grad():
            next_obs = context["next_observations"]
            rewards = context["rewards"].to(self.device)
            dones = context["dones"].to(self.device)
            terminated = context.get("terminated", dones).to(self.device)
            bootstrap_on_truncated = bool(
                getattr(self.config, "bootstrap_on_truncated", False)
            )
            terminal_mask = terminated if bootstrap_on_truncated else dones

            # Action Masking Logic
            next_masks = context["next_legal_moves_masks"].to(self.device)
            next_infos = [
                {"legal_moves": torch.nonzero(m).view(-1).tolist()} for m in next_masks
            ]

            # COMPUTE (Double DQN)
            # 1. Select best action using Online Network
            curr_next_q = agent.predict(next_obs)

            # Use action selector if provided, otherwise use argmax
            if self.action_selector is not None:
                # Select actions batch-wise
                next_actions = []
                for i in range(self.config.minibatch_size):
                    action, _ = self.action_selector.select_action(
                        agent_network=agent.model,
                        obs=next_obs[i : i + 1],
                        network_output=InferenceOutput(q_values=curr_next_q[i : i + 1]),
                        exploration=False,
                        info=next_infos[i],
                    )
                    next_actions.append(action)
                next_actions = torch.stack(next_actions).squeeze()
            else:
                # Fallback: simple greedy with masking
                from utils.utils import action_mask

                masked_q = torch.stack(
                    [
                        action_mask(
                            curr_next_q[i],
                            next_infos[i].get(
                                "legal_moves", list(range(curr_next_q.shape[-1]))
                            ),
                            mask_value=-float("inf"),
                            device=self.device,
                        )
                        for i in range(self.config.minibatch_size)
                    ]
                )
                next_actions = masked_q.argmax(dim=-1)

            # 2. Evaluate that action using Target Network
            target_next_q = agent.predict_target(next_obs)
            max_q_next = target_next_q[range(self.config.minibatch_size), next_actions]

            # 3. Bellman Calculation
            targets = rewards + self.config.discount_factor * (~terminal_mask) * max_q_next

        # POPULATE
        context["target_q_values"] = targets

    def compute_loss(
        self,
        agent=None,
        context: dict = None,
        k: int = None,
        predictions: dict = None,
        targets: dict = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """DQN-style: Returns (scalar_loss, elementwise_loss)"""
        preds = context["online_q_values"]
        targets_vals = context["target_q_values"]
        weights = context["weights"].to(torch.float32).to(self.device)

        # Calculate Elementwise (MSE or Huber)
        elementwise = self.config.loss_function(preds, targets_vals)

        # Apply PER weights
        loss = (elementwise * weights).mean()

        return loss, elementwise


class C51Loss(LossModule):
    def __init__(self, config, device, action_selector=None):
        super().__init__(config, device)
        self.support = torch.linspace(config.v_min, config.v_max, config.atom_size).to(
            device
        )
        self.sequence_loss = False
        self.action_selector = action_selector

    def ensure_predictions(self, agent, context: dict):
        if "online_dist" in context:
            return

        observations = context["observations"]
        actions = context["actions"].to(self.device).long()

        # (B, outputs, atom_size) -> Index by Action -> (B, atom_size)
        all_dists = agent.predict(observations)
        selected_dist = all_dists[range(self.config.minibatch_size), actions]

        context["online_dist"] = selected_dist

    def ensure_targets(self, agent, context: dict):
        if "target_dist" in context:
            return

        with torch.no_grad():
            # Setup
            next_obs = context["next_observations"]
            rewards = context["rewards"].to(self.device).view(-1, 1)
            dones = context["dones"].to(self.device).view(-1, 1)
            terminated = context.get("terminated", dones).to(self.device).view(-1, 1)
            bootstrap_on_truncated = bool(
                getattr(self.config, "bootstrap_on_truncated", False)
            )
            terminal_mask = terminated if bootstrap_on_truncated else dones

            # Masking
            next_masks = context["next_legal_moves_masks"].to(self.device)
            next_infos = [
                {"legal_moves": torch.nonzero(m).view(-1).tolist()} for m in next_masks
            ]

            # 1. Select Actions (Online Net) - Double DQN: online net picks action
            online_next_logits = agent.predict(next_obs)
            online_next_probs = torch.softmax(online_next_logits, dim=-1)

            # Convert distributions to Q-values for action selection
            online_q_values = (online_next_probs * self.support).sum(dim=-1)

            # Use action selector if provided, otherwise use argmax
            next_actions = []
            for i in range(self.config.minibatch_size):
                action, _ = self.action_selector.select_action(
                    agent_network=agent.model,
                    obs=next_obs[i : i + 1],
                    network_output=InferenceOutput(q_values=online_q_values[i : i + 1]),
                    exploration=False,
                    info=next_infos[i],
                )
                next_actions.append(action)
            next_actions = torch.stack(next_actions).squeeze()

            # 2. Get Target Distributions (Target Net) - Target net evaluates the action
            target_next_logits = agent.predict_target(next_obs)
            target_next_probs = torch.softmax(target_next_logits, dim=-1)
            probabilities = target_next_probs[
                range(self.config.minibatch_size), next_actions
            ]

            # 3. Project Distribution (C51 Logic)
            discount = self.config.discount_factor**self.config.n_step
            delta_z = (self.config.v_max - self.config.v_min) / (
                self.config.atom_size - 1
            )

            Tz = (rewards + discount * (~terminal_mask) * self.support).clamp(
                self.config.v_min, self.config.v_max
            )
            b = (Tz - self.config.v_min) / delta_z
            l = b.floor().long().clamp(0, self.config.atom_size - 1)
            u = b.ceil().long().clamp(0, self.config.atom_size - 1)

            # Distribute probability mass
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.config.atom_size - 1)) * (l == u)] += 1

            m = torch.zeros_like(probabilities)
            m.scatter_add_(dim=1, index=l, src=probabilities * (u.float() - b))
            m.scatter_add_(dim=1, index=u, src=probabilities * (b - l.float()))

        context["target_dist"] = m

    def compute_loss(
        self,
        agent=None,
        context: dict = None,
        k: int = None,
        predictions: dict = None,
        targets: dict = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """C51-style: Returns (scalar_loss, elementwise_loss)"""
        logit_preds = context["online_dist"]  # Now contains Logits
        targets_vals = context["target_dist"]  # Contains Probs (m)
        weights = context["weights"].to(torch.float32).to(self.device)

        # Cross Entropy: -Sum(target * log_softmax(pred))
        # This is numerically stable
        log_probs = F.log_softmax(logit_preds, dim=1)
        elementwise = -torch.sum(targets_vals * log_probs, dim=1)

        loss = (elementwise * weights).mean()

        return loss, elementwise


# ============================================================================
# NEW MUZERO-STYLE LOSSES (Updated to work with unified interface)
# ============================================================================


class ValueLoss(LossModule):
    """Value prediction loss module."""

    def __init__(self, config, device):
        super().__init__(config, device)
        self.sequence_loss = True

    def should_compute(self, k: int, context: dict) -> bool:
        return True  # Compute at all steps

    def compute_loss(
        self,
        agent=None,
        context: dict = None,
        k: int = None,
        predictions: dict = None,
        targets: dict = None,
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
        value_loss = self.config.value_loss_factor * self.config.value_loss_function(
            predicted_values_k, target_values_k
        )

        return value_loss


class PolicyLoss(LossModule):
    """Policy prediction loss module."""

    def __init__(self, config, device):
        super().__init__(config, device)
        self.sequence_loss = True

    def should_compute(self, k: int, context: dict) -> bool:
        return True  # Compute at all steps

    def get_mask(self, k: int, context: dict) -> torch.Tensor:
        # IMPORTANT: Policy Loss uses Policy Mask (excludes terminal)
        return context["has_valid_action_mask"][:, k]

    def compute_loss(
        self,
        agent=None,
        context: dict = None,
        k: int = None,
        predictions: dict = None,
        targets: dict = None,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B,)"""
        policies_k = predictions["policies"]
        target_policies_k = targets["policies"]
        # Policy Loss: (B,)
        policy_loss = self.config.policy_loss_function(policies_k, target_policies_k)

        return policy_loss


class RewardLoss(LossModule):
    """Reward prediction loss module."""

    def __init__(self, config, device):
        super().__init__(config, device)
        self.sequence_loss = True

    def should_compute(self, k: int, context: dict) -> bool:
        return k > 0  # Only for k > 0

    def compute_loss(
        self,
        agent=None,
        context: dict = None,
        k: int = None,
        predictions: dict = None,
        targets: dict = None,
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
        reward_loss = self.config.reward_loss_function(
            predicted_rewards_k, target_rewards_k
        )

        return reward_loss


class ToPlayLoss(LossModule):
    """To-play (turn indicator) prediction loss module."""

    def __init__(self, config, device):
        super().__init__(config, device)
        self.sequence_loss = True

    def should_compute(self, k: int, context: dict) -> bool:
        # Only compute for multi-player games and k > 0
        return k > 0 and self.config.game.num_players != 1

    def compute_loss(
        self,
        agent=None,
        context: dict = None,
        k: int = None,
        predictions: dict = None,
        targets: dict = None,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B,)"""
        to_plays_k = predictions["to_plays"]
        target_to_plays_k = targets["to_plays"]

        # To-Play Loss: (B,)
        # important to correctly predict whos turn it is on a terminal state, but unimportant afterwards
        to_play_loss = (
            self.config.to_play_loss_factor
            * self.config.to_play_loss_function(to_plays_k, target_to_plays_k)
        )

        return to_play_loss


class ConsistencyLoss(LossModule):
    """Consistency loss module (EfficientZero style)."""

    def __init__(self, config, device, model):
        super().__init__(config, device)
        self.sequence_loss = True
        self.model = model

    def should_compute(self, k: int, context: dict) -> bool:
        return k > 0  # Only for k > 0

    def get_mask(self, k: int, context: dict) -> torch.Tensor:
        # Consistency valid if policy is valid (step is not terminal)
        return context["has_valid_action_mask"][:, k]

    def compute_loss(
        self,
        agent=None,
        context: dict = None,
        k: int = None,
        predictions: dict = None,
        targets: dict = None,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B,)"""
        latent_states_k = predictions["latent_states"]
        target_features_k = targets["consistency_targets"]
        if isinstance(latent_states_k, dict):
            latent_states_k = latent_states_k["dynamics"]

        # Process the predicted latent (Prediction)
        # We project, then predict (SimSiam style predictor head)
        proj_preds = self.model.project(latent_states_k, grad=True)
        f2 = F.normalize(proj_preds, p=2.0, dim=-1, eps=1e-5)

        # Compare against learner-precomputed target features.
        current_consistency = -(target_features_k * f2).sum(dim=1)
        consistency_loss = self.config.consistency_loss_factor * current_consistency

        return consistency_loss


class ChanceQLoss(LossModule):
    """Q-value loss for chance nodes (stochastic MuZero)."""

    def __init__(self, config, device):
        super().__init__(config, device)
        self.sequence_loss = True

    def should_compute(self, k: int, context: dict) -> bool:
        return self.config.stochastic and k > 0

    def compute_loss(
        self,
        agent=None,
        context: dict = None,
        k: int = None,
        predictions: dict = None,
        targets: dict = None,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B,)"""
        # Note: chance values are indexed at k-1 in the stochastic arrays
        chance_values_k = predictions["chance_values"]
        # Target is derived from replay values at k+1, not a separate learner target head.
        target_chance_values_k = context["target_values_next"]

        # TODO: HAVE WE ALREADY RECOMPUTED TARGET Q IN THE CASE OF REANALYZE? I THINK WE HAVE

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

        q_loss = self.config.value_loss_factor * self.config.value_loss_function(
            predicted_chance_values_k,
            target_chance_values_k,
        )

        return q_loss

    def get_mask(self, k: int, context: dict) -> torch.Tensor:
        return context["has_valid_obs_mask"][:, k]


class SigmaLoss(LossModule):
    """Sigma (chance code prediction) loss for stochastic MuZero."""

    def __init__(self, config, device):
        super().__init__(config, device)
        self.sequence_loss = True

    def should_compute(self, k: int, context: dict) -> bool:
        return self.config.stochastic and k > 0

    def get_mask(self, k: int, context: dict) -> torch.Tensor:
        # no chance nodes from terminal -> absorbing
        return context["has_valid_action_mask"][:, k]

    def compute_loss(
        self,
        agent=None,
        context: dict = None,
        k: int = None,
        predictions: dict = None,
        targets: dict = None,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B,)"""
        # Note: indexed at k-1 in the stochastic arrays
        latent_code_probabilities_k = predictions["chance_codes"]
        target_codes_k = targets["chance_codes"].squeeze(-1).long()
        # Default config uses cross entropy (logits + class index). Keep one-hot fallback
        # for custom losses expecting distribution targets.
        if self.config.sigma_loss == F.cross_entropy:
            sigma_loss = self.config.sigma_loss(latent_code_probabilities_k, target_codes_k)
        else:
            encoder_onehot_k_plus_1 = F.one_hot(
                target_codes_k, num_classes=latent_code_probabilities_k.shape[-1]
            ).float()
            sigma_loss = self.config.sigma_loss(
                latent_code_probabilities_k,
                encoder_onehot_k_plus_1.detach(),
            )

        return sigma_loss


class VQVAECommitmentLoss(LossModule):
    """VQ-VAE commitment cost for encoder (stochastic MuZero)."""

    def __init__(self, config, device):
        super().__init__(config, device)
        self.sequence_loss = True

    def should_compute(self, k: int, context: dict) -> bool:
        return (
            self.config.stochastic and k > 0 and not self.config.use_true_chance_codes
        )

    def get_mask(self, k: int, context: dict) -> torch.Tensor:
        # no chance nodes from terminal -> absorbing
        return context["has_valid_action_mask"][:, k]

    def compute_loss(
        self,
        agent=None,
        context: dict = None,
        k: int = None,
        predictions: dict = None,
        targets: dict = None,
    ) -> torch.Tensor:
        """MuZero-style: Returns elementwise_loss of shape (B,)"""
        # Note: indexed at k-1 in the stochastic arrays
        encoder_softmax_k_plus_1 = predictions["encoder_softmaxes"]
        target_codes_k = targets["chance_codes"].squeeze(-1).long()
        encoder_onehot_k_plus_1 = F.one_hot(
            target_codes_k, num_classes=encoder_softmax_k_plus_1.shape[-1]
        ).float()

        # VQ-VAE commitment cost between c_t+k+1 and (c^e)_t+k+1 ||c_t+k+1 - (c^e)_t+k+1||^2
        # If using true chance codes, we can use them directly
        diff = (
            encoder_softmax_k_plus_1 - encoder_onehot_k_plus_1.detach()
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
    """

    def __init__(self, modules: list[LossModule]):
        self.modules = modules
        self.has_sequence_losses = any(m.sequence_loss for m in modules)
        self.has_nonsequence_losses = any(not m.sequence_loss for m in modules)

    def run(self, agent=None, context: dict = None, **kwargs):
        """
        Dispatch to appropriate pipeline based on loss types.

        For non-sequence losses (DQN, C51):
            run(agent, context) -> (total_loss, primary_elementwise)

        For sequence losses (MuZero):
            run(predictions_tensor, targets_tensor, context, weights, gradient_scales)
            -> (total_loss, loss_dict, priorities)
        """
        if self.has_sequence_losses and not self.has_nonsequence_losses:
            # Pure sequence losses (MuZero)
            return self._run_sequence_pipeline(**kwargs, context=context)
        elif self.has_nonsequence_losses and not self.has_sequence_losses:
            # Pure non-sequence losses (DQN)
            return self._run_nonsequence_pipeline(agent, context)
        else:
            raise ValueError(
                "Cannot mix sequence and non-sequence losses in the same pipeline"
            )

    def _run_nonsequence_pipeline(self, agent, context: dict):
        """
        For non-sequence losses (DQN, C51, etc.)
        Runs the full pipeline: Preparation -> Target Calc -> Loss Calc
        Returns combined loss and the elementwise loss of the *primary* (first) module.
        """
        # 1. Prepare all Predictions (Iterative check & populate)
        for module in self.modules:
            module.ensure_predictions(agent, context)

        # 2. Prepare all Targets (Iterative check & populate)
        for module in self.modules:
            module.ensure_targets(agent, context)

        # 3. Compute Losses
        total_loss = 0
        primary_elementwise = None

        for idx, module in enumerate(self.modules):
            loss, elementwise = module.compute_loss(agent=agent, context=context)
            total_loss += loss

            # We use the first module's elementwise loss for PER priorities
            if idx == 0:
                primary_elementwise = elementwise

        return total_loss, primary_elementwise

    def _run_sequence_pipeline(
        self,
        predictions: dict,
        targets: dict,
        context: dict,
        weights: torch.Tensor,
        gradient_scales: torch.Tensor,
        config=None,
        device=None,
    ) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        """
        Run the loss pipeline across all unroll steps for sequence losses.

        Args:
            predictions: Dict of tensors with shape (B, K+1, ...)
            targets: Dict of tensors with shape (B, K+1, ...)
            context: Additional context (masks, observations, etc.)
            weights: PER weights of shape (B,)
            gradient_scales: Gradient scales of shape (1, K+1)

        Returns:
            total_loss: Scalar loss for backpropagation
            loss_dict: Dictionary of accumulated losses for logging
            priorities: Priority tensor of shape (B,) for PER
        """
        from modules.utils import support_to_scalar, scale_gradient

        # Get config from first module
        if config is None:
            config = self.modules[0].config
        if device is None:
            device = self.modules[0].device

        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {module.name: 0.0 for module in self.modules}
        priorities = torch.zeros(config.minibatch_size, device=device)

        unroll_steps = gradient_scales.shape[1] - 1

        expected_steps = unroll_steps + 1

        for k in range(unroll_steps + 1):
            # Extract predictions and targets for step k
            preds_k = self._extract_step_data(predictions, k, expected_steps)
            targets_k = self._extract_step_data(targets, k, expected_steps)

            # --- 1. Priority Update (Only for k=0) ---
            if k == 0:
                values_k = preds_k["values"]
                target_values_k = targets_k["values"]

                if config.support_range is not None:
                    # Convert predicted value support to scalar for priority calculation
                    pred_scalar = support_to_scalar(values_k, config.support_range)
                    assert pred_scalar.shape == target_values_k.shape
                    priority = torch.abs(target_values_k - pred_scalar)
                else:
                    priority = torch.abs(target_values_k - values_k.squeeze(-1))
                priorities = priority.detach()  # Keep the B-length tensor

            # --- 2. Compute losses for this step ---
            step_loss = torch.zeros(config.minibatch_size, device=device)
            if "values" in targets and k > 0 and targets["values"].shape[1] > k:
                context["target_values_next"] = targets["values"][:, k]

            for module in self.modules:
                if not module.should_compute(k, context):
                    continue

                if not self._can_compute_module(module, preds_k, targets_k):
                    continue

                # Compute elementwise loss: (B,)
                loss_k = module.compute_loss(
                    predictions=preds_k, targets=targets_k, k=k, context=context
                )

                # Apply mask if using absorbing states
                if config.mask_absorbing:
                    mask_k = module.get_mask(k, context)
                    loss_k = loss_k * mask_k

                # Accumulate for this step
                step_loss = step_loss + loss_k

                # Accumulate for logging (unweighted)
                loss_dict[module.name] += loss_k.sum().item()

            # --- 3. Apply gradient scaling and PER weights ---
            scales_k = gradient_scales[:, k]  # (1,)
            scaled_loss_k = scale_gradient(step_loss, scales_k.item())
            weighted_scaled_loss_k = scaled_loss_k * weights

            # Accumulate total loss (scalar)
            total_loss += weighted_scaled_loss_k.sum()

        # Average the total loss by batch size
        loss_mean = total_loss / config.minibatch_size

        # Average accumulated losses for logging
        for key in loss_dict:
            loss_dict[key] /= config.minibatch_size

        return loss_mean, loss_dict, priorities

    def _extract_step_data(self, tensor_dict: dict, k: int, expected_steps: int) -> dict:
        """
        Extract data for unroll step `k`.

        Supports:
        - state-aligned tensors with shape (B, K+1, ...)
        - transition-aligned tensors with shape (B, K, ...), mapped by k -> k-1
        """
        step_data = {}
        for key, tensor in tensor_dict.items():
            if tensor is None or not torch.is_tensor(tensor) or tensor.ndim <= 1:
                continue

            steps = tensor.shape[1]
            if steps == expected_steps:
                step_data[key] = tensor[:, k]
            elif steps == expected_steps - 1:
                # Transition-aligned tensors (e.g., rewards/chance) correspond to k in [1..K].
                if k > 0:
                    step_data[key] = tensor[:, k - 1]
        return step_data

    def _can_compute_module(self, module: LossModule, predictions: dict, targets: dict) -> bool:
        required_predictions = {
            "ValueLoss": {"values"},
            "PolicyLoss": {"policies"},
            "RewardLoss": {"rewards"},
            "ToPlayLoss": {"to_plays"},
            "ConsistencyLoss": {"latent_states"},
            "ChanceQLoss": {"chance_values"},
            "SigmaLoss": {"chance_codes"},
            "VQVAECommitmentLoss": {"encoder_softmaxes"},
        }
        required_targets = {
            "ValueLoss": {"values"},
            "PolicyLoss": {"policies"},
            "RewardLoss": {"rewards"},
            "ToPlayLoss": {"to_plays"},
            "ConsistencyLoss": {"consistency_targets"},
            "ChanceQLoss": {"values"},
            "SigmaLoss": {"chance_codes"},
            "VQVAECommitmentLoss": {"chance_codes"},
        }
        module_name = module.__class__.__name__
        pred_required = required_predictions.get(module_name, set())
        target_required = required_targets.get(module_name, set())
        return pred_required.issubset(predictions.keys()) and target_required.issubset(
            targets.keys()
        )


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def create_dqn_loss_pipeline(config, device):
    """Create a standard DQN loss pipeline."""
    modules = [
        StandardDQNLoss(config, device),
    ]
    return LossPipeline(modules)


def create_c51_loss_pipeline(config, device):
    """Create a C51 distributional RL loss pipeline."""
    modules = [
        C51Loss(config, device),
    ]
    return LossPipeline(modules)


def create_muzero_loss_pipeline(config, device, model):
    """
    Factory function to create the standard MuZero loss pipeline.

    Args:
        config: Configuration object
        device: Device to run on
        model: Model instance (needed for consistency)

    Returns:
        LossPipeline instance configured for MuZero
    """
    modules = [
        ValueLoss(config, device),
        PolicyLoss(config, device),
        RewardLoss(config, device),
    ]

    # Optional modules
    if config.game.num_players != 1:
        modules.append(ToPlayLoss(config, device))

    modules.append(ConsistencyLoss(config, device, model))

    if config.stochastic:
        modules.extend(
            [
                ChanceQLoss(config, device),
                SigmaLoss(config, device),
                VQVAECommitmentLoss(config, device),
            ]
        )

    return LossPipeline(modules)
