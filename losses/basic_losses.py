from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from losses.losses import LossModule
from modules.world_models.inference_output import InferenceOutput


def _select_next_actions(
    q_values: torch.Tensor, legal_moves_masks: Optional[torch.Tensor]
) -> torch.Tensor:
    if legal_moves_masks is None:
        return q_values.argmax(dim=-1)

    mask = legal_moves_masks.to(device=q_values.device).bool()
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    # Match legacy behavior: if no legal moves are provided, do not mask that row.
    any_legal = mask.any(dim=-1, keepdim=True)
    mask = torch.where(any_legal, mask, torch.ones_like(mask))
    masked_q = q_values.masked_fill(~mask.bool(), -float("inf"))
    return masked_q.argmax(dim=-1)


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


class StandardDQNLossModule(LossModule):
    def __init__(self, config, device, action_selector: Optional[object] = None):
        super().__init__(config, device)
        self.action_selector = action_selector

    @property
    def required_predictions(self) -> set[str]:
        return {"q_values", "next_q_values"}

    @property
    def required_targets(self) -> set[str]:
        return {"target_next_q_values", "actions", "rewards", "dones"}

    def compute_loss(
        self, predictions: dict, targets: dict, context: dict, k: int = 0
    ) -> torch.Tensor:
        if "values" in predictions and predictions["values"] is not None:
            # Blind Network: Network already selected Q for taken actions
            selected_q = predictions["values"].squeeze(-1)
        else:
            # Legacy/Fallback: Extract from full Q-values
            actions = context["actions"].to(self.device).long()
            batch_size = actions.shape[0]
            selected_q = online_q_values[
                torch.arange(batch_size, device=self.device), actions
            ]

        rewards = context["rewards"].to(self.device)
        dones = context["dones"].to(self.device)
        terminated = context.get("terminated", dones).to(self.device)
        bootstrap_on_truncated = bool(self.config.bootstrap_on_truncated)
        terminal_mask = terminated if bootstrap_on_truncated else dones

        if "q_values" in targets:
            # Blind Learner: Builder already computed Bellman targets
            targets_val = targets["q_values"]
        else:
            # Fallback legacy logic
            next_online_q_values = predictions["next_q_values"]
            target_next_q_values = targets.get("target_next_q_values")
            next_masks = context.get("next_legal_moves_masks")
            next_observations = context.get("next_observations")
            if next_observations is not None:
                next_observations = next_observations.to(self.device)

            agent_network = context.get("agent_network")
            next_actions = None
            if (
                self.action_selector is not None
                and agent_network is not None
                and next_observations is not None
                and next_masks is not None
            ):
                next_infos = [
                    {"legal_moves": torch.nonzero(m).view(-1).tolist()}
                    for m in next_masks.to(self.device)
                ]
                selected_actions = []
                for i in range(next_online_q_values.shape[0]):
                    action, _ = self.action_selector.select_action(
                        agent_network=agent_network,
                        obs=next_observations[i : i + 1],
                        network_output=InferenceOutput(
                            q_values=next_online_q_values[i : i + 1]
                        ),
                        exploration=False,
                        info=next_infos[i],
                    )
                    selected_actions.append(action)
                next_actions = torch.stack(selected_actions).squeeze()
            if next_actions is None:
                next_actions = _select_next_actions(next_online_q_values, next_masks)

            max_q_next = target_next_q_values[
                torch.arange(next_online_q_values.shape[0], device=self.device),
                next_actions,
            ]

            targets_val = (
                rewards
                + (self.config.discount_factor**self.config.n_step)
                * (~terminal_mask.bool())
                * max_q_next
            )

        # Return elementwise loss (B,)
        return self.config.loss_function(selected_q, targets_val, reduction="none")


class C51LossModule(LossModule):
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
        return {"q_logits", "next_q_logits"}

    @property
    def required_targets(self) -> set[str]:
        return {"target_next_q_logits", "actions", "rewards", "dones"}

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

        projected = torch.zeros((batch_size, self.config.atom_size), device=self.device)

        for j in range(self.config.atom_size):
            bj = b[:, j]
            lj = l[:, j]
            uj = u[:, j]

            mask_equal = lj == uj

            # Non-equal case
            dist_l = uj.float() - bj
            dist_u = bj - lj.float()

            # Equal case (bj is integer)
            dist_l = torch.where(mask_equal, torch.ones_like(dist_l), dist_l)

            prob_j = next_probs[:, j]

            projected.scatter_add_(1, lj.view(-1, 1), (prob_j * dist_l).view(-1, 1))
            projected.scatter_add_(1, uj.view(-1, 1), (prob_j * dist_u).view(-1, 1))

        return projected

    def compute_loss(
        self, predictions: dict, targets: dict, context: dict, k: int = 0
    ) -> torch.Tensor:
        online_q_logits = predictions["q_logits"]

        if "values" in predictions and predictions["values"] is not None:
            # Blind Network: Network already selected Q for taken actions
            # Note: For C51, values would be the expected value or we could use the distribution.
            # But compute_loss needs the chosen logits. So we still need actions.
            pass

        actions = context["actions"].to(self.device).long()
        batch_size = actions.shape[0]

        rewards = context["rewards"].to(self.device)
        dones = context["dones"].to(self.device)
        terminated = context.get("terminated", dones).to(self.device)
        bootstrap_on_truncated = bool(self.config.bootstrap_on_truncated)
        terminal_mask = terminated if bootstrap_on_truncated else dones

        online_next_q_logits = predictions["next_q_logits"]
        online_next_probs = torch.softmax(online_next_q_logits, dim=-1)
        online_next_q = (online_next_probs * self.support).sum(dim=-1)

        if "target_dist" in targets:
            # Blind Learner: Builder already computed C51 targets
            target_dist = targets["target_dist"]
        else:
            # Fallback legacy logic
            target_next_q_logits = targets.get("target_next_q_logits")
            next_masks = context.get("next_legal_moves_masks")
            next_observations = context.get("next_observations")
            if next_observations is not None:
                next_observations = next_observations.to(self.device)

            agent_network = context.get("agent_network")
            next_actions = None
            if (
                self.action_selector is not None
                and agent_network is not None
                and next_observations is not None
                and next_masks is not None
            ):
                next_infos = [
                    {"legal_moves": torch.nonzero(m).view(-1).tolist()}
                    for m in next_masks.to(self.device)
                ]
                selected_actions = []
                for i in range(batch_size):
                    action, _ = self.action_selector.select_action(
                        agent_network=agent_network,
                        obs=next_observations[i : i + 1],
                        network_output=InferenceOutput(
                            q_values=online_next_q[i : i + 1]
                        ),
                        exploration=False,
                        info=next_infos[i],
                    )
                    selected_actions.append(action)
                next_actions = torch.stack(selected_actions).squeeze()
            if next_actions is None:
                next_actions = _select_next_actions(online_next_q, next_masks)

            target_next_probs = torch.softmax(target_next_q_logits, dim=-1)
            chosen_next_probs = target_next_probs[
                torch.arange(batch_size, device=self.device), next_actions
            ]
            target_dist = self._project_target_distribution(
                rewards=rewards,
                terminal_mask=terminal_mask,
                next_probs=chosen_next_probs,
            )

        chosen_logits = online_q_logits[
            torch.arange(batch_size, device=self.device), actions
        ]
        log_probs = F.log_softmax(chosen_logits, dim=-1)
        # Return elementwise loss (B,)
        return -(target_dist * log_probs).sum(dim=-1)


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
