from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
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
    masked_q = q_values.masked_fill(~mask, -float("inf"))
    return masked_q.argmax(dim=-1)


def _resolve_weights(
    batch: dict, device: torch.device, batch_size: int
) -> torch.Tensor:
    weights = batch.get("weights")
    if weights is None:
        return torch.ones(batch_size, device=device, dtype=torch.float32)
    return weights.to(device=device, dtype=torch.float32)


@dataclass
class PPOPolicyLoss:
    clip_param: float
    entropy_coefficient: float
    policy_strategy: Optional[object] = None

    def compute(
        self,
        policy_logits: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        entropy = dist.entropy().mean()
        loss = -torch.min(surr1, surr2).mean() - self.entropy_coefficient * entropy

        with torch.no_grad():
            approx_kl = (old_log_probs - log_probs).mean()
        return loss, approx_kl


@dataclass
class PPOValueLoss:
    critic_coefficient: float
    atom_size: int = 1
    v_min: Optional[float] = None
    v_max: Optional[float] = None
    value_strategy: Optional[object] = None

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

    def compute(
        self, value_logits: torch.Tensor, returns: torch.Tensor
    ) -> torch.Tensor:
        values = self._to_scalar_values(value_logits)
        return self.critic_coefficient * ((returns - values) ** 2).mean()


@dataclass
class StandardDQNLossModule:
    config: object
    device: torch.device
    action_selector: Optional[object] = None

    def compute(
        self,
        online_q_values: torch.Tensor,
        next_online_q_values: torch.Tensor,
        target_next_q_values: torch.Tensor,
        batch: dict,
        agent_network: Optional[torch.nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = batch["actions"].to(self.device).long()
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device)
        terminated = batch.get("terminated", dones).to(self.device)
        next_masks = batch.get("next_legal_moves_masks")
        next_observations = batch.get("next_observations")
        if next_observations is not None:
            next_observations = next_observations.to(self.device)

        bootstrap_on_truncated = bool(self.config.bootstrap_on_truncated)
        terminal_mask = terminated if bootstrap_on_truncated else dones
        batch_size = actions.shape[0]

        selected_q = online_q_values[
            torch.arange(batch_size, device=self.device), actions
        ]

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
            torch.arange(batch_size, device=self.device), next_actions
        ]

        targets = (
            rewards
            + (self.config.discount_factor**self.config.n_step)
            * (~terminal_mask)
            * max_q_next
        )

        elementwise = self.config.loss_function(selected_q, targets)
        weights = _resolve_weights(batch, self.device, batch_size)
        loss = (elementwise * weights).mean()
        return loss, elementwise


@dataclass
class C51LossModule:
    config: object
    device: torch.device
    action_selector: Optional[object] = None

    def __post_init__(self) -> None:
        self.support = torch.linspace(
            self.config.v_min,
            self.config.v_max,
            self.config.atom_size,
            device=self.device,
        )

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
            + discount * (~terminal_mask).view(-1, 1) * self.support.view(1, -1)
        ).clamp(self.config.v_min, self.config.v_max)

        # Map back to index space: b = (Tz - v_min) / delta_z
        b = (tz - self.config.v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        # Fix for points that are exactly on the boundary or exactly integers
        # If l == u, the probability should all go to index l.
        # But we can simplify by just using the standard projection form:
        # projected[l] += probs * (u - b)
        # projected[u] += probs * (b - l)
        # If l == u, then (u - b) = 0 and (b - l) = 0 if b is integer, which is wrong.
        # So we handle l == u by making u = l + 1 and clamping, OR just using a small epsilon.
        # Standard approach from "A Distributional Perspective on Reinforcement Learning":

        projected = torch.zeros((batch_size, self.config.atom_size), device=self.device)

        # Interpolate probabilities into the nearest atoms
        # For each atom j:
        # bj = (Tz_j - v_min) / delta_z
        # l = floor(bj), u = ceil(bj)
        # projected[l] += next_probs_j * (u - bj)
        # projected[u] += next_probs_j * (bj - l)

        # We need to iterate over the atoms of the next state distribution (self.support)
        # But wait, next_probs is already [batch, atom_size].
        # So we are projecting next_probs[i, j] (prob of atom j) onto the new support atoms.

        for j in range(self.config.atom_size):
            bj = b[:, j]
            lj = l[:, j]
            uj = u[:, j]

            # If l == u, bj is an integer. Weight should be 1 for l.
            # We can use (uj - bj) and (bj - lj). If lj == uj, these are 0.
            # Fix:
            mask_equal = lj == uj

            # Non-equal case
            dist_l = uj.float() - bj
            dist_u = bj - lj.float()

            # Equal case (bj is integer)
            dist_l = torch.where(mask_equal, torch.ones_like(dist_l), dist_l)
            # dist_u remains 0 for equal case if we want to add nothing to uj (which is lj)
            # but we scatter_add twice to the same index if lj == uj?
            # No, scatter_add is additive, so it would be fine if we ensure they sum to 1.

            prob_j = next_probs[:, j]

            projected.scatter_add_(1, lj.view(-1, 1), (prob_j * dist_l).view(-1, 1))
            projected.scatter_add_(1, uj.view(-1, 1), (prob_j * dist_u).view(-1, 1))

        return projected

    def compute(
        self,
        online_q_logits: torch.Tensor,
        next_online_q_logits: torch.Tensor,
        target_next_q_logits: torch.Tensor,
        batch: dict,
        agent_network: Optional[torch.nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = batch["actions"].to(self.device).long()
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device)
        terminated = batch.get("terminated", dones).to(self.device)
        next_masks = batch.get("next_legal_moves_masks")
        next_observations = batch.get("next_observations")
        if next_observations is not None:
            next_observations = next_observations.to(self.device)

        bootstrap_on_truncated = bool(self.config.bootstrap_on_truncated)
        terminal_mask = terminated if bootstrap_on_truncated else dones
        batch_size = actions.shape[0]

        online_next_probs = torch.softmax(next_online_q_logits, dim=-1)
        online_next_q = (online_next_probs * self.support).sum(dim=-1)
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
                    network_output=InferenceOutput(q_values=online_next_q[i : i + 1]),
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
        elementwise = -(target_dist * log_probs).sum(dim=-1)

        weights = _resolve_weights(batch, self.device, batch_size)
        loss = (elementwise * weights).mean()
        return loss, elementwise
