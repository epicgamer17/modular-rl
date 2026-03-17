from __future__ import annotations

from typing import Dict, Any, Iterable

import torch

from agents.learners.base import Callback, EarlyStopIteration, StepResult


class MetricsCallback(Callback):
    """Tracks MuZero-specific learner diagnostics after each step."""

    def on_step_end(
        self,
        learner,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_dict,
        stats=None,
        **kwargs,
    ) -> None:
        if stats is None:
            return

        if learner.config.stochastic:
            self._track_stochastic_stats(predictions, targets, stats, learner.device)

        if learner.training_step % learner.config.latent_viz_interval == 0:
            self._track_latent_visualization(
                predictions=predictions,
                targets=targets,
                stats=stats,
                method=learner.config.latent_viz_method,
            )

    def _track_latent_visualization(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        stats,
        method: str,
    ) -> None:
        latent_states = predictions.get("latent_states")
        actions = targets.get("actions")
        if latent_states is None or actions is None:
            return
        s0 = latent_states[:, 0].detach().cpu()
        a0 = actions[:, 0].detach().cpu()
        stats.add_latent_visualization("latent_root", s0, labels=a0, method=method)

    def _track_stochastic_stats(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        stats,
        device,
    ) -> None:
        latent_code_logits_tensor = predictions.get("chance_codes")
        chance_codes = targets.get("chance_codes")
        if latent_code_logits_tensor is None or chance_codes is None:
            return
        latent_code_probs_tensor = torch.softmax(latent_code_logits_tensor, dim=-1)

        if latent_code_probs_tensor.ndim == 3:
            prob_sums = latent_code_probs_tensor.sum(dim=-1)
            mask = prob_sums > 0.001
        else:
            mask = torch.ones(
                latent_code_probs_tensor.shape[:-1],
                dtype=torch.bool,
                device=device,
            )

        # chance_codes comes in as [B, T+1, 1]
        codes = chance_codes.squeeze(-1).long()
        valid_codes = codes[mask] if mask.shape == codes.shape else codes.flatten()

        unique_codes_all = torch.unique(valid_codes)
        stats.append("num_codes", int(unique_codes_all.numel()))

        if latent_code_probs_tensor.ndim == 3:
            valid_probs = latent_code_probs_tensor[mask]
            if valid_probs.shape[0] > 0:
                mean_probs = valid_probs.mean(dim=0)
            else:
                mean_probs = torch.zeros(
                    latent_code_probs_tensor.shape[-1],
                    device=latent_code_probs_tensor.device,
                )
        else:
            mean_probs = latent_code_probs_tensor.mean(dim=0)
        stats.append("chance_probs", mean_probs.detach().cpu())

        probs = latent_code_probs_tensor
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        entropy = entropy[mask] if entropy.shape == mask.shape else entropy.flatten()
        stats.append(
            "chance_entropy",
            entropy.mean().item() if entropy.numel() > 0 else 0.0,
        )


class TargetNetworkSyncCallback(Callback):
    """Syncs target network weights at the end of each training step."""

    def on_training_step_end(
        self,
        learner,
        stats=None,
    ) -> None:
        assert learner.target_agent_network is not None

        transfer_interval = learner.config.transfer_interval
        if transfer_interval is None or transfer_interval == 0:
            should_sync = True
        else:
            should_sync = learner.training_step % transfer_interval == 0

        if not should_sync:
            return

        from modules.utils import get_clean_state_dict

        with torch.no_grad():
            clean_state = get_clean_state_dict(learner.agent_network)
            if getattr(learner.config, "soft_update", False):
                target_state = learner.target_agent_network.state_dict()
                ema_beta = getattr(learner.config, "ema_beta", 0.99)
                for k, v in clean_state.items():
                    if k not in target_state:
                        continue
                    if target_state[k].is_floating_point():
                        target_state[k].mul_(ema_beta).add_(
                            v.detach(), alpha=1.0 - ema_beta
                        )
                    else:
                        target_state[k].copy_(v.detach())
            else:
                learner.target_agent_network.load_state_dict(clean_state, strict=False)


class ResetNoiseCallback(Callback):
    """Resets noisy network parameters after every optimizer step."""

    def on_optimizer_step_end(
        self,
        learner,
    ) -> None:
        if hasattr(learner.agent_network, "reset_noise") and callable(
            learner.agent_network.reset_noise
        ):
            learner.agent_network.reset_noise()


class ImitationMetricsCallback(Callback):
    """Collects and reports policy metrics for ImitationLearner."""

    def __init__(self):
        self._policy_total = None
        self._policy_count = 0

    def on_step_begin(
        self,
        learner,
        iterator: Iterable[Dict[str, Any]],
    ) -> None:
        self._policy_total = torch.zeros(learner.num_actions, device=learner.device)
        self._policy_count = 0

    def on_backward_end(
        self,
        learner,
        step_result: StepResult,
        stats=None,
    ) -> None:
        if step_result.meta is not None and "policy_mean" in step_result.meta:
            if self._policy_total is None:
                self._policy_total = torch.zeros(
                    learner.num_actions, device=learner.device
                )
            self._policy_total += step_result.meta["policy_mean"]
            self._policy_count += 1

    def on_training_step_end(
        self,
        learner,
        stats=None,
    ) -> None:
        if stats is not None and self._policy_count > 0:
            stats.set("sl_policy", self._policy_total / self._policy_count)


class PPOEarlyStoppingCallback(Callback):
    """Early stops the optimization loop if KL divergence exceeds target_kl."""

    def __init__(self, target_kl: float, key: str = "approx_kl"):
        self.target_kl = target_kl
        self.key = key

    def on_backward_end(
        self,
        learner,
        step_result: StepResult,
        stats=None,
    ) -> None:
        """Checks KL divergence against target threshold."""
        kl = step_result.loss_dict.get(self.key)
        if kl is not None and kl > 1.5 * self.target_kl:
            if stats is not None:
                stats.append("ppo_early_stop", 1.0)
            raise EarlyStopIteration(f"KL divergence {kl:.4f} > 1.5 * {self.target_kl}")


class PriorityUpdaterCallback(Callback):
    """Updates Prioritized Experience Replay (PER) buffer priorities."""

    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer

    def on_step_end(
        self,
        learner,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_dict: Dict[str, float],
        stats=None,
        **kwargs,
    ) -> None:
        batch = kwargs.get("batch")
        step_result = kwargs.get("meta", {}).get(
            "step_result"
        )  # We will pass step_result in meta

        # If step_result wasn't explicitly passed in meta, we can try to fall back,
        # but the cleanest way is for UniversalLearner to pass result.priorities directly.
        # Actually, let's just look for priorities in kwargs since UniversalLearner can pass it.
        priorities = kwargs.get("priorities")

        if priorities is not None and batch is not None and "indices" in batch:
            if hasattr(self.replay_buffer, "update_priorities"):
                ids = batch.get("ids")
                self.replay_buffer.update_priorities(
                    batch["indices"], priorities, ids=ids
                )
