from __future__ import annotations

from typing import Dict, Any, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from agents.learners.base import StepResult, UniversalLearner
    from modules.agent_nets.modular import ModularAgentNetwork
from abc import ABC, abstractmethod


class EarlyStopIteration(Exception):
    """Raised by callbacks to break the batch iteration loop early."""

    pass


class Callback(ABC):
    """Learner callback interface with event hooks."""

    def on_step_begin(
        self,
        learner: UniversalLearner,
        iterator: Iterable[Dict[str, Any]],
    ) -> None:
        """Fired at the very beginning of the learner step, before fetching batches."""
        pass  # pragma: no cover

    def on_backward_end(
        self,
        learner: UniversalLearner,
        step_result: StepResult,
        stats=None,
    ) -> None:
        """Fired after loss.backward() but before optimizer.step().

        Raise EarlyStopIteration to break the batch iteration loop.
        """
        pass  # pragma: no cover

    def on_optimizer_step_end(
        self,
        learner: UniversalLearner,
    ) -> None:
        """Fired exactly after optimizer.step() and optional lr_scheduler.step()."""
        pass  # pragma: no cover

    def on_step_end(
        self,
        learner: UniversalLearner,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_dict: Dict[str, float],
        stats=None,
        **kwargs,
    ) -> None:
        """Fired after each sub-batch optimization (forward + backward + step)."""
        pass  # pragma: no cover

    def on_training_step_end(
        self,
        learner: UniversalLearner,
        stats=None,
    ) -> None:
        """Fired after the full training step (all iterations complete)."""
        pass  # pragma: no cover


class CallbackList:
    """Lightweight callback collection wrapper."""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def on_step_begin(
        self,
        learner: UniversalLearner,
        iterator: Iterable[Dict[str, Any]],
    ) -> None:
        for callback in self.callbacks:
            callback.on_step_begin(learner=learner, iterator=iterator)

    def on_backward_end(
        self,
        learner: UniversalLearner,
        step_result: StepResult,
        stats=None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_backward_end(
                learner=learner,
                step_result=step_result,
                stats=stats,
            )

    def on_optimizer_step_end(
        self,
        learner: UniversalLearner,
    ) -> None:
        for callback in self.callbacks:
            callback.on_optimizer_step_end(learner=learner)

    def on_step_end(
        self,
        learner: UniversalLearner,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_dict: Dict[str, float],
        stats=None,
        **kwargs,
    ) -> None:
        for callback in self.callbacks:
            callback.on_step_end(
                learner=learner,
                predictions=predictions,
                targets=targets,
                loss_dict=loss_dict,
                stats=stats,
                **kwargs,
            )

    def on_training_step_end(
        self,
        learner: UniversalLearner,
        stats=None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_training_step_end(learner=learner, stats=stats)


class StochasticMetricsCallback(Callback):
    """Tracks stochastic-specific learner diagnostics."""

    def on_step_end(
        self,
        learner: UniversalLearner,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_dict: Dict[str, float],
        stats=None,
        **kwargs,
    ) -> None:
        if stats is None:
            return

        self._track_stochastic_stats(predictions, targets, stats, learner.device)

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


class LatentMetricsCallback(Callback):
    """Tracks latent visualization diagnostics."""

    def __init__(self, viz_interval: int, viz_method: str = "tsne"):
        self.viz_interval = viz_interval
        self.viz_method = viz_method

    def on_step_end(
        self,
        learner: UniversalLearner,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_dict: Dict[str, float],
        stats=None,
        **kwargs,
    ) -> None:
        if stats is None:
            return

        if self.viz_interval > 0 and learner.training_step % self.viz_interval == 0:
            self._track_latent_visualization(
                predictions=predictions,
                targets=targets,
                stats=stats,
                method=self.viz_method,
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


class TargetNetworkSyncCallback(Callback):
    """Syncs target network weights at the end of each training step."""

    def __init__(
        self,
        target_network: ModularAgentNetwork,
        sync_interval: int,
        soft_update: bool = False,
        ema_beta: float = 0.99,
    ):
        self.target_network = target_network
        self.sync_interval = sync_interval
        self.soft_update = soft_update
        self.ema_beta = ema_beta

    def on_training_step_end(
        self,
        learner: UniversalLearner,
        stats=None,
    ) -> None:
        if self.sync_interval > 0 and learner.training_step % self.sync_interval != 0:
            return

        from modules.utils import get_clean_state_dict

        with torch.no_grad():
            clean_state = get_clean_state_dict(learner.agent_network)
            if self.soft_update:
                target_state = self.target_network.state_dict()
                ema_beta = self.ema_beta
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
                self.target_network.load_state_dict(clean_state, strict=False)


class ResetNoiseCallback(Callback):
    """Resets noisy network parameters after every optimizer step."""

    def on_optimizer_step_end(
        self,
        learner,
    ) -> None:
        learner.agent_network.reset_noise()


class ImitationMetricsCallback(Callback):
    """Collects and reports policy metrics for supervised/imitation learning."""

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
        assert target_kl > 0, f"target_kl must be positive, got {target_kl}"
        self.target_kl = target_kl
        self.key = key

    def on_backward_end(
        self,
        learner,
        step_result: StepResult,
        stats=None,
    ) -> None:
        """Checks KL divergence against target threshold."""
        assert self.key in step_result.loss_dict, (
            f"Key '{self.key}' missing from StepResult.loss_dict. "
            f"Available keys: {list(step_result.loss_dict.keys())}. "
            "Ensure the LossPipeline is propagating this metric from context."
        )
        kl = step_result.loss_dict[self.key]
        if kl > 1.5 * self.target_kl:
            if stats is not None:
                stats.append("ppo_early_stop", 1.0)
            raise EarlyStopIteration(f"KL divergence {kl:.4f} > 1.5 * {self.target_kl}")


class PriorityUpdaterCallback(Callback):
    """Updates Prioritized Experience Replay (PER) buffer priorities via a callback."""

    def __init__(
        self,
        priority_update_fn: Callable,
        set_beta_fn: Callable,
        per_beta_schedule: Any,
    ):
        """
        Initializes the PriorityUpdaterCallback.

        Args:
            priority_update_fn: A callable that accepts (indices, priorities, ids).
            set_beta_fn: A callable to update PER beta (accepts float).
            per_beta_schedule: A schedule for updating PER beta.
        """
        self.priority_update_fn = priority_update_fn
        self.set_beta_fn = set_beta_fn
        self.per_beta_schedule = per_beta_schedule

    def on_step_end(
        self,
        learner: UniversalLearner,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_dict: Dict[str, float],
        stats=None,
        **kwargs,
    ) -> None:
        batch = kwargs["batch"]
        priorities = kwargs["priorities"]

        ids = batch.get("ids")
        priorities_np = priorities.detach().cpu().numpy()
        self.priority_update_fn(batch["indices"], priorities_np, ids=ids)

    def on_training_step_end(
        self,
        learner: UniversalLearner,
        stats=None,
    ) -> None:
        self.set_beta_fn(self.per_beta_schedule.get_value())


class WeightBroadcastCallback(Callback):
    """Broadcasts network weights to remote workers/executors."""

    def __init__(self, weight_broadcast_fn: Callable[[Dict[str, Any]], None]):
        """
        Initializes the WeightBroadcastCallback.

        Args:
            weight_broadcast_fn: A callable that accepts the state_dict of the agent network.
        """
        self.weight_broadcast_fn = weight_broadcast_fn

    def on_training_step_end(
        self,
        learner: UniversalLearner,
        stats=None,
    ) -> None:
        """Broadcast weights at the end of the full training step (after all iterations/batches)."""
        self.weight_broadcast_fn(learner.agent_network.state_dict())


class EpsilonGreedySchedulerCallback(Callback):
    """Syncs target network weights at the end of each training step."""

    # TODO: should we have the init like this or should the learner store the target_agent_network too?
    def __init__(self, epsilon_schedule):
        self.epsilon_schedule = epsilon_schedule

    def on_training_step_end(
        self,
        learner,
        stats=None,
    ) -> None:
        self.epsilon_schedule.step()
