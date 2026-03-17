from __future__ import annotations

from typing import Dict

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
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], stats, method: str
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
