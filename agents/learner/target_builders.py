from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
from torch import Tensor


class BaseTargetBuilder(ABC):
    """
    Abstract base class for Reinforcement Learning target calculation modules.
    """

    @abstractmethod
    def build_targets(
        self,
        batch: Dict[str, Tensor],
        predictions: Dict[str, Tensor],
        network: nn.Module,
    ) -> Dict[str, Tensor]:
        """
        Build target tensors for the loss calculation.

        Args:
            batch: Dictionary of tensors from the replay buffer.
            predictions: Current network predictions (LearningOutput).
            network: The neural network module (may be used for target network calls).

        Returns:
            Dictionary containing the computed target tensors.
        """
        pass  # pragma: no cover


class SingleStepTargetBuilder(BaseTargetBuilder):
    """
    Base class for all non-sequence algorithms (PPO, DQN, Imitation).
    Automatically upgrades tensors to Universal T=1 and generates masks.
    """

    def format_single_step(
        self, raw_targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        formatted = {}
        for k, v in raw_targets.items():
            # If it's a tensor and has at least a batch dimension, inject the T=1 dimension
            if torch.is_tensor(v) and v.ndim >= 1:
                formatted[k] = v.unsqueeze(1)
            else:
                formatted[k] = v

        # Automatically generate the Universal T=1 Mask for all standard semantics
        if "value_mask" not in formatted and len(formatted) > 0:
            # Grab the device and batch size from the first tensor we can find
            B, T = 1, 1
            for key, tensor in formatted.items():
                if key != "gradient_scales" and tensor.ndim >= 2:
                    B, T = tensor.shape[:2]
                    break  # We found a valid batch tensor (like rewards, actions, values)

            # Create a generic all-ones mask [B, 1]
            generic_mask = torch.ones((B, 1), device=tensor.device, dtype=torch.bool)

            # Route it to all potential semantic keys expected by LossModules
            formatted["value_mask"] = generic_mask
            formatted["reward_mask"] = generic_mask
            formatted["policy_mask"] = generic_mask
            formatted["q_mask"] = generic_mask

        return formatted


class TargetBuilderPipeline(BaseTargetBuilder):
    """
    Target builder that delegates target building to a pipeline of builders.
    All builders sequentially update the target dictionary.
    """

    def __init__(self, builders: List[BaseTargetBuilder]):
        self.builders = builders

    def build_targets(
        self,
        batch: Dict[str, Tensor],
        predictions: Dict[str, Tensor],
        network: nn.Module,
    ) -> Dict[str, Tensor]:
        targets = {}
        for builder in self.builders:
            targets.update(builder.build_targets(batch, predictions, network))
        return targets


class TemporalDifferenceBuilder(SingleStepTargetBuilder):
    """
    Standard TD target builder for Q-learning (Double DQN).
    Calculates: target = reward + gamma^n * (1 - done) * max_a' Q_target(s', a')
    """

    def __init__(
        self,
        target_network: nn.Module,
        gamma: float = 0.99,
        n_step: int = 1,
        bootstrap_on_truncated: bool = False,
    ):
        self.target_network = target_network
        self.gamma = gamma
        self.n_step = n_step
        self.bootstrap_on_truncated = bootstrap_on_truncated

    def build_targets(
        self,
        batch: Dict[str, Tensor],
        predictions: Dict[str, Tensor],
        network: nn.Module,
    ) -> Dict[str, Tensor]:
        rewards = batch["rewards"].float()
        dones = batch["dones"].bool()
        terminated = batch.get("terminated", dones).bool()
        next_obs = batch.get("next_observations")
        next_masks = batch.get("next_legal_moves_masks")

        terminal_mask = terminated if self.bootstrap_on_truncated else dones
        batch_size = rewards.shape[0]
        discount = self.gamma**self.n_step

        with torch.inference_mode():
            # Double DQN: Use online network for action selection
            online_next_out = network.learner_inference({"observations": next_obs})
            next_q_values = online_next_out["q_values"]

            # Use target network for value estimation
            target_out = self.target_network.learner_inference(
                {"observations": next_obs}
            )
            target_q_values = target_out["q_values"]

        # Ensure shapes are [B, Actions]
        if next_q_values.dim() == 3:
            next_q_values = next_q_values.squeeze(1)
        if target_q_values.dim() == 3:
            target_q_values = target_q_values.squeeze(1)

        if next_masks is not None:
            next_q_values = next_q_values.masked_fill(~next_masks.bool(), -float("inf"))

        next_actions = next_q_values.argmax(dim=-1)
        max_next_q = target_q_values[
            torch.arange(batch_size, device=rewards.device), next_actions
        ]

        target_q = rewards + (1 - terminal_mask.float()) * discount * max_next_q

        # Whitelist the mathematical labels required for the loss
        raw_targets = {
            "values": target_q,
            "rewards": rewards,
            "dones": terminal_mask.float(),
            "next_actions": next_actions,
            "actions": batch["actions"],
            "q_logits": target_out["q_logits"],
        }

        # Upgrade to Universal T=1 and add masks
        return self.format_single_step(raw_targets)


class PPOTargetBuilder(SingleStepTargetBuilder):
    """
    Whitelist-based builder for PPO targets.
    Ensures targets (values, returns, etc.) have a Universal T dimension [B, 1, ...].
    """

    def build_targets(
        self,
        batch: Dict[str, Tensor],
        predictions: Dict[str, Tensor],
        network: nn.Module,
    ) -> Dict[str, Tensor]:
        # Explicit Whitelist of mathematical labels needed for PPO Loss
        raw_targets = {}
        target_keys = ["values", "returns", "actions", "old_log_probs", "advantages"]

        for key in target_keys:
            if key in batch:
                raw_targets[key] = batch[key]

        # Upgrade them to [B, 1] and add masks
        return self.format_single_step(raw_targets)


class ImitationTargetBuilder(SingleStepTargetBuilder):
    """
    Whitelist-based builder for Behavioral Cloning / Imitation Learning.
    """

    def build_targets(
        self,
        batch: Dict[str, Tensor],
        predictions: Dict[str, Tensor],
        network: nn.Module,
    ) -> Dict[str, Tensor]:
        # Explicit Whitelist: we only need the actions and optional policy labels
        raw_targets = {
            "actions": batch["actions"],
        }
        if "target_policies" in batch:
            raw_targets["policies"] = batch["target_policies"]

        # Upgrade them to [B, 1] and add masks
        return self.format_single_step(raw_targets)


class MuZeroTargetBuilder(BaseTargetBuilder):
    """
    Ensures MuZero targets follow Universal T [B, T, ...].
    Specifically pads transition-aligned data (rewards, actions) from [B, K] to [B, T].
    """

    def __init__(self, unroll_steps: int):
        self.T = unroll_steps + 1

    def build_targets(
        self,
        batch: Dict[str, Tensor],
        predictions: Dict[str, Tensor],
        network: nn.Module,
    ) -> Dict[str, Tensor]:
        res = {}

        # Explicit Whitelist of mathematical labels required for MuZero losses
        # Transition-aligned: rewards, actions (K length)
        # State-aligned: values, policies (T length)
        target_keys = ["values", "rewards", "policies", "actions", "to_plays"]

        for key in target_keys:
            if key not in batch:
                continue

            v = batch[key]
            if torch.is_tensor(v) and v.ndim >= 2:
                B, K = v.shape[:2]
                if K == self.T - 1:
                    # Pad one dummy element at the BEGINNING of the time dimension (step 0 / root)
                    # This aligns k-th transition index with k-th state index in the LossPipeline.
                    padding_shape = list(v.shape)
                    padding_shape[1] = 1
                    padding = torch.zeros(padding_shape, device=v.device, dtype=v.dtype)
                    # [B, 1, ...] + [B, K, ...] -> [B, K+1, ...]
                    res[key] = torch.cat([padding, v], dim=1)
                else:
                    res[key] = v
            else:
                res[key] = v

        # MuZero Semantic Menu: Calculate specialized masks
        # Standard [B, T] base mask from is_same_game (1s until episode truly ends)
        base_mask = batch.get("is_same_game")
        if base_mask is None:
            # Fallback: find any tensor we just added to get the shape
            B, T = 1, 1
            for key, tensor in res.items():
                if key != "gradient_scales" and tensor.ndim >= 2:
                    B, T = tensor.shape[:2]
                    break  # We found a valid batch tensor (like rewards, actions, values)

            base_mask = torch.ones((B, T), device=self.device, dtype=torch.bool)

        res["value_mask"] = base_mask.clone()
        res["policy_mask"] = batch.get("has_valid_obs_mask", base_mask).clone()

        # Reward Mask: Zero out the root step (we predict r_k for k > 0)
        reward_mask = base_mask.clone()
        reward_mask[:, 0] = False
        res["reward_mask"] = reward_mask

        # To-Play Mask: Zero out the root AND post-terminal steps
        # Similar logic to reward_map, but can be more restrictive if needed
        to_play_mask = batch.get("has_valid_obs_mask", base_mask).clone()
        to_play_mask[:, 0] = False
        # If any step is terminal, the to_play for THAT state is not useful (no action follows)
        # So we can effectively use the same logic as reward_mask for alignment
        # TODO: to_play targets may be coming in a batch size of 6 so we and im not sure what index 0 represents, need to verify this
        res["to_play_mask"] = to_play_mask.clone()
        res[
            "to_play_mask"
        ] &= reward_mask  # ensuring root is zero and terminal transitions aligned

        return res


class LatentConsistencyBuilder(BaseTargetBuilder):
    """
    Build detached target embeddings for EfficientZero consistency loss.
    Takes None in __init__.
    """

    def __init__(self):
        pass

    def build_targets(
        self,
        batch: Dict[str, Tensor],
        predictions: Dict[str, Tensor],
        network: nn.Module,
    ) -> Dict[str, Tensor]:
        # Use unroll_observations from buffer [B, T+1, C, H, W]
        # UniversalLearner already passed through the batch, so it's in batch
        real_obs = batch["unroll_observations"].float()
        batch_size, unroll_len = real_obs.shape[:2]
        flat_obs = real_obs.flatten(0, 1)

        initial_out = network.obs_inference(flat_obs)
        real_latents = initial_out.network_state.dynamics

        # Clone to promote from inference_mode tensors to normal autograd-tracked tensors
        real_latents = real_latents.clone()

        proj_targets = network.project(real_latents, grad=False)
        normalized_targets = torch.nn.functional.normalize(
            proj_targets, p=2.0, dim=-1, eps=1e-5
        )

        consistency_targets = normalized_targets.reshape(
            batch_size, unroll_len, -1
        ).detach()
        return {"consistency_targets": consistency_targets}


class TrajectoryGradientScaleBuilder(BaseTargetBuilder):
    """
    Builds gradient scaling tensors for BPTT unrolling.
    Ensures gradients are correctly weighted across the sequence.
    """

    def __init__(self, unroll_steps: int):
        self.unroll_steps = unroll_steps

    def build_targets(
        self,
        batch: Dict[str, Tensor],
        predictions: Dict[str, Tensor],
        network: nn.Module,
    ) -> Dict[str, Tensor]:
        # Gradient Scales for MuZero sequence unrolling
        # Typically [1.0] for root, then [1/unroll_steps] for subsequent steps
        scales = [1.0] + [1.0 / self.unroll_steps] * self.unroll_steps
        # batch["rewards"] is used to get the device
        device = batch["rewards"].device
        scales_tensor = torch.tensor(scales, device=device).reshape(1, -1)
        return {"gradient_scales": scales_tensor}
