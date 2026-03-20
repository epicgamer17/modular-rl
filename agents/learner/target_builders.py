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
        current_targets: Dict[str, Tensor],
    ) -> None:
        """
        Build target tensors for the loss calculation and update 'current_targets' in place.

        Args:
            batch: Dictionary of tensors from the replay buffer.
            predictions: Current network predictions as a dictionary of tensors.
            network: The neural network module (may be used for target network calls).

        Returns:
            Dictionary containing the computed target tensors.
        """
        pass  # pragma: no cover


class SingleStepFormatter(BaseTargetBuilder):
    """
    Upgrades [B, ...] to Universal T=1 [B, 1, ...] and generates Masks.
    This is a pure modifier intended to be placed at the END of a single-step builder pipeline.
    """

    def build_targets(
        self,
        batch: Dict[str, Tensor],
        predictions: Dict[str, Tensor],
        network: nn.Module,
        current_targets: Dict[str, Tensor],
    ) -> None:
        if "actions" not in current_targets:
            # We need actions to determine batch size and device
            return

        batch_size = current_targets["actions"].shape[0]
        device = current_targets["actions"].device

        # 1. Inject T=1
        for k, v in current_targets.items():
            if torch.is_tensor(v) and v.ndim >= 1 and (v.ndim == 1 or v.shape[1] != 1):
                current_targets[k] = v.unsqueeze(1)

        # 2. Generate Universal T=1 Masks
        generic_mask = torch.ones((batch_size, 1), device=device, dtype=torch.bool)
        for mask_key in ["value_mask", "reward_mask", "policy_mask", "q_mask", "masks"]:
            if mask_key not in current_targets:
                current_targets[mask_key] = generic_mask

        # 3. Weights and Gradient Scales Anchors
        if "weights" not in current_targets:
            current_targets["weights"] = batch.get(
                "weights", torch.ones(batch_size, device=device)
            )
        if "gradient_scales" not in current_targets:
            current_targets["gradient_scales"] = torch.ones((1, 1), device=device)


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
        current_targets: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        if current_targets is None:
            current_targets = {}

        for builder in self.builders:
            # Capture keys before mutation to check for illegal collisions
            pre_keys = set(current_targets.keys())

            builder.build_targets(batch, predictions, network, current_targets)

            # The Fail-Fast Collision Check
            new_keys = set(current_targets.keys()) - pre_keys
            collisions = pre_keys.intersection(new_keys)

            # Anchors are allowed to be updated by subsequent builders
            collisions -= {"weights", "gradient_scales"}
            if collisions:
                raise RuntimeError(
                    f"TargetBuilder collision! Builder {builder.__class__.__name__} tried to overwrite keys: {collisions}. "
                    "Ensure builders have disjoint responsibilities."
                )

        return current_targets


class TemporalDifferenceBuilder(BaseTargetBuilder):
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
        current_targets: Dict[str, Tensor],
    ) -> None:
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
        if "values" in current_targets:
            raise RuntimeError("Collision on 'values' in TemporalDifferenceBuilder.")

        current_targets["values"] = target_q
        current_targets["rewards"] = rewards
        current_targets["dones"] = terminal_mask.float()
        current_targets["next_actions"] = next_actions
        current_targets["actions"] = batch["actions"]


class DistributionalTargetBuilder(BaseTargetBuilder):
    """
    Builder for C51/Distributional RL targets.
    Handles the Bellman Shift (MDP math) and delegates projection to the Representation.
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
        current_targets: Dict[str, Tensor],
    ) -> None:
        # 1. Extract MDP elements
        rewards = batch["rewards"].float()
        dones = batch["dones"].bool()
        terminated = batch.get("terminated", dones).bool()
        next_obs = batch.get("next_observations")
        next_masks = batch.get("next_legal_moves_masks")

        terminal_mask = terminated if self.bootstrap_on_truncated else dones
        batch_size = rewards.shape[0]
        discount = self.gamma**self.n_step

        with torch.inference_mode():
            # Double DQN: Use online network for action selection (argmax a' over Q)
            # Use learner_inference to get [B, Actions] q_values
            online_next_out = network.learner_inference({"observations": next_obs})
            next_q_values = online_next_out["q_values"]
            if next_q_values.ndim == 3:
                next_q_values = next_q_values.squeeze(1)

            if next_masks is not None:
                next_q_values = next_q_values.masked_fill(
                    ~next_masks.bool(), -float("inf")
                )
            next_actions = next_q_values.argmax(dim=-1)

            # Target network evaluation: Get probabilities of atoms for the chosen action
            target_out = self.target_network.learner_inference(
                {"observations": next_obs}
            )
            next_q_logits = target_out["q_logits"]  # [B, 1, Actions, Atoms]
            if next_q_logits.ndim == 4:
                next_q_logits = next_q_logits.squeeze(1)

            # [B, Actions, Atoms] -> [B, Atoms]
            next_probs = torch.softmax(
                next_q_logits[torch.arange(batch_size, device=rewards.device), next_actions],
                dim=-1,
            )

        # 2. Get the base grid geometry from the network's representation
        # It MUST be a C51Representation (or similar with support)
        representation = getattr(network.q_head, "representation", None)
        assert hasattr(
            representation, "project_onto_grid"
        ), "DistributionalTargetBuilder requires a representation with project_onto_grid API."

        base_support = representation.support.to(rewards.device)

        # 3. Do the MDP Math: Shift the support! (Tz = r + gamma * z)
        # [B, 1] + [B, 1] * [1, Atoms] -> [B, Atoms]
        shifted_support = rewards.unsqueeze(1) + discount * (
            1.0 - terminal_mask.float()
        ).unsqueeze(1) * base_support.unsqueeze(0)

        # 4. Delegate the pure geometric projection back to the representation
        target_distribution = representation.project_onto_grid(
            shifted_support=shifted_support, probabilities=next_probs
        )

        # 5. Whitelist the labels for the loss module
        if "q_logits" in current_targets:
            raise RuntimeError("Collision on 'q_logits' in DistributionalTargetBuilder.")

        current_targets["q_logits"] = target_distribution
        current_targets["rewards"] = rewards
        current_targets["actions"] = batch["actions"]
        current_targets["next_actions"] = next_actions
        current_targets["dones"] = terminal_mask.float()


class PassThroughTargetBuilder(BaseTargetBuilder):
    """
    Generic whitelist-based builder that passes specific keys
    from the batch through to the loss modules.
    Ensures targets have a Universal T dimension [B, 1, ...].
    """

    def __init__(self, keys_to_keep: List[str]):
        self.keys_to_keep = keys_to_keep

    def build_targets(
        self,
        batch: Dict[str, Tensor],
        predictions: Dict[str, Tensor],
        network: nn.Module,
        current_targets: Dict[str, Tensor],
    ) -> None:
        for key in self.keys_to_keep:
            if key in batch:
                if key in current_targets:
                    raise RuntimeError(f"Collision on '{key}' in PassThroughTargetBuilder.")
                current_targets[key] = batch[key]


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
        current_targets: Dict[str, Tensor],
    ) -> None:
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

        base_mask = batch.get("is_same_game")
        if base_mask is None:
            # The Anchor: MuZero must have actions to unroll
            B = batch["actions"].shape[0]
            T = self.T

            # Use the device from the actions tensor
            base_mask = torch.ones((B, T), device=batch["actions"].device, dtype=torch.bool)

        res["value_mask"] = base_mask.clone()
        res["masks"] = base_mask.clone()
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

        # --- PURGING HACKS: Explicitly build derived targets ---
        # 1. Weights Bridge: Ensure weights [B] are in targets
        B = batch["actions"].shape[0] if "actions" in batch else next(iter(batch.values())).shape[0]
        if "weights" not in res:
            res["weights"] = batch.get("weights", torch.ones(B, device=batch["actions"].device))

        # 2. Chance Shifting: Stochastic MuZero needs target value at step k+1
        if "values" in res:
            v = res["values"]
            v_next = torch.zeros_like(v)
            v_next[:, :-1] = v[:, 1:]
            res["chance_values_next"] = v_next

        # 3. Secure T Anchor: Ensure gradient_scales [1, T] exists
        if "gradient_scales" not in res:
            unroll_steps = self.T - 1
            scales = [1.0] + [1.0 / unroll_steps] * unroll_steps if unroll_steps > 0 else [1.0]
            res["gradient_scales"] = torch.tensor(scales, device=batch["actions"].device).reshape(1, -1)

        current_targets.update(res)


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
        current_targets: Dict[str, Tensor],
    ) -> None:
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
        current_targets["consistency_targets"] = consistency_targets


