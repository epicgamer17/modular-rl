from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
from torch import Tensor


class BaseTargetBuilder(ABC):
    @abstractmethod
    def build_targets(
        self,
        batch: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
        network: nn.Module,
        current_targets: Dict[str, torch.Tensor],
    ) -> None:
        """Mutates current_targets in-place."""
        pass  # pragma: no cover


class SingleStepFormatter(BaseTargetBuilder):
    """
    Upgrades explicitly whitelisted keys to Universal T=1 [B, 1, ...] and generates Masks.
    This is a pure modifier intended to be placed at the END of a single-step builder pipeline.
    """

    def __init__(self, temporal_keys: Optional[List[str]] = None):
        # Default whitelist for standard single-step RL
        self.temporal_keys = temporal_keys or [
            "actions",
            "next_actions",
            "rewards",
            "dones",
            "values",
            "policies",
            "q_logits",
            "q_values",
            "returns",
            "advantages",
            "old_log_probs",
        ]

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

        # 1. Inject T=1 ONLY on the explicitly defined temporal keys
        for k in self.temporal_keys:
            if k in current_targets:
                v = current_targets[k]
                if torch.is_tensor(v) and (
                    v.ndim == 1 or (v.ndim >= 2 and v.shape[1] != 1)
                ):
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
        batch: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
        network: nn.Module,
        current_targets: Dict[str, torch.Tensor],
    ) -> None:
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
            # Handle potential variation in Q-value naming
            next_q = online_next_out.get("q_values", online_next_out.get("q_logits"))

            # Cleanly strip temporal dimension if present: [B, 1, Actions] -> [B, Actions]
            if next_q.dim() == 3 and next_q.shape[1] == 1:
                next_q = next_q.squeeze(1)

            if next_masks is not None:
                if next_masks.dim() == 3 and next_masks.shape[1] == 1:
                    next_masks = next_masks.squeeze(1)
                next_q = next_q.masked_fill(~next_masks.bool(), -float("inf"))

            next_actions = next_q.argmax(dim=-1)

            # Use target network for value estimation
            target_out = self.target_network.learner_inference(
                {"observations": next_obs}
            )
            target_q = target_out.get("q_values", target_out.get("q_logits"))
            if target_q.dim() == 3 and target_q.shape[1] == 1:
                target_q = target_q.squeeze(1)

        max_next_q = target_q[
            torch.arange(batch_size, device=rewards.device), next_actions
        ]

        target_q_val = rewards + (1 - terminal_mask.float()) * discount * max_next_q

        # Whitelist the mathematical labels required for the loss pipeline
        if "q_values" in current_targets:
            raise RuntimeError("Collision on 'q_values' in TemporalDifferenceBuilder.")

        current_targets["q_values"] = target_q_val
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

        # We need the representation to compute expected Q-values for action selection
        representation = network.components["behavior_heads"]["q_logits"].representation

        with torch.inference_mode():
            # Double DQN: Use online network for action selection
            online_next_out = network.learner_inference({"observations": next_obs})
            next_q_logits = online_next_out["q_logits"]

            # Cleanly strip temporal dimension if present: [B, 1, Actions, Atoms] -> [B, Actions, Atoms]
            if next_q_logits.dim() == 4 and next_q_logits.shape[1] == 1:
                next_q_logits = next_q_logits.squeeze(1)

            # Get Expected Q-Values for argmax (resolves the bug of argmaxing the atoms dim)
            expected_next_q = representation.to_expected_value(next_q_logits)

            if next_masks is not None:
                if next_masks.dim() == 3 and next_masks.shape[1] == 1:
                    next_masks = next_masks.squeeze(1)
                expected_next_q = expected_next_q.masked_fill(
                    ~next_masks.bool(), -float("inf")
                )

            # Argmax over expected values: [B, Actions] -> [B]
            next_actions = expected_next_q.argmax(dim=-1)

            # Target network evaluation: Get probabilities of atoms for the chosen action
            target_out = self.target_network.learner_inference(
                {"observations": next_obs}
            )
            target_next_logits = target_out["q_logits"]

            if target_next_logits.dim() == 4 and target_next_logits.shape[1] == 1:
                target_next_logits = target_next_logits.squeeze(1)

            # Extract logits for chosen actions: [B, Actions, Atoms] -> [B, Atoms]
            chosen_next_logits = target_next_logits[
                torch.arange(batch_size, device=rewards.device), next_actions
            ]

            # Compute probabilities for the projected atoms
            next_probs = torch.softmax(chosen_next_logits, dim=-1)

        # 2. Get the base grid geometry from the network's representation
        assert hasattr(
            representation, "project_onto_grid"
        ), "Distributional targets require a C51Representation."
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
            raise RuntimeError(
                "Collision on 'q_logits' in DistributionalTargetBuilder."
            )

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
                    raise RuntimeError(
                        f"Collision on '{key}' in PassThroughTargetBuilder."
                    )
                current_targets[key] = batch[key]


class MCTSExtractor(BaseTargetBuilder):
    """Generates targets by extracting MCTS search statistics from the batch."""

    def build_targets(self, batch, predictions, network, current_targets) -> None:
        target_keys = ["values", "rewards", "policies", "actions", "to_plays"]
        for key in target_keys:
            if key in batch and key not in current_targets:
                current_targets[key] = batch[key]


class SequencePadder(BaseTargetBuilder):
    """Modifier: Pads transition-aligned data (length K) to state-aligned length (T)."""

    def __init__(self, unroll_steps: int):
        self.T = unroll_steps + 1

    def build_targets(self, batch, predictions, network, current_targets) -> None:
        for key, v in current_targets.items():
            if torch.is_tensor(v) and v.ndim >= 2 and v.shape[1] == self.T - 1:
                padding_shape = list(v.shape)
                padding_shape[1] = 1
                # Insert t=0 padding: [B, 1, ...] + [B, K, ...] -> [B, K+1, ...]
                padding = torch.zeros(padding_shape, device=v.device, dtype=v.dtype)
                current_targets[key] = torch.cat([padding, v], dim=1)


class SequenceMaskBuilder(BaseTargetBuilder):
    """Modifier: Generates Universal [B, T] sequence masks."""

    def build_targets(self, batch, predictions, network, current_targets) -> None:
        B, T = current_targets["actions"].shape[:2]
        device = current_targets["actions"].device

        base_mask = batch.get(
            "is_same_game", torch.ones((B, T), device=device, dtype=torch.bool)
        )

        current_targets["value_mask"] = base_mask.clone()
        current_targets["masks"] = base_mask.clone()
        current_targets["policy_mask"] = batch.get(
            "has_valid_obs_mask", base_mask
        ).clone()

        # Rewards are transitions; root (t=0) does not get a reward prediction
        reward_mask = base_mask.clone()
        reward_mask[:, 0] = False
        current_targets["reward_mask"] = reward_mask

        # To-Play is state-aligned, but we might not want loss on the root if the representation doesn't predict it
        to_play_mask = batch.get("has_valid_obs_mask", base_mask).clone()
        to_play_mask[:, 0] = False
        to_play_mask &= reward_mask  # Cap at terminal states
        current_targets["to_play_mask"] = to_play_mask


class SequenceInfrastructureBuilder(BaseTargetBuilder):
    """Modifier: Generates non-sequence infrastructure tensors (weights, gradient scales)."""

    def __init__(self, unroll_steps: int):
        self.unroll_steps = unroll_steps

    def build_targets(self, batch, predictions, network, current_targets) -> None:
        device = current_targets["actions"].device
        B = current_targets["actions"].shape[0]

        if "weights" not in current_targets:
            current_targets["weights"] = batch.get(
                "weights", torch.ones(B, device=device)
            )

        if "gradient_scales" not in current_targets:
            scales = (
                [1.0] + [1.0 / self.unroll_steps] * self.unroll_steps
                if self.unroll_steps > 0
                else [1.0]
            )
            current_targets["gradient_scales"] = torch.tensor(
                scales, device=device
            ).reshape(1, -1)


class ChanceTargetBuilder(BaseTargetBuilder):
    """Generator: Calculates chance outcomes for Stochastic MuZero."""

    def build_targets(self, batch, predictions, network, current_targets) -> None:
        # Stochastic MuZero shifts the value target by 1 step for chance nodes
        if "values" in current_targets and "chance_values_next" not in current_targets:
            v = current_targets["values"]
            v_next = torch.zeros_like(v)
            v_next[:, :-1] = v[:, 1:]  # Shift left
            current_targets["chance_values_next"] = v_next


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

        # 1. The Builder explicitly owns the detached context
        # We use no_grad() so the resulting tensors are standard detached tensors,
        # NOT strict inference_mode tensors which crash loss functions.
        with torch.no_grad():
            initial_out = network.obs_inference(flat_obs)
            real_latents = initial_out.network_state.dynamics

            # No more .clone() hacks!
            # real_latents is already safely detached.
            proj_targets = network.project(real_latents, grad=False)
            normalized_targets = torch.nn.functional.normalize(
                proj_targets, p=2.0, dim=-1, eps=1e-5
            )

        consistency_targets = normalized_targets.reshape(
            batch_size, unroll_len, -1
        ).detach()
        current_targets["consistency_targets"] = consistency_targets
