import torch
import torch.nn as nn
from typing import Optional
from core import PipelineComponent
from core import Blackboard


class TDTargetComponent(PipelineComponent):
    """
    Lazy Learner-Side Target Builder.
    Reads rewards/dones from batch, bootstraps using target_network, writes 'values' to targets.
    Supports Double DQN action selection if online_network is provided.
    """

    def __init__(
        self,
        target_network: nn.Module,
        online_network: Optional[nn.Module] = None,
        gamma: float = 0.99,
        n_step: int = 1,
        bootstrap_on_truncated: bool = False,
    ):
        self.target_network = target_network
        self.online_network = online_network
        self.discount = gamma**n_step
        self.bootstrap_on_truncated = bootstrap_on_truncated

    @property
    def reads(self) -> set[str]:
        keys = {"data.rewards", "data.dones", "data.next_observations"}
        # 'terminated' and 'next_legal_moves_masks' are optional in some buffers
        # but if we want strict validation, we should decide if they are required.
        # For now, let's assume 'terminated' is a fallback for 'dones' if missing.
        return keys

    @property
    def writes(self) -> set[str]:
        w = {"targets.values"}
        if self.online_network is not None:
            w.add("targets.next_actions")
        return w

    def execute(self, blackboard: Blackboard) -> None:
        data = blackboard.data
        rewards = data["rewards"].float()
        dones = data["dones"].bool()
        terminated = data.get("terminated", dones).bool()
        next_obs = data.get("next_observations")
        next_masks = data.get("next_legal_moves_masks")

        terminal_mask = terminated if self.bootstrap_on_truncated else dones
        batch_size = rewards.shape[0]

        with torch.no_grad():
            if self.online_network is not None:
                # Double DQN: Use online network for action selection
                online_next_out = self.online_network.learner_inference(
                    {"observations": next_obs}
                )
                next_q_values = online_next_out["q_values"]
                if next_q_values.ndim == 3:
                    next_q_values = next_q_values.squeeze(1)

                if next_masks is not None:
                    next_q_values = next_q_values.masked_fill(
                        ~next_masks.bool(), -float("inf")
                    )

                next_actions = next_q_values.argmax(dim=-1)

                # Use target network for value estimation
                target_out = self.target_network.learner_inference(
                    {"observations": next_obs}
                )
                target_q_values = target_out["q_values"]
                if target_q_values.ndim == 3:
                    target_q_values = target_q_values.squeeze(1)

                max_next_q = target_q_values[
                    torch.arange(batch_size, device=rewards.device), next_actions
                ]
                blackboard.targets["next_actions"] = next_actions.unsqueeze(1)
            else:
                # Standard Q-Learning: Evaluate next states on target network
                target_out = self.target_network.learner_inference(
                    {"observations": next_obs}
                )
                target_q_values = target_out["q_values"]
                if target_q_values.ndim == 3:
                    target_q_values = target_q_values.squeeze(1)

                if next_masks is not None:
                    target_q_values = target_q_values.masked_fill(
                        ~next_masks.bool(), -float("inf")
                    )

                max_next_q = target_q_values.max(dim=-1).values

        # Bellman Math
        target_q = rewards + (1.0 - terminal_mask.float()) * self.discount * max_next_q

        # Write directly to the Universal targets dict
        # Force [B, T=1] dimension to satisfy the Universal Time Mandate
        blackboard.targets["values"] = target_q.unsqueeze(1)
        if self.online_network is not None:
             blackboard.targets["next_actions"] = next_actions.unsqueeze(1)


class DistributionalTargetComponent(PipelineComponent):
    """
    Component for C51/Distributional RL targets.
    Handles the Bellman Shift (MDP math) and delegates projection to the Representation.
    """

    def __init__(
        self,
        target_network: nn.Module,
        online_network: nn.Module,
        gamma: float = 0.99,
        n_step: int = 1,
        bootstrap_on_truncated: bool = False,
    ):
        self.target_network = target_network
        self.online_network = online_network
        self.discount = gamma**n_step
        self.bootstrap_on_truncated = bootstrap_on_truncated

    @property
    def reads(self) -> set[str]:
        return {"data.rewards", "data.dones", "data.next_observations"}

    @property
    def writes(self) -> set[str]:
        return {"targets.q_logits", "targets.next_actions"}

    def execute(self, blackboard: Blackboard) -> None:
        data = blackboard.data
        rewards = data["rewards"].float()
        dones = data["dones"].bool()
        terminated = data.get("terminated", dones).bool()
        next_obs = data.get("next_observations")
        next_masks = data.get("next_legal_moves_masks")

        terminal_mask = terminated if self.bootstrap_on_truncated else dones
        batch_size = rewards.shape[0]

        with torch.no_grad():
            # Double DQN action selection
            online_next_out = self.online_network.learner_inference(
                {"observations": next_obs}
            )
            next_q_values = online_next_out["q_values"]
            if next_q_values.ndim == 3:
                next_q_values = next_q_values.squeeze(1)

            if next_masks is not None:
                next_q_values = next_q_values.masked_fill(
                    ~next_masks.bool(), -float("inf")
                )
            next_actions = next_q_values.argmax(dim=-1)

            # Target network evaluation
            target_out = self.target_network.learner_inference(
                {"observations": next_obs}
            )
            next_q_logits = target_out["q_logits"]
            if next_q_logits.ndim == 4:
                next_q_logits = next_q_logits.squeeze(1)

            next_probs = torch.softmax(
                next_q_logits[
                    torch.arange(batch_size, device=rewards.device), next_actions
                ],
                dim=-1,
            )

        # Get the base grid geometry from the network's representation
        # For NFSP/Rainbow, it's usually in network.components["q_head"].representation
        components = getattr(self.online_network, "components", {})
        q_head = None
        if "q_head" in components:
            q_head = components["q_head"]
        elif hasattr(self.online_network, "q_head"):
            q_head = self.online_network.q_head

        if q_head is None:
            # Fallback or error
            raise RuntimeError(
                "Could not find q_head/representation for DistributionalTargetComponent"
            )

        representation = q_head.representation

        base_support = representation.support.to(rewards.device)

        # MDP Math: Shift the support
        shifted_support = rewards.unsqueeze(1) + self.discount * (
            1.0 - terminal_mask.float()
        ).unsqueeze(1) * base_support.unsqueeze(0)

        # Delegate projection
        target_distribution = representation.project_onto_grid(
            shifted_support=shifted_support, probabilities=next_probs
        )

        blackboard.targets["q_logits"] = target_distribution.unsqueeze(1)
        blackboard.targets["next_actions"] = next_actions.unsqueeze(1)
