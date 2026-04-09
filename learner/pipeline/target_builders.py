import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from learner.pipeline.base import PipelineComponent
from learner.core import Blackboard

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
        bootstrap_on_truncated: bool = False
    ):
        self.target_network = target_network
        self.online_network = online_network
        self.discount = gamma ** n_step
        self.bootstrap_on_truncated = bootstrap_on_truncated

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
                online_next_out = self.online_network.learner_inference({"observations": next_obs})
                next_q_values = online_next_out["q_values"]
                if next_q_values.ndim == 3:
                    next_q_values = next_q_values.squeeze(1)
                
                if next_masks is not None:
                    next_q_values = next_q_values.masked_fill(~next_masks.bool(), -float("inf"))
                
                next_actions = next_q_values.argmax(dim=-1)
                
                # Use target network for value estimation
                target_out = self.target_network.learner_inference({"observations": next_obs})
                target_q_values = target_out["q_values"]
                if target_q_values.ndim == 3:
                    target_q_values = target_q_values.squeeze(1)
                    
                max_next_q = target_q_values[torch.arange(batch_size, device=rewards.device), next_actions]
                blackboard.targets["next_actions"] = next_actions.unsqueeze(1)
            else:
                # Standard Q-Learning: Evaluate next states on target network
                target_out = self.target_network.learner_inference({"observations": next_obs})
                target_q_values = target_out["q_values"]
                if target_q_values.ndim == 3:
                    target_q_values = target_q_values.squeeze(1)
                max_next_q = target_q_values.max(dim=-1).values

        # Bellman Math
        target_q = rewards + (1.0 - terminal_mask.float()) * self.discount * max_next_q

        # Write directly to the Universal targets dict
        # Force [B, T=1] dimension to satisfy the Universal Time Mandate
        blackboard.targets["values"] = target_q.unsqueeze(1)
        blackboard.targets["rewards"] = rewards.unsqueeze(1)
        blackboard.targets["dones"] = terminal_mask.float().unsqueeze(1)
        blackboard.targets["actions"] = data["actions"].unsqueeze(1)

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
        self.discount = gamma ** n_step
        self.bootstrap_on_truncated = bootstrap_on_truncated

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
            online_next_out = self.online_network.learner_inference({"observations": next_obs})
            next_q_values = online_next_out["q_values"]
            if next_q_values.ndim == 3:
                next_q_values = next_q_values.squeeze(1)

            if next_masks is not None:
                next_q_values = next_q_values.masked_fill(~next_masks.bool(), -float("inf"))
            next_actions = next_q_values.argmax(dim=-1)

            # Target network evaluation
            target_out = self.target_network.learner_inference({"observations": next_obs})
            next_q_logits = target_out["q_logits"]
            if next_q_logits.ndim == 4:
                next_q_logits = next_q_logits.squeeze(1)

            next_probs = torch.softmax(
                next_q_logits[torch.arange(batch_size, device=rewards.device), next_actions],
                dim=-1,
            )

        # Get the base grid geometry from the network's representation
        # Assuming we can find the representation from the online network or similar
        # For NFSP/Rainbow, it's usually in network.components["q_head"].representation
        representation = getattr(self.online_network, "components", {}).get("q_head", {}).representation
        if representation is None:
            # Fallback or error
            raise RuntimeError("Could not find representation for DistributionalTargetComponent")

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
        blackboard.targets["rewards"] = rewards.unsqueeze(1)
        blackboard.targets["actions"] = data["actions"].unsqueeze(1)
        blackboard.targets["next_actions"] = next_actions.unsqueeze(1)
        blackboard.targets["dones"] = terminal_mask.float().unsqueeze(1)

class PassThroughTargetComponent(PipelineComponent):
    """
    Generic whitelist-based component that passes specific keys
    from the batch through to the targets.
    """
    def __init__(self, keys_to_keep: List[str]):
        self.keys_to_keep = keys_to_keep

    def execute(self, blackboard: Blackboard) -> None:
        data = blackboard.data
        for key in self.keys_to_keep:
            if key in data:
                val = data[key]
                # Ensure T dimension [B, 1, ...] if not already present
                if torch.is_tensor(val) and (val.ndim == 1 or (val.ndim >= 2 and val.shape[1] != 1)):
                    blackboard.targets[key] = val.unsqueeze(1)
                else:
                    blackboard.targets[key] = val

class MCTSExtractorComponent(PipelineComponent):
    """Extracts MCTS search statistics from the batch into targets."""
    def execute(self, blackboard: Blackboard) -> None:
        data = blackboard.data
        target_keys = ["values", "rewards", "policies", "actions", "to_plays"]
        for key in target_keys:
            if key in data:
                blackboard.targets[key] = data[key]

class SequencePadderComponent(PipelineComponent):
    """Modifier: Pads transition-aligned data to state-aligned length."""
    def __init__(self, unroll_steps: int):
        self.T = unroll_steps + 1

    def execute(self, blackboard: Blackboard) -> None:
        for key, v in blackboard.targets.items():
            if torch.is_tensor(v) and v.ndim >= 2 and v.shape[1] == self.T - 1:
                padding_shape = list(v.shape)
                padding_shape[1] = 1
                padding = torch.zeros(padding_shape, device=v.device, dtype=v.dtype)
                blackboard.targets[key] = torch.cat([padding, v], dim=1)

class SequenceMaskComponent(PipelineComponent):
    """Modifier: Generates Universal [B, T] sequence masks."""
    def execute(self, blackboard: Blackboard) -> None:
        data = blackboard.data
        blackboard.targets["value_mask"] = data["is_same_game"].clone()
        blackboard.targets["masks"] = data["is_same_game"].clone()
        blackboard.targets["policy_mask"] = data["has_valid_obs_mask"].clone()
        blackboard.targets["policy_mask"] &= ~data["dones"]

        blackboard.targets["reward_mask"] = data["is_same_game"].clone()
        blackboard.targets["reward_mask"][:, 0] = False

        blackboard.targets["to_play_mask"] = data["is_same_game"].clone()
        blackboard.targets["to_play_mask"][:, 0] = False

class SequenceInfrastructureComponent(PipelineComponent):
    """Modifier: Generates weight and gradient scale tensors."""
    def __init__(self, unroll_steps: int):
        self.unroll_steps = unroll_steps

    def execute(self, blackboard: Blackboard) -> None:
        data = blackboard.data
        device = next(iter(blackboard.targets.values())).device if blackboard.targets else torch.device("cpu")
        B = data["actions"].shape[0]

        if "weights" not in blackboard.meta:
            blackboard.meta["weights"] = data.get("weights", torch.ones(B, device=device))

        if "gradient_scales" not in blackboard.meta:
            scales = [1.0] + [1.0 / self.unroll_steps] * self.unroll_steps if self.unroll_steps > 0 else [1.0]
            blackboard.meta["gradient_scales"] = torch.tensor(scales, device=device).reshape(1, -1)

class LatentConsistencyComponent(PipelineComponent):
    """EfficientZero consistency loss target builder."""
    def __init__(self, agent_network: nn.Module):
        self.agent_network = agent_network

    def execute(self, blackboard: Blackboard) -> None:
        real_obs = blackboard.data["unroll_observations"].float()
        batch_size, unroll_len = real_obs.shape[:2]
        flat_obs = real_obs.flatten(0, 1)

        with torch.no_grad():
            initial_out = self.agent_network.obs_inference(flat_obs)
            real_latents = initial_out.network_state.dynamics
            proj_targets = self.agent_network.project(real_latents, grad=False)
            normalized_targets = torch.nn.functional.normalize(proj_targets, p=2.0, dim=-1, eps=1e-5)

        blackboard.targets["consistency_targets"] = normalized_targets.reshape(batch_size, unroll_len, -1).detach()

class ChanceTargetComponent(PipelineComponent):
    """Generator: Calculates chance outcomes for Stochastic MuZero."""
    def execute(self, blackboard: Blackboard) -> None:
        # Stochastic MuZero shifts the value target by 1 step for chance nodes
        if "values" in blackboard.targets and "chance_values_next" not in blackboard.targets:
            v = blackboard.targets["values"]
            v_next = torch.zeros_like(v)
            v_next[:, :-1] = v[:, 1:]  # Shift left
            blackboard.targets["chance_values_next"] = v_next

class TargetFormatterComponent(PipelineComponent):
    """Formats target keys using their respective representations."""
    def __init__(self, target_mapping: Dict[str, Any]):
        self.target_mapping = target_mapping

    def execute(self, blackboard: Blackboard) -> None:
        for key, rep in self.target_mapping.items():
            if key in blackboard.targets:
                blackboard.targets[key] = rep.format_target(blackboard.targets, target_key=key)

class UniversalInfrastructureComponent(PipelineComponent):
    """
    Standard Infrastructure Component for single-step learners.
    Ensures masks, weights, and gradient scales exist.
    """
    def execute(self, blackboard: Blackboard) -> None:
        if not blackboard.targets:
            return
            
        any_val = next(iter(blackboard.targets.values()))
        batch_size = any_val.shape[0]
        device = any_val.device
        
        # 1. Generate Universal T=1 Masks if missing
        generic_mask = torch.ones((batch_size, 1), device=device, dtype=torch.bool)
        for mask_key in ["value_mask", "reward_mask", "policy_mask", "q_mask", "masks"]:
            if mask_key not in blackboard.targets:
                blackboard.targets[mask_key] = generic_mask

        # 2. Weights and Gradient Scales
        if "weights" not in blackboard.meta:
            blackboard.meta["weights"] = blackboard.data.get("weights", torch.ones(batch_size, device=device))
        if "gradient_scales" not in blackboard.meta:
            blackboard.meta["gradient_scales"] = torch.ones((1, 1), device=device)
