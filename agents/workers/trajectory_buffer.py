import torch
import numpy as np
from typing import Dict, Any, Optional

class TrajectoryBuffer:
    """
    Pre-allocated tensor storage for rollout chunks.
    Designed for massive speedups in hot loops by avoiding 
    per-step dictionary allocations and per-environment Python loops.
    
    Fields are stored in [T, B, ...] format where T is chunk_size.
    """

    def __init__(
        self,
        num_envs: int,
        chunk_size: int,
        obs_shape: tuple,
        num_actions: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_envs = num_envs
        self.chunk_size = chunk_size
        self.device = device

        # 1. Base Transition Fields (Pre-allocated)
        self.obs = torch.zeros((chunk_size, num_envs, *obs_shape), device=self.device)
        self.actions = torch.zeros(
            (chunk_size, num_envs), dtype=torch.long, device=self.device
        )
        self.rewards = torch.zeros((chunk_size, num_envs), device=self.device)
        self.terminals = torch.zeros(
            (chunk_size, num_envs), dtype=torch.bool, device=self.device
        )
        self.truncations = torch.zeros(
            (chunk_size, num_envs), dtype=torch.bool, device=self.device
        )

        # 2. Math & Policy Fields
        self.values = torch.zeros((chunk_size, num_envs), device=self.device)
        self.log_probs = torch.zeros((chunk_size, num_envs), device=self.device)

        self.probs = None
        if num_actions is not None:
            self.probs = torch.zeros(
                (chunk_size, num_envs, num_actions), device=self.device
            )

        # 3. Environmental Metadata
        self.player_ids = torch.zeros(
            (chunk_size, num_envs), dtype=torch.int64, device=self.device
        )
        
        self.legal_moves_masks = None
        if num_actions is not None:
            self.legal_moves_masks = torch.ones(
                (chunk_size, num_envs, num_actions), dtype=torch.bool, device=self.device
            )

        self.step_idx = 0

    def insert(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        truncations: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        log_probs: Optional[torch.Tensor] = None,
        player_ids: Optional[torch.Tensor] = None,
        legal_moves_masks: Optional[torch.Tensor] = None,
    ):
        """
        FAST: Vectorized block insertion. 
        Expects Tensors of shape [B, ...].
        """
        if self.step_idx >= self.chunk_size:
            # Silent reset or raise error? Usually better to raise error in a hot loop
            # and let the actor manage its own resetting.
            raise ValueError(f"TrajectoryBuffer Overflow: index {self.step_idx} >= chunk_size {self.chunk_size}")

        t = self.step_idx
        
        # Ensure data is moved if needed, but copy_ handles device differences (slowly)
        # Ideally info/rewards/actions are already on the target device
        self.obs[t].copy_(obs)
        self.actions[t].copy_(actions)
        self.rewards[t].copy_(rewards)
        self.terminals[t].copy_(terminals)
        self.truncations[t].copy_(truncations)

        if values is not None:
            self.values[t].copy_(values.squeeze(-1) if values.dim() > 1 else values)
        if probs is not None and self.probs is not None:
            self.probs[t].copy_(probs)
        if log_probs is not None:
            self.log_probs[t].copy_(log_probs)
        if player_ids is not None:
            self.player_ids[t].copy_(player_ids)
        if legal_moves_masks is not None and self.legal_moves_masks is not None:
            self.legal_moves_masks[t].copy_(legal_moves_masks)

        self.step_idx += 1

    def clear(self):
        """Resets the pointer for the next chunk."""
        self.step_idx = 0

    def get_step_dict(self, t: int, env_idx: int) -> Dict[str, Any]:
        """
        Retrieves a single transition as a dictionary for downstream 
        compatibility with SequenceManager.
        """
        transition = {
            "observation": self.obs[t, env_idx].cpu().numpy(),
            "action": int(self.actions[t, env_idx].item()),
            "reward": float(self.rewards[t, env_idx].item()),
            "terminated": bool(self.terminals[t, env_idx].item()),
            "truncated": bool(self.truncations[t, env_idx].item()),
        }

        # Optional fields: only include if they were tracked
        if self.values[t, env_idx] != 0:
            transition["value"] = float(self.values[t, env_idx].item())
            
        if self.probs is not None:
            transition["policy"] = self.probs[t, env_idx].cpu().numpy()
        
        if self.log_probs[t, env_idx] != 0:
            transition["log_prob"] = float(self.log_probs[t, env_idx].item())
            
        if self.player_ids[t, env_idx] != 0:
            transition["player_id"] = int(self.player_ids[t, env_idx].item())

        if self.legal_moves_masks is not None:
            mask = self.legal_moves_masks[t, env_idx]
            transition["legal_moves"] = torch.where(mask)[0].cpu().numpy().tolist()

        return transition
