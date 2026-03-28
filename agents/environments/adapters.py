import torch
import numpy as np
import gymnasium as gym
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, List, Union

class BaseAdapter(ABC):
    """
    Abstract base class for all environment adapters.
    Ensures a consistent interface: PyTorch tensors, leading batch dimension, correct device.
    """
    def __init__(self, device: torch.device):
        self.device = device

    @abstractmethod
    def reset(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Resets the environment.
        Returns:
            obs: torch.Tensor of shape [B, ...]
            info: Dict containing transition details, possibly batched.
        """
        pass

    @abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Performs one step in the environment.
        Args:
            actions: torch.Tensor of shape [B, ...]
        Returns:
            next_obs: torch.Tensor of shape [B, ...]
            rewards: torch.Tensor of shape [B]
            terminals: torch.Tensor of shape [B] (boolean)
            truncations: torch.Tensor of shape [B] (boolean)
            infos: Dict containing transition details.
        """
        pass


class GymAdapter(BaseAdapter):
    """
    Wraps standard single-player Gymnasium environments.
    Handles dimension expansion for observations and rewards to shape [1, ...].
    """
    def __init__(self, env_or_factory: Any, device: torch.device, num_actions: Optional[int] = None):
        super().__init__(device)
        if callable(env_or_factory):
            self.env = env_or_factory()
        else:
            self.env = env_or_factory
        
        # Try to infer num_actions for the legal_moves_mask
        if num_actions is not None:
            self.num_actions = num_actions
        elif hasattr(self.env.action_space, "n"):
            self.num_actions = self.env.action_space.n
        else:
            self.num_actions = None
        self.num_envs = 1

    def reset(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        res = self.env.reset()
        if isinstance(res, tuple) and len(res) == 2:
            obs, info = res
        else:
            obs, info = res, {}
            
        if info is None:
            info = {}
            
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        processed_info = self._process_info(info)
        return obs_tensor, processed_info

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # actions is [1, ...]
        if actions.numel() == 1:
            action = actions.item()
        else:
            action = actions[0].cpu().numpy()
            
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Auto-reset logic: preserve terminal obs, then reset for next episode
        if terminated or truncated:
            info["terminal_observation"] = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            new_obs, reset_info = self.env.reset()
            if reset_info:
                info.update(reset_info)
            obs = new_obs
            
        return (
            torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0),
            torch.tensor([reward], dtype=torch.float32, device=self.device),
            torch.tensor([terminated], dtype=torch.bool, device=self.device),
            torch.tensor([truncated], dtype=torch.bool, device=self.device),
            self._process_info(info)
        )

    def _process_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        processed = info.copy()
        # Standardize player_id for Actor/Search consumption
        # Guaranteed to be Tensor[B] where B=1 for single-player Gym
        processed["player_id"] = torch.tensor([0], dtype=torch.int64, device=self.device)
        
        # Expand legal_moves into a boolean mask [1, num_actions]
        if "legal_moves" in processed and self.num_actions is not None:
            mask = torch.zeros((1, self.num_actions), dtype=torch.bool, device=self.device)
            legal = processed["legal_moves"]
            if len(legal) > 0:
                mask[0, legal] = True
            else:
                mask.fill_(True)
            processed["legal_moves_mask"] = mask
        elif self.num_actions is not None:
            # Default to all legal if not provided
            mask = torch.ones((1, self.num_actions), dtype=torch.bool, device=self.device)
            processed["legal_moves_mask"] = mask
            
        return processed

class VectorAdapter(BaseAdapter):
    """
    Wraps native batched environments (PufferLib or Gym Vector).
    Acts as a pass-through, casting NumPy arrays to PyTorch tensors.
    """
    def __init__(self, vec_env_or_factory: Any, device: torch.device, num_actions: int):
        super().__init__(device)
        if callable(vec_env_or_factory):
            self.vec_env = vec_env_or_factory()
        else:
            self.vec_env = vec_env_or_factory
        self.num_actions = num_actions
        self.num_envs = getattr(self.vec_env, "num_envs", 1)

    def reset(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        result = self.vec_env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs = result
            info = {}
            
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_tensor.dim() == len(self.vec_env.observation_space.shape):
            obs_tensor = obs_tensor.unsqueeze(0)
            
        return obs_tensor, self._process_info(info)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # Map actions back to NumPy for the vectorized environment
        actions_np = actions.cpu().numpy()
        ans = self.vec_env.step(actions_np)
        
        # Unpack, being robust to different return lengths (e.g. PufferLib)
        obs, rewards, terminals, truncs, infos = ans[:5]
        
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        terminals_tensor = torch.as_tensor(terminals, dtype=torch.bool, device=self.device)
        truncs_tensor = torch.as_tensor(truncs, dtype=torch.bool, device=self.device)

        # Force batch dimension if squeezed by the environment
        if obs_tensor.dim() == len(self.vec_env.observation_space.shape):
            obs_tensor = obs_tensor.unsqueeze(0)
            rewards_tensor = rewards_tensor.unsqueeze(0)
            terminals_tensor = terminals_tensor.unsqueeze(0)
            truncs_tensor = truncs_tensor.unsqueeze(0)

        return (
            obs_tensor,
            rewards_tensor,
            terminals_tensor,
            truncs_tensor,
            self._process_info(infos)
        )

    def _process_info(self, info: Any) -> Dict[str, Any]:
        """Maps info dictionaries into batched tensor masks."""
        if info is None:
            return {"player_id": torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)}
        
        processed = {}
        # Normalize list-of-dicts (PufferLib style) to dict-of-lists
        if isinstance(info, list):
            keys = set()
            for d in info: 
                if d: keys.update(d.keys())
            for k in keys:
                processed[k] = [d.get(k) for d in info]
        else:
            processed = info.copy()

        # Build batched legal_moves_mask [B, num_actions]
        if "legal_moves" in processed:
            legal_batch = processed["legal_moves"]
            mask = torch.zeros((self.num_envs, self.num_actions), dtype=torch.bool, device=self.device)
            for i in range(self.num_envs):
                legal = legal_batch[i]
                if legal is not None and len(legal) > 0:
                    mask[i, legal] = True
                else:
                    mask[i].fill_(True)
            processed["legal_moves_mask"] = mask
        else:
            # Default to all legal
            mask = torch.ones((self.num_envs, self.num_actions), dtype=torch.bool, device=self.device)
            processed["legal_moves_mask"] = mask
            
        # Standardize player indexing: Guaranteed to be Tensor[B]
        p_id_raw = processed.get("player_id", processed.get("player"))
        # If we have multiple envs/players, we MUST have a player identity
        if getattr(self, "num_players", 1) > 1 or self.num_envs > 1:
            assert p_id_raw is not None, (
                "For multi-player or vectorized environments, 'player_id' or 'player' must be in info. "
                "Check your environment wrappers/adapters."
            )
            
        if p_id_raw is not None:
             p_id = torch.as_tensor(p_id_raw, dtype=torch.int64, device=self.device)
        else:
            p_id = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
            
        # Ensure it is at least 1D (B,)
        if p_id.dim() == 0:
            p_id = p_id.unsqueeze(0)
        processed["player_id"] = p_id
        # Keep legacy key for backward compatibility
        processed["player"] = p_id
            
        return processed

class PettingZooAdapter(BaseAdapter):
    """
    Wraps multi-agent AEC or Parallel environments.
    Exposes current player_id as a tensor and aligns rewards.
    """
    def __init__(self, env_or_factory: Any, device: torch.device, num_actions: Optional[int] = None):
        super().__init__(device)
        if callable(env_or_factory):
            self.env = env_or_factory()
        else:
            self.env = env_or_factory
        
        # Detect environment type
        self.is_aec = hasattr(self.env, "agent_selection")
        if not self.is_aec and hasattr(self.env, "unwrapped"):
            self.is_aec = hasattr(self.env.unwrapped, "agent_selection")
            
        self.agents = self.env.possible_agents
        # Try to infer num_actions
        if num_actions is not None:
            self.num_actions = num_actions
        else:
            first_space = self.env.action_space(self.agents[0])
            self.num_actions = first_space.n if hasattr(first_space, "n") else None
        
        self.num_envs = 1 if self.is_aec else len(self.agents)

    def reset(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.is_aec:
            self.env.reset()
            obs, reward, term, trunc, info = self.env.last()
            return torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0), self._process_info_aec(info)
        else:
            # Parallel API
            obs_dict, info_dict = self.env.reset()
            obs_list = np.stack([obs_dict[a] for a in self.agents])
            return torch.as_tensor(obs_list, dtype=torch.float32, device=self.device), self._process_info_parallel(info_dict)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self.is_aec:
            acting_player = self.env.agent_selection
            # AEC takes a single scalar action for the active player
            action = actions.item() if actions.numel() == 1 else actions[0].item()
            self.env.step(action)
            
            # Capture ALL rewards before possible auto-reset
            reward = float(self.env.rewards.get(acting_player, 0.0))
            all_rewards = self.env.rewards.copy() if hasattr(self.env, "rewards") else {acting_player: reward}
            
            obs, _, term, trunc, env_info = self.env.last()
            info = env_info.copy()
            info["all_rewards"] = all_rewards

            # Auto-reset for AEC: preserve terminal obs, then reset
            if term or trunc:
                info["terminal_observation"] = torch.as_tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                self.env.reset()
                new_obs, _, _, _, reset_info = self.env.last()
                obs = new_obs
                if reset_info:
                    info.update(reset_info)
            
            return (
                torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0),
                torch.tensor([reward], dtype=torch.float32, device=self.device),
                torch.tensor([term], dtype=torch.bool, device=self.device),
                torch.tensor([trunc], dtype=torch.bool, device=self.device),
                self._process_info_aec(info)
            )
        else:
            # Parallel API takes a dict mapping agent names to actions
            action_dict = {
                a: actions[i].item() if actions[i].numel() == 1 else actions[i].item() 
                for i, a in enumerate(self.agents)
            }
            obs_dict, reward_dict, term_dict, trunc_dict, info_dict = self.env.step(action_dict)
            for a in self.agents:
                if a not in info_dict:
                    info_dict[a] = {}
                info_dict[a]["all_rewards"] = reward_dict

            # Auto-reset for Parallel API: preserve terminal obs, then reset
            episode_over = any(term_dict.values()) or any(trunc_dict.values())
            if episode_over:
                terminal_obs_list = np.stack([obs_dict[a] for a in self.agents])
                for i, a in enumerate(self.agents):
                    if a not in info_dict:
                        info_dict[a] = {}
                    info_dict[a]["terminal_observation"] = terminal_obs_list[i]
                new_obs_dict, reset_info_dict = self.env.reset()
                obs_dict = new_obs_dict
                if reset_info_dict:
                    info_dict.update(reset_info_dict)

            obs_list = np.stack([obs_dict[a] for a in self.agents])
            reward_list = [reward_dict[a] for a in self.agents]
            term_list = [term_dict[a] for a in self.agents]
            trunc_list = [trunc_dict[a] for a in self.agents]
            
            return (
                torch.as_tensor(obs_list, dtype=torch.float32, device=self.device),
                torch.tensor(reward_list, dtype=torch.float32, device=self.device),
                torch.tensor(term_list, dtype=torch.bool, device=self.device),
                torch.tensor(trunc_list, dtype=torch.bool, device=self.device),
                self._process_info_parallel(info_dict)
            )

    def _process_info_aec(self, info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        info = (info or {}).copy()
        agent = self.env.agent_selection
        # PettingZoo AEC: agent_selection is the only source of truth for current player
        if len(self.agents) > 1:
            assert agent is not None, "PettingZoo AEC environment missing agent_selection for multi-agent game."
            
        idx = self.agents.index(agent) if agent in self.agents else 0
        p_id = torch.tensor([idx], dtype=torch.int64, device=self.device)
        info["player_id"] = p_id
        info["player"] = p_id
        
        if "legal_moves" in info and self.num_actions is not None:
            mask = torch.zeros((1, self.num_actions), dtype=torch.bool, device=self.device)
            legal = info["legal_moves"]
            if len(legal) > 0:
                mask[0, legal] = True
            else:
                mask.fill_(True)
            info["legal_moves_mask"] = mask
        elif self.num_actions is not None:
            info["legal_moves_mask"] = torch.ones((1, self.num_actions), dtype=torch.bool, device=self.device)
            
        return info

    def _process_info_parallel(self, info_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregates per-agent info from Parallel environments into batched tensors."""
        processed = {
            "player_id": torch.tensor(list(range(len(self.agents))), dtype=torch.int64, device=self.device),
            "player": torch.tensor(list(range(len(self.agents))), dtype=torch.int64, device=self.device)
        }
        
        if self.num_actions is not None:
            num_agents = len(self.agents)
            mask = torch.zeros((num_agents, self.num_actions), dtype=torch.bool, device=self.device)
            for i, a in enumerate(self.agents):
                info = info_dict.get(a, {})
                legal = info.get("legal_moves", [])
                if len(legal) > 0:
                    mask[i, legal] = True
                else:
                    mask[i].fill_(True)
            processed["legal_moves_mask"] = mask
            
        return processed
