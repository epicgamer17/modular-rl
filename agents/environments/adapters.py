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

    def _to_tensor(self, data: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Helper to move data to the correct device/dtype and ensure leading dimension."""
        if isinstance(data, list):
            try:
                # Handle lists of numbers or booleans
                if all(isinstance(x, (bool, int, float, np.bool_, type(None))) for x in data):
                    # Filter out None and convert
                    clean_data = [x if x is not None else 0 for x in data]
                    data = np.array(clean_data)
                else:
                    # Possibly a list of numpy arrays (like observations in vector env)
                    if all(isinstance(x, np.ndarray) for x in data):
                        data = np.stack(data)
            except (ValueError, TypeError):
                pass
                
        if isinstance(data, (np.ndarray, float, int, bool, np.bool_)):
            return torch.tensor(data, device=self.device, dtype=dtype)
        elif isinstance(data, torch.Tensor):
            t = data.to(self.device)
            if dtype is not None:
                t = t.to(dtype)
            return t
        
        # Fallback for complex nesting or other types
        return torch.as_tensor(data, device=self.device, dtype=dtype)

class GymAdapter(BaseAdapter):
    """
    Wraps standard single-player Gymnasium environments.
    Handles dimension expansion for observations and rewards to shape [1, ...].
    """
    def __init__(self, env: gym.Env, device: torch.device, num_actions: Optional[int] = None):
        super().__init__(device)
        self.env = env
        # Try to infer num_actions for the legal_moves_mask
        if num_actions is not None:
            self.num_actions = num_actions
        elif hasattr(env.action_space, "n"):
            self.num_actions = env.action_space.n
        else:
            self.num_actions = None

    def reset(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        res = self.env.reset()
        if isinstance(res, tuple) and len(res) == 2:
            obs, info = res
        else:
            obs, info = res, {}
            
        if info is None:
            info = {}
            
        obs_tensor = self._to_tensor(obs).unsqueeze(0) 
        processed_info = self._process_info(info)
        return obs_tensor, processed_info

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # actions is [1, ...]
        if actions.numel() == 1:
            action = actions.item()
        else:
            action = actions[0].cpu().numpy()
            
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        return (
            self._to_tensor(obs).unsqueeze(0),
            self._to_tensor([reward], dtype=torch.float32),
            self._to_tensor([terminated], dtype=torch.bool),
            self._to_tensor([truncated], dtype=torch.bool),
            self._process_info(info)
        )

    def _process_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        processed = info.copy()
        # Standardize player_id for Actor/Search consumption
        processed["player_id"] = self._to_tensor([0], dtype=torch.int64)
        
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
    def __init__(self, vec_env: Any, device: torch.device, num_actions: int):
        super().__init__(device)
        self.vec_env = vec_env
        self.num_actions = num_actions
        self.num_envs = getattr(vec_env, "num_envs", 1)

    def reset(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        result = self.vec_env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs = result
            info = {}
            
        return self._to_tensor(obs), self._process_info(info)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # Map actions back to NumPy for the vectorized environment
        actions_np = actions.cpu().numpy()
        ans = self.vec_env.step(actions_np)
        
        # Unpack, being robust to different return lengths (e.g. PufferLib)
        obs, rewards, terminals, truncs, infos = ans[:5]
        
        return (
            self._to_tensor(obs),
            self._to_tensor(rewards, dtype=torch.float32),
            self._to_tensor(terminals, dtype=torch.bool),
            self._to_tensor(truncs, dtype=torch.bool),
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
            
        # Standardize player indexing
        if "player" in processed:
            processed["player_id"] = self._to_tensor(processed["player"], dtype=torch.int64)
        else:
            processed["player_id"] = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
            
        return processed

class PettingZooAdapter(BaseAdapter):
    """
    Wraps multi-agent AEC or Parallel environments.
    Exposes current player_id as a tensor and aligns rewards.
    """
    def __init__(self, env: Any, device: torch.device, num_actions: Optional[int] = None):
        super().__init__(device)
        self.env = env
        
        # Detect environment type
        self.is_aec = hasattr(env, "agent_selection")
        if not self.is_aec and hasattr(env, "unwrapped"):
            self.is_aec = hasattr(env.unwrapped, "agent_selection")
            
        self.agents = env.possible_agents
        # Try to infer num_actions
        if num_actions is not None:
            self.num_actions = num_actions
        else:
            first_space = env.action_space(self.agents[0])
            self.num_actions = first_space.n if hasattr(first_space, "n") else None

    def reset(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.is_aec:
            self.env.reset()
            obs, reward, term, trunc, info = self.env.last()
            return self._to_tensor(obs).unsqueeze(0), self._process_info_aec(info)
        else:
            # Parallel API
            obs_dict, info_dict = self.env.reset()
            obs_list = [obs_dict[a] for a in self.agents]
            return self._to_tensor(obs_list), self._process_info_parallel(info_dict)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self.is_aec:
            acting_player = self.env.agent_selection
            # AEC takes a single scalar action for the active player
            action = actions.item() if actions.numel() == 1 else actions[0].item()
            self.env.step(action)
            
            # Formatting rewards: return reward for the agent who just moved
            reward = float(self.env.rewards.get(acting_player, 0.0))
            
            obs, _, term, trunc, info = self.env.last()
            
            return (
                self._to_tensor(obs).unsqueeze(0),
                self._to_tensor([reward], dtype=torch.float32),
                self._to_tensor([term], dtype=torch.bool),
                self._to_tensor([trunc], dtype=torch.bool),
                self._process_info_aec(info)
            )
        else:
            # Parallel API takes a dict mapping agent names to actions
            action_dict = {
                a: actions[i].item() if actions[i].numel() == 1 else actions[i].item() 
                for i, a in enumerate(self.agents)
            }
            obs_dict, reward_dict, term_dict, trunc_dict, info_dict = self.env.step(action_dict)
            
            obs_list = [obs_dict[a] for a in self.agents]
            reward_list = [reward_dict[a] for a in self.agents]
            term_list = [term_dict[a] for a in self.agents]
            trunc_list = [trunc_dict[a] for a in self.agents]
            
            return (
                self._to_tensor(obs_list),
                self._to_tensor(reward_list, dtype=torch.float32),
                self._to_tensor(term_list, dtype=torch.bool),
                self._to_tensor(trunc_list, dtype=torch.bool),
                self._process_info_parallel(info_dict)
            )

    def _process_info_aec(self, info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        info = (info or {}).copy()
        agent = self.env.agent_selection
        idx = self.agents.index(agent) if agent in self.agents else 0
        info["player_id"] = self._to_tensor([idx], dtype=torch.int64)
        
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
            "player_id": self._to_tensor(list(range(len(self.agents))), dtype=torch.int64)
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
