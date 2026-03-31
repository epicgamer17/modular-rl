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
    def __init__(self, device: torch.device, num_players: int = 1):
        self.device = device
        self.num_players = num_players
        self.num_envs = 0  # To be set by subclasses
        self._current_player_ids = None
        self._current_scores = None
        self._current_lengths = None
        self._batch_scores = []
        self._batch_lengths = []

    def _init_tracking(self):
        """Initializes internal buffers once num_envs is known."""
        self._current_scores = np.zeros((self.num_envs, self.num_players))
        self._current_lengths = np.zeros(self.num_envs)
        # Default to player 0 as acting
        self._current_player_ids = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

    def _update_metrics(self, rewards: torch.Tensor, terminals: torch.Tensor, truncations: torch.Tensor, infos: Dict[str, Any]):
        """Updates internal score and length tracking. Should be called at the end of step()."""
        if self._current_scores is None:
            self._init_tracking()

        for i in range(self.num_envs):
            # 1. Update Scores
            all_rewards = infos.get("all_rewards")
            if all_rewards is not None:
                # Support AEC-style index dicts or batched tensors
                if isinstance(all_rewards, dict):
                    for p_idx, r_val in all_rewards.items():
                        self._current_scores[i, p_idx] += float(r_val)
                elif isinstance(all_rewards, (list, np.ndarray, torch.Tensor)):
                    r_vec = all_rewards[i]
                    if torch.is_tensor(r_vec):
                        r_vec = r_vec.cpu().numpy()
                    self._current_scores[i] += r_vec
            else:
                # Single-reward env: credit the currently acting player
                p_id = int(self._current_player_ids[i].item())
                self._current_scores[i, p_id] += float(rewards[i].item())
            
            self._current_lengths[i] += 1
            
            # 2. Check for completion
            if terminals[i] or truncations[i]:
                ep_score = self._current_scores[i].copy()
                # If single player, return as scalar for cleaner logging
                if self.num_players == 1:
                    ep_score = float(ep_score[0])
                
                self._batch_scores.append(ep_score)
                self._batch_lengths.append(int(self._current_lengths[i]))
                
                # Reset
                self._current_scores[i].fill(0)
                self._current_lengths[i] = 0

    def get_metrics(self) -> Tuple[List[Any], List[int]]:
        """Returns completed episode scores and lengths since last call."""
        scores = self._batch_scores
        lengths = self._batch_lengths
        self._batch_scores = []
        self._batch_lengths = []
        return scores, lengths

    @property
    def current_lengths(self) -> np.ndarray:
        """Returns the current step count for each active episode."""
        if self._current_lengths is None:
            self._init_tracking()
        return self._current_lengths

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
    def __init__(self, env_or_factory: Any, device: torch.device, num_actions: Optional[int] = None, num_players: int = 1):
        super().__init__(device, num_players=num_players)
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

        # Update player tracking (Gym is always 0)
        self._current_player_ids = processed_info["player_id"]
        
        return obs_tensor, processed_info

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # actions is [1, ...]
        if actions.numel() == 1:
            action = actions.item()
        else:
            action = actions[0].cpu().numpy()
            
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Auto-reset logic: if episode ends, reset and return fresh obs for next step
        if terminated or truncated:
            # Preserve terminal state for MuZero/Sequence processing
            info["terminal_observation"] = obs
            info["terminal_info"] = self._process_info(info)

            new_obs, reset_info = self.env.reset()
            if reset_info:
                info.update(reset_info)
            # We return the new_obs so the Actor can start the next episode immediately
            obs = new_obs
            
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        rewards = torch.tensor([reward], dtype=torch.float32, device=self.device)
        terminals = torch.tensor([terminated], dtype=torch.bool, device=self.device)
        truncations = torch.tensor([truncated], dtype=torch.bool, device=self.device)
        processed_info = self._process_info(info)

        # Update Metrics
        self._update_metrics(rewards, terminals, truncations, processed_info)
        
        # After metrics are updated, update the next step's player
        self._current_player_ids = processed_info["player_id"]

        return (
            obs_tensor,
            rewards,
            terminals,
            truncations,
            processed_info
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
    def __init__(self, vec_env_or_factory: Any, device: torch.device, num_actions: int, num_players: int = 1):
        super().__init__(device, num_players=num_players)
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
        if obs_tensor.dim() < len(self.vec_env.observation_space.shape):
            obs_tensor = obs_tensor.unsqueeze(0)
            
        processed_info = self._process_info(info)
        self._current_player_ids = processed_info["player_id"]
        return obs_tensor, processed_info

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
        if obs_tensor.dim() < len(self.vec_env.observation_space.shape):
            obs_tensor = obs_tensor.unsqueeze(0)
            rewards_tensor = rewards_tensor.unsqueeze(0)
            terminals_tensor = terminals_tensor.unsqueeze(0)
            truncs_tensor = truncs_tensor.unsqueeze(0)

        processed_info = self._process_info(infos)
        
        # Update Metrics
        self._update_metrics(rewards_tensor, terminals_tensor, truncs_tensor, processed_info)
        
        # Update player tracking for next step
        self._current_player_ids = processed_info["player_id"]

        return (
            obs_tensor,
            rewards_tensor,
            terminals_tensor,
            truncs_tensor,
            processed_info
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
            
        # 3. Standardize player indexing: Guaranteed to be Tensor[B]
        if "player" in processed:
            p_id = torch.as_tensor(processed["player"], dtype=torch.int64, device=self.device)
            # Ensure it is at least 1D (B,)
            if p_id.dim() == 0:
                p_id = p_id.unsqueeze(0)
            processed["player_id"] = p_id
        else:
            processed["player_id"] = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
            
        return processed

class PettingZooAdapter(BaseAdapter):
    """
    Wraps multi-agent AEC or Parallel environments.
    Exposes current player_id as a tensor and aligns rewards.
    """
    def __init__(self, env_or_factory: Any, device: torch.device, num_actions: Optional[int] = None, num_players: Optional[int] = None):
        if callable(env_or_factory):
            env = env_or_factory()
        else:
            env = env_or_factory
        
        # Determine num_players if not provided
        agents = env.possible_agents
        if num_players is None:
            num_players = len(agents)
            
        super().__init__(device, num_players=num_players)
        self.env = env
        self.agents = agents
        
        # Detect environment type
        self.is_aec = hasattr(self.env, "agent_selection")
        if not self.is_aec and hasattr(self.env, "unwrapped"):
            self.is_aec = hasattr(self.env.unwrapped, "agent_selection")
            
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
            
            # Handle dict observations (common in PettingZoo AEC)
            if isinstance(obs, dict):
                if "action_mask" in obs and "legal_moves" not in info:
                    # Update info with legal moves from the mask
                    info = info.copy()
                    info["legal_moves"] = np.where(obs["action_mask"] == 1)[0]
                obs = obs.get("observation", obs)
            
            processed_info = self._process_info_aec(info)
            self._current_player_ids = processed_info["player_id"]
            return torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0), processed_info
        else:
            # Parallel API
            obs_dict, info_dict = self.env.reset()
            
            # Handle dict observations in parallel mode
            obs_list = []
            for a in self.agents:
                o = obs_dict[a]
                if isinstance(o, dict):
                    if "action_mask" in o:
                        if a not in info_dict: info_dict[a] = {}
                        info_dict[a]["legal_moves"] = np.where(o["action_mask"] == 1)[0]
                    o = o.get("observation", o)
                obs_list.append(o)
            
            processed_info = self._process_info_parallel(info_dict)
            self._current_player_ids = processed_info["player_id"]
            return torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=self.device), processed_info

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self.is_aec:
            acting_player = self.env.agent_selection
            # AEC takes a single scalar action for the active player
            action = actions.item() if actions.numel() == 1 else actions[0].item()
            self.env.step(action)
            
            # Capture ALL rewards before possible auto-reset
            reward = float(self.env.rewards.get(acting_player, 0.0))
            all_rewards = self.env.rewards.copy() if hasattr(self.env, "rewards") else {acting_player: reward}
            all_rewards_idx = {self.agents.index(a): r for a, r in all_rewards.items()}
            
            obs, _, term, trunc, env_info = self.env.last()
            info = (env_info or {}).copy()
            
            # Handle observation being a dict (with action_mask / observation)
            if isinstance(obs, dict):
                if "action_mask" in obs and "legal_moves" not in info:
                    info["legal_moves"] = np.where(obs["action_mask"] == 1)[0]
                obs = obs.get("observation", obs)
            
            info["all_rewards"] = all_rewards_idx
            
            is_done = term or trunc
            # Auto-reset for AEC: if episode is over, reset and get fresh root state
            if is_done:
                # Capture terminal data BEFORE reset
                info["terminal_observation"] = obs
                info["terminal_info"] = self._process_info_aec(info)

                self.env.reset()
                new_obs, _, _, _, reset_info = self.env.last()
                
                # Handle reset observation dict
                if isinstance(new_obs, dict):
                    if "action_mask" in new_obs and "legal_moves" not in (reset_info or {}):
                        reset_info = (reset_info or {}).copy()
                        reset_info["legal_moves"] = np.where(new_obs["action_mask"] == 1)[0]
                    new_obs = new_obs.get("observation", new_obs)
                
                obs = new_obs
                if reset_info:
                    info.update(reset_info)
            
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            rewards = torch.tensor([reward], dtype=torch.float32, device=self.device)
            terminals = torch.tensor([term], dtype=torch.bool, device=self.device)
            truncations = torch.tensor([trunc], dtype=torch.bool, device=self.device)
            processed_info = self._process_info_aec(info)

            # Update Metrics
            self._update_metrics(rewards, terminals, truncations, processed_info)
            
            # Update player tracking for next step
            self._current_player_ids = processed_info["player_id"]

            return (
                obs_tensor,
                rewards,
                terminals,
                truncations,
                processed_info
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

            # Auto-reset for Parallel API
            episode_over = any(term_dict.values()) or any(trunc_dict.values())
            if episode_over:
                new_obs_dict, reset_info_dict = self.env.reset()
                obs_dict = new_obs_dict
                if reset_info_dict:
                    info_dict.update(reset_info_dict)

            # Handle dict observations in parallel mode
            obs_list = []
            for a in self.agents:
                o = obs_dict[a]
                if isinstance(o, dict):
                    if "action_mask" in o:
                        if a not in info_dict: info_dict[a] = {}
                        info_dict[a]["legal_moves"] = np.where(o["action_mask"] == 1)[0]
                    o = o.get("observation", o)
                obs_list.append(o)
                
            reward_list = [reward_dict[a] for a in self.agents]
            term_list = [term_dict[a] for a in self.agents]
            trunc_list = [trunc_dict[a] for a in self.agents]
            
            obs_tensor = torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=self.device)
            rewards = torch.tensor(reward_list, dtype=torch.float32, device=self.device)
            terminals = torch.tensor(term_list, dtype=torch.bool, device=self.device)
            truncations = torch.tensor(trunc_list, dtype=torch.bool, device=self.device)
            processed_info = self._process_info_parallel(info_dict)

            # Update Metrics
            self._update_metrics(rewards, terminals, truncations, processed_info)
            
            # Update player tracking for next step
            self._current_player_ids = processed_info["player_id"]

            return (
                obs_tensor,
                rewards,
                terminals,
                truncations,
                processed_info
            )

    def _process_info_aec(self, info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        info = (info or {}).copy()
        agent = self.env.agent_selection
        idx = self.agents.index(agent) if agent in self.agents else 0
        info["player_id"] = torch.tensor([idx], dtype=torch.int64, device=self.device)
        
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
            "player_id": torch.tensor(list(range(len(self.agents))), dtype=torch.int64, device=self.device)
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
