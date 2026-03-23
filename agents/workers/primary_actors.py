import torch
import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from modules.models.agent_network import AgentNetwork
from agents.action_selectors.policy_sources import BasePolicySource
from agents.environments.adapters import BaseAdapter
from agents.workers.state_management import SequenceManager
from replay_buffers.modular_buffer import ModularReplayBuffer

class BaseActor(ABC):
    """
    Abstract base class for all primary actors.
    Provides a standardized API for Orchestrators and Executors.
    """
    @abstractmethod
    def setup(self) -> None:
        """Initializes the actor and its internal state."""
        pass

    @abstractmethod
    def update_parameters(self, state_dict: Dict[str, Any]) -> None:
        """Updates model weights or hyperparameters."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Returns the current state/metrics of the actor."""
        pass

class RolloutActor(BaseActor):
    """
    The main worker for data collection.
    Iterates between inference and environment stepping, storing results locally
    via a SequenceManager before flushing them to a centralized ReplayBuffer.
    """
    def __init__(
        self,
        adapter: BaseAdapter,
        network: AgentNetwork,
        policy_source: BasePolicySource,
        buffer: ModularReplayBuffer,
        num_players: int = 1
    ):
        """
        Initializes the RolloutActor.

        Args:
            adapter: The EnvironmentAdapter instance.
            network: Neural network for value/policy estimation.
            policy_source: Strategy for retrieving InferenceResults.
            buffer: Replay Buffer reference for storing completed sequences.
            num_players: Number of players in the environment.
        """
        self.adapter = adapter
        self.network = network
        self.policy_source = policy_source
        self.buffer = buffer
        
        self.num_envs = adapter.num_envs
        self.seq_manager = SequenceManager(num_players, self.num_envs)
        
        # Internal state for the collection loop
        self.obs, self.info = self.adapter.reset()
        
        # Seed the initial state for each environment sequence
        # Sequence contract: len(obs) == len(action) + 1
        for i in range(self.num_envs):
            player_id = 0
            if "player_id" in self.info:
                p_id_batch = self.info["player_id"]
                player_id = p_id_batch[i].item() if torch.is_tensor(p_id_batch) else p_id_batch[i]

            self.seq_manager.append(i, {
                "observation": self.obs[i].cpu().numpy(),
                "terminated": False,
                "truncated": False,
                "player_id": player_id
            })
            
        self.total_steps = 0
        self.episodes_completed = 0
        self.total_reward = 0.0

    def setup(self) -> None:
        """Prepares the network for rollout (eval mode)."""
        self.network.eval()

    def update_parameters(self, state_dict: Dict[str, Any]) -> None:
        """Updates network weights (from state_dict) and/or selector hyperparameters."""
        if any(isinstance(v, (torch.Tensor, dict)) for v in state_dict.values()):
            # Handle potentially compiled model state dicts
            clean_params = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            self.network.load_state_dict(clean_params, strict=False)
        else:
            # hyperparameters update (e.g. epsilon, temperature)
            # This relies on policy_source or search engines having an update_parameters method if needed.
            if hasattr(self.policy_source, "update_parameters"):
                self.policy_source.update_parameters(state_dict)

    def get_state(self) -> Dict[str, Any]:
        """Returns rolling statistics for logging."""
        return {
            "total_steps": self.total_steps,
            "episodes_completed": self.episodes_completed,
            "total_reward": self.total_reward,
            "average_reward": self.total_reward / max(1, self.episodes_completed)
        }

    @torch.inference_mode()
    def collect(self, num_steps: int) -> Dict[str, Any]:
        """
        Main execution loop. Performs num_steps environment steps across managed environments.
        Returns metrics for the collection call.
        """
        steps_this_call = 0
        start_time = time.time()
        
        while steps_this_call < num_steps:
            # 1. Unified Inference Pass
            result = self.policy_source.get_inference(self.obs, self.info, agent_network=self.network)
            
            # 2. Resilient Action Extraction
            if result.action is not None:
                actions = result.action
            elif result.probs is not None:
                # Sample from policy if specific action not provided
                actions = torch.multinomial(result.probs, num_samples=1).squeeze(-1)
            elif "best_actions" in result.extras:
                actions = result.extras["best_actions"]
            else:
                # Greedy fallback
                actions = result.probs.argmax(dim=-1)
                
            # 3. Step the Adapter (Hides messy environment library logic)
            next_obs, rewards, terminals, truncations, infos = self.adapter.step(actions)
            
            # 4. Route Transitions to SequenceManager
            for i in range(self.num_envs):
                transition = {
                    "observation": next_obs[i].cpu().numpy(),
                    "action": actions[i].item(),
                    "reward": rewards[i].item(),
                    "terminated": terminals[i].item(),
                    "truncated": truncations[i].item(),
                }
                
                # Enrich with inference math (targets for learner)
                if result.value is not None:
                    # Search value is usually (B,)
                    val = result.value[i].item() if result.value.dim() > 0 else result.value.item()
                    transition["value"] = val
                if result.probs is not None:
                    transition["policy"] = result.probs[i].cpu().numpy()
                
                # Enrich with environment metadata
                if "player_id" in self.info:
                    p_id = self.info["player_id"]
                    transition["player_id"] = p_id[i].item() if torch.is_tensor(p_id) else p_id[i]
                
                if "legal_moves_mask" in self.info:
                    mask = self.info["legal_moves_mask"][i]
                    transition["legal_moves"] = torch.where(mask)[0].cpu().numpy().tolist()

                self.seq_manager.append(i, transition)
                
                # 5. Boundary Logic: Flush and Reseed
                if terminals[i] or truncations[i]:
                    completed_seq = self.seq_manager.flush(i)
                    self.buffer.store_aggregate(completed_seq)
                    
                    self.episodes_completed += 1
                    
                    # Seed the START of the new episode (hidden in next_obs for auto-resetting envs)
                    new_p_id = 0
                    if "player_id" in infos:
                        next_p_ids = infos["player_id"]
                        new_p_id = next_p_ids[i].item() if torch.is_tensor(next_p_ids) else next_p_ids[i]

                    self.seq_manager.append(i, {
                        "observation": next_obs[i].cpu().numpy(),
                        "terminated": False,
                        "truncated": False,
                        "player_id": new_p_id
                    })
                    
            self.obs = next_obs
            self.info = infos
            
            # Record metrics
            batch_steps = self.num_envs
            steps_this_call += batch_steps
            self.total_steps += batch_steps
            self.total_reward += rewards.sum().item()
            
        return {
            **self.get_state(),
            "steps_this_call": steps_this_call,
            "duration": time.time() - start_time,
            "steps_per_second": steps_this_call / (time.time() - start_time)
        }

class EvaluatorActor(BaseActor):
    """
    Dedicated worker for evaluation and testing.
    Runs greedy episodes without a Replay Buffer or Sequence Manager.
    Accumulates internal metrics for averaged reporting.
    """
    def __init__(self, adapter: BaseAdapter, network: AgentNetwork, policy_source: BasePolicySource):
        """
        Initializes the EvaluatorActor.

        Args:
            adapter: EnvironmentAdapter instance.
            network: Greedy network for performance evaluation.
            policy_source: Inference provider (will be forced to exploration=False).
        """
        self.adapter = adapter
        self.network = network
        self.policy_source = policy_source
        self.num_envs = adapter.num_envs
        
        self.total_reward = 0.0
        self.episodes_completed = 0

    def setup(self) -> None:
        self.network.eval()

    def update_parameters(self, state_dict: Dict[str, Any]) -> None:
        if any(isinstance(v, (torch.Tensor, dict)) for v in state_dict.values()):
            clean_params = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            self.network.load_state_dict(clean_params, strict=False)

    def get_state(self) -> Dict[str, Any]:
        return {
            "episodes_completed": self.episodes_completed,
            "total_reward": self.total_reward,
            "average_reward": self.total_reward / max(1, self.episodes_completed)
        }

    @torch.inference_mode()
    def evaluate(self, num_episodes: int) -> Dict[str, Any]:
        """
        Runs evaluation episodes until num_episodes are completed.
        Returns final averaged metrics.
        """
        obs, info = self.adapter.reset()
        current_rewards = torch.zeros(self.num_envs)
        finished_rewards = []
        
        while len(finished_rewards) < num_episodes:
            # Force greedy behavior via exploration=False
            result = self.policy_source.get_inference(obs, info, agent_network=self.network, exploration=False)
            
            # Determine greedy action
            if result.action is not None:
                actions = result.action
            else:
                actions = result.probs.argmax(dim=-1)
                
            next_obs, rewards, terminals, truncations, infos = self.adapter.step(actions)
            current_rewards += rewards
            
            for i in range(self.num_envs):
                if terminals[i] or truncations[i]:
                    reward_val = current_rewards[i].item()
                    finished_rewards.append(reward_val)
                    self.total_reward += reward_val
                    self.episodes_completed += 1
                    current_rewards[i] = 0.0
                    
            obs, info = next_obs, infos
            
        avg_score = sum(finished_rewards) / len(finished_rewards) if finished_rewards else 0.0
        return {
            "score": avg_score,
            "num_episodes": len(finished_rewards),
            "total_reward_accumulated": self.total_reward
        }
