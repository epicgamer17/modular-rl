import torch
import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

from modules.models.agent_network import AgentNetwork
from agents.action_selectors.policy_sources import BasePolicySource
from agents.environments.adapters import BaseAdapter
from agents.workers.state_management import SequenceManager
from replay_buffers.modular_buffer import ModularReplayBuffer
from agents.action_selectors.selectors import BaseActionSelector


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
    def update_parameters(
        self,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> None:
        """Updates model weights and/or hyperparameters."""
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
        adapter_cls: Type[BaseAdapter],
        adapter_args: Tuple[Any, ...],
        network: AgentNetwork,
        policy_source: BasePolicySource,
        buffer: ModularReplayBuffer,        # Index 4
        config: Any,
        action_selector: Optional[BaseActionSelector] = None, # Index 6
        test_agents: Optional[List[Any]] = None,
        worker_id: int = 0,
    ):
        """
        Initializes the RolloutActor.

        Args:
            adapter_cls: The EnvironmentAdapter class.
            adapter_args: Arguments for the adapter class (e.g. env_factory).
            network: Neural network for value/policy estimation.
            policy_source: Strategy for retrieving InferenceResults.
            buffer: Replay Buffer reference for storing completed sequences.
            config: Algorithm configuration.
            action_selector: Optional action selector.
            worker_id: Unique ID for this worker.
        """
        self.worker_id = worker_id
        # 1. Build the adapter for this worker
        # Most adapters take (env_factory, device, num_actions)
        device = torch.device("cpu")  # Rollout always on CPU
        num_actions = getattr(config.game, "num_actions", None)

        self.adapter = adapter_cls(
            *adapter_args, device=device, num_actions=num_actions
        )
        self.agent_network = network
        self.policy_source = policy_source
        self.action_selector = action_selector
        self.buffer = buffer

        num_players = getattr(config.game, "num_players", 1)
        self.num_envs = self.adapter.num_envs
        self.seq_manager = SequenceManager(num_players, self.num_envs)

        # Internal state for the collection loop
        self.obs, self.info = self.adapter.reset()

        # Seed the initial state for each environment sequence
        # Sequence contract: len(obs) == len(action) + 1
        for i in range(self.num_envs):
            player_id = 0
            if "player_id" in self.info:
                p_id_batch = self.info["player_id"]
                player_id = (
                    p_id_batch[i].item()
                    if torch.is_tensor(p_id_batch)
                    else p_id_batch[i]
                )

            self.seq_manager.append(
                i,
                {
                    "observation": self.obs[i].cpu().numpy(),
                    "terminated": False,
                    "truncated": False,
                    "player_id": player_id,
                },
            )

        self.total_steps = 0
        self.episodes_completed = 0
        self.completed_scores = []
        self.completed_lengths = []

        # Track current episode progress across envs
        self.current_scores = np.zeros(self.num_envs)
        self.current_lengths = np.zeros(self.num_envs)

    def setup(self) -> None:
        """Prepares the network for rollout (eval mode)."""
        self.agent_network.eval()

    def update_parameters(
        self,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> None:
        """Updates network weights (from weights) and/or selector hyperparameters."""
        if weights:
            # Handle potentially compiled model state dicts
            clean_params = {
                k.replace("_orig_mod.", ""): v for k, v in weights.items()
            }
            self.agent_network.load_state_dict(clean_params, strict=False)
            
            # Reset noise if present on the network
            if hasattr(self.agent_network, "reset_noise"):
                self.agent_network.reset_noise()

        if hyperparams:
            if self.action_selector and hasattr(self.action_selector, "update_parameters"):
                self.action_selector.update_parameters(hyperparams)
            
            if hasattr(self.policy_source, "update_parameters"):
                self.policy_source.update_parameters(hyperparams)

    def get_state(self) -> Dict[str, Any]:
        """Returns rolling statistics for logging."""
        return {
            "total_steps": self.total_steps,
            "episodes_completed": self.episodes_completed,
            "avg_score": (
                np.mean(self.completed_scores) if self.completed_scores else 0.0
            ),
            "avg_length": (
                np.mean(self.completed_lengths) if self.completed_lengths else 0.0
            ),
        }

    @torch.inference_mode()
    def collect(self, num_steps: int) -> Dict[str, Any]:
        """
        Main execution loop. Performs num_steps environment steps across managed environments.
        Returns metrics for the collection call.
        """
        steps_this_call = 0
        batch_scores = []
        batch_lengths = []
        start_time = time.time()

        while steps_this_call < num_steps:
            # 1. Unified Inference Pass
            result = self.policy_source.get_inference(
                self.obs, self.info, agent_network=self.agent_network
            )

            # 2. Resilient Action Extraction
            if self.action_selector is not None:
                # Use standard action selector (handles masking, temperature, etc)
                actions, metadata = self.action_selector.select_action(
                    result, self.info, exploration=True
                )
            elif result.action is not None:
                actions = result.action
                metadata = result.extras
            elif result.probs is not None:
                # Sample from policy if specific action not provided
                actions = torch.multinomial(result.probs, num_samples=1).squeeze(-1)
                metadata = result.extras
            elif "best_actions" in result.extras:
                actions = result.extras["best_actions"]
                metadata = result.extras
            else:
                # Greedy fallback
                actions = result.probs.argmax(dim=-1)
                metadata = result.extras

            # 3. Step the Adapter (Hides messy environment library logic)
            next_obs, rewards, terminals, truncations, infos = self.adapter.step(
                actions
            )

            # 4. Route Transitions to SequenceManager
            self.current_scores += rewards.cpu().numpy()
            self.current_lengths += 1

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
                    val = (
                        result.value[i].item()
                        if result.value.dim() > 0
                        else result.value.item()
                    )
                    transition["value"] = val
                if result.probs is not None:
                    # [num_actions]
                    transition["policy"] = result.probs[i].cpu().numpy()

                # 4.5 Capture Log Probs (Essential for PPO)
                # Check metadata from ActionSelector first
                if "log_prob" in metadata:
                    lp = metadata["log_prob"]
                    transition["log_prob"] = lp[i].item() if torch.is_tensor(lp) else lp
                elif result.policy is not None and hasattr(result.policy, "log_prob"):
                    # Compute manually from distribution if selector didn't provide it
                    lp = result.policy.log_prob(actions).cpu().numpy()
                    transition["log_prob"] = lp[i]

                # Enrich with environment metadata
                if "player_id" in self.info:
                    p_id = self.info["player_id"]
                    transition["player_id"] = (
                        p_id[i].item() if torch.is_tensor(p_id) else p_id[i]
                    )

                if "legal_moves_mask" in self.info:
                    mask = self.info["legal_moves_mask"][i]
                    transition["legal_moves"] = (
                        torch.where(mask)[0].cpu().numpy().tolist()
                    )

                self.seq_manager.append(i, transition)

                # 5. Boundary Logic: Flush and Reseed
                if terminals[i] or truncations[i]:
                    completed_seq = self.seq_manager.flush(i)
                    self.buffer.store_aggregate(completed_seq)

                    self.episodes_completed += 1
                    ep_score = float(self.current_scores[i])
                    ep_len = int(self.current_lengths[i])

                    self.completed_scores.append(ep_score)
                    self.completed_lengths.append(ep_len)
                    batch_scores.append(ep_score)
                    batch_lengths.append(ep_len)

                    self.current_scores[i] = 0.0
                    self.current_lengths[i] = 0

                    if len(self.completed_scores) > 100:
                        self.completed_scores.pop(0)
                        self.completed_lengths.pop(0)

                    # Seed the START of the new episode (hidden in next_obs for auto-resetting envs)
                    new_p_id = 0
                    if "player_id" in infos:
                        next_p_ids = infos["player_id"]
                        new_p_id = (
                            next_p_ids[i].item()
                            if torch.is_tensor(next_p_ids)
                            else next_p_ids[i]
                        )

                    self.seq_manager.append(
                        i,
                        {
                            "observation": next_obs[i].cpu().numpy(),
                            "terminated": False,
                            "truncated": False,
                            "player_id": new_p_id,
                        },
                    )

            self.obs = next_obs
            self.info = infos

            # Record metrics
            batch_steps = self.num_envs
            steps_this_call += batch_steps
            self.total_steps += batch_steps

        # 6. Force-Flush at Chunk Boundary (PPO Bootstrap)
        # Calculate bootstrap values for all active trajectories
        with torch.inference_mode():
            final_result = self.policy_source.get_inference(
                self.obs,
                self.info,
                agent_network=self.agent_network,
                exploration=False,
            )
            final_values = final_result.value

        for i in range(self.num_envs):
            seq = self.seq_manager.flush(i)
            if seq and len(seq.observation_history) > 1:
                # Use scalar value extraction (handles search or direct V)
                v = (
                    final_values[i].item()
                    if final_values.dim() > 0
                    else final_values.item()
                )
                self.buffer.store_aggregate(seq, bootstrap_value=v)

            # 7. Seed the NEXT chunk starting with this observation
            # (Ensures the first transition of the next collect() has s_0)
            p_id = 0
            if "player_id" in self.info:
                p_id_batch = self.info["player_id"]
                p_id = (
                    p_id_batch[i].item()
                    if torch.is_tensor(p_id_batch)
                    else (
                        p_id_batch[i]
                        if isinstance(p_id_batch, (list, np.ndarray))
                        else p_id_batch
                    )
                )

            self.seq_manager.append(
                i,
                {
                    "observation": self.obs[i].cpu().numpy(),
                    "terminated": False,
                    "truncated": False,
                    "player_id": p_id,
                },
            )

        return {
            **self.get_state(),
            "steps_this_call": steps_this_call,
            "batch_scores": batch_scores,
            "batch_lengths": batch_lengths,
            "duration": time.time() - start_time,
            "steps_per_second": steps_this_call / (time.time() - (start_time + 1e-6)),
        }


class EvaluatorActor(BaseActor):
    """
    Dedicated worker for evaluation and testing.
    Runs greedy episodes without a Replay Buffer or Sequence Manager.
    Accumulates internal metrics for averaged reporting.
    """

    def __init__(
        self,
        adapter_cls: Type[BaseAdapter],
        adapter_args: Tuple[Any, ...],
        network: AgentNetwork,
        policy_source: BasePolicySource,
        buffer: Optional[ModularReplayBuffer],  # Index 4 (ignored by EvaluatorActor)
        config: Any,  # Index 5
        action_selector: Optional[BaseActionSelector] = None,  # Index 6
        test_agents: Optional[List[Any]] = None,  # Index 7
        worker_id: int = 0,  # Index 8
    ):
        """
        Initializes the EvaluatorActor.

        Args:
            adapter_cls: EnvironmentAdapter class.
            adapter_args: Arguments for the adapter class.
            network: Greedy network for performance evaluation.
            policy_source: Inference provider.
            buffer: Placeholder for argument consistency (not used).
            config: Algorithm configuration.
            worker_id: Worker identifier.
            action_selector: Optional action selection provider for evaluation.
        """
        self.worker_id = worker_id
        device = torch.device("cpu")
        num_actions = getattr(config.game, "num_actions", None)
        self.adapter = adapter_cls(
            *adapter_args, device=device, num_actions=num_actions
        )

        self.agent_network = network
        self.policy_source = policy_source
        self.action_selector = action_selector
        self.test_agents = test_agents
        self.num_envs = self.adapter.num_envs

        self.total_reward = 0.0
        self.episodes_completed = 0

    def setup(self) -> None:
        self.agent_network.eval()

    def update_parameters(
        self,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> None:
        if weights:
            clean_params = {
                k.replace("_orig_mod.", ""): v for k, v in weights.items()
            }
            self.agent_network.load_state_dict(clean_params, strict=False)

    def get_state(self) -> Dict[str, Any]:
        return {
            "episodes_completed": self.episodes_completed,
            "total_reward": self.total_reward,
            "average_reward": self.total_reward / max(1, self.episodes_completed),
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
            result = self.policy_source.get_inference(
                obs, info, agent_network=self.agent_network, exploration=False
            )

            # 1. Routing Trick: Check for hardcoded test agents
            player_ids = info.get("player_id", [])
            # For simplicity in evaluation, assume a single player per environment at each step
            # or use the first if batched (Evaluator usually uses B=1)
            current_player = player_ids[0] if (player_ids and len(player_ids) > 0) else None
            
            # Resolve agent if available
            agent = None
            if self.test_agents and current_player is not None:
                if isinstance(self.test_agents, dict):
                    # Handle dict-based routing
                    agent = self.test_agents.get(current_player.item() if hasattr(current_player, "item") else current_player)
                elif isinstance(self.test_agents, list) and current_player < len(self.test_agents):
                    # Handle list-based routing
                    agent = self.test_agents[current_player]

            if agent is not None:
                # Test agent contract: act(obs, info)
                if hasattr(agent, "act"):
                    actions = agent.act(obs, info)
                elif hasattr(agent, "select_action"):
                    # Handle if it was another action selector passed as an agent
                    actions, _ = agent.select_action(result, info, exploration=False)
                else:
                    # Fallback to Student if agent is None or invalid
                    actions, _ = self.action_selector.select_action(
                        result, info, exploration=False
                    )
            elif self.action_selector is not None:
                # 2. Standard Student Inference
                actions, _ = self.action_selector.select_action(
                    result, info, exploration=False
                )
            else:
                # Fallback to result check (deprecated but safe for simple cases)
                if result.action is not None:
                    actions = result.action
                elif result.probs is not None:
                    actions = result.probs.argmax(dim=-1)
                elif result.logits is not None:
                    actions = result.logits.argmax(dim=-1)
                else:
                    raise AttributeError(
                        "EvaluatorActor failed to determine action: "
                        "no action_selector provided and results lack actions/probs/logits."
                    )

            next_obs, rewards, terminals, truncations, infos = self.adapter.step(
                actions
            )
            current_rewards += rewards

            for i in range(self.num_envs):
                if terminals[i] or truncations[i]:
                    reward_val = current_rewards[i].item()
                    finished_rewards.append(reward_val)
                    self.total_reward += reward_val
                    self.episodes_completed += 1
                    current_rewards[i] = 0.0

            obs, info = next_obs, infos

        avg_score = (
            sum(finished_rewards) / len(finished_rewards) if finished_rewards else 0.0
        )
        return {
            "score": avg_score,
            "num_episodes": len(finished_rewards),
            "total_reward_accumulated": self.total_reward,
        }
