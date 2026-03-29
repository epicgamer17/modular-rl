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
from agents.workers.payloads import TaskRequest, TaskType, WorkerPayload
from agents.action_selectors.selectors import BaseActionSelector
from agents.action_selectors.types import InferenceResult


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
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Updates model weights and/or hyperparameters."""
        pass

    @abstractmethod
    def execute(self, request: TaskRequest) -> WorkerPayload:
        """Single entry point for all worker tasks."""
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
        buffer: ModularReplayBuffer,
        action_selector: Optional[BaseActionSelector] = None,
        actor_device: str = "cpu",
        num_actions: Optional[int] = None,
        num_players: int = 1,
        test_agents: Optional[List[Any]] = None,
        flush_incomplete: bool = True,
        worker_id: int = 0,
        **kwargs,
    ):
        """
        Initializes the RolloutActor.

        Args:
            adapter_cls: The EnvironmentAdapter class.
            adapter_args: Arguments for the adapter class (e.g. env_factory).
            network: Neural network for value/policy estimation.
            policy_source: Strategy for retrieving InferenceResults.
            buffer: Replay Buffer reference for storing completed sequences.
            action_selector: Optional action selector.
            actor_device: Device to use for the environment and network.
            num_actions: Number of actions in the environment.
            num_players: Number of players in the game.
            test_agents: Optional list of agents for evaluation.
            flush_incomplete: If True (PPO default), force-flush active sequences at the
                end of each collect() call to produce fixed-length chunks. If False
                (MuZero), only store sequences that terminated naturally so every stored
                slot has a valid MCTS policy.
            worker_id: Unique ID for this worker.
        """
        self.worker_id = worker_id
        # 1. Build the adapter for this worker
        # Most adapters take (env_factory, device, num_actions)
        device = torch.device(actor_device)

        self.adapter = adapter_cls(
            *adapter_args, device=device, num_actions=num_actions
        )
        self.agent_network = network
        self.policy_source = policy_source
        assert (
            action_selector is not None
        ), "ActionSelector is mandatory for RolloutActor (Null Object Pattern). Provide a valid selector (e.g., CategoricalSelector, ArgmaxSelector)."
        self.action_selector = action_selector
        self.buffer = buffer
        self.flush_incomplete = flush_incomplete

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
        self.num_players = num_players
        self.current_lengths = np.zeros(self.num_envs)
        self.current_scores = np.zeros((self.num_envs, num_players))

    def setup(self) -> None:
        """Prepares the network for rollout (eval mode)."""
        self.agent_network.eval()

    def update_parameters(
        self,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Updates network weights (from weights) and/or selector hyperparameters."""
        if weights:
            # Handle potentially compiled model state dicts
            clean_params = {k.replace("_orig_mod.", ""): v for k, v in weights.items()}
            self.agent_network.load_state_dict(clean_params, strict=False)

            # Reset noise if present on the network
            if hasattr(self.agent_network, "reset_noise"):
                self.agent_network.reset_noise()

        if hyperparams:
            if self.action_selector and hasattr(
                self.action_selector, "update_parameters"
            ):
                self.action_selector.update_parameters(hyperparams)

            if hasattr(self.policy_source, "update_parameters"):
                self.policy_source.update_parameters(hyperparams)

    def get_state(self) -> Dict[str, Any]:
        """Returns rolling statistics for logging."""
        scores = np.array(self.completed_scores) if self.completed_scores else np.zeros((0, self.num_players))
        avg_score_vec = np.mean(scores, axis=0) if scores.size > 0 else np.zeros(self.num_players)
        
        # Format scores based on player count
        if self.num_players == 1:
            score_metrics = {"avg_score": float(avg_score_vec[0])}
        else:
            score_metrics = {f"avg_score_p{p}": float(avg_score_vec[p]) for p in range(self.num_players)}
            score_metrics["avg_score"] = float(np.mean(avg_score_vec))
            
        return {
            "total_steps": self.total_steps,
            "episodes_completed": self.episodes_completed,
            **score_metrics,
            "avg_length": (
                np.mean(self.completed_lengths) if self.completed_lengths else 0.0
            ),
        }

    def execute(self, request: TaskRequest) -> WorkerPayload:
        if request.task_type != TaskType.COLLECT:
            raise ValueError(
                f"RolloutActor only supports {TaskType.COLLECT}, got {request.task_type}"
            )

        metrics = self.collect(request.batch_size)
        return WorkerPayload(worker_type=self.__class__.__name__, metrics=metrics)

    @torch.inference_mode()
    def collect(self, num_steps: int) -> Dict[str, Any]:
        """
        Main execution loop. Performs num_steps environment steps across managed environments.
        Returns metrics for the collection call.
        """
        steps_this_call = 0

        steps_this_call = 0
        batch_scores = []
        batch_lengths = []
        start_time = time.time()

        while steps_this_call < num_steps:
            # 1. Unified Inference Pass
            result = self.policy_source.get_inference(
                self.obs, self.info, agent_network=self.agent_network
            )

            # 2. Resilient Action Extraction (Null Object Pattern)
            # NO BRANCHING. Responsibility live strictly in the selector.
            actions, metadata = self.action_selector.select_action(
                result, self.info, exploration=True
            )

            # 3. Step the Adapter (Hides messy environment library logic)
            next_obs, rewards, terminals, truncations, infos = self.adapter.step(
                actions
            )

            # 4. Process Transitions and Boundary Logic
            for i in range(self.num_envs):
                # Capture player info for scores before possibly using next state's info
                p_id_acting = 0
                if "player_id" in self.info:
                    p_ids_acting = self.info["player_id"]
                    p_id_acting = int(p_ids_acting[i].item() if torch.is_tensor(p_ids_acting) else p_ids_acting[i])

                self.current_scores[i, p_id_acting] += rewards[i].item()
                self.current_lengths[i] += 1

                # Determine next state's player ID
                p_id_next = 0
                if "player_id" in infos:
                    p_ids_next = infos["player_id"]
                    p_id_next = int(p_ids_next[i].item() if torch.is_tensor(p_ids_next) else p_ids_next[i])

                # Build step dictionary. Default to next_obs, but check for terminal data.
                obs_history_val = next_obs[i].cpu().numpy()
                player_id_val = p_id_next
                
                is_done = bool(terminals[i].item() or truncations[i].item())
                if is_done and "terminal_observation" in infos:
                    # Use preserved terminal observation instead of reset observation
                    obs_history_val = infos["terminal_observation"]
                    if isinstance(obs_history_val, torch.Tensor):
                        obs_history_val = obs_history_val[i].cpu().numpy()
                    elif isinstance(obs_history_val, list):
                        obs_history_val = obs_history_val[i]
                    
                    # Use preserved terminal info for player_id
                    if "terminal_info" in infos:
                        t_info = infos["terminal_info"]
                        if "player_id" in t_info:
                            p_t_ids = t_info["player_id"]
                            player_id_val = int(p_t_ids[i].item() if torch.is_tensor(p_t_ids) else p_t_ids[i])

                step_dict = {
                    "observation": obs_history_val,
                    "action": actions[i].item(),
                    "reward": rewards[i].item(),
                    "terminated": terminals[i].item(),
                    "truncated": truncations[i].item(),
                    "player_id": player_id_val,
                }

                # Optional metadata
                if result.value is not None:
                    step_dict["value"] = result.value[i].item()
                
                # Search target policies
                target_p = result.extras.get("target_policies")
                if target_p is not None:
                    step_dict["policy"] = target_p[i].cpu().numpy()
                elif result.probs is not None:
                    step_dict["policy"] = result.probs[i].cpu().numpy()

                lp = metadata.get("log_prob")
                if lp is not None:
                    step_dict["log_prob"] = lp[i].item()

                mask = self.info.get("legal_moves_mask")
                if mask is not None:
                    step_dict["legal_moves"] = (
                        torch.where(mask[i])[0].cpu().numpy().tolist()
                    )

                # Store in SequenceManager
                self.seq_manager.append(i, step_dict)

                if is_done:
                    # Flush the completed episode
                    completed_seq = self.seq_manager.flush(i)
                    self.buffer.store_aggregate(completed_seq)

                    self.episodes_completed += 1
                    ep_score = self.current_scores[i].copy()
                    ep_len = int(self.current_lengths[i])
                    
                    self.completed_scores.append(ep_score)
                    self.completed_lengths.append(ep_len)
                    batch_scores.append(ep_score)
                    batch_lengths.append(ep_len)
                    
                    self.current_scores[i].fill(0.0)
                    self.current_lengths[i] = 0

                    if len(self.completed_scores) > 100:
                        self.completed_scores.pop(0)
                        self.completed_lengths.pop(0)

                    # Seed the NEW episode with the reset observation (already in next_obs)
                    self.seq_manager.append(
                        i,
                        {
                            "observation": next_obs[i].cpu().numpy(),
                            "terminated": False,
                            "truncated": False,
                            "player_id": p_id_next, 
                        },
                    )

            # Update global state for next step
            self.obs, self.info = next_obs, infos

            # Record metrics
            batch_steps = self.num_envs
            steps_this_call += batch_steps
            self.total_steps += batch_steps

        # 6. Force-Flush at Chunk Boundary (PPO Bootstrap)
        # Only flush incomplete sequences when explicitly enabled (PPO needs fixed-length
        # chunks for GAE; MuZero must store complete episodes so every slot has a policy).
        if not self.flush_incomplete:
            return {
                **self.get_state(),
                "steps_this_call": steps_this_call,
                "batch_scores": batch_scores,
                "batch_lengths": batch_lengths,
                "duration": time.time() - start_time,
                "steps_per_second": steps_this_call / (time.time() - (start_time + 1e-6)),
            }

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
            active_seq = self.seq_manager.get_sequence(i)
            # Only force-flush the chunk if we actually took steps in it (len > 1)
            if len(active_seq.observation_history) > 1:
                seq = self.seq_manager.flush(i)

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
                        else p_id_batch[i]
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
        buffer: Optional[ModularReplayBuffer],
        action_selector: Optional[BaseActionSelector] = None,
        actor_device: str = "cpu",
        num_actions: Optional[int] = None,
        num_players: int = 1,
        test_agents: Optional[List[Any]] = None,
        worker_id: int = 0,
        **kwargs,
    ):
        """
        Initializes the EvaluatorActor.

        Args:
            adapter_cls: EnvironmentAdapter class.
            adapter_args: Arguments for the adapter class.
            network: Greedy network for performance evaluation.
            policy_source: Inference provider.
            buffer: Placeholder for argument consistency (not used).
            action_selector: Optional action selection provider for evaluation.
            actor_device: Device to use for the environment and network.
            num_actions: Number of actions in the environment.
            num_players: Number of players in the environment.
            test_agents: Optional list of agents for evaluation.
            worker_id: Worker identifier.
        """
        self.worker_id = worker_id
        device = torch.device(actor_device)
        self.adapter = adapter_cls(
            *adapter_args, device=device, num_actions=num_actions
        )

        self.agent_network = network
        self.policy_source = policy_source
        assert (
            action_selector is not None
        ), "ActionSelector is mandatory for EvaluatorActor (Null Object Pattern). Provide a valid selector (e.g., ArgmaxSelector)."
        self.action_selector = action_selector
        self.num_players = num_players
        self.test_agents = test_agents
        self.num_envs = self.adapter.num_envs

        self.total_reward = 0.0
        self.episodes_completed = 0

    def setup(self) -> None:
        self.agent_network.eval()

    def update_parameters(
        self,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        if weights:
            clean_params = {k.replace("_orig_mod.", ""): v for k, v in weights.items()}
            self.agent_network.load_state_dict(clean_params, strict=False)

    def get_state(self) -> Dict[str, Any]:
        return {
            "episodes_completed": self.episodes_completed,
            "total_reward": self.total_reward,
            "average_reward": self.total_reward / max(1, self.episodes_completed),
        }

    def execute(self, request: TaskRequest) -> WorkerPayload:
        if request.task_type != TaskType.EVALUATE:
            raise ValueError(
                f"EvaluatorActor only supports {TaskType.EVALUATE}, got {request.task_type}"
            )

        metrics = self.evaluate(request.batch_size, **request.kwargs)
        return WorkerPayload(worker_type=self.__class__.__name__, metrics=metrics)

    @torch.inference_mode()
    def evaluate(self, num_episodes: int, **kwargs) -> Dict[str, Any]:
        """
        Performs Matrix Evaluation: Students plays round-robin tournament
        as each player position against each opponent in the test_agents roster.
        """
        results = {}
        all_avg_scores = []
        total_lengths = []
        
        # Determine if we have opponents. If not, just evaluate student alone.
        opponents = self.test_agents if self.test_agents else [None]
        
        for opponent in opponents:
            opp_name = opponent.name if opponent else "self"
            opp_scores = []
            player_scores = {p: [] for p in range(self.num_players)}
            
            # Rounds: Student takes each position
            for student_pos in range(self.num_players):
                for _ in range(num_episodes):
                    obs, info = self.adapter.reset()
                    episode_length = 0
                    student_total_reward = 0.0
                    done = False
                    trunc = False
                    
                    while not (done or trunc):
                        # 1. Identify current player from adapter info
                        current_player_idx = info.get("player_id")
                        if current_player_idx is not None:
                            # Standardized player_id is a tensor [1] or scalar
                            if torch.is_tensor(current_player_idx):
                                current_player_idx = int(current_player_idx.item())
                            else:
                                current_player_idx = int(current_player_idx)
                        else:
                            current_player_idx = 0
                            
                        # 2. Assign policy source (Student or Expert)
                        if current_player_idx == student_pos:
                            # Student uses search toggle
                            use_search = kwargs.get("use_search", True)
                            from agents.action_selectors.policy_sources import SearchPolicySource
                            if use_search and isinstance(self.policy_source, SearchPolicySource):
                                result = self.policy_source.get_inference(
                                    obs, info, agent_network=self.agent_network, exploration=False
                                )
                            else:
                                output = self.agent_network.obs_inference(obs)
                                result = InferenceResult.from_inference_output(output)
                        else:
                            # Expert/Opponent
                            if opponent is not None:
                                output = opponent.obs_inference(obs, info=info, adapter=self.adapter)
                                result = InferenceResult.from_inference_output(output)
                            else:
                                # Fallback if playing against self/None
                                output = self.agent_network.obs_inference(obs)
                                result = InferenceResult.from_inference_output(output)

                        # 3. Select action
                        from agents.action_selectors.types import InferenceResult
                        if not isinstance(result, InferenceResult):
                             result = InferenceResult.from_inference_output(result)

                        actions, _ = self.action_selector.select_action(
                            result, info, exploration=False
                        )
                        
                        # 4. Step environment
                        obs, rewards, dones, truncs, info = self.adapter.step(actions)
                        
                        # Accumulate reward for the student position from ALL sources
                        # PettingZooAdapter.step now provides 'all_rewards' in info
                        all_rewards = info.get("all_rewards")
                        if all_rewards is not None:
                            if isinstance(all_rewards, dict):
                                agent_id = self.adapter.agents[student_pos]
                                student_total_reward += all_rewards.get(agent_id, 0.0)
                            elif torch.is_tensor(all_rewards):
                                # Parallel batched info
                                student_total_reward += all_rewards[student_pos].item()
                        else:
                            # Fallback for Gym: rewards is a scalar for the only player
                            student_total_reward += rewards.item()

                        # In AEC, episode ends when done=True for any agent in some envs, 
                        # or specifically for the current agent.
                        done = bool(dones.any().item())
                        trunc = bool(truncs.any().item())
                        episode_length += 1

                    opp_scores.append(student_total_reward)
                    player_scores[student_pos].append(student_total_reward)
                    total_lengths.append(episode_length)

            # Aggregate for this opponent
            results[f"vs_{opp_name}_score"] = {
                "avg": float(np.mean(opp_scores)),
                **{f"p{p}": float(np.mean(player_scores[p])) for p in range(self.num_players)}
            }
            all_avg_scores.append(np.mean(opp_scores))

        results["score"] = float(np.mean(all_avg_scores)) if all_avg_scores else 0.0
        results["avg_length"] = float(np.mean(total_lengths)) if total_lengths else 0.0
        return results
