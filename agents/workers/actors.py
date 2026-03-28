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
            self.seq_manager.append(
                i,
                {
                    "observation": self.obs[i].cpu().numpy(),
                    "terminated": False,
                    "truncated": False,
                    "player_id": self._get_player_id(self.info, i),
                    "legal_moves": self._get_legal_moves(self.info, i),
                },
            )

        self.total_steps = 0
        self.episodes_completed = 0
        self.completed_scores = []
        self.completed_lengths = []

        # Track current episode progress across envs
        self.num_players = num_players
        self.current_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self.current_scores = np.zeros((self.num_envs, num_players))

    def _get_batched_value(self, value: Any, index: int) -> Any:
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.dim() == 0:
                return value
            return value[index]
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.item()
            return value[index]
        if isinstance(value, (list, tuple)):
            return value[index]
        return value

    def _get_info_value(
        self, info: Optional[Dict[str, Any]], key: str, index: int, default: Any = None
    ) -> Any:
        if not isinstance(info, dict):
            return default
        return self._get_batched_value(info.get(key), index) if key in info else default

    def _get_player_id(self, info: Optional[Dict[str, Any]], index: int) -> int:
        player = self._get_info_value(info, "player_id", index, default=None)
        if player is None:
            player = self._get_info_value(info, "player", index, default=0)

        if torch.is_tensor(player):
            player = player.item()
        elif isinstance(player, np.generic):
            player = player.item()

        return int(player)

    def _legal_moves_from_value(self, value: Any) -> Optional[List[int]]:
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.dtype == torch.bool:
                return torch.where(value)[0].cpu().tolist()
            return value.view(-1).cpu().tolist()
        if isinstance(value, np.ndarray):
            if value.dtype == np.bool_:
                return np.flatnonzero(value).tolist()
            return value.reshape(-1).tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [int(value)]

    def _get_legal_moves(
        self, info: Optional[Dict[str, Any]], index: int
    ) -> Optional[List[int]]:
        if not isinstance(info, dict):
            return None

        mask = self._get_info_value(info, "legal_moves_mask", index, default=None)
        if mask is not None:
            return self._legal_moves_from_value(mask)

        legal_moves = self._get_info_value(info, "legal_moves", index, default=None)
        return self._legal_moves_from_value(legal_moves)

    def _merge_selection_metadata(
        self, result: InferenceResult, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        merged = dict(result.extras or {})
        for key, value in metadata.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value

        if merged.get("policy") is None and result.probs is not None:
            merged["policy"] = result.probs.detach()
        if merged.get("value") is None and result.value is not None:
            merged["value"] = result.value.detach()
        return merged

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
                result, self.info, exploration=True, episode_step=self.current_lengths
            )
            merged_metadata = self._merge_selection_metadata(result, metadata)

            # 3. Step the Adapter (Hides messy environment library logic)
            next_obs, rewards, terminals, truncations, infos = self.adapter.step(
                actions
            )

            # 4. Process Transitions and Boundary Logic
            for i in range(self.num_envs):
                # Build the step dictionary manually for each environment
                p_id = self._get_player_id(self.info, i)

                # Update rolling accumulators for CURRENT PLAYER
                self.current_scores[i, p_id] += rewards[i].item()
                self.current_lengths[i] += 1

                # Use terminal observation if adapter preserved it, else fall back to next_obs
                is_done = terminals[i].item() or truncations[i].item()
                if is_done and "terminal_observation" in infos:
                    term_obs = infos["terminal_observation"]
                    if torch.is_tensor(term_obs):
                        obs_np = term_obs[i].cpu().numpy() if term_obs.dim() > 1 else term_obs.cpu().numpy()
                    else:
                        obs_np = term_obs[i] if hasattr(term_obs, '__getitem__') else term_obs
                else:
                    obs_np = next_obs[i].cpu().numpy()

                step_dict = {
                    "observation": obs_np,
                    "action": actions[i].item(),
                    "reward": rewards[i].item(),
                    "terminated": terminals[i].item(),
                    "truncated": truncations[i].item(),
                    "player_id": p_id,
                }

                # Optional metadata
                value = self._get_batched_value(merged_metadata.get("value"), i)
                if value is not None:
                    step_dict["value"] = (
                        value.item() if torch.is_tensor(value) else float(value)
                    )

                policy = merged_metadata.get(
                    "target_policies", merged_metadata.get("policy")
                )
                policy = self._get_batched_value(policy, i)
                if policy is not None:
                    if torch.is_tensor(policy):
                        step_dict["policy"] = policy.detach().cpu().numpy()
                    elif isinstance(policy, np.ndarray):
                        step_dict["policy"] = policy
                    else:
                        step_dict["policy"] = np.asarray(policy)

                lp = self._get_batched_value(merged_metadata.get("log_prob"), i)
                if lp is not None:
                    step_dict["log_prob"] = (
                        lp.item() if torch.is_tensor(lp) else float(lp)
                    )

                # IMPORTANT: Use MASK[t+1] from INFOS for the NEW OBSERVATION.
                # The reset mask [0] was added to the seed above.
                next_legal_moves = None
                if not (
                    is_done and isinstance(infos, dict) and "terminal_observation" in infos
                ):
                    next_legal_moves = self._get_legal_moves(infos, i)
                if next_legal_moves is not None:
                    step_dict["legal_moves"] = next_legal_moves

                # Store in SequenceManager
                self.seq_manager.append(i, step_dict)

                if terminals[i].item() or truncations[i].item():
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

                    # Seed the START of the new episode (hidden in next_obs for auto-resetting envs)
                    self.seq_manager.append(
                        i,
                        {
                            "observation": next_obs[i].cpu().numpy(),
                            "terminated": False,
                            "truncated": False,
                            "player_id": self._get_player_id(infos, i),
                            "legal_moves": self._get_legal_moves(infos, i),
                        },
                    )

            self.obs = next_obs
            self.info = infos

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
                self.seq_manager.append(
                    i,
                    {
                        "observation": self.obs[i].cpu().numpy(),
                        "terminated": False,
                        "truncated": False,
                        "player_id": self._get_player_id(self.info, i),
                        "legal_moves": self._get_legal_moves(self.info, i),
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


# EvaluatorActor moved to agents/workers/evaluator.py
