"""
PufferActor Module

This module defines the PufferActor subclasses, which are 'Fat Workers' that run PufferLib
and pivot the data from horizontal batches back into vertical, chronological Sequence objects.
"""

import numpy as np
import torch
import time
from typing import Callable, Any, Optional, Tuple, Dict, List
from abc import abstractmethod

import pufferlib.vector
from data.samplers.sequence import Sequence
from agents.workers.actors import BaseActor
from utils.wrappers import AECSequentialWrapper
from data.storage.circular import ModularReplayBuffer
from modules.agent_nets.modular import ModularAgentNetwork
from agents.action_selectors.selectors import BaseActionSelector
from agents.action_selectors.types import InferenceResult
from agents.action_selectors.policy_sources import (
    BasePolicySource,
    NetworkPolicySource,
    SearchPolicySource,
)


def _make_puffer_env(env_factory, buf=None, seed=None):
    raw_env = env_factory()

    is_aec = hasattr(raw_env, "agent_selection")
    if not is_aec and hasattr(raw_env, "unwrapped"):
        is_aec = hasattr(raw_env.unwrapped, "agent_selection")

    if is_aec:
        # Wrap the AEC env into a Sequential Gym env safely
        wrapped_env = AECSequentialWrapper(raw_env)
        # Because we flattened it, PufferLib can treat it like a standard Gym Env!
        return pufferlib.emulation.GymnasiumPufferEnv(
            env=wrapped_env, buf=buf, seed=seed
        )
    else:
        return pufferlib.emulation.GymnasiumPufferEnv(env=raw_env, buf=buf, seed=seed)


class BasePufferActor(BaseActor):
    """
    Abstract base class for Puffer-based 'Fat Workers'.
    Handles vectorized execution and sequence pivoting.
    """

    def __init__(
        self,
        env_factory: Callable[[], Any],
        agent_network: ModularAgentNetwork,
        action_selector: BaseActionSelector,
        replay_buffer: ModularReplayBuffer,
        num_players: int,
        device: torch.device,
        name: str = "puffer_actor",
        *,
        num_envs: int = 1,
        num_puffer_workers: int = 2,
        worker_id: int = 0,
        input_shape: Tuple[int, ...] = (0,),
        num_actions: int = 0,
        policy_source: Optional[BasePolicySource] = None,
        # Explicit search/compilation/video args
        search_engine: Optional[Any] = None,
        record_video: bool = False,
        record_video_interval: int = 1000,
        compilation_enabled: bool = False,
        compilation_mode: str = "default",
        compilation_fullgraph: bool = False,
    ):
        super().__init__(
            env_factory=env_factory,
            agent_network=agent_network,
            action_selector=action_selector,
            replay_buffer=replay_buffer,
            num_players=num_players,
            device=device,
            name=name,
            input_shape=input_shape,
            num_actions=num_actions,
            worker_id=worker_id,
            policy_source=policy_source,
            search_engine=search_engine,
            record_video=record_video,
            record_video_interval=record_video_interval,
            compilation_enabled=compilation_enabled,
            compilation_mode=compilation_mode,
            compilation_fullgraph=compilation_fullgraph,
        )

        print("PufferActor initialized", worker_id)

        self.num_envs = num_envs
        self.num_puffer_workers = num_puffer_workers
        self.vec_env = None
        self.active_sequences = None
        self._initialized = False

        self._obs = None
        self._infos = None

    def reset(self) -> Tuple[Any, Any]:
        """Override reset for vectorized puffer environments.

        The base class ``reset()`` calls ``_sanitize_boundary_data(info)`` which
        expects a single dict, but puffer's ``_reset_env()`` returns a *list* of
        dicts (one per vectorized env).  We store the raw list in ``self._infos``
        and leave vectorization to ``_extract_batched_info``.
        """
        self._state, self._infos = self._reset_env()
        self._done = False
        self._episode_reward = 0.0
        self._episode_length = 0
        return self._state, self._infos

    def _reset_env(self) -> Tuple[Any, Dict[str, Any]]:
        """Initializes the vectorized environment on the first call."""
        if not self._initialized:
            self.vec_env = pufferlib.vector.Multiprocessing(
                env_creators=[_make_puffer_env] * self.num_envs,
                env_args=[(self.env_factory,)] * self.num_envs,
                env_kwargs=[{}] * self.num_envs,
                num_envs=self.num_envs,
                num_workers=self.num_puffer_workers,
            )
            self.active_sequences = [
                Sequence(self.num_players) for _ in range(self.num_envs)
            ]

            self.vec_env.async_reset()
            self._obs, _, _, _, self._infos, _, _ = self.vec_env.recv()
            self._initialized = True

        return self._obs, self._infos

    def _step_env(self, action: Any) -> Any:
        raise NotImplementedError(
            "PufferActors use play_sequence() for vectorized stepping"
        )

    @abstractmethod
    def _extract_batched_info(
        self, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[Any]]:
        """Extracts player IDs and legal moves from a batch of infos."""
        pass

    @torch.inference_mode()
    def play_sequence(self, stats_tracker: Optional[Any] = None) -> Dict[str, Any]:
        """Runs vectorized environments until at least one episode completes."""
        start_time = time.time()
        if not self._initialized:
            self.reset()

        completed_stats = []
        mcts_sims_total = 0
        mcts_search_total = 0.0

        while not completed_stats:
            # 1. Batched Action Selection
            batched_info = self._extract_batched_info(self._infos)

            obs_tensor = torch.as_tensor(
                self._obs, dtype=torch.float32, device=self.device
            )
            # Ensure batch dimension
            if obs_tensor.dim() == len(self.input_shape):
                obs_tensor = obs_tensor.unsqueeze(0)

            # Perform batched inference via PolicySource
            # Note: batched_info now contains pre-vectorized 'player' and 'legal_moves'
            result = self.policy_source.get_inference(
                obs=obs_tensor,
                info=batched_info,
                agent_network=self.agent_network,
                exploration=True,
                to_play=0,  # Default for batched search for now
            )

            actions_tensor, metadata = self.selector.select_action(
                result=result,
                info=batched_info,
                exploration=True,
            )
            # Merge search_metadata (and other extras) from the policy source result
            if result.extra_metadata:
                metadata.update(result.extra_metadata)
            actions = actions_tensor.cpu().numpy()

            # Collect MCTS metrics from search metadata (list for batched, dict for single)
            search_meta = metadata.get("search_metadata")
            if search_meta:
                if isinstance(search_meta, list):
                    for sm in search_meta:
                        mcts_sims_total += int(sm.get("mcts_simulations", 0))
                        mcts_search_total += float(sm.get("mcts_search_time", 0.0))
                else:
                    mcts_sims_total += int(search_meta.get("mcts_simulations", 0))
                    mcts_search_total += float(search_meta.get("mcts_search_time", 0.0))

            # 2. Step Environments
            self.vec_env.send(actions)
            next_obs, rewards, terminals, truncs, next_infos, _, _ = self.vec_env.recv()

            # 3. Pivot Batch to Chronological Sequences
            policies = metadata.get("target_policies", metadata.get("policy"))
            # Fallback: CategoricalSelector/TemperatureSelector don't set "value"; use MCTS root.
            values = metadata.get("value")
            if values is None and result.value is not None:
                values = result.value

            for i in range(self.num_envs):
                # Robustly get current and next info
                info = self._infos[i] if (self._infos and i < len(self._infos)) else {}
                next_info = (
                    next_infos[i] if (next_infos and i < len(next_infos)) else {}
                )

                # Handle potential None from PufferLib
                if info is None:
                    info = {}
                if next_info is None:
                    next_info = {}

                self.active_sequences[i].append(
                    observation=np.copy(self._obs[i]),
                    action=actions[i],
                    reward=float(rewards[i]),
                    policy=policies[i] if policies is not None else None,
                    value=values[i] if values is not None else None,
                    # IMPORTANT: terminated/truncated describe the STATE of the
                    # observation, NOT the result of the action.  self._obs[i]
                    # is the pre-action state and is NEVER terminal.  The done
                    # flags belong exclusively on the terminal-close append.
                    terminated=False,
                    truncated=False,
                    player_id=batched_info["player"][i].item(),
                    legal_moves=info.get("legal_moves", []),
                    all_player_rewards=next_info.get("all_player_rewards"),
                )

                # 4. Handle End of Episode — Unwrap the Info-Stash
                if terminals[i] or truncs[i]:
                    # --- Extract the TRUE terminal state from the stash ---
                    # The wrapper stashes the real final board state before
                    # PufferLib's auto-reset overwrites it.  Fall back to
                    # next_obs[i] (auto-reset obs) if stash is missing
                    # (e.g. single-player Gym envs without the wrapper).
                    terminal_obs = next_info.get("terminal_observation")
                    assert terminal_obs is not None
                    terminal_legal_moves = next_info.get(
                        "terminal_legal_moves",
                        next_info.get("legal_moves", []),
                    )

                    # Close the old sequence with the TRUE terminal data
                    self.active_sequences[i].append(
                        observation=terminal_obs,
                        terminated=terminals[i],
                        truncated=truncs[i],
                        legal_moves=terminal_legal_moves,
                    )
                    completed_seq = self.active_sequences[i]
                    completed_seq.duration_seconds = time.time() - start_time

                    # Store MCTS metrics in sequence stats for executor aggregation
                    completed_seq.stats["mcts_simulations"] = mcts_sims_total
                    completed_seq.stats["mcts_search_time"] = mcts_search_total
                    if mcts_search_total > 0:
                        completed_seq.stats["mcts_sps"] = (
                            mcts_sims_total / mcts_search_total
                        )

                    # Use stashed terminal rewards for correct score tracking
                    terminal_info = {}
                    if "terminal_all_player_rewards" in next_info:
                        terminal_info["all_player_rewards"] = next_info[
                            "terminal_all_player_rewards"
                        ]
                    elif "all_player_rewards" in next_info:
                        terminal_info["all_player_rewards"] = next_info[
                            "all_player_rewards"
                        ]
                    self._finalize_episode_info(completed_seq, terminal_info)
                    self.replay_buffer.store_aggregate(completed_seq)

                    score = self._get_score(completed_seq)
                    ep_stats = {
                        "episode_length": len(completed_seq),
                        "score": score,
                        "duration_seconds": completed_seq.duration_seconds,
                        "mcts_simulations": mcts_sims_total,
                        "mcts_search_time": mcts_search_total,
                        "mcts_sps": completed_seq.stats.get("mcts_sps", 0.0),
                    }
                    completed_stats.append(ep_stats)

                    if stats_tracker:
                        stats_tracker.append("score", score)
                        stats_tracker.append("episode_length", len(completed_seq))
                        stats_tracker.increment_steps(len(completed_seq))
                        # Log MCTS SPS to stats_tracker
                        if "mcts_sps" in completed_seq.stats:
                            stats_tracker.append(
                                "mcts_sps", completed_seq.stats["mcts_sps"]
                            )

                    # --- Start a NEW sequence for the next episode ---
                    # PufferLib already reset the env; next_obs[i] is step-0
                    # of the new episode.  Do NOT seed it here — the next
                    # loop iteration will append self._obs[i] (= next_obs[i])
                    # with its action, maintaining the Sequence invariant:
                    #   len(obs_history) == len(action_history) + 1
                    self.active_sequences[i] = Sequence(self.num_players)

            self._obs, self._infos = next_obs, next_infos

        return completed_stats[0]

    @abstractmethod
    def _finalize_episode_info(
        self, sequence: Sequence, final_info: Dict[str, Any]
    ) -> None:
        pass


class GymPufferActor(BasePufferActor):
    """PufferActor specialized for Gymnasium single-player environments."""

    def _detect_num_players(self) -> int:
        return 1

    def _get_player_id(self) -> None:
        return None

    def _finalize_episode_info(
        self, sequence: Sequence, final_info: Dict[str, Any]
    ) -> None:
        pass

    def _get_score(self, sequence: Sequence) -> float:
        return sum(sequence.rewards)

    def _extract_batched_info(self, infos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ready-to-use vectorized tensors for Gymnasium environments."""
        num_actions = self.num_actions

        # 1. Vectorize Legal Moves and Players
        mask = torch.zeros(
            (self.num_envs, num_actions), dtype=torch.bool, device=self.device
        )
        player_tensor = torch.zeros(self.num_envs, dtype=torch.int8, device=self.device)

        if not infos:
            mask.fill_(True)
            return {
                "legal_moves": mask,
                "legal_moves_mask": mask,
                "player": player_tensor,
            }

        for i in range(self.num_envs):
            info = infos[i] if (infos and i < len(infos)) else {}
            if info is None:
                info = {}

            legal = info.get("legal_moves", [])
            if isinstance(legal, (list, np.ndarray, torch.Tensor)) and len(legal) > 0:
                mask[i, legal] = True
            else:
                mask[i, :].fill_(True)

        return {
            "legal_moves": mask,
            "legal_moves_mask": mask,
            "player": player_tensor,
        }


class PettingZooPufferActor(BasePufferActor):
    """PufferActor specialized for PettingZoo AEC multi-player environments."""

    def _detect_num_players(self) -> int:
        return len(self.env.possible_agents)

    def _get_player_id(self) -> Optional[str]:
        return None

    def _finalize_episode_info(
        self, sequence: Sequence, final_info: Dict[str, Any]
    ) -> None:
        if final_info and "all_player_rewards" in final_info:
            sequence.stats["final_player_rewards"] = final_info["all_player_rewards"]

    def _get_score(self, sequence: Sequence) -> float:
        if "final_player_rewards" in sequence.stats:
            final_rewards = sequence.stats["final_player_rewards"]
            # player_0 mapping
            return final_rewards.get("player_0", 0.0)
        return sum(sequence.rewards)

    def _extract_batched_info(self, infos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ready-to-use vectorized tensors for PettingZoo environments."""
        num_actions = self.num_actions

        mask = torch.zeros(
            (self.num_envs, num_actions), dtype=torch.bool, device=self.device
        )
        player_ids = []

        if not infos:
            mask.fill_(True)
            return {
                "legal_moves": mask,
                "legal_moves_mask": mask,
                "player": torch.zeros(
                    self.num_envs, dtype=torch.int8, device=self.device
                ),
            }

        batch_indices = []
        action_indices = []

        for i in range(self.num_envs):
            info = infos[i] if (infos and i < len(infos)) else {}
            if info is None:
                info = {}

            legal = info.get("legal_moves", [])
            if isinstance(legal, (list, np.ndarray, torch.Tensor)) and len(legal) > 0:
                batch_indices.extend([i] * len(legal))
                action_indices.extend(legal)
            else:
                mask[i, :].fill_(True)

            player_ids.append(info.get("player", 0))

        if batch_indices:
            mask[batch_indices, action_indices] = True

        player_tensor = torch.tensor(player_ids, dtype=torch.int8, device=self.device)

        return {
            "legal_moves": mask,
            "legal_moves_mask": mask,
            "player": player_tensor,
        }
