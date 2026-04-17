"""
Sequence Pivot Component
========================
Converts PufferLib's *horizontal* (per-env-index) batch format back into
*vertical* (chronological), per-episode ``Sequence`` objects.

Why this exists
---------------
PufferLib keeps N environments running in lock-step.  After every call to
``vec_env.recv()`` you receive a batch of N observations – one row per
*environment*, not one row per *episode*.  A replay buffer (especially
MuZero's) wants complete, contiguous episode sequences.

``VectorToSequencePivotComponent`` maintains ``num_envs`` in-flight
``Sequence`` objects and appends one step to each on every pipeline tick.
When an environment signals episode-end it closes the sequence, attaches
aggregate statistics, and writes it to
``blackboard.meta["completed_sequences"]`` for a downstream
``BufferPushComponent`` to consume.

Blackboard contract
-------------------
Reads:
    ``data["obs"]``         – torch.Tensor  [B, *obs_shape] (pre-action obs)
    ``data["infos"]``       – List[Dict]    length B       (per-env info dicts)
    ``meta["actions"]``     – np.ndarray   [B]            (int actions taken)
    ``meta["action_metadata"]`` – Dict[str, Any]
        Expected optional keys (all shapes [B, …]):
            * ``"target_policies"`` or ``"policy"`` – policy vectors
            * ``"value"``                             – value estimates
    ``data["rewards"]``     – np.ndarray   [B]            float
    ``data["terminals"]``   – np.ndarray   [B]            bool
    ``data["truncations"]`` – np.ndarray   [B]            bool
    ``data["next_infos"]``  – List[Dict]   length B

Writes:
    ``meta["completed_sequences"]`` – List[Sequence]
        One entry for every env that finished an episode this tick.
        Empty list if no env terminated.
"""

import time
import numpy as np
from typing import Any, Dict, List, Optional, Set

from core import PipelineComponent, Blackboard
from core.contracts import Key, Observation, Action, Reward, Done, SemanticType
from data.ingestion import Sequence


# ---------------------------------------------------------------------------
# Sentinel: no value supplied for a given batch entry
# ---------------------------------------------------------------------------
_MISSING = object()


class VectorToSequencePivotComponent(PipelineComponent):
    """
    Pivots a vectorised PufferLib step into chronological ``Sequence`` objects.

    One instance of this component is shared across all pipeline ticks for the
    lifetime of an actor run.  It owns ``num_envs`` mutable ``Sequence``
    objects and flushes completed ones each tick.

    Args:
        num_envs:    Number of parallel PufferLib environments.
        num_players: Number of players per environment (1 for single-agent).
        policy_shape:
            Shape of a single policy vector (e.g. ``(num_actions,)``).
            Used to construct a zero policy when the action-selector does
            not emit one.  Pass ``None`` to skip the zero-policy fallback
            (a ``None`` policy will be stored in the sequence instead).
        episode_start_time:
            Optional epoch time (``time.time()``) marking when the current
            batch of episodes started. If ``None`` the component measures
            time from its own construction.
    """

    def __init__(
        self,
        num_envs: int,
        num_players: int,
        policy_shape: Optional[tuple] = None,
        episode_start_time: Optional[float] = None,
    ) -> None:
        assert num_envs > 0, f"num_envs must be positive, got {num_envs}"
        assert num_players > 0, f"num_players must be positive, got {num_players}"

        self.num_envs = num_envs
        self.num_players = num_players
        self.policy_shape = policy_shape

        # One in-flight Sequence per environment.
        self.active_sequences: List[Sequence] = [
            Sequence(num_players) for _ in range(num_envs)
        ]

        self._start_time: float = (
            episode_start_time if episode_start_time is not None else time.time()
        )

        self._requires = {
            Key("data.obs", Observation),
            Key("meta.actions", Action),
            Key("data.rewards", Reward),
            Key("data.terminals", Done),
            Key("data.truncations", Done),
            Key("data.next_infos", SemanticType),
        }
        self._provides = {Key("meta.completed_sequences", SemanticType): "new"}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures all required batch inputs exist and have matching shapes."""
        obs_tensor = blackboard.data.get("obs")
        assert obs_tensor is not None, (
            "VectorToSequencePivotComponent: 'obs' missing from blackboard.data"
        )
        actions = blackboard.meta.get("actions")
        assert actions is not None, (
            "VectorToSequencePivotComponent: 'actions' missing from blackboard.meta"
        )
        assert len(actions) == self.num_envs, (
            f"VectorToSequencePivotComponent: actions length {len(actions)} != num_envs {self.num_envs}"
        )
        assert blackboard.data.get("rewards") is not None, (
            "VectorToSequencePivotComponent: 'rewards' missing from blackboard.data"
        )
        assert blackboard.data.get("terminals") is not None, (
            "VectorToSequencePivotComponent: 'terminals' missing from blackboard.data"
        )
        assert blackboard.data.get("truncations") is not None, (
            "VectorToSequencePivotComponent: 'truncations' missing from blackboard.data"
        )

    # ------------------------------------------------------------------
    # PipelineComponent interface
    # ------------------------------------------------------------------

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        """
        Append one step to every active sequence, flush completed episodes.

        After execution, the returned updates contain "meta.completed_sequences" 
        with a (possibly empty) list of ``Sequence`` objects ready for the replay
        buffer.

        Args:
            blackboard: The shared Blackboard for the current pipeline tick.

        Returns:
            Dictionary containing "meta.completed_sequences".

        Raises:
            AssertionError: If any required Blackboard key is missing or has
                an unexpected shape.
        """
        # ---- Read required inputs (validated by validate()) ----------------
        obs_tensor = blackboard.data["obs"]

        infos: List[Dict[str, Any]] = blackboard.data.get("infos") or [
            {} for _ in range(self.num_envs)
        ]
        actions: np.ndarray = blackboard.meta["actions"]  # type: ignore[assignment]
        rewards: np.ndarray = blackboard.data["rewards"]  # type: ignore[assignment]
        terminals: np.ndarray = blackboard.data["terminals"]  # type: ignore[assignment]
        truncations: np.ndarray = blackboard.data["truncations"]  # type: ignore[assignment]

        next_infos: List[Dict[str, Any]] = blackboard.data.get("next_infos") or [
            {} for _ in range(self.num_envs)
        ]

        # ---- Read optional action-metadata ---------------------------------
        metadata: Dict[str, Any] = blackboard.meta.get("action_metadata", {})

        # Policy vectors – shape [B, num_actions] or None
        policies = metadata.get("target_policies", metadata.get("policy"))

        # Value estimates – shape [B] or None
        values = metadata.get("value")

        # Convert obs tensor to a numpy array once for cheap per-env slicing.
        # Shape: [B, *obs_shape]
        obs_np: np.ndarray = obs_tensor.cpu().numpy()

        # ---- Per-environment pivot loop ------------------------------------
        completed: List[Sequence] = []

        for i in range(self.num_envs):
            info: Dict[str, Any] = infos[i] if i < len(infos) else {}
            next_info: Dict[str, Any] = next_infos[i] if i < len(next_infos) else {}

            # Defensive: PufferLib can occasionally yield None info dicts.
            if info is None:
                info = {}
            if next_info is None:
                next_info = {}

            # -- Retrieve per-env scalars -----------------------------------
            action = int(actions[i])
            reward = float(rewards[i])
            terminal = bool(terminals[i])
            truncated = bool(truncations[i])

            # Policy for this env (or zero-filled fallback).
            policy = self._get_policy(policies, i)

            # Value for this env.
            value: Optional[float] = (
                float(values[i]) if values is not None else None
            )

            # Player ID: PufferLib stores it in info["player"] (as an int).
            player_id: Optional[int] = info.get("player", None)

            # Legal moves for the *current* state.
            legal_moves: List[int] = info.get("legal_moves", [])

            all_player_rewards = next_info.get("all_player_rewards")

            # -- Append the current OBSERVATION + action to the sequence ----
            # The pre-action state is never terminal; done-flags are only
            # valid on the terminal-close append (see below).
            self.active_sequences[i].append(
                observation=np.copy(obs_np[i]),
                action=action,
                reward=reward,
                policy=policy,
                value=value,
                terminated=False,   # pre-action state is never terminal
                truncated=False,
                player_id=player_id,
                legal_moves=legal_moves,
                all_player_rewards=all_player_rewards,
            )

            # -- Handle episode termination ---------------------------------
            if terminal or truncated:
                # PufferLib (via GymnasiumPufferEnv) stores the true final
                # state in next_info["terminal_observation"] before auto-reset.
                terminal_obs = next_info.get("terminal_observation")
                assert terminal_obs is not None, (
                    f"VectorToSequencePivotComponent: env[{i}] finished but "
                    f"next_infos[{i}]['terminal_observation'] is missing. "
                    "Ensure the environment is wrapped with a GymnasiumPufferEnv "
                    "or equivalent that stashes the final state."
                )

                terminal_legal_moves: List[int] = next_info.get(
                    "terminal_legal_moves",
                    next_info.get("legal_moves", []),
                )

                # Close the sequence with the TRUE terminal observation.
                self.active_sequences[i].append(
                    observation=terminal_obs,
                    terminated=terminal,
                    truncated=truncated,
                    legal_moves=terminal_legal_moves,
                )

                completed_seq = self.active_sequences[i]
                completed_seq.duration_seconds = time.time() - self._start_time
                completed.append(completed_seq)

                # Start a fresh sequence; next loop tick will seed it.
                self.active_sequences[i] = Sequence(self.num_players)
                # Reset per-episode timer so subsequent episodes get a
                # meaningful duration.
                self._start_time = time.time()

        # ---- Return completed sequences updates ---------------------------
        return {"meta.completed_sequences": completed}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_policy(
        self,
        policies: Optional[Any],
        env_idx: int,
    ) -> Optional[np.ndarray]:
        """
        Extract the policy vector for a single env index.

        Returns a zero-filled array of ``self.policy_shape`` if *policies* is
        ``None`` and ``policy_shape`` was provided; otherwise returns ``None``.

        Args:
            policies:  Batch of policy vectors (shape [B, A]) or ``None``.
            env_idx:   Environment index to slice.

        Returns:
            1-D NumPy float32 array for the env, or ``None``.
        """
        if policies is not None:
            return policies[env_idx]

        if self.policy_shape is not None:
            return np.zeros(self.policy_shape, dtype=np.float32)

        return None
