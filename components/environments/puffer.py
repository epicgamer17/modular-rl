"""
Puffer Environment Components
==============================
Pipeline components that bridge PufferLib's vectorized ``vec_env`` interface
to the ``Blackboard`` data-flow model.

Architecture boundary
---------------------
These components own the **I/O layer** with ``vec_env``; they know nothing
about sequences, replay buffers, or episode logic.  That pivoting work lives
in ``components/memory/sequence_pivot.py``.

Blackboard contract
-------------------
After ``PufferObservationComponent.execute()``:
    blackboard.data["obs"]   : torch.Tensor  [B, *obs_shape]
    blackboard.data["infos"] : List[Dict[str, Any]]

After ``PufferStepComponent.execute()``:
    blackboard.data["next_obs"]   : np.ndarray   [B, *obs_shape]
    blackboard.data["rewards"]    : np.ndarray   [B]
    blackboard.data["terminals"]  : np.ndarray   [B]  (bool)
    blackboard.data["truncations"]: np.ndarray   [B]  (bool)
    blackboard.data["next_infos"] : List[Dict[str, Any]]
"""

import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple

from core import PipelineComponent, Blackboard


class PufferObservationComponent(PipelineComponent):
    """
    Reads the latest batch of observations from a PufferLib vectorised
    environment and writes them to the Blackboard.

    On the very first call the environment must already have been reset
    externally (e.g. by ``BasePufferActor._reset_env()``).  Subsequent
    calls simply re-publish whatever ``_obs`` / ``_infos`` were previously
    received; the *step* component is responsible for advancing those
    pointers after each ``vec_env.send()`` / ``vec_env.recv()`` cycle.

    Args:
        vec_env: A PufferLib vectorised environment (``pufferlib.vector.*``).
        num_envs: Number of parallel environments inside ``vec_env``.
        device: The torch device on which the observation tensor is placed.
        input_shape: Shape of a single observation (without the batch dim).
            Used to validate that the batched tensor has the expected rank.
            Pass ``None`` to skip the rank check.

    Blackboard outputs:
        ``data["obs"]``   – Float32 tensor of shape ``[num_envs, *input_shape]``.
        ``data["infos"]`` – Raw list of per-env info dicts from ``vec_env.recv()``.
    """

    def __init__(
        self,
        vec_env: Any,
        num_envs: int,
        device: torch.device,
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        self.vec_env = vec_env
        self.num_envs = num_envs
        self.device = device
        self.input_shape = input_shape

        # Internal cache, populated externally before the first execute() call.
        # BasePufferActor sets these after _reset_env(); the StepComponent keeps
        # them updated after each environment step.
        self._obs: Optional[np.ndarray] = None
        self._infos: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # Public helpers – called by BasePufferActor to seed initial state
    # ------------------------------------------------------------------

    def set_obs(self, obs: np.ndarray, infos: List[Dict[str, Any]]) -> None:
        """
        Seed the internal observation and info caches.

        Called once after the environment has been reset (before the first
        pipeline tick) and again by ``PufferStepComponent`` after every step.

        Args:
            obs:   Raw NumPy array of shape ``[num_envs, *obs_shape]``.
            infos: List of per-env info dicts, length ``num_envs``.
        """
        self._obs = obs
        self._infos = infos

    # ------------------------------------------------------------------
    # PipelineComponent interface
    # ------------------------------------------------------------------

    def execute(self, blackboard: Blackboard) -> None:
        """
        Publish the current observation batch onto the Blackboard.

        Converts the cached NumPy array to a Float32 tensor (on
        ``self.device``) and writes it to ``blackboard.data["obs"]``.
        The raw info list is stored at ``blackboard.data["infos"]``.

        Args:
            blackboard: The shared Blackboard for the current pipeline tick.

        Raises:
            AssertionError: If ``set_obs`` has not been called yet.
            AssertionError: If the obs batch size does not match ``num_envs``.
            AssertionError: If ``input_shape`` is given and the tensor rank is wrong.
        """
        assert self._obs is not None, (
            "PufferObservationComponent: obs cache is None. "
            "Call set_obs() with the result of vec_env.recv() before executing."
        )
        assert self._infos is not None, (
            "PufferObservationComponent: infos cache is None. "
            "Call set_obs() with the result of vec_env.recv() before executing."
        )

        # [B, *obs_shape]
        obs_tensor = torch.as_tensor(
            self._obs, dtype=torch.float32, device=self.device
        )

        assert obs_tensor.shape[0] == self.num_envs, (
            f"PufferObservationComponent: expected batch size {self.num_envs}, "
            f"got {obs_tensor.shape[0]}. Check num_envs."
        )

        if self.input_shape is not None:
            expected_ndim = 1 + len(self.input_shape)  # batch dim + obs dims
            assert obs_tensor.dim() == expected_ndim, (
                f"PufferObservationComponent: expected {expected_ndim}-D tensor "
                f"[{self.num_envs}, *{self.input_shape}], "
                f"got shape {tuple(obs_tensor.shape)}."
            )

        blackboard.data["obs"] = obs_tensor          # [B, *obs_shape]
        blackboard.data["infos"] = self._infos        # List[Dict]


class PufferStepComponent(PipelineComponent):
    """
    Sends the batched action to PufferLib and receives the resulting
    observations, rewards, and done-flags via ``vec_env.send()`` /
    ``vec_env.recv()``.

    The component reads ``blackboard.meta["actions"]`` (a NumPy int array of
    shape ``[num_envs]``) and writes the full step result back onto the
    Blackboard so that downstream components (e.g. ``VectorToSequencePivotComponent``)
    can consume them without knowing anything about ``vec_env``.

    It also updates the ``PufferObservationComponent`` internal cache so that
    the *next* pipeline tick sees the fresh observations returned by the reset.

    Args:
        vec_env: The same PufferLib vectorised environment used by the
            paired ``PufferObservationComponent``.
        obs_component: The companion observation component whose internal
            ``_obs`` / ``_infos`` caches will be advanced after each step.

    Blackboard inputs (reads):
        ``meta["actions"]`` – ``np.ndarray`` of shape ``[num_envs]`` (int).

    Blackboard outputs (writes):
        ``data["next_obs"]``    – ``np.ndarray``  [B, *obs_shape]  (from recv)
        ``data["rewards"]``     – ``np.ndarray``  [B]              float
        ``data["terminals"]``   – ``np.ndarray``  [B]              bool
        ``data["truncations"]`` – ``np.ndarray``  [B]              bool
        ``data["next_infos"]``  – ``List[Dict]``  length B
    """

    def __init__(
        self,
        vec_env: Any,
        obs_component: PufferObservationComponent,
    ) -> None:
        self.vec_env = vec_env
        self.obs_component = obs_component

    def execute(self, blackboard: Blackboard) -> None:
        """
        Send actions to the vectorised environment and receive the next batch.

        The step follows PufferLib's async convention: ``send()`` triggers the
        environment workers, and ``recv()`` blocks until all results arrive.

        Args:
            blackboard: The shared Blackboard for the current pipeline tick.

        Raises:
            KeyError: If ``blackboard.meta["actions"]`` is not set by an
                upstream action-selection component.
            AssertionError: If the actions array does not have the expected
                batch size.
        """
        assert "actions" in blackboard.meta, (
            "PufferStepComponent: 'actions' key missing from blackboard.meta. "
            "An upstream action-selection component must write "
            "blackboard.meta['actions'] (np.ndarray of shape [num_envs]) "
            "before PufferStepComponent executes."
        )

        actions: np.ndarray = blackboard.meta["actions"]

        # Send actions; PufferLib workers start stepping asynchronously.
        self.vec_env.send(actions)

        # Block until all workers have finished; unpack the 7-tuple PufferLib returns.
        # Shape: next_obs [B, *obs], rewards [B], terminals [B], truncs [B],
        #        next_infos List[Dict], episode_returns, episode_lengths
        next_obs, rewards, terminals, truncations, next_infos, _, _ = (
            self.vec_env.recv()
        )

        # Write step results to the Blackboard for downstream components.
        blackboard.data["next_obs"] = next_obs           # np.ndarray [B, *obs_shape]
        blackboard.data["rewards"] = rewards             # np.ndarray [B]
        blackboard.data["terminals"] = terminals         # np.ndarray [B]  bool
        blackboard.data["truncations"] = truncations     # np.ndarray [B]  bool
        blackboard.data["next_infos"] = next_infos       # List[Dict]

        # Advance the observation-component's cache so the next tick sees the
        # post-reset observations (PufferLib auto-resets completed envs).
        self.obs_component.set_obs(next_obs, next_infos)
