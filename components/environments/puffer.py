from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
import torch
from core import PipelineComponent, Blackboard
from core.contracts import Key, Observation, Action, Reward, Done, SemanticType


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

    @property
    def requires(self) -> Set[Key]:
        return set()

    @property
    def provides(self) -> Dict[Key, str]:
        return {
            Key("data.obs", Observation): "new",
            Key("data.infos", SemanticType): "new",
        }

    def validate(self, blackboard: Blackboard) -> None:
        assert self._obs is not None, (
            "PufferObservationComponent: obs cache is None. "
            "Call set_obs() before executing."
        )

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

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        """
        Publish the current observation batch onto the Blackboard.

        Converts the cached NumPy array to a Float32 tensor (on
        ``self.device``) and returns it for writing to ``blackboard.data["obs"]``.
        The raw info list is returned for ``blackboard.data["infos"]``.

        Args:
            blackboard: The shared Blackboard for the current pipeline tick.

        Returns:
            Dictionary containing "data.obs" and "data.infos".

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

        return {
            "data.obs": obs_tensor,          # [B, *obs_shape]
            "data.infos": self._infos        # List[Dict]
        }


class PufferStepComponent(PipelineComponent):
    """
    Sends the batched action to PufferLib and receives the resulting
    observations, rewards, and done-flags via ``vec_env.send()`` /
    ``vec_env.recv()``.

    The component reads ``blackboard.meta["actions"]`` (a NumPy int array of
    shape ``[num_envs]``) and writes the full step result back onto the
    Blackboard via return so that downstream components can consume them.

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

    @property
    def requires(self) -> Set[Key]:
        return {Key("meta.actions", Action)}

    @property
    def provides(self) -> Dict[Key, str]:
        return {
            Key("data.next_obs", Observation): "new",
            Key("data.rewards", Reward): "new",
            Key("data.terminals", Done): "new",
            Key("data.truncations", Done): "new",
            Key("data.dones", Done): "new",
            Key("data.next_infos", SemanticType): "new",
        }

    def validate(self, blackboard: Blackboard) -> None:
        assert "actions" in blackboard.meta, (
            "PufferStepComponent: 'actions' missing from blackboard.meta"
        )

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        """
        Send actions to the vectorised environment and receive the next batch.

        The step follows PufferLib's async convention: ``send()`` triggers the
        environment workers, and ``recv()`` blocks until all results arrive.

        Args:
            blackboard: The shared Blackboard for the current pipeline tick.

        Returns:
            Dictionary containing "data.next_obs", "data.rewards", 
            "data.terminals", "data.truncations", and "data.next_infos".

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
        next_obs, rewards, terminals, truncations, next_infos, _, _ = (
            self.vec_env.recv()
        )

        # Advance the observation-component's cache so the next tick sees the
        # post-reset observations (PufferLib auto-resets completed envs).
        self.obs_component.set_obs(next_obs, next_infos)

        return {
            "data.next_obs": next_obs,           # np.ndarray [B, *obs_shape]
            "data.rewards": rewards,             # np.ndarray [B]
            "data.terminals": terminals,         # np.ndarray [B]  bool
            "data.truncations": truncations,     # np.ndarray [B]  bool
            "data.dones": terminals | truncations,
            "data.next_infos": next_infos        # List[Dict]
        }
