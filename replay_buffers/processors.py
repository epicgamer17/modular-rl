from typing import Dict, List, Optional
import numpy as np
import torch
from abc import ABC, abstractmethod
from collections import deque
from replay_buffers.utils import discounted_cumulative_sums
from utils.utils import legal_moves_mask
from logging import warning

# ==========================================
# Base Classes
# ==========================================


class InputProcessor(ABC):
    """
    Processes data BEFORE it is written to the Writer/Storage.
    """

    @abstractmethod
    def process_single(self, *args, **kwargs):
        """
        Processes a single transition.
        Returns:
            processed_data: Data ready to be stored (or None if accumulating).
        """
        pass  # pragma: no cover

    def process_sequence(self, sequence, **kwargs):
        """Optional hook for processing entire sequence objects."""
        # Default behavior: iterate over transitions and apply process_single
        transitions = kwargs.get("transitions")
        if transitions is None:
            transitions = self._sequence_to_transitions(sequence)

        processed = []
        for t in transitions:
            res = self.process_single(**t)
            if res is not None:
                processed.append(res)
        return {"transitions": processed}

    def _sequence_to_transitions(self, sequence):
        """
        Helper to convert a Sequence object into a list of transition dictionaries
        compatible with process_single.
        """
        transitions = []
        for i in range(len(sequence.action_history)):
            t = {
                "observations": sequence.observation_history[i],
                "actions": sequence.action_history[i],
                "rewards": (
                    float(sequence.rewards[i]) if i < len(sequence.rewards) else 0.0
                ),
                "next_observations": sequence.observation_history[i + 1],
                "terminated": bool(sequence.terminated_history[i + 1]),
                "truncated": bool(sequence.truncated_history[i + 1]),
                "dones": bool(sequence.done_history[i + 1]),
                "player": (
                    sequence.player_id_history[i]
                    if i < len(sequence.player_id_history)
                    else 0
                ),
                "values": (
                    sequence.value_history[i]
                    if i < len(sequence.value_history)
                    else 0.0
                ),
                "policies": (
                    sequence.policy_history[i]
                    if i < len(sequence.policy_history)
                    else None
                ),
                "legal_moves": (
                    sequence.legal_moves_history[i]
                    if i < len(sequence.legal_moves_history)
                    else None
                ),
                "next_legal_moves": (
                    sequence.legal_moves_history[i + 1]
                    if i + 1 < len(sequence.legal_moves_history)
                    else None
                ),
                "log_prob": (
                    sequence.log_prob_history[i]
                    if i < len(sequence.log_prob_history)
                    else None
                ),
            }
            # Add next_infos if available (usually not in Sequence but for completeness)
            if hasattr(sequence, "next_infos") and i < len(sequence.next_infos):
                t["next_infos"] = sequence.next_infos[i]

            transitions.append(t)
        return transitions

    def finish_trajectory(self, buffers, trajectory_slice, **kwargs):
        """
        Optional hook called when a trajectory ends.
        Override to compute trajectory-level computations (e.g., GAE).
        Returns:
            Optional dict of computed values to store in buffer.
        """
        return None

    def clear(self):
        pass


class OutputProcessor(ABC):
    """
    Processes indices indices retrieved from the Sampler into a final batch.
    """

    @abstractmethod
    def process_batch(
        self, indices: List[int], buffers: Dict[str, torch.Tensor], **kwargs
    ):
        """
        Args:
            indices: List of indices selected by the Sampler.
            buffers: A dictionary reference to the ReplayBuffer's internal storage
                     (e.g., {'obs': self.observation_buffer, 'rew': self.reward_buffer}).
        Returns:
            batch: A dictionary containing the final tensors for training.
        """
        pass  # pragma: no cover

    def clear(self):
        pass


# ==========================================
# Stacked Processors (Pipeline)
# ==========================================


class StackedInputProcessor(InputProcessor):
    """
    Chains multiple InputProcessors sequentially.
    Output of processor i is passed as input to processor i+1.
    """

    def __init__(self, processors: List[InputProcessor]):
        self.processors = processors

    def process_single(self, *args, **kwargs):
        data = kwargs.copy()
        if args:
            raise NotImplementedError(
                "Positional arguments are not supported in StackedInputProcessor."
            )  # pragma: no cover
        for p in self.processors:
            data = p.process_single(**data)
            if data is None:
                return None

        return data

    def process_sequence(self, sequence, **kwargs):
        data = {"sequence": sequence, **kwargs}
        for p in self.processors:
            result = p.process_sequence(**data)
            if result is None:
                return None
            data.update(result)
        return data

    def finish_trajectory(self, buffers, trajectory_slice, **kwargs):
        """
        Calls finish_trajectory on all processors, aggregating results.
        """
        result = {}
        for p in self.processors:
            traj_result = p.finish_trajectory(buffers, trajectory_slice, **kwargs)
            if traj_result:
                result.update(traj_result)
        return result

    def get_processor(self, processor_type):
        """
        Find a processor by type in the stack.
        Returns the first matching processor or None.
        """
        for p in self.processors:
            if isinstance(p, processor_type):
                return p
        return None

    def clear(self):
        for p in self.processors:
            p.clear()


class StackedOutputProcessor(OutputProcessor):
    """
    Chains multiple OutputProcessors.
    Each processor updates the 'batch' dictionary.
    """

    def __init__(self, processors: List[OutputProcessor]):
        self.processors = processors

    def process_batch(
        self,
        indices: List[int],
        buffers: Dict[str, torch.Tensor],
        batch: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ):
        if batch is None:
            batch = {}

        for p in self.processors:
            # Processors should return a dict of new/updated keys
            # They receive the 'batch' so far to allow transformation (e.g. normalization)
            result = p.process_batch(indices, buffers, batch=batch, **kwargs)
            if result:
                batch.update(result)

        return batch

    def clear(self):
        for p in self.processors:
            p.clear()


# ==========================================
# Input Processors
# ==========================================


class IdentityInputProcessor(InputProcessor):
    """Pass-through processor."""

    def process_single(self, **kwargs):
        return kwargs


class LegalMovesMaskProcessor(InputProcessor):
    """
    Creates a boolean mask from a list of legal moves.
    """

    def __init__(
        self,
        num_actions: int,
        input_key: str = "legal_moves",
        output_key: str = "legal_moves_masks",
    ):
        self.num_actions = num_actions
        self.input_key = input_key
        self.output_key = output_key

    def process_single(self, **kwargs):
        legal_moves = kwargs.get(self.input_key, [])
        if legal_moves is None:
            legal_moves = []
        mask = legal_moves_mask(self.num_actions, legal_moves)
        kwargs[self.output_key] = mask
        return kwargs


class ToPlayInputProcessor(InputProcessor):
    """
    Extracts 'player_id' from kwargs.
    """

    def __init__(
        self,
        num_players: int,
        input_key: str = "player",
        output_key: str = "to_plays",
    ):
        self.num_players = num_players
        self.input_key = input_key
        self.output_key = output_key

    def process_single(self, **kwargs):
        val = kwargs.get(self.input_key, 0)
        kwargs[self.output_key] = val
        return kwargs


class FilterKeysInputProcessor(InputProcessor):
    """
    Filters the input dictionary to only include specific keys (whitelist).
    """

    def __init__(self, whitelisted_keys: List[str]):
        self.whitelisted_keys = whitelisted_keys

    def process_single(self, **kwargs):
        return {k: v for k, v in kwargs.items() if k in self.whitelisted_keys}


class TerminationFlagsInputProcessor(InputProcessor):
    """
    Ensures 'terminated'/'truncated' flags are present for transition pipelines.
    Defaults are conservative and backward-compatible with older call sites.
    """

    def __init__(
        self,
        done_key: str = "dones",
        terminated_key: str = "terminated",
        truncated_key: str = "truncated",
    ):
        self.done_key = done_key
        self.terminated_key = terminated_key
        self.truncated_key = truncated_key

    def process_single(self, **kwargs):
        done = bool(kwargs.get(self.done_key, False))
        kwargs[self.terminated_key] = bool(kwargs.get(self.terminated_key, done))
        kwargs[self.truncated_key] = bool(kwargs.get(self.truncated_key, False))
        return kwargs


class NStepInputProcessor(InputProcessor):
    """
    Handles N-Step return calculation.
    Accumulates transitions in a buffer and emits them when N steps are available.
    """

    def __init__(
        self,
        n_step: int,
        gamma: float,
        num_players: int = 1,
        reward_key="rewards",
        done_key="dones",
        terminated_key="terminated",
        truncated_key="truncated",
    ):
        self.n_step = n_step
        self.gamma = gamma
        self.num_players = num_players
        self.reward_key = reward_key
        self.done_key = done_key
        self.terminated_key = terminated_key
        self.truncated_key = truncated_key
        self.n_step_buffers = [deque(maxlen=n_step) for _ in range(num_players)]

    def process_single(self, **kwargs):
        # Determine player index
        player = kwargs.get("player", 0)

        # Store current step data
        self.n_step_buffers[player].append(kwargs)

        if len(self.n_step_buffers[player]) < self.n_step:
            return None

        res = self._emit_oldest(player)
        self.n_step_buffers[player].popleft()
        return res

    def _emit_oldest(self, player):
        """
        Calculates the n-step return for the oldest transition in the buffer
        and returns the processed transition.
        """
        buffer = self.n_step_buffers[player]
        if not buffer:
            return None

        # Calculate N-Step Return
        # We look at the buffer to calculate discounted reward sum
        # The 'transition' to be returned is the oldest one in the deque (s_t)
        # The 'next_observation' will be the one from the newest transition (s_t+n)

        # 1. Calculate Discounted Reward
        final_reward = 0.0
        final_next_obs = buffer[-1].get("next_observations")
        final_next_info = buffer[-1].get("next_infos")
        final_next_legal_moves = buffer[-1].get("next_legal_moves")
        final_done = buffer[-1].get(self.done_key, False)
        final_terminated = buffer[-1].get(self.terminated_key, final_done)
        final_truncated = buffer[-1].get(self.truncated_key, False)

        # Iterate reversed from newest to oldest
        for transition in reversed(list(buffer)):
            r = transition.get(self.reward_key, 0.0)
            d = transition.get(self.done_key, False)

            # If a step was terminal, it cuts the n-step chain
            if d:
                final_reward = r
                final_next_obs = transition.get("next_observations")
                final_next_info = transition.get("next_infos")
                final_next_legal_moves = transition.get("next_legal_moves")
                final_done = True
                final_terminated = transition.get(self.terminated_key, True)
                final_truncated = transition.get(self.truncated_key, False)
            else:
                final_reward = r + self.gamma * final_reward

        # 2. Prepare the output
        # The output is the oldest transition, but with updated reward/next_obs/done
        head_transition = buffer[0].copy()
        head_transition[self.reward_key] = final_reward
        head_transition["next_observations"] = final_next_obs
        head_transition["next_infos"] = final_next_info
        head_transition["next_legal_moves"] = final_next_legal_moves
        head_transition[self.done_key] = final_done
        head_transition[self.terminated_key] = final_terminated
        head_transition[self.truncated_key] = final_truncated

        return head_transition

    def process_sequence(self, sequence, **kwargs):
        """
        Processes a sequence of transitions.
        """
        self.clear()
        transitions = kwargs.get("transitions")
        if transitions is None:
            transitions = self._sequence_to_transitions(sequence)
        processed_transitions = []

        for t in transitions:
            processed = self.process_single(**t)
            if processed is not None:
                processed_transitions.append(processed)

        # Flush remaining transitions if sequence is terminal
        is_done = sequence.done_history[-1] if sequence.done_history else False
        if is_done:
            active_players = set()
            for t in transitions:
                active_players.add(t.get("player", 0))

            for p in active_players:
                while self.n_step_buffers[p]:
                    res = self._emit_oldest(p)
                    if res:
                        processed_transitions.append(res)
                    self.n_step_buffers[p].popleft()

        return {"transitions": processed_transitions}

    def clear(self):
        self.n_step_buffers = [
            deque(maxlen=self.n_step) for _ in range(self.num_players)
        ]


class SequenceTensorProcessor(InputProcessor):
    """
    Converts a complete Sequence object into tensor format for storage.
    Handles observation stacking, action/reward tensors, and legal move masks.
    """

    def __init__(
        self,
        num_actions: int,
        num_players: int,
        player_id_mapping: Dict[str, int],
        device="cpu",
    ):
        self.num_actions = num_actions
        self.num_players = num_players
        self.player_id_mapping = player_id_mapping
        self.device = device

    def _resolve_player_id(self, player_id) -> int:
        if isinstance(player_id, int):
            return player_id
        if player_id not in self.player_id_mapping:
            raise ValueError(
                f"player_id '{player_id}' not found in player_id_mapping. "
                f"Available keys: {list(self.player_id_mapping.keys())}"
            )
        return self.player_id_mapping[player_id]

    def process_single(self, **kwargs):
        raise NotImplementedError(
            "SequenceTensorProcessor only supports process_sequence."
        )  # pragma: no cover

    def process_sequence(self, sequence, **kwargs):
        # 1. Prepare Observations
        obs_history = sequence.observation_history
        n_states = len(obs_history)
        n_transitions = len(sequence.action_history)
        if n_transitions + 1 != n_states:
            raise ValueError(
                "observation_history must have exactly one more entry than action_history"
            )
        if len(sequence.terminated_history) != n_states:
            raise ValueError(
                "Sequence terminated_history length must equal observation_history length"
            )
        if len(sequence.truncated_history) != n_states:
            raise ValueError(
                "Sequence truncated_history length must equal observation_history length"
            )
        if len(sequence.done_history) != n_states:
            raise ValueError(
                "Sequence done_history length must equal observation_history length"
            )
        obs_tensor = torch.from_numpy(np.stack(obs_history))  # .to(self.device)

        # 2. Prepare transition-aligned tensors (no algorithm-specific padding here).
        acts_t = torch.tensor(sequence.action_history, dtype=torch.float16)
        rews_t = torch.tensor(sequence.rewards, dtype=torch.float32)

        if sequence.policy_history:
            pols_t = (
                torch.stack(
                    [
                        p.detach() if torch.is_tensor(p) else torch.as_tensor(p)
                        for p in sequence.policy_history
                    ]
                )
                .cpu()
                .float()
            )
        else:
            pols_t = torch.empty((0, self.num_actions), dtype=torch.float32)

        # Values
        vals_t = torch.tensor(sequence.value_history, dtype=torch.float32)

        # To Plays
        tps_t = torch.tensor(
            [
                (
                    self._resolve_player_id(sequence.player_id_history[i])
                    if i < len(sequence.player_id_history)
                    else 0
                )
                for i in range(n_states)
            ],
            dtype=torch.int16,
        )

        # Chances
        chance_t = torch.tensor(
            [
                sequence.chance_history[i] if i < len(sequence.chance_history) else 0
                for i in range(n_states)
            ],
            dtype=torch.int16,
        ).unsqueeze(1)

        # Legal Moves Mask
        legal_masks_t = torch.stack(
            [
                legal_moves_mask(
                    self.num_actions,
                    (
                        sequence.legal_moves_history[i]
                        if i < len(sequence.legal_moves_history)
                        else []
                    ),
                )
                for i in range(n_states)
            ]
        )

        # Episode end signals per state
        terminated_t = torch.tensor(sequence.terminated_history, dtype=torch.bool)
        truncated_t = torch.tensor(sequence.truncated_history, dtype=torch.bool)
        dones_t = torch.tensor(sequence.done_history, dtype=torch.bool)

        return {
            "observations": obs_tensor,
            "actions": acts_t,
            "rewards": rews_t,
            "policies": pols_t,
            "values": vals_t,
            "to_plays": tps_t,
            "chances": chance_t,
            "terminated": terminated_t,
            "truncated": truncated_t,
            "dones": dones_t,
            "legal_masks": legal_masks_t,
            "n_states": n_states,
        }


class ObservationCompressionProcessor(InputProcessor):
    """
    Compresses and/or quantizes observations before storage.
    Applied during process_sequence or process_single.

    Supports:
    - Quantization: float32 -> float16 (50% reduction)
    - Compression: zlib or lz4 (additional ~70% reduction)
    """

    def __init__(self, quantization=None, compression=None, max_compressed_length=None):
        self.quantization = quantization
        self.compression = compression
        self.max_compressed_length = max_compressed_length

        if compression == "lz4":
            try:
                import lz4.frame as _lz4_frame

                self._lz4 = _lz4_frame
            except ImportError:
                raise ImportError(
                    "lz4 compression requires 'lz4' package. "
                    "Install with: pip install lz4"
                )
        elif compression not in (None, "zlib"):
            raise ValueError(
                f"Unsupported compression: {compression}. Use None, 'zlib', or 'lz4'"
            )

        if quantization not in (None, "float16"):
            raise ValueError(
                f"Unsupported quantization: {quantization}. Use None or 'float16'"
            )

        if compression and max_compressed_length is None:
            warning(
                "Compression is enabled but max_compressed_length is not set. "
                "process_single will fail to pad arrays to fit pre-allocated buffer sizes. "
                "Set max_compressed_length to match your BufferConfig.shape for observations."
            )

    def process_single(self, **kwargs):
        """Compresses a single step observation into a padded 1D uint8 tensor."""
        if "observations" not in kwargs:
            return kwargs

        obs = kwargs["observations"]
        # Ensure it's a tensor before processing
        if not torch.is_tensor(obs):
            obs_tensor = torch.from_numpy(np.array(obs))
        else:
            obs_tensor = obs.clone()

        if self.quantization == "float16":
            obs_tensor = obs_tensor.to(torch.float16)

        if self.compression:
            obs_bytes = obs_tensor.cpu().numpy().tobytes()

            if self.compression == "zlib":
                import zlib

                compressed = zlib.compress(obs_bytes, level=1)
            else:
                compressed = self._lz4.compress(obs_bytes, compression_level=0)

            # Prepend the 4-byte size header
            size_bytes = len(compressed).to_bytes(4, byteorder="little")
            compressed_data = size_bytes + compressed

            # Pad to max_compressed_length to fit in pre-allocated ReplayBuffer
            if self.max_compressed_length is not None:
                pad_len = self.max_compressed_length - len(compressed_data)
                if pad_len < 0:
                    raise ValueError(
                        f"Compressed observation length ({len(compressed_data)}) "
                        f"exceeds max_compressed_length ({self.max_compressed_length}). "
                        "Increase max_compressed_length in the processor config."
                    )
                compressed_data += b"\x00" * pad_len

            # Return as a 1D uint8 tensor so the replay buffer can store it
            obs_tensor = torch.frombuffer(bytearray(compressed_data), dtype=torch.uint8)

        kwargs["observations"] = obs_tensor
        return kwargs

    def process_sequence(self, sequence, **kwargs):
        """Compresses a sequence of observations into a padded 2D uint8 tensor."""
        if "observations" not in kwargs:
            return kwargs

        obs_tensor = kwargs["observations"]

        if self.quantization == "float16":
            obs_tensor = obs_tensor.to(torch.float16)

        if self.compression:
            obs_tensor = self._compress_observations(obs_tensor)

        kwargs["observations"] = obs_tensor
        return kwargs

    def _compress_observations(self, obs_tensor):
        n_states = obs_tensor.shape[0]
        compressed_list = []

        for i in range(n_states):
            obs = obs_tensor[i]
            obs_bytes = obs.numpy().tobytes()

            if self.compression == "zlib":
                import zlib

                compressed = zlib.compress(obs_bytes, level=1)
            else:
                compressed = self._lz4.compress(obs_bytes, compression_level=0)

            size_bytes = len(compressed).to_bytes(4, byteorder="little")
            compressed_list.append(size_bytes + compressed)

        # If storing sequences, we pad to either the local max of the sequence
        # or the global max_compressed_length
        target_len = (
            self.max_compressed_length
            if self.max_compressed_length
            else max(len(c) for c in compressed_list)
        )

        result = torch.zeros((n_states, target_len), dtype=torch.uint8)
        for i, c in enumerate(compressed_list):
            if len(c) > target_len:
                raise ValueError(
                    f"Compressed observation length ({len(c)}) "
                    f"exceeds max_compressed_length ({target_len}). "
                    "Increase max_compressed_length in the processor config."
                )
            result[i, : len(c)] = torch.frombuffer(bytearray(c), dtype=torch.uint8)

        return result


class LazyDecompressedBuffer:
    """
    A buffer-like wrapper that decompresses observations on-demand.
    Supports tensor indexing like the original buffer.
    """

    def __init__(self, compressed_buffer, decompressor, obs_shape, obs_dtype, device):
        self.compressed_buffer = compressed_buffer
        self.decompressor = decompressor
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self._device = device
        self._cache = {}

    @property
    def device(self):
        return self._device

    def __getitem__(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist() if indices.dim() > 0 else [indices.item()]
        elif isinstance(indices, int):
            indices = [indices]

        # Freeze the indices tuple to use as a cache key
        cache_key = tuple(indices)
        if cache_key in self._cache:
            return self._cache[cache_key]

        batch_size = len(indices)
        result = torch.zeros(
            (batch_size, *self.obs_shape), dtype=self.obs_dtype, device=self._device
        )

        for i, idx in enumerate(indices):
            compressed_data = self.compressed_buffer[idx]

            # Read the 4-byte header to get the size
            size_bytes = bytes(compressed_data[:4].tolist())
            size = int.from_bytes(size_bytes, byteorder="little")

            # Check if empty based on size, NOT by filtering > 0
            # (since valid compressed data often contains 0x00 bytes)
            if size == 0:
                continue

            compressed = bytes(compressed_data[4 : 4 + size].tolist())

            if self.decompressor.compression == "zlib":
                import zlib

                decompressed = zlib.decompress(compressed)
            else:
                decompressed = self.decompressor._lz4.decompress(compressed)

            dtype = (
                torch.float16
                if self.decompressor.quantization == "float16"
                else self.obs_dtype
            )

            obs = torch.frombuffer(bytearray(decompressed), dtype=dtype).reshape(
                self.obs_shape
            )

            if self.decompressor.quantization == "float16":
                obs = obs.to(self.obs_dtype)

            result[i] = obs

        self._cache[cache_key] = result
        return result


class ObservationDecompressionProcessor(OutputProcessor):
    """
    Decompresses and/or dequantizes observations during batch sampling.
    Wraps an inner OutputProcessor (e.g., NStepUnrollProcessor).
    """

    def __init__(
        self,
        inner_processor,
        quantization=None,
        compression=None,
        obs_shape=None,
        obs_dtype=torch.float32,
    ):
        self.inner_processor = inner_processor
        self.quantization = quantization
        self.compression = compression
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype

        if compression == "lz4":
            try:
                import lz4.frame as _lz4_frame

                self._lz4 = _lz4_frame
            except ImportError:
                raise ImportError(
                    "lz4 compression requires 'lz4' package. "
                    "Install with: pip install lz4"
                )
        elif compression not in (None, "zlib"):
            raise ValueError(
                f"Unsupported compression: {compression}. Use None, 'zlib', or 'lz4'"
            )

    def process_batch(
        self, indices: List[int], buffers: Dict[str, torch.Tensor], **kwargs
    ):
        decompressed_buffers = dict(buffers)

        if self.compression or self.quantization:
            device = buffers["observations"].device
            decompressed_buffers["observations"] = LazyDecompressedBuffer(
                compressed_buffer=buffers["observations"],
                decompressor=self,
                obs_shape=self.obs_shape,
                obs_dtype=self.obs_dtype,
                device=device,
            )

        return self.inner_processor.process_batch(
            indices, decompressed_buffers, **kwargs
        )

    def clear(self):
        self.inner_processor.clear()


class GAEProcessor(InputProcessor):
    """
    Computes Generalized Advantage Estimation (GAE) at trajectory end.
    Passes through single steps unchanged, then finalizes with GAE calculation.
    """

    def __init__(self, gamma, gae_lambda):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def process_single(self, *args, **kwargs):
        return kwargs

    def process_sequence(self, sequence, **kwargs):
        """
        Computes GAE over a full sequence of transitions.
        """
        transitions = kwargs.get("transitions")
        if transitions is None:
            transitions = self._sequence_to_transitions(sequence)

        if transitions is None or len(transitions) == 0:
            return {"transitions": []}

        # Extract rewards and values
        rewards_np = np.array(
            [t.get("rewards", 0.0) for t in transitions], dtype=np.float32
        )
        values_np = np.array(
            [t.get("values", 0.0) for t in transitions], dtype=np.float32
        )

        # GAE requires a bootstrap value for the final next state.
        # 1. Use external bootstrap from kwargs (RolloutActor force-flush)
        # 2. Fallback to sequence history (if it's n+1 long - i.e. finished trajectory)
        last_value = kwargs.get("last_value", 0.0)

        if (
            sequence is not None
            and hasattr(sequence, "value_history")
            and len(sequence.value_history) > len(transitions)
        ):
            # If the sequence has its own final value (e.g. from end of trajectory), use it.
            # (In PPO we usually don't rely on history for chunk bootstrap, but it's safe)
            last_value = sequence.value_history[-1]

        dones_np = np.array([t.get("dones", False) for t in transitions], dtype=bool)

        rewards_pad = np.append(rewards_np, last_value)
        values_pad = np.append(values_np, last_value)

        # Vectorized GAE
        deltas = (
            rewards_pad[:-1]
            + self.gamma * values_pad[1:] * (~dones_np)
            - values_pad[:-1]
        )

        advantages = discounted_cumulative_sums(
            deltas, self.gamma * self.gae_lambda
        ).copy()
        returns = discounted_cumulative_sums(rewards_pad, self.gamma).copy()[:-1]

        adv_list = advantages.tolist()
        ret_list = returns.tolist()

        for t, adv, ret in zip(transitions, adv_list, ret_list):
            t["advantages"] = adv
            t["returns"] = ret

            # Extract scalar log_prob for PPO
            if t.get("policy") is not None:
                pol = t["policy"]
                if np.isscalar(pol) or (hasattr(pol, "ndim") and pol.ndim == 0):
                    t["log_prob"] = float(pol)
                elif isinstance(pol, (np.ndarray, torch.Tensor)) and pol.ndim == 1:
                    # Fallback: if it's a distribution and we have the action, compute log_prob
                    # Assuming policy is probs
                    action = t.get("actions")
                    if action is not None:
                        idx = int(action)
                        # Avoid log(0)
                        prob = float(pol[idx])
                        t["log_prob"] = float(np.log(max(prob, 1e-10)))
            elif t.get("policies") is not None:
                # Handle plural naming consistency
                pol = t["policies"]
                if np.isscalar(pol) or (hasattr(pol, "ndim") and pol.ndim == 0):
                    t["log_prob"] = float(pol)

        return {"transitions": transitions}

    def finish_trajectory(self, buffers, trajectory_slice, last_value=0):
        """
        Compute GAE advantages and returns for a trajectory segment in old single-step mode.
        """
        rewards = torch.cat(
            (
                buffers["rewards"][trajectory_slice],
                torch.tensor([last_value], dtype=torch.float32),
            )
        )
        values = torch.cat(
            (
                buffers["values"][trajectory_slice],
                torch.tensor([last_value], dtype=torch.float32),
            )
        )

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        deltas_np = deltas.detach().cpu().numpy()
        rewards_np = rewards.detach().cpu().numpy()

        advantages = torch.from_numpy(
            discounted_cumulative_sums(deltas_np, self.gamma * self.gae_lambda).copy()
        ).to(torch.float32)
        returns = torch.from_numpy(
            discounted_cumulative_sums(rewards_np, self.gamma).copy()
        ).to(torch.float32)[:-1]

        return {"advantages": advantages, "returns": returns}


# ==========================================
# Output Processors
# ==========================================


class StandardOutputProcessor(OutputProcessor):
    """Returns data indices directly."""

    def process_batch(
        self, indices: List[int], buffers: Dict[str, torch.Tensor], **kwargs
    ):
        return {key: buf[indices] for key, buf in buffers.items()}


class NStepUnrollProcessor(OutputProcessor):
    """
    Handles window unrolling, validity masking, and N-step target calculation.
    Samples indices and creates unrolled sequences with N-step value bootstrapping.
    """

    def __init__(
        self,
        unroll_steps,
        n_step,
        gamma,
        num_actions,
        num_players,
        max_size,
        lstm_horizon_len=10,
        value_prefix=False,
        tau=0.3,
    ):
        self.unroll_steps = unroll_steps
        self.n_step = n_step
        self.gamma = gamma
        self.num_actions = num_actions
        self.num_players = num_players
        self.max_size = max_size
        self.lstm_horizon_len = lstm_horizon_len
        self.value_prefix = value_prefix
        self.tau = tau

    def process_batch(
        self, indices: List[int], buffers: Dict[str, torch.Tensor], **kwargs
    ):
        # buffers dict should contain: obs, rew, val, pol, act, to_play, chance, game_id, legal_mask, training_step

        device = buffers["observations"].device
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        batch_size = len(indices)

        # 1. Define Window
        lookahead_window = self.unroll_steps + self.n_step
        offsets = torch.arange(
            0, lookahead_window + 1, dtype=torch.long, device=device
        ).unsqueeze(0)
        all_indices = (indices_tensor.unsqueeze(1) + offsets) % self.max_size

        # 2. Fetch Raw Data
        raw_rewards = buffers["rewards"][all_indices]
        raw_values = buffers["values"][all_indices]
        raw_policies = buffers["policies"][all_indices]
        raw_actions = buffers["actions"][all_indices]
        raw_to_plays = buffers["to_plays"][all_indices]
        raw_chances = buffers["chances"][all_indices]
        raw_game_ids = buffers["game_ids"][all_indices]
        raw_legal_masks = buffers["legal_masks"][all_indices]
        raw_terminated = (
            buffers["terminated"][all_indices]
            if "terminated" in buffers
            else buffers["dones"][all_indices]
        )
        raw_truncated = (
            buffers["truncated"][all_indices]
            if "truncated" in buffers
            else torch.zeros_like(raw_terminated)
        )
        raw_dones = raw_terminated | raw_truncated

        # 3. Validity Masks
        base_game_ids = raw_game_ids[:, 0].unsqueeze(1)
        same_game = raw_game_ids == base_game_ids

        # Calculate episode boundaries using dones (terminated/truncated)
        # We mask out any steps that occur AFTER a done signal in the sequence
        # cumsum gives us a mask of [0, 0, 1, 1, 1] if done happens at index 2
        cumulative_dones = torch.cumsum(raw_dones.float(), dim=1)

        # We want to mask steps *after* the done, not the done itself (which is a valid terminal state)
        # Shift cumsum right by 1: [0, 0, 0, 1, 1]
        post_done_mask = (
            torch.cat(
                [torch.zeros((batch_size, 1), device=device), cumulative_dones[:, :-1]],
                dim=1,
            )
            > 0
        )

        # Obs/Value Mask: Valid states (including terminal states), consistent with game ID and episode boundary
        obs_mask = same_game & (~post_done_mask)

        # Dynamics/Policy Mask: Valid transitions (excluding terminal states)
        # We cannot predict next state or policy FROM a terminal state
        dynamics_mask = obs_mask & (~raw_dones)

        # 5. Compute N-Step Targets
        target_values, target_rewards = self._compute_n_step_targets(
            batch_size,
            raw_rewards,
            raw_values,
            raw_to_plays,
            raw_terminated,
            raw_truncated,
            dynamics_mask,
            device,
        )

        # 6. Prepare Unroll Targets
        target_policies = torch.zeros(
            (batch_size, self.unroll_steps + 1, self.num_actions),
            dtype=torch.float32,
            device=device,
        )
        target_actions = torch.zeros(
            (batch_size, self.unroll_steps), dtype=torch.int64, device=device
        )
        target_to_plays = torch.zeros(
            (batch_size, self.unroll_steps + 1, self.num_players),
            dtype=torch.float32,
            device=device,
        )
        target_chances = torch.zeros(
            (batch_size, self.unroll_steps + 1, 1), dtype=torch.int64, device=device
        )
        target_dones = torch.ones(
            (batch_size, self.unroll_steps + 1), dtype=torch.bool, device=device
        )

        for u in range(self.unroll_steps + 1):
            is_consistent = dynamics_mask[:, u]

            target_policies[is_consistent, u] = raw_policies[is_consistent, u]
            target_policies[~is_consistent, u] = 1.0 / self.num_actions

            # Pass the clean raw integer value directly
            tp_indices = torch.clamp(raw_to_plays[:, u].long(), 0, self.num_players - 1)
            target_to_plays[range(batch_size), u, tp_indices] = 1.0
            target_to_plays[~is_consistent, u] = 0

            target_dones[is_consistent, u] = raw_dones[is_consistent, u]
            target_dones[~is_consistent, u] = True

            target_chances[is_consistent, u, 0] = (
                raw_chances[is_consistent, u].squeeze(-1).long()
            )

            if u < self.unroll_steps:
                target_actions[:, u] = raw_actions[:, u].long()
                target_actions[~is_consistent, u] = int(
                    np.random.randint(0, self.num_actions)
                )

        # 7. Unroll Observations
        obs_indices = all_indices[:, : self.unroll_steps + 1]
        # Valid observations include terminal states
        obs_valid_mask = obs_mask[:, : self.unroll_steps + 1]

        unroll_observations = buffers["observations"][obs_indices].clone()

        for step in range(1, self.unroll_steps + 1):
            is_absorbing = ~obs_valid_mask[:, step]
            if is_absorbing.any():
                unroll_observations[is_absorbing, step] = unroll_observations[
                    is_absorbing, step - 1
                ]

        # 8. Chance Encoder Inputs
        # Stack current and next observations along the channel dimension (dim=2)
        # resulting in [B, T, 2*C, H, W]
        chance_encoder_inputs = None
        if self.unroll_steps > 0:
            chance_encoder_inputs = torch.cat(
                [unroll_observations[:, :-1], unroll_observations[:, 1:]], dim=2
            )

        return dict(
            observations=buffers["observations"][indices_tensor],
            unroll_observations=unroll_observations,
            chance_encoder_inputs=chance_encoder_inputs,
            has_valid_obs_mask=obs_valid_mask,
            has_valid_action_mask=dynamics_mask[:, : self.unroll_steps + 1],
            rewards=target_rewards,
            policies=target_policies,
            values=target_values,
            actions=target_actions,
            to_plays=target_to_plays,
            chance_codes=target_chances,
            dones=target_dones,
            is_same_game=same_game[:, : self.unroll_steps + 1],
            ids=buffers["ids"][indices_tensor].clone(),
            legal_moves_masks=buffers["legal_masks"][indices_tensor],
            indices=indices,
            training_steps=buffers["training_steps"][indices_tensor],
        )

    def _compute_n_step_targets(
        self,
        batch_size,
        raw_rewards,
        raw_values,
        raw_to_plays,
        raw_terminated,
        raw_truncated,
        valid_mask,
        device,
    ):
        """
        Vectorized N-step target calculation.
        """
        # 1. Setup Dimensions and Windows
        num_windows = self.unroll_steps + 1
        n_step = self.n_step
        gamma = self.gamma

        # Pre-compute gamma vector: [1, g, g^2, ..., g^(n-1)]
        # Shape: [1, 1, n_step]
        gammas = (
            gamma ** torch.arange(n_step, dtype=torch.float32, device=device)
        ).reshape(1, 1, n_step)

        # 2. Slice/Pad inputs to ensure we don't go out of bounds for the unfold
        # The 'raw' tensors typically have length >= num_windows + n_step
        # We need to create 'num_windows' windows of size 'n_step'.
        # The length required for this is: num_windows + n_step - 1.

        required_len = num_windows + n_step - 1
        raw_dones = raw_terminated | raw_truncated

        # Helper to pad if needed
        def safe_slice_unfold(tensor, length, size, step):
            if tensor.shape[1] < length:
                pad_len = length - tensor.shape[1]
                # Pad last dim (time)
                # pad format: (pad_left, pad_right, pad_top, pad_bottom...)
                # we want to pad dim 1. 2D tensor (B, L) -> (0, pad_len)
                # 3D tensor (B, L, C) -> (0, 0, 0, pad_len) ?? No, F.pad works on last dim first.
                if tensor.dim() == 2:
                    tensor = torch.nn.functional.pad(tensor, (0, pad_len))
                elif tensor.dim() == 3:
                    # (B, L, C) -> pad L means pad dim 1.
                    # F.pad(dHW, dC, dL, dB) order?
                    # "Padding size: The padding size by which to pad some dimensions of input are described starting from the last dimension and moving forward."
                    # For (B, L, C): last is C (pad 0,0). Next is L (pad 0, pad_len).
                    tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_len))

            return tensor[:, :length].unfold(1, size, step)

        # [B, num_windows, n_step]
        rewards_windows = safe_slice_unfold(raw_rewards, required_len, n_step, 1)
        to_plays_windows = safe_slice_unfold(raw_to_plays, required_len, n_step, 1)
        dones_windows = safe_slice_unfold(raw_dones, required_len, n_step, 1)
        valid_windows = safe_slice_unfold(valid_mask, required_len, n_step, 1)

        # Current ToPlay for each window start (u)
        # We need raw_to_plays to be long enough for simple indexing too
        if raw_to_plays.shape[1] < num_windows:
            raw_to_plays = torch.nn.functional.pad(
                raw_to_plays, (0, num_windows - raw_to_plays.shape[1])
            )

        # Shape: [B, num_windows, 1]
        current_to_plays = raw_to_plays[:, :num_windows].unsqueeze(2)

        # 3. Compute Value Masks
        # Game Boundary Mask (from valid_mask) & Done Mask
        # We need to compute a mask that handles the "break" when a done occurs.
        # Original logic:
        #   valid = valid_mask & (~has_ended)
        #   has_ended |= done & valid

        # This implies:
        # - If done[k] is True, then mask[k] is VALID (we count the terminal reward).
        # - But mask[k+1] and onwards are INVALID.

        dones_float = dones_windows.float()
        # cumsum gives [0, 0, 1, 1] for dones [0, 0, 1, 0]
        # We want to forbid steps AFTER the first done.
        # Shift cumsum right by 1 to get "was done before this step?"
        was_done_before = torch.cat(
            [
                torch.zeros((batch_size, num_windows, 1), device=device),
                torch.cumsum(dones_float, dim=2)[:, :, :-1],
            ],
            dim=2,
        )

        # Check if "was done before" > 0 or if "was done before" depends on valid mask?
        # Typically simple cumsum is enough if we assume raw_dones are correct.

        # Valid Reward Steps:
        # 1. Original valid_mask is True (same game)
        # 2. No done happened BEFORE this step in the window
        valid_steps_mask = valid_windows & (was_done_before == 0)

        # 4. Compute Discounted Rewards
        # Player Sign: compared to current_to_play
        # [B, W, N]
        signs = torch.where(current_to_plays == to_plays_windows, 1.0, -1.0)

        # Weighted Rewards
        # sum(gamma^k * R_k * sign_k * valid_k)

        weighted_rewards = rewards_windows * gammas * signs * valid_steps_mask.float()

        # [B, num_windows]
        summed_rewards = weighted_rewards.sum(dim=2)

        # 5. Compute Bootstrap Value
        # Value at u + n_step (the boot_idx)
        # Boot indices: u + n_step
        boot_indices = torch.arange(n_step, n_step + num_windows, device=device)
        # Clamp to safeguard against OOB (though logic should prevent using it)
        safe_boot_indices = torch.clamp(boot_indices, max=raw_values.shape[1] - 1)

        # [B, num_windows]
        boot_values = raw_values[:, safe_boot_indices]
        boot_to_plays = raw_to_plays[:, safe_boot_indices]

        # Bootstrap Validity:
        # 1. Boot index must be within valid range (num_windows + n_step < max_size ideally)
        # 2. boot index valid_mask must be true
        # 3. Game must NOT have ended inside the n_step window.

        # Check if any done occurred in the valid part of the window
        # If valid_steps_mask has valid done, it's fine for reward, but kills bootstrap.
        # If any 'done' & 'valid' occurred in window -> no bootstrap.
        # We can sum up (dones_windows & valid_steps_mask).
        # If > 0, then we hit a done.
        terminated_windows = safe_slice_unfold(raw_terminated, required_len, n_step, 1)
        hit_terminated_in_window = (
            terminated_windows.float() * valid_steps_mask.float()
        ).sum(dim=2) > 0

        boot_is_valid = (
            valid_mask[:, safe_boot_indices]
            & (~hit_terminated_in_window)
            & (~raw_terminated[:, safe_boot_indices])
        )

        # Boot Sign
        boot_signs = torch.where(
            current_to_plays.squeeze(2) == boot_to_plays, 1.0, -1.0
        )

        boot_term = (gamma**n_step) * boot_values * boot_signs

        # Final Target Values
        target_values = summed_rewards + torch.where(
            boot_is_valid, boot_term, torch.tensor(0.0, device=device)
        )

        # Grounding: Ensure that any targets for padded steps (past game end) are explicitly 0.0
        # This is important for "safe absorbing states" logic where we want to learn V(s_absorbing) = 0
        target_values = target_values * valid_mask[:, :num_windows].float()

        # 6. Target Rewards (Value Prefix or Instant)
        target_rewards = torch.zeros(
            (batch_size, num_windows), dtype=torch.float32, device=device
        )

        if self.value_prefix and self.lstm_horizon_len > 0:
            # Reimplement prefix accumulation loop (fast enough for small unroll_steps)
            prefix_sum = torch.zeros(batch_size, device=device)
            horizon_id = 0

            for u in range(1, num_windows):
                if horizon_id % self.lstm_horizon_len == 0:
                    prefix_sum = torch.zeros(batch_size, device=device)
                horizon_id += 1

                # Reward at u-1
                if (u - 1) < raw_rewards.shape[1]:
                    # We assume raw_rewards are valid for the prefix calculation
                    # (ignoring valid_mask logic here as per original implementation appearance
                    # or assuming consistency)
                    prefix_sum = prefix_sum + raw_rewards[:, u - 1]

                # Check if the TRANSITION to this node is valid (state u-1 must be non-terminal)
                is_valid = valid_mask[:, u - 1]
                target_rewards[is_valid, u] = prefix_sum[is_valid]
        else:
            # Standard: target_rewards[u] = raw_rewards[u-1]
            # The reward at step u is for the transition FROM state u-1,
            # so validity is determined by dynamics_mask at position u-1.
            t_rew = raw_rewards[:, : num_windows - 1]
            mask_slice = valid_mask[:, : num_windows - 1]  # positions 0..K-1

            # We assign to target_rewards[:, 1:]
            # But we must respect the mask
            # Safe way: fill zero, then assign masked

            # Slice match: [B, num_windows-1]
            target_slice = torch.zeros_like(target_rewards[:, 1:])
            target_slice[mask_slice] = t_rew[mask_slice]
            target_rewards[:, 1:] = target_slice

        return target_values, target_rewards


class AdvantageNormalizer(OutputProcessor):
    """
    Normalizes advantages and formats batches for policy gradient methods.
    """

    def process_batch(
        self, indices: List[int], buffers: Dict[str, torch.Tensor], **kwargs
    ):
        # In PPO we usually sample the whole filled rollout and then minibatch in the learner.
        sl = slice(None) if indices is None else indices

        advantages = buffers["advantages"][sl].to(torch.float32)
        advantage_mean = advantages.mean()
        advantage_std = advantages.std()
        normalized_advantages = (advantages - advantage_mean) / (advantage_std + 1e-10)

        return dict(
            observations=buffers["observations"][sl],
            actions=buffers["actions"][sl],
            advantages=normalized_advantages,
            returns=buffers["returns"][sl],
            log_prob=buffers["log_prob"][sl],
            legal_moves_masks=buffers["legal_moves_masks"][sl],
        )
