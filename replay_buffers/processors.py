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
        # Fail-fast: Ensure that if an actor flushes an unfinished chunk, they provide a bootstrap_value.
        # Failing to do so will result in GAE calculating a future return of 0.0, destroying the value network.
        if (
            sequence is not None
            and hasattr(sequence, "terminated_history")
            and len(sequence.terminated_history) > 0
        ):
            if (
                not sequence.terminated_history[-1]
                and not sequence.truncated_history[-1]
            ):
                assert (
                    "bootstrap_value" in kwargs or "last_value" in kwargs
                ), f"[Fail-Fast] Unfinished sequence (len={len(sequence)}) flushed without a bootstrap_value!"

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
    Ensures 'terminated'/'truncated' and 'dones' flags are present for transition pipelines.
    Computes an authoritative 'done' if missing.
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
        # 1. Read explicitly provided flags
        terminated = bool(kwargs.get(self.terminated_key))
        truncated = bool(kwargs.get(self.truncated_key))

        assert (
            terminated is not None and truncated is not None
        ), "terminated and truncated flags must be present"

        # 2. Compute authoritative done
        is_done = terminated or truncated

        # 3. Explicitly inject ALL keys back into kwargs
        kwargs[self.terminated_key] = terminated
        kwargs[self.truncated_key] = truncated
        kwargs[self.done_key] = is_done

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

        # Corrected N-Step Summation (Forward)
        for i, transition in enumerate(list(buffer)):
            r = transition.get(self.reward_key, 0.0)
            d = transition.get(self.done_key, False)

            # Apply discount based on step index (gamma^i * r_i)
            final_reward += (self.gamma**i) * r

            # If a step was terminal, it cuts the n-step chain
            if d:
                final_next_obs = transition.get("next_observations")
                final_next_info = transition.get("next_infos")
                final_next_legal_moves = transition.get("next_legal_moves")
                final_done = True
                final_terminated = transition.get(self.terminated_key, True)
                final_truncated = transition.get(self.truncated_key, False)
                break

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
    def __init__(self, gamma, gae_lambda):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def process_single(self, *args, **kwargs):
        # We only compute GAE on full sequences/chunks
        return kwargs

    def process_sequence(self, sequence, **kwargs):
        # 1. Fail-Fast Boundary Check
        # If the episode hasn't ended, the actor MUST provide the bootstrap value.
        is_terminal = bool(sequence.terminated_history[-1]) or bool(
            sequence.done_history[-1]
        )

        if not is_terminal:
            assert (
                "bootstrap_value" in kwargs
            ), "[Fail-Fast] Unfinished PPO chunk requires a bootstrap_value in kwargs!"
            last_val = kwargs["bootstrap_value"]
        else:
            last_val = 0.0

        # 2. Extract arrays from the sequence object
        rewards = np.array(sequence.rewards, dtype=np.float32)
        values = np.array(sequence.value_history, dtype=np.float32)

        # 4. Calculate GAE and Returns using pure math from functional/advantages.py
        from agents.learner.functional.advantages import compute_gae

        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            bootstrap_value=last_val,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # 7. Package the flattened transitions for the Replay Buffer
        processed_transitions = []
        for i in range(len(rewards)):
            processed_transitions.append(
                {
                    "observations": sequence.observation_history[i],
                    "actions": sequence.action_history[i],
                    "log_prob": sequence.log_prob_history[i],  # Unified singular key
                    "values": values[i],  # Old values needed for Value Clipping
                    "advantages": advantages[i],
                    "returns": returns[i],  # Target for Value Loss
                }
            )

        return {"transitions": processed_transitions}


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

        # Obs/Value Mask: Valid states (including terminal states), consistent with game ID.
        # We allow post-terminal steps for regression targets (EfficientZero)
        # constrained only by game ID boundaries.
        obs_mask = same_game

        # Dynamics/Policy Mask: Valid transitions (excluding terminal states and post-terminal)
        dynamics_mask = same_game & (~post_done_mask) & (~raw_dones)

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

            # Defensive: even if 'consistent' by same_game/done logic, if the policy is missing (all zeros),
            raw_p = raw_policies[:, u]
            target_policies[is_consistent, u] = raw_p[is_consistent]
            # No fallback; let it be zero if raw is zero to surface errors

            # To_play targets follow a specific contract:
            # - Masked at root (u=0)
            # - Masked post-terminal (post_done_mask)
            # - Included at terminal
            is_consistent_tp = (u > 0) & (~post_done_mask[:, u]) & same_game[:, u]

            tp_indices = torch.clamp(raw_to_plays[:, u].long(), 0, self.num_players - 1)
            target_to_plays[range(batch_size), u, tp_indices] = 1.0
            target_to_plays[~is_consistent_tp, u] = 0

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

        # Contract: Initial state (root) should always have a terminal flag of False
        # to correctly signal state/mask resets in recurrent unrolls.
        assert not target_dones[
            :, 0
        ].all(), "Initial state (root) should always have a terminal flag of False"

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
        Delegates to pure math in functional/returns.py
        """
        from agents.learner.functional.returns import compute_unrolled_n_step_targets

        return compute_unrolled_n_step_targets(
            raw_rewards=raw_rewards,
            raw_values=raw_values,
            raw_to_plays=raw_to_plays,
            raw_terminated=raw_terminated | raw_truncated,
            valid_mask=valid_mask,
            gamma=self.gamma,
            n_step=self.n_step,
            unroll_steps=self.unroll_steps,
            lstm_horizon_len=self.lstm_horizon_len,
            value_prefix=self.value_prefix,
        )


# TODO: shoud we remove this?
class PPOBatchProcessor(OutputProcessor):
    """
    Formats batches for policy gradient methods.
    Does NOT normalize advantages (handled at the mini-batch iterator level).
    """

    def process_batch(
        self, indices: List[int], buffers: Dict[str, torch.Tensor], **kwargs
    ):
        # In PPO we usually sample the whole filled rollout and then minibatch in the learner.
        sl = slice(None) if indices is None else indices

        return dict(
            observations=buffers["observations"][sl],
            actions=buffers["actions"][sl],
            values=buffers["values"][sl],
            advantages=buffers["advantages"][sl],
            returns=buffers["returns"][sl],
            log_prob=buffers["log_prob"][sl],
            legal_moves_masks=buffers["legal_moves_masks"][sl],
        )
