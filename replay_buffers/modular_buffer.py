from logging import warning
import torch
import numpy as np
import torch.multiprocessing as mp
from dataclasses import dataclass
from typing import List, Optional, Any

from replay_buffers.processors import IdentityInputProcessor, StandardOutputProcessor
from replay_buffers.writers import CircularWriter
from replay_buffers.samplers import UniformSampler
from replay_buffers.concurrency import ConcurrencyBackend, LocalBackend
from utils.utils import numpy_dtype_to_torch_dtype


@dataclass
class BufferConfig:
    name: str
    shape: tuple
    dtype: torch.dtype
    is_shared: bool = False

    # Optional: fill value for initialization
    fill_value: Any = 0


class ModularReplayBuffer:
    def __init__(
        self,
        max_size: int,
        buffer_configs: List[BufferConfig],
        batch_size: int = 32,
        writer=None,
        sampler=None,
        input_processor=None,
        output_processor=None,
        backend: Optional[ConcurrencyBackend] = None,
        # For mapping tuple outputs from legacy input processors to buffer names
    ):
        self.max_size: int = max_size
        self.batch_size: int = batch_size if batch_size is not None else max_size

        self.buffer_configs = buffer_configs
        self.backend = backend if backend is not None else LocalBackend()

        # 2. Initialize Buffers dynamically
        self.buffers = {}
        for config in buffer_configs:
            self._create_buffer(config)

        # 3. Synchronization (using backend)
        self.is_shared = self.backend.is_shared
        self.write_lock = self.backend.create_lock()
        self.priority_lock = self.backend.create_lock()

        # MuZero specific counters (only initialized if shared context is detected/needed)
        # You could also make this a specific config option
        if self.is_shared:
            self._next_id = self.backend.create_tensor(
                (1,), dtype=torch.int64, fill_value=0
            )
            self._next_game_id = self.backend.create_tensor(
                (1,), dtype=torch.int64, fill_value=0
            )

        self.sampler = sampler if sampler is not None else UniformSampler()
        self.writer = writer if writer is not None else CircularWriter(max_size)
        self.input_processor = (
            input_processor if input_processor else IdentityInputProcessor()
        )
        self.output_processor = (
            output_processor if output_processor else StandardOutputProcessor()
        )
        print("Max size:", max_size)

        self.clear()
        assert self.size == 0, "Replay buffer should be empty at initialization"
        assert self.max_size > 0, "Replay buffer should have a maximum size"
        assert self.batch_size > 0, "Replay buffer batch size should be greater than 0"

    def _create_buffer(self, config: BufferConfig):
        """Creates a single tensor buffer based on config."""
        final_shape = (self.max_size,) + config.shape

        # Handle numpy to torch conversion if necessary
        dtype = config.dtype
        if not isinstance(dtype, torch.dtype):
            dtype = numpy_dtype_to_torch_dtype(dtype)

        # We ignore config.is_shared if we have a backend that handles it
        # but for compatibility, we use the backend to decide if it's shared
        if self.backend.is_shared:
            tensor = self.backend.create_tensor(
                final_shape, dtype=dtype, fill_value=config.fill_value
            )
        else:
            tensor = torch.full(final_shape, config.fill_value, dtype=dtype)

        self.buffers[config.name] = tensor

    def store(self, **kwargs):
        """
        Stores a single transition (DQN, PPO, NFSP style).
        """
        # 1. Process Input
        processed = self.input_processor.process_single(**kwargs)

        if processed is None:
            return None  # Processor indicates accumulation (e.g. N-step)

        # 2. Determine Write Index
        # Locking priority_lock before write_lock ensures the sampler (which also uses priority_lock)
        # sees an atomic update of size and priorities.
        with self.priority_lock:
            with self.write_lock:
                idx = self.writer.store()
                if idx is None:
                    return None

                # 3. Map Processed Data to Buffers
                if isinstance(processed, dict):
                    # Direct mapping (Processor returns dict keys matching buffer names)
                    for key, val in processed.items():
                        # warn if key not in buffers
                        if key in self.buffers:
                            self._write_to_buffer(key, idx, val)
                        else:
                            # warn only once per key
                            if not hasattr(self, "_warned_keys"):
                                self._warned_keys = set()
                            if key not in self._warned_keys:
                                warning(
                                    f"Key '{key}' from input processor not found in buffers."
                                )
                                self._warned_keys.add(key)
                else:
                    raise ValueError(
                        "Input processor must return a dict mapping buffer names to values"
                    )

            # 4. Update Sampler (Priorities)
            priority = kwargs.get("priority", None)

            # We assume tree_pointer logic is handled by the sampler or writer if specific
            # For CircularWriter, the writer doesn't track 'tree_pointer' distinct from 'pointer' usually
            # But specific samplers might need hooks.
            self.sampler.on_store(
                idx,
                priority=priority,
            )

        return idx

    def store_aggregate(self, sequence_object, **kwargs):
        """
        Stores a complete sequence/trajectory (MuZero style).
        Uses process_sequence instead of process_single.
        """
        # 1. Process Sequence
        # Expecting a dictionary of tensors: {'observations': ..., 'actions': ...}
        data = self.input_processor.process_sequence(sequence_object, **kwargs)

        # We need to know how many items to write to reserve space
        # We assume the input processor returns a dict where values are arrays/tensors of equal length
        # or it returns a 'n_states' key (like MuZeroGameInputProcessor)
        n_items = data.get("n_states", len(next(iter(data.values()))))
        priorities = kwargs.get("priorities", [None] * n_items)

        with self.priority_lock:
            with self.write_lock:
                # 2. Reserve Batch Indices
                slices = self.writer.store_batch(n_items)

                # 3. Handle IDs (MuZero specific logic - integrated generally)
                if self.is_shared and "ids" in self.buffers:
                    start_id = int(self._next_id.item())
                    self._next_id[0] = start_id + n_items

                    # Generate IDs on the fly
                    # We put this into the 'data' dict so the loop below handles it generically
                    data["ids"] = torch.arange(
                        start_id + 1, start_id + n_items + 1, dtype=torch.int64
                    )

                if self.is_shared and "game_ids" in self.buffers:
                    start_game_id = int(self._next_game_id.item()) + 1
                    self._next_game_id[0] = start_game_id
                    data["game_ids"] = torch.full(
                        (n_items,), start_game_id, dtype=torch.int64
                    )

                # 4. Write Data to Buffers
                data_offset = 0
                for sl in slices:
                    slice_len = sl.stop - sl.start
                    rng = sl

                    for key, tensor_data in data.items():
                        # Only write if we have a matching buffer
                        if key in self.buffers:
                            # Slice the input data (tensor_data) matching the buffer slice
                            batch_slice = tensor_data[
                                data_offset : data_offset + slice_len
                            ]

                            # Handle Numpy/Torch mismatch
                            if isinstance(batch_slice, np.ndarray):
                                batch_slice = torch.from_numpy(batch_slice)

                            self.buffers[key][rng] = batch_slice

                    data_offset += slice_len

            # 5. Update Priorities
            # Reconstruct indices from slices
            all_indices = []
            for sl in slices:
                all_indices.extend(range(sl.start, sl.stop))

            for i, (idx, p) in enumerate(zip(all_indices, priorities)):
                # TODO: MAKE THIS A SEPERATE PROCESSOR?
                is_terminal = i == n_items - 1
                if is_terminal:
                    # print("Storing terminal with zero priority")
                    self.sampler.on_store(
                        idx, sum_tree_val=0.0, min_tree_val=float("inf")
                    )
                else:
                    self.sampler.on_store(idx, priority=p)

    def _write_to_buffer(self, name, idx, val):
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val)
        self.buffers[name][idx] = val

    def sample(self):
        # 1. Acquire Lock before checking size or sampling
        with self.priority_lock:
            assert (
                self.size >= self.batch_size
            ), f"Not enough items in buffer to sample: {self.size} < {self.batch_size}"

            # 2. Sample Indices
            indices, weights = self.sampler.sample(self.size, self.batch_size)

        # no indices greater than current buffer size:
        # 2. Collect Raw Data
        # We pass self.buffers directly to the output processor
        # The output processor knows which keys it needs

        # 3. Process Batch
        # Output processors expect (indices, buffers)
        batch = self.output_processor.process_batch(indices, self.buffers)

        # 4. Add Sampler Metadata
        if weights is not None:
            batch["weights"] = weights
            batch["indices"] = indices

        return batch

    def update_priorities(self, indices, priorities, ids=None):
        with self.priority_lock:
            if self.is_shared and ids is None:
                warning(
                    "Updating priorities without IDs in a shared buffer may lead to incorrect updates."
                )

            # Support optional ID checking if 'ids' buffer exists
            buffer_ids = None
            if "ids" in self.buffers:
                buffer_ids = self.buffers["ids"]

            self.sampler.update_priorities(
                indices, priorities, ids=ids, buffer_ids=buffer_ids
            )

    def clear(self):
        with self.write_lock:
            with self.priority_lock:
                self.sampler.clear()
                self.writer.clear()
                self.input_processor.clear()  # Clear processor state if necessary
                self.output_processor.clear()  # Clear output processor state if necessary
                # Zero out buffers
                for buf in self.buffers.values():
                    buf.zero_()

                if self.is_shared:
                    self._next_id.zero_()
                    self._next_game_id.zero_()

    # Accessors for properties required by some utils (like beta)
    def set_beta(self, beta):
        self.sampler.set_beta(beta)

    @property
    def size(self):
        # Delegate size to writer (handles shared memory wrappers)
        return self.writer.size

    def __len__(self):
        return self.size

    @property
    def beta(self):
        return self.sampler.beta

    def sample_sequence(self):
        """
        Retrieves all stored states for a specific sequence ID.
        Useful for debugging or visualization, but slow (O(N) scan).
        """
        if "game_ids" not in self.buffers:
            raise ValueError("Buffer does not have 'game_ids' key")

        game_ids = list(set(self.buffers["game_ids"][: self.size].tolist()))
        if not game_ids:
            return None

        game_id = np.random.choice(game_ids, 1)[0]
        mask = self.buffers["game_ids"][: self.size] == game_id
        indices = torch.nonzero(mask).view(-1).tolist()

        if not indices:
            return None

        indices.sort()
        return self.output_processor.process_batch(indices, self.buffers)

    def reanalyze_sequence(
        self,
        indices,
        new_policies,
        new_values,
        ids=None,
        training_step: Optional[int] = None,
        total_training_steps: Optional[int] = None,
    ):
        """
        Updates the raw values and policies in the buffer.
        Much faster: No N-step recalculation required during write.
        """
        if len(indices) == 0:
            return

        assert (
            "values" in self.buffers and "policies" in self.buffers
        ), "Buffer does not have 'values' or 'policies' keys"
        assert len(new_policies) == len(
            indices
        ), f"Length of new_policies must match length of indices: {len(new_policies)} != {len(indices)}"

        assert len(new_values) == len(
            indices
        ), f"Length of new_values must match length of indices: {len(new_values)} != {len(indices)}"
        for i, idx in enumerate(indices):
            with self.write_lock:
                if ids is not None and "ids" in self.buffers:
                    if int(self.buffers["ids"][idx].item()) != ids[i]:
                        continue
                self.buffers["values"][idx] = new_values[i]
                self.buffers["policies"][idx] = new_policies[i]
