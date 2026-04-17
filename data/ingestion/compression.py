from typing import Dict, List, Optional
import numpy as np
import torch
from abc import ABC, abstractmethod
from collections import deque
from data.utils import discounted_cumulative_sums
from utils.utils import legal_moves_mask
from logging import warning

from data.processors.input_processors import InputProcessor
from data.processors.output_processors import OutputProcessor


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