"""
DataRef abstraction for the RL IR.
Provides a unified interface for referencing data (tensors, buffers, streams)
without necessarily holding the concrete value in the IR structure itself.
"""

from typing import Any, Optional
import torch

class DataRef:
    """
    A reference to a piece of data (typically a tensor).
    
    Attributes:
        data: The underlying data (initially a PyTorch tensor).
    """
    def __init__(self, data: Any):
        self._data = data

    @property
    def data(self) -> Any:
        return self._data

    def __repr__(self) -> str:
        return f"DataRef(type={type(self._data)}, shape={getattr(self._data, 'shape', 'N/A')})"

class BufferRef(DataRef):
    """
    A reference to a persistent buffer (e.g., replay buffer slice).
    """
    def __repr__(self) -> str:
        return f"BufferRef(type={type(self._data)}, shape={getattr(self._data, 'shape', 'N/A')})"

class StreamRef(DataRef):
    """
    A reference to a stream of data (e.g., environment observations).
    """
    def __repr__(self) -> str:
        return f"StreamRef(type={type(self._data)}, shape={getattr(self._data, 'shape', 'N/A')})"
