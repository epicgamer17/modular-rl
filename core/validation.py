import torch
from typing import Any, Dict, Optional, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from core.blackboard import Blackboard

def same_batch(t1: Any, t2: Any) -> bool:
    """True if tensors/arrays have the same batch dimension (dim 0)."""
    if not torch.is_tensor(t1) or not torch.is_tensor(t2):
        return True # Non-tensor objects skip batch check
    
    assert t1.shape[0] == t2.shape[0], f"Batch size mismatch: {t1.shape[0]} vs {t2.shape[0]}"
    return True

def has_time_dim(t: Any, expected_t: Optional[int] = None) -> bool:
    """True if tensor has exactly 3 dimensions (B, T, ...) or matches expected_t."""
    assert torch.is_tensor(t), f"Expected tensor for time dim check, got {type(t)}"
    assert t.ndim >= 2, f"Tensor must have at least 2 dims (B, T), got {t.ndim}"
    
    if expected_t is not None:
        assert t.shape[1] == expected_t, f"Time dimension mismatch: expected {expected_t}, got {t.shape[1]}"
    
    return True

def is_distributional(t: Any) -> bool:
    """True if tensor has categorical atoms (usually ndim >= 3)."""
    if not torch.is_tensor(t):
        return False
    return t.ndim >= 3

def assert_same_batch(t1: Any, t2: Any, msg: str = ""):
    """Asserts that two tensors have the same batch size."""
    if torch.is_tensor(t1) and torch.is_tensor(t2):
        assert t1.shape[0] == t2.shape[0], f"Batch size mismatch {msg}: {t1.shape[0]} vs {t2.shape[0]}"

def assert_time_dim(t: Any, expected_t: int, msg: str = ""):
    """Asserts that a tensor has the correct time dimension."""
    if torch.is_tensor(t):
        assert t.ndim >= 2, f"Tensor {msg} lacks time dimension"
        assert t.shape[1] == expected_t, f"Time mismatch {msg}: expected {expected_t}, got {t.shape[1]}"

def assert_compatible_value(pred: torch.Tensor, target: torch.Tensor, msg: str = ""):
    """Asserts that value predictions and targets are compatible (both scalar or both distributional)."""
    if is_distributional(pred):
        assert is_distributional(target) or target.ndim == 2, f"Distributional prediction requires distributional target or scalar [B, T] {msg}"
    else:
        assert not is_distributional(target), f"Scalar prediction cannot match distributional target {msg}"

def assert_in_blackboard(bb: "Blackboard", key: str, msg: str = ""):
    """Asserts that a key exists in the blackboard (either path or direct)."""
    # Simple check for top-level dicts first
    parts = key.split(".")
    if len(parts) > 1:
        # It's a path
        from core.path_resolver import resolve_blackboard_path
        try:
            resolve_blackboard_path(bb, key)
        except (KeyError, AttributeError):
            assert False, f"Key {key} not found in blackboard {msg}"
    else:
        assert hasattr(bb, parts[0]) or parts[0] in bb.data or parts[0] in bb.meta, f"Key {key} not found in blackboard {msg}"

def assert_is_tensor(obj: Any, msg: str = ""):
    """Asserts that an object is a torch.Tensor."""
    assert torch.is_tensor(obj), f"Expected torch.Tensor {msg}, got {type(obj)}"

def assert_shape_sanity(t: torch.Tensor, min_ndim: int = 1, max_ndim: int = 3, msg: str = ""):
    """Asserts that a tensor has a standard rank (e.g. [B], [B, T], [B, T, D])."""
    assert min_ndim <= t.ndim <= max_ndim, (
        f"Tensor rank mismatch {msg}: expected {min_ndim}-{max_ndim} dims, got {t.ndim} ({t.shape})"
    )

def assert_representation_supports(representation: Any, tensor: torch.Tensor, msg: str = ""):
    """Asserts that a representation strategy can correctly process a tensor."""
    if hasattr(representation, "validate_logits"):
        representation.validate_logits(tensor)
    elif hasattr(representation, "supports"):
        assert representation.supports(tensor), f"Representation {type(representation).__name__} does not support tensor {msg} with shape {tensor.shape}"

