"""Runtime tensor shape/dtype validation against Key shape contracts."""

import torch
from typing import Any, Dict, List

from core.contracts import Key, ShapeContract


def validate_tensor(key: Key, value: Any) -> None:
    """
    Validate a tensor against its Key's shape contract using semantic axis names.
    
    Raises AssertionError on mismatch.
    """
    if not torch.is_tensor(value):
        return  # Non-tensors skip validation
    
    contract = key.shape
    if contract is None:
        return  # No contract = skip validation
    
    actual = value
    errors: List[str] = []
    
    # Check rank and semantic dimensions
    if contract.semantic_shape is not None:
        if actual.ndim != len(contract.semantic_shape):
            errors.append(
                f"rank mismatch: contract requires {len(contract.semantic_shape)} dims {contract.semantic_shape}, "
                f"but tensor has {actual.ndim} ({list(actual.shape)})"
            )
        else:
            # Axis-by-axis verification
            event_axes_seen = 0
            for i, axis_type in enumerate(contract.semantic_shape):
                # "*" is a wildcard at runtime too
                if axis_type == "*":
                    continue
                
                # Check explicit time value
                if axis_type == "T" and contract.time_val is not None:
                    if actual.shape[i] != contract.time_val:
                        errors.append(
                            f"time_val: expected T={contract.time_val} at dim {i}, got {actual.shape[i]}"
                        )
                
                # Check explicit event value (any axis that isn't B, T, or *)
                if axis_type not in ("B", "T", "*") and contract.event_shape is not None:
                    if event_axes_seen < len(contract.event_shape):
                        expected_val = contract.event_shape[event_axes_seen]
                        if actual.shape[i] != expected_val:
                            errors.append(
                                f"event_shape mismatch: axis '{axis_type}' at dim {i} expects {expected_val}, "
                                f"but tensor has {actual.shape[i]}"
                            )
                        event_axes_seen += 1
    
    # Check dtype
    if contract.dtype is not None:
        if actual.dtype != contract.dtype:
            errors.append(
                f"dtype: expected {contract.dtype}, got {actual.dtype}"
            )
    
    if errors:
        raise AssertionError(
            f"Shape validation failed for {key.path}: {'; '.join(errors)} "
            f"(actual shape: {list(actual.shape)}, dtype: {actual.dtype})"
        )



def validate_batch_outputs(
    outputs: Dict[str, Any],
    provides_keys: Dict[Key, Any],
) -> None:
    """
    Validate all tensor outputs against their Key contracts.
    
    Called in debug mode after each component executes.
    """
    for key, value in provides_keys.items():
        if key.path in outputs:
            validate_tensor(key, outputs[key.path])