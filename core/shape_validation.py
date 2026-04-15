"""Runtime tensor shape/dtype validation against Key shape contracts."""

import torch
from typing import Any, Dict, List

from core.contracts import Key, ShapeContract


def validate_tensor(key: Key, value: Any) -> None:
    """
    Validate a tensor against its Key's shape contract.
    
    Raises AssertionError on mismatch.
    """
    if not torch.is_tensor(value):
        return  # Non-tensors skip validation
    
    contract = key.shape
    if contract is None:
        return  # No contract = skip validation
    
    actual = value
    errors: List[str] = []
    
    # Validate ndim
    if contract.ndim is not None:
        if actual.ndim != contract.ndim:
            errors.append(
                f"ndim: expected {contract.ndim}, got {actual.ndim}"
            )
    
    # Validate dtype
    if contract.dtype is not None:
        if actual.dtype != contract.dtype:
            errors.append(
                f"dtype: expected {contract.dtype}, got {actual.dtype}"
            )
    
    # Validate event_shape (non-batch, non-time dimensions)
    if contract.event_shape is not None:
        if actual.ndim < len(contract.event_shape):
            errors.append(
                f"ndim {actual.ndim} too small for event_shape {contract.event_shape}"
            )
        else:
            # Check from the end, accounting for batch/time dims
            if len(contract.event_shape) == 0:
                # For scalar events, if it's just [Batch] or [Batch, Time], 
                # we don't expect any more trailing dimensions.
                # However, PyTorch tensors often have shape [B] for a batch of scalars.
                # If we're at this point, actual.ndim >= len(event_shape) is 0, always true.
                # We need to decide if we allow [B] or strictly nothing extra.
                # Existing behavior: actual_event was actual.shape[-0:] which is the whole shape.
                # New behavior: if event_shape is (), we check that no extra dims exist 
                # beyond batch (and time if present).
                expected_ndim = 1 + (1 if contract.time_dim is not None else 0)
                if actual.ndim > expected_ndim:
                     actual_event = tuple(actual.shape[expected_ndim:])
                else:
                     actual_event = ()
            else:
                actual_event = tuple(actual.shape[-len(contract.event_shape):])
            if actual_event != contract.event_shape:
                errors.append(
                    f"event_shape: expected {contract.event_shape}, got {actual_event}"
                )
    
    # Validate time_dim consistency
    if contract.time_dim is not None:
        # If time_dim is set, the tensor MUST have at least that many dimensions
        if actual.ndim <= contract.time_dim:
             errors.append(
                f"time_dim validation: contract requires sequence dim at {contract.time_dim}, "
                f"but tensor only has {actual.ndim} dimensions."
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