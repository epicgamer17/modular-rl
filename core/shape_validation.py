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
    
    # Validate dtype (as string)
    if contract.dtype is not None:
        actual_dtype = str(actual.dtype).split(".")[-1]  # e.g. "float32" from "torch.float32"
        if actual_dtype != contract.dtype:
            errors.append(
                f"dtype: expected {contract.dtype}, got {actual_dtype}"
            )
    
    # Validate feature_shape (non-batch, non-time dimensions)
    if contract.feature_shape is not None:
        if actual.ndim < len(contract.feature_shape):
            errors.append(
                f"ndim {actual.ndim} too small for feature_shape {contract.feature_shape}"
            )
        else:
            # Check from the end, accounting for batch/time dims
            actual_feature = tuple(actual.shape[-len(contract.feature_shape):])
            if actual_feature != contract.feature_shape:
                errors.append(
                    f"feature_shape: expected {contract.feature_shape}, got {actual_feature}"
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