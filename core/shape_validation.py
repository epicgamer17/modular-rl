"""Runtime tensor shape/dtype validation against Key shape contracts."""

import torch
from typing import Any, Dict, List

import numpy as np
from core.contracts import Key, ShapeContract, BufferSchema


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


def validate_buffer_write(batch: Dict[str, Any]) -> None:
    """
    Mandatory invariant gate for any data entering the replay buffer.
    Enforces BufferSchema and strict structural constraints.

    Invariants:
    1. Mandatory fields (episode_id, step_id, done) MUST exist.
    2. NO Python objects: lists, dicts, custom dataclasses are REJECTED.
    3. NO numpy 'object' dtype (ragged arrays) are REJECTED.
    4. Every field MUST be a torch.Tensor (once in storage).
    5. Every field MUST have at least one dimension [T, ...].
    """
    mandatory = BufferSchema.get_mandatory_fields()

    for path, value in batch.items():
        # 1. Reject forbidden containers
        if isinstance(value, (list, dict)):
            raise TypeError(
                f"Forbidden container type in buffer write for key '{path}': {type(value)}. "
                f"Only Tensors are allowed in the replay buffer."
            )

        # 2. Reject custom dataclasses
        if hasattr(value, "__dataclass_fields__"):
            raise TypeError(
                f"Forbidden dataclass in buffer write for key '{path}'. "
                f"Only Tensors are allowed in the replay buffer."
            )

        # 3. Reject numpy object dtypes
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                raise TypeError(
                    f"Forbidden numpy 'object' dtype for key '{path}'. "
                    f"Ragged/Variable-length lists are not allowed in the buffer."
                )

        # 4. Final safety check: must be a Tensor
        if not (torch.is_tensor(value) or isinstance(value, np.ndarray)):
            raise TypeError(
                f"Field '{path}' must be a torch.Tensor or np.ndarray, got {type(value)}"
            )

    # 4. Final pass: ensure all mandatory fields exist
    for m_field in mandatory:
        if m_field not in batch:
            raise KeyError(
                f"Buffer write is missing mandatory BufferSchema field: '{m_field}'. "
                f"Available: {list(batch.keys())}"
            )

    # 5. Length check (Self-consistency of T)
    first_dim = None
    for path, value in batch.items():
        batch_dim = value.shape[0]
        if first_dim is None:
            first_dim = batch_dim
        elif first_dim != batch_dim:
            raise ValueError(
                f"Buffer write contains ragged data. All fields must have the same first dimension [T, ...]. "
                f"Found '{path}' with length {batch_dim}, but previous field had length {first_dim}."
            )

    # 6. Shape check [T, ...]
    for path, value in batch.items():
        if value.ndim < 1:
            raise ValueError(
                f"Buffer field '{path}' must have at least one dimension [T, ...], "
                f"got scalar shape {value.shape}"
            )