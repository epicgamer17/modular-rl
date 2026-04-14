import torch
from typing import Any
from core.blackboard import Blackboard

def resolve_blackboard_path(blackboard: Blackboard, path: str) -> Any:
    """
    Resolves a path to a value on the blackboard.
    Supports dotted notation for nested access (e.g., 'data.rewards', 'meta.info.legal_moves').
    
    If the path has no dots, it tries:
    1. blackboard.targets[path]
    2. blackboard.data[path]
    3. blackboard.predictions[path]
    
    If the result is a 1D Tensor [B], it is automatically reshaped to [B, 1] 
    to satisfy the Learner's [B, T, ...] shape requirements.
    """
    if not path:
        return None
        
    parts = path.split(".")
    
    # 1. Resolve Root Container
    if parts[0] in ["data", "targets", "predictions", "meta", "losses"]:
        container = getattr(blackboard, parts[0])
        sub_parts = parts[1:]
    else:
        # Search priority for unqualified keys
        if parts[0] in blackboard.targets:
            container = blackboard.targets
            sub_parts = parts
        elif parts[0] in blackboard.data:
            container = blackboard.data
            sub_parts = parts
        elif parts[0] in blackboard.predictions:
            container = blackboard.predictions
            sub_parts = parts
        else:
            raise KeyError(f"Path root '{parts[0]}' not found in any blackboard section.")

    # 2. Traverse nested keys
    try:
        val = container
        for key in sub_parts:
            val = val[key]
    except (KeyError, TypeError, AttributeError) as e:
        raise KeyError(f"Failed to resolve path '{path}' on blackboard: {e}")

    # 3. Canonical Shape Enforcement [B] -> [B, 1]
    # Most learner components expect a Time dimension. If data is sourced
    # directly from 'data' (raw buffer), it typically lacks this dimension.
    if torch.is_tensor(val) and val.ndim == 1:
        val = val.unsqueeze(1)
        
    return val

def write_blackboard_path(blackboard: Blackboard, path: str, value: Any) -> None:
    """
    Writes a value to a path on the blackboard.
    Supports dotted notation (e.g., 'losses.value_loss').
    """
    parts = path.split(".")
    if len(parts) < 2:
        # Default to meta if unqualified
        blackboard.meta[parts[0]] = value
        return

    root = parts[0]
    if root not in ["data", "targets", "predictions", "meta", "losses"]:
        # Default to meta if root is unknown
        blackboard.meta[path] = value
        return

    container = getattr(blackboard, root)
    sub_parts = parts[1:]
    
    # Traverse to the parent of the final key
    current = container
    for key in sub_parts[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    
    # Write the value
    current[sub_parts[-1]] = value

