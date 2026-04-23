"""
Schema-driven collation for RL data.
Ensures that sampled batches from Replay Buffers are correctly structured.
"""

import torch
from typing import List, Dict, Any, Optional
from core.schema import Schema, TensorSpec

class ReplayCollator:
    """
    Collates a list of transition dictionaries into a single dictionary of batched tensors.
    Uses the provided Schema to determine correct dtypes and shapes.
    """
    def __init__(self, schema: Schema):
        self.schema = schema
        self.field_map = schema.get_field_map()

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collates a batch of transitions.
        
        Args:
            batch: List of dictionaries (one per transition).
            
        Returns:
            Dictionary of batched data.
        """
        if not batch:
            return {}

        collated = {}
        
        # 1. Collate fields defined in the schema
        for field_name, spec in self.field_map.items():
            if field_name not in batch[0]:
                continue
                
            values = [item[field_name] for item in batch]
            
            # Map schema dtypes to torch dtypes
            dtype_map = {
                "float32": torch.float32,
                "float64": torch.float64,
                "int64": torch.int64,
                "int32": torch.int32,
                "long": torch.int64,
                "bool": torch.bool,
            }
            torch_dtype = dtype_map.get(spec.dtype, torch.float32)
            
            first_val = values[0]
            
            if isinstance(first_val, torch.Tensor):
                # Handle existing tensors (usually observations)
                if first_val.dim() == 0:
                    # Scalar tensor -> vector tensor [B]
                    collated[field_name] = torch.stack(values).to(torch_dtype)
                else:
                    # Multi-dim tensor -> [B, ...]
                    collated[field_name] = torch.stack(values).to(torch_dtype)
            elif isinstance(first_val, (int, float, bool)):
                # Handle raw scalars -> tensor [B]
                collated[field_name] = torch.tensor(values, dtype=torch_dtype)
            else:
                # Fallback for structured objects (metadata)
                collated[field_name] = values

        # 2. Handle fields NOT in the schema (e.g. metadata)
        for k in batch[0].keys():
            if k not in collated:
                collated[k] = [item[k] for item in batch]

        return collated
