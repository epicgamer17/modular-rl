from dataclasses import dataclass, field
from typing import Dict, Any, Mapping
from types import MappingProxyType
import torch

@dataclass
class Blackboard:
    """
    The Absolute Truth. All keys are strings. 
    All tensor values MUST conform to [B, T, ...] where possible.
    """
    # 1. Incoming Data (Replay Buffer Batches, or Env Observations)
    data: Dict[str, Any] = field(default_factory=dict)
    
    # 2. Network Outputs (Action logits, values, etc.)
    predictions: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # 3. Ground Truth / Targets (MCTS targets, TD-targets)
    targets: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # 4. Pure Scalar Losses
    losses: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # 5. Non-Graph Metadata (Logging, PER priorities)
    meta: Dict[str, Any] = field(default_factory=dict)


    def frozen(self) -> "Blackboard":
        """Returns a read-only view of the blackboard."""
        return Blackboard(
            data=MappingProxyType(self.data),  # type: ignore
            predictions=MappingProxyType(self.predictions),  # type: ignore
            targets=MappingProxyType(self.targets),  # type: ignore
            losses=MappingProxyType(self.losses),  # type: ignore
            meta=MappingProxyType(self.meta),  # type: ignore
        )

    @staticmethod
    def validate_write(batch: Dict[str, Any]) -> None:
        """
        Validate a batch of data before it is written to the replay buffer.
        Enforces BufferSchema and strict structural invariants.
        """
        from core.shape_validation import validate_buffer_write
        validate_buffer_write(batch)
