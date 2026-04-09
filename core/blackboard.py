from dataclasses import dataclass, field
from typing import Dict, Any
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
