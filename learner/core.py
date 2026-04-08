from dataclasses import dataclass, field
from typing import Dict, Any, Iterable, Iterator
import torch
import time

from learner.pipeline.base import PipelineComponent

@dataclass
class Blackboard:
    """
    The Absolute Truth. All keys are strings. 
    All tensor values MUST conform to [B, T, ...] where possible.
    """
    # 1. Hardware/Routing
    batch: Dict[str, Any] = field(default_factory=dict)
    
    # 2. Forward Pass Outputs
    predictions: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # 3. Target Math Outputs
    targets: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # 4. Pure Scalar Losses & Aggregation
    losses: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # 5. Non-Graph Metadata (Logging, PER priorities)
    meta: Dict[str, Any] = field(default_factory=dict)


class UniversalLearner:
    """
    The Unchanging Orchestrator. Blindly routes the Blackboard through components.
    """
    def __init__(self, components: list[PipelineComponent], device: torch.device):
        self.components = components
        self.device = device
        self.training_step = 0

    def step(self, batch_iterator: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        t_last = time.perf_counter()

        for batch in batch_iterator:
            # 1. Universal Time Mandate: Move Batch to Device explicitly here
            device_batch = {
                k: v.to(self.device) if torch.is_tensor(v) else v 
                for k, v in batch.items()
            }
            blackboard = Blackboard(batch=device_batch)
            
            # 2. Sequential Pipeline Execution (Radical Transparency)
            for component in self.components:
                component.execute(blackboard)
            
            # 3. Telemetry Output
            t_now = time.perf_counter()
            throughput = len(device_batch.get("actions", [0])) / (t_now - t_last)
            t_last = t_now
            
            blackboard.meta["learner_throughput"] = throughput
            self.training_step += 1
            
            yield blackboard.meta
