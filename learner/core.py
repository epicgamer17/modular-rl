from dataclasses import dataclass, field
from typing import Dict, Any, List, Iterable, Iterator, Optional
import torch
import time

from telemetry.stats import finalize_metrics
from learner.pipeline.base import PipelineComponent

@dataclass
class Blackboard:
    """
    Central data store for the execution graph.
    All PipelineComponents read from and write to this shared dataclass.
    """
    batch: Dict[str, Any] = field(default_factory=dict)
    predictions: Dict[str, torch.Tensor] = field(default_factory=dict)
    targets: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # Total loss calculated per optimizer key
    losses: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # Mathematical scalar output and telemetry metrics
    loss_dict: Dict[str, float] = field(default_factory=dict)
    
    # PER priorities and other state tracking
    priorities: Optional[torch.Tensor] = None
    meta: Dict[str, Any] = field(default_factory=dict)

class UniversalLearner:
    """
    Algorithm-agnostic learner that orchestrates the optimization loop
    using the 'Unchanging Orchestrator' / Blackboard pattern.
    """
    def __init__(self, components: List[PipelineComponent]):
        self.components = components
        self.training_step = 0

    def step(self, batch_iterator: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        t_last = time.perf_counter()

        for batch in batch_iterator:
            blackboard = Blackboard(batch=batch)
            
            # 1. Execute Pipeline Components sequentially
            for component in self.components:
                component.execute(blackboard)
            
            # 2. Pipeline-agnostic throughput computation
            t_now = time.perf_counter()
            dt = t_now - t_last
            t_last = t_now
            
            # Try to infer batch dimensions for telemetry
            B, T = 1, 1
            if blackboard.predictions:
                any_pred = next((p for p in blackboard.predictions.values() if torch.is_tensor(p)), None)
                if any_pred is not None and any_pred.ndim >= 2:
                    B, T = any_pred.shape[:2]
            
            throughput = (B * T) / dt if dt > 0 else 0
            blackboard.loss_dict["learner_throughput"] = throughput
            
            self.training_step += 1
            yield self._build_step_metrics(blackboard)

    def _build_step_metrics(self, blackboard: Blackboard) -> Dict[str, Any]:
        metrics = dict(blackboard.loss_dict)
        if blackboard.losses:
            metrics["loss"] = sum(loss.item() for loss in blackboard.losses.values())

        finalized = finalize_metrics(blackboard.meta.setdefault("metrics", {}))
        if finalized:
            metrics["metrics"] = finalized

        return metrics
