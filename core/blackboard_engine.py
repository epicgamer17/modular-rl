from typing import Dict, Any, Iterable, Iterator
import torch
import time

from core.blackboard import Blackboard
from core.component import PipelineComponent

class BlackboardEngine:
    """
    The Unchanging Orchestrator. Manages the lifecycle of the Blackboard dictionary by routing it through sequential components.
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
            blackboard = Blackboard(data=device_batch)
            
            # 2. Sequential Pipeline Execution (Radical Transparency)
            for component in self.components:
                component.execute(blackboard)
                if blackboard.meta.get("stop_execution"):
                    break
            
            # 3. Telemetry Output
            t_now = time.perf_counter()
            throughput = len(device_batch.get("actions", [0])) / (t_now - t_last)
            t_last = t_now
            
            blackboard.meta["learner_throughput"] = throughput
            self.training_step += 1
            yield {
                "losses": {k: v.item() for k, v in blackboard.losses.items() if k != "total_loss"},
                "total_losses": {k: v.item() for k, v in blackboard.losses.get("total_loss", {}).items()},
                "meta": blackboard.meta,
            }

            if blackboard.meta.get("stop_execution"):
                break
