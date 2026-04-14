from typing import Dict, Any, Iterable, Iterator
import torch
import time

from core.blackboard import Blackboard
from core.component import PipelineComponent

def validate_recipe(components: list[PipelineComponent], initial_keys: set[str] = frozenset()):
    """
    Validates that the pipeline components' read/write contracts are satisfied.
    Simulates the data flow through the Blackboard to catch DAG topology errors early.
    """
    available_keys = set(initial_keys)
    
    for i, component in enumerate(components):
        # 1. Check if the component's read requirements are met by prior writes
        missing_keys = {key for key in component.requires if key not in available_keys}
        
        if missing_keys:
            raise RuntimeError(
                f"DAG Topology Error at Component [{i}] '{component.__class__.__name__}':\n"
                f"Required keys {missing_keys} have not been written by any previous component or provided in the initial batch.\n"
                f"Available keys at this stage: {sorted(list(available_keys))}"
            )
            
        # 2. Simulate the component executing by adding its writes to the available pool
        available_keys.update(component.provides)
        
    print(f"DAG Validation Passed: {len(components)} components verified.")


class BlackboardEngine:
    """
    The Unchanging Orchestrator. Manages the lifecycle of the Blackboard dictionary by routing it through sequential components.
    """
    def __init__(
        self, 
        components: list[PipelineComponent], 
        device: torch.device, 
        initial_keys: set[str] | None = None,
        strict: bool = False
    ):
        self.components = components
        self.device = device
        self.training_step = 0
        self.strict = strict

        # Validate the DAG before the first training step
        validate_recipe(components, initial_keys or set())

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
                # Runtime Validation (Strict Mode)
                if self.strict:
                    component.validate(blackboard)
                
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
