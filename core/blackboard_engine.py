from typing import Dict, Any, Iterable, Iterator, List, Set, Type, Optional
import torch
import time

from core.blackboard import Blackboard
from core.component import PipelineComponent
from core.contracts import Key
from core.path_resolver import resolve_blackboard_path, write_blackboard_path

def apply_updates(blackboard: Blackboard, updates: Dict[str, Any]) -> None:
    """
    Applies a dictionary of path-based updates to the blackboard.
    Used by the engine and meta-components to apply returned mutations.
    """
    if not updates:
        return
    for path, value in updates.items():
        write_blackboard_path(blackboard, path, value)

def validate_recipe(components: List[PipelineComponent], initial_keys: Set[Key]) -> None:
    """
    Validates that all components in the pipeline have their requirements met.
    Performs both path existence and semantic type checks.
    """
    available_contracts: Dict[str, Type] = {k.path: k.semantic_type for k in initial_keys}

    for i, component in enumerate(components):
        missing = []
        type_mismatches = []
        
        for req in component.requires:
            if req.path not in available_contracts:
                missing.append(req.path)
            elif not issubclass(available_contracts[req.path], req.semantic_type):
                type_mismatches.append(
                    f"'{req.path}' expected {req.semantic_type.__name__}, "
                    f"but found {available_contracts[req.path].__name__}"
                )

        if missing or type_mismatches:
            error_msg = f"DAG Topology Error at Component [{i}] '{type(component).__name__}':\n"
            if missing:
                error_msg += f"  Missing keys: {missing}\n"
            if type_mismatches:
                error_msg += f"  Type mismatches: {type_mismatches}\n"
            error_msg += f"  Available keys at this stage: {list(available_contracts.keys())}"
            raise RuntimeError(error_msg)

        # Update available keys with what this component provides
        provides = component.provides
        # Support both old Set[Key] and new Dict[Key, str] during transition
        provides_items = provides.items() if isinstance(provides, dict) else [(k, "new") for k in provides]
        
        for prov, mode in provides_items:
            if mode == "new" and prov.path in available_contracts:
                # Optional: warn or error if "new" is used but key exists
                pass
            elif mode == "overwrite":
                if prov.path not in available_contracts:
                    raise RuntimeError(f"DAG Topology Error: Component '{type(component).__name__}' attempts to overwrite non-existent key '{prov.path}'")
            
            available_contracts[prov.path] = prov.semantic_type

    print(f"DAG Validation Passed: {len(components)} components verified.")


class BlackboardEngine:
    """
    The Unchanging Orchestrator. Manages the lifecycle of the Blackboard dictionary by routing it through sequential components.
    """
    def __init__(
        self, 
        components: list[PipelineComponent], 
        device: torch.device, 
        initial_keys: Set[Key] = set(),
        strict: bool = False
    ):
        self.components = components
        self.device = device
        self.training_step = 0
        self.strict = strict

        # Validate the DAG before the first training step
        validate_recipe(components, initial_keys)

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
                
                # Support both in-place mutation and explicit return dicts
                outputs = component.execute(blackboard)
                apply_updates(blackboard, outputs)

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
