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
    Performs path existence, semantic type checks, and parameter compatibility checks.
    """
    # Track the full Key object for every available path
    available_contracts: Dict[str, Key] = {k.path: k for k in initial_keys}

    for i, component in enumerate(components):
        missing = []
        incompatibilities = []
        
        for req in component.requires:
            if req.path not in available_contracts:
                missing.append(req.path)
                continue
            
            found_key = available_contracts[req.path]
            
            # 1. Type Check (Semantic)
            if not issubclass(found_key.semantic_type, req.semantic_type):
                incompatibilities.append(
                    f"Type mismatch for '{req.path}': expected {req.semantic_type.__name__}, "
                    f"but found {found_key.semantic_type.__name__}"
                )
            
            # 2. Metadata Check (Parameter-Aware Contracts)
            # If a requirement specifies metadata (e.g. bins=51), the provider must match it.
            for m_name, m_value in req.metadata.items():
                if m_name not in found_key.metadata:
                    incompatibilities.append(
                        f"Missing metadata '{m_name}' for '{req.path}': expected {m_value}"
                    )
                elif found_key.metadata[m_name] != m_value:
                    incompatibilities.append(
                        f"Metadata mismatch for '{req.path}.{m_name}': expected {m_value}, "
                        f"found {found_key.metadata[m_name]}"
                    )

        if missing or incompatibilities:
            error_msg = f"DAG Topology Error at Component [{i}] '{type(component).__name__}':\n"
            if missing:
                error_msg += f"  Missing keys: {missing}\n"
            if incompatibilities:
                error_msg += f"  Incompatibilities: \n    - " + "\n    - ".join(incompatibilities) + "\n"
            error_msg += f"  Available keys at this stage: {list(available_contracts.keys())}"
            raise RuntimeError(error_msg)

        # Update available keys with what this component provides
        provides = component.provides
        provides_items = provides.items() if isinstance(provides, dict) else [(k, "new") for k in provides]
        
        for prov, mode in provides_items:
            if mode == "overwrite" and prov.path not in available_contracts:
                raise RuntimeError(f"DAG Topology Error: Component '{type(component).__name__}' attempts to overwrite non-existent key '{prov.path}'")
            
            # Update the global dataflow state
            available_contracts[prov.path] = prov

    print(f"DAG Validation Passed: {len(components)} components verified.")


class BlackboardEngine:
    """
    The Unchanging Orchestrator. Manages the lifecycle of the Blackboard dictionary by routing it through sequential components.
    
    Now includes basic Execution Graph capabilities for #6 Optimization:
    - Pruning: Skips components whose outputs are never read by downstream components or the engine output.
    """
    def __init__(
        self, 
        components: List[PipelineComponent], 
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
        
        # Build the optimized execution order
        self._execution_plan = self._build_execution_plan(components, initial_keys)

    def _build_execution_plan(self, components: List[PipelineComponent], initial_keys: Set[Key]) -> List[PipelineComponent]:
        """
        Builds the final list of components to execute.
        Currently performs simple 'Terminal-Value Pruning':
        If a component's provides are never required by a later component,
        it is potentially removable (unless it provides 'meta', 'losses', or terminal outputs).
        """
        # 1. Determine which keys are 'Terminal Targets' (always required for engine output)
        terminal_paths = {
            "meta.stop_execution",
            "meta.learner_throughput"
        }
        
        active_components = []
        required_downstream = set(terminal_paths)
        
        # Iterate backwards to find dependencies
        # Any component providing a key required by a later component (or terminal) is kept.
        for component in reversed(components):
            provides = component.provides
            provides_keys = provides.keys() if isinstance(provides, dict) else provides
            provides_paths = {k.path for k in provides_keys}
            
            # Heuristic: Components writing to 'losses' or 'meta' are currently never pruned
            is_telemetry = any(p.startswith("losses.") or p.startswith("meta.") for p in provides_paths)
            
            # If this component produces something someone else needs, we must keep it.
            # OR if it's a telemetry/output component.
            # OR if it has NO provides (side-effect component like a buffer writer)
            if (provides_paths & required_downstream) or is_telemetry or not provides_paths:
                active_components.append(component)
                # Add its requirements to the list of what we need from earlier components
                required_downstream.update({k.path for k in component.requires})
        
        plan = list(reversed(active_components))
        print(f"Execution Graph Built: Optimized {len(components)} -> {len(plan)} active components.")
        return plan

    def step(self, batch_iterator: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        t_last = time.perf_counter()

        for batch in batch_iterator:
            # 1. Universal Time Mandate: Move Batch to Device explicitly here
            device_batch = {
                k: v.to(self.device) if torch.is_tensor(v) else v 
                for k, v in batch.items()
            }
            blackboard = Blackboard(data=device_batch)
            
            # 2. Optimized Pipeline Execution with Profiling
            profiles = {}
            for component in self._execution_plan:
                comp_name = type(component).__name__
                t0 = time.perf_counter()
                
                # Runtime Validation (Strict Mode)
                if self.strict:
                    component.validate(blackboard)
                
                outputs = component.execute(blackboard)
                apply_updates(blackboard, outputs)
                
                profiles[comp_name] = profiles.get(comp_name, 0) + (time.perf_counter() - t0)

                if blackboard.meta.get("stop_execution"):
                    break
            
            # 3. Telemetry Output
            t_now = time.perf_counter()
            throughput = len(device_batch.get("actions", [0])) / (t_now - t_last)
            t_last = t_now
            
            blackboard.meta["learner_throughput"] = throughput
            # Add profiling metadata (bottleneck detection)
            blackboard.meta["component_profiles_ms"] = {
                k: round(v * 1000, 3) for k, v in profiles.items()
            }
            
            self.training_step += 1
            
            # Aggregate losses and metrics
            log_losses = {}
            total_losses = {}
            for k, v in blackboard.losses.items():
                if k.startswith("total_loss"):
                    # Handle both flat total_loss and nested total_loss.opt_key
                    if isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            total_losses[sub_k] = sub_v.item()
                    else:
                        val = v.item()
                        total_losses["total"] = val
                        # Backward compatibility for 'default' key if missing
                        if "default" not in total_losses:
                            total_losses["default"] = val
                else:
                    # Individual diagnostic losses
                    if torch.is_tensor(v):
                        log_losses[k] = v.item()
            
            yield {
                "losses": log_losses,
                "total_losses": total_losses,
                "meta": blackboard.meta,
            }

            if blackboard.meta.get("stop_execution"):
                break
