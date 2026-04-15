from typing import Dict, Any, Iterable, Iterator, List, Set, Type, Optional
import torch
import time

from core.blackboard import Blackboard
from core.component import PipelineComponent
from core.contracts import Key, check_shape_compatibility
from core.execution_graph import ExecutionGraph, build_execution_graph, _get_provides_keys, _get_provides_with_modes
from core.blackboard_diff import snapshot_blackboard, diff_snapshots, BlackboardDiff
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
    Validates the pipeline DAG at build-time.
    Stages:
    1. Dependency Resolution (Path check)
    2. Semantic Compatibility (Type check)
    3. Representation Consistency (Metadata/Parameters check)
    4. Shape Integrity (Lightweight tensor shape check)
    """
    # Track the full Key object for every available path
    available_contracts: Dict[str, Key] = {k.path: k for k in initial_keys}

    for i, component in enumerate(components):
        missing = []
        incompatibilities = []
        
        for req in component.requires:
            # Stage 1: Dependency Resolution
            if req.path not in available_contracts:
                missing.append(req.path)
                continue
            
            found_key = available_contracts[req.path]
            
            # Stage 2: Semantic Compatibility
            # Generic semantic types (e.g., Reward) must match or found_key must be a subclass
            if not issubclass(found_key.semantic_type, req.semantic_type):
                incompatibilities.append(
                    f"SEMANTIC MISMATCH for '{req.path}': expected {req.semantic_type}, "
                    f"but found {found_key.semantic_type}"
                )
            
            # Stage 3: Representation Consistency (Metadata)
            # This ensures bins, vmin, vmax etc. match exactly between provider and consumer.
            for m_name, m_value in req.metadata.items():
                if m_name not in found_key.metadata:
                    incompatibilities.append(
                        f"REPRESENTATION GAP for '{req.path}': consumer expects '{m_name}={m_value}', "
                        f"but provider metadata is missing this parameter."
                    )
                elif found_key.metadata[m_name] != m_value:
                    incompatibilities.append(
                        f"REPRESENTATION MISMATCH for '{req.path}.{m_name}': "
                        f"expected {m_value}, provider has {found_key.metadata[m_name]}"
                    )

            # Stage 4: Shape Integrity
            shape_issues = check_shape_compatibility(provider=found_key, consumer=req)
            for issue in shape_issues:
                incompatibilities.append(f"SHAPE ERROR for '{req.path}': {issue}")

        if missing or incompatibilities:
            error_header = f"DAG Topology Error at Component [{i}] '{type(component).__name__}':"
            error_msg = [error_header]
            if missing:
                error_msg.append(f"  Missing Dependencies: {missing}")
            if incompatibilities:
                error_msg.append("  Contract Violations:")
                for inc in incompatibilities:
                    error_msg.append(f"    - {inc}")
            
            error_msg.append(f"  Available keys in namespace: {sorted(list(available_contracts.keys()))}")
            raise RuntimeError("\n".join(error_msg))

        # Update available keys with component provisions
        provides = component.provides
        provides_items = provides.items() if isinstance(provides, dict) else [(k, "new") for k in provides]
        
        for prov, mode in provides_items:
            # Mode 'overwrite' requires the path to already exist
            if mode == "overwrite" and prov.path not in available_contracts:
                raise RuntimeError(
                    f"STAGE OVERWRITE ERROR: Component '{type(component).__name__}' "
                    f"attempts to overwrite non-existent key '{prov.path}'"
                )
            
            # Register the key for downstream components
            available_contracts[prov.path] = prov

    print(f"DAG Validation Passed: {len(components)} components verified.")


def _blackboard_has_path(blackboard: Blackboard, path: str) -> bool:
    """Check if a dotted path already exists on the blackboard."""
    parts = path.split(".")
    if parts[0] in ("data", "targets", "predictions", "meta", "losses"):
        container = getattr(blackboard, parts[0])
        sub_parts = parts[1:]
    else:
        return False

    current = container
    for key in sub_parts:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def _resolve_lazy_plan(
    graph: ExecutionGraph,
    blackboard: Blackboard,
) -> List[int]:
    """Demand-driven backward resolution for lazy execution.

    Walks the execution order in reverse. A component is included if:
    - It is a terminal sink (always executes), OR
    - It uses a mutating write mode (overwrite/append) on any path, OR
    - At least one of its provided paths is demanded by a downstream
      component that will execute, AND that path is not already present
      on the blackboard.

    Returns the filtered execution order (indices into graph.components).
    """
    must_execute: Set[int] = set()
    demanded_keys: Set[Key] = set()

    for idx in reversed(graph.execution_order):
        comp = graph.components[idx]
        provides_modes = _get_provides_with_modes(comp)
        provides_keys = set(provides_modes.keys())
        is_terminal = idx in graph.terminal_indices

        # Components that mutate existing data can never be skipped
        has_mutation = any(
            mode in ("overwrite", "append") for mode in provides_modes.values()
        )

        if is_terminal or has_mutation:
            must_execute.add(idx)
            demanded_keys.update(comp.requires)
        elif provides_keys & demanded_keys:
            # This component provides something demanded downstream.
            # Only skip if ALL demanded outputs are already on the blackboard.
            needed = provides_keys & demanded_keys
            already_satisfied = all(
                _blackboard_has_path(blackboard, k.path) for k in needed
            )
            if not already_satisfied:
                must_execute.add(idx)
                demanded_keys.update(comp.requires)

    return [idx for idx in graph.execution_order if idx in must_execute]


class BlackboardEngine:
    """
    The Unchanging Orchestrator. Manages the lifecycle of the Blackboard
    dictionary by routing it through a DAG-sorted pipeline of components.

    Uses an ExecutionGraph for:
    - Topological ordering derived from requires/provides contracts
    - Backward-reachability pruning from terminal sinks (losses.*, meta.*)
    - Cycle detection at build time
    - Optional lazy execution: skip components whose outputs are already
      present on the blackboard (enabled via ``lazy=True``)
    """
    def __init__(
        self, 
        components: List[PipelineComponent], 
        device: torch.device, 
        initial_keys: Set[Key] = set(),
        strict: bool = False,
        lazy: bool = False,
        diff: bool = False,
        **kwargs: Any,
    ):
        self.components = components
        self.device = device
        self.training_step = 0
        self.strict = strict
        self.lazy = lazy
        self.diff = diff

        # Build the DAG execution graph (includes dependency + cycle checks)
        self._graph: ExecutionGraph = build_execution_graph(components, initial_keys)

        # Validate semantic contracts on the active (topologically sorted) components
        validate_recipe(self._graph.active_components(), initial_keys)

    @property
    def execution_graph(self) -> ExecutionGraph:
        """Read-only access to the execution graph for introspection."""
        return self._graph

    def step(self, batch_iterator: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        t_last = time.perf_counter()

        # Pre-resolve the static execution plan (used when lazy=False)
        static_plan = self._graph.active_components()

        for batch in batch_iterator:
            # 1. Universal Time Mandate: Move Batch to Device explicitly here
            device_batch = {
                k: v.to(self.device) if torch.is_tensor(v) else v 
                for k, v in batch.items()
            }
            blackboard = Blackboard(data=device_batch)
            
            # 2. Resolve execution plan (lazy or static)
            if self.lazy:
                lazy_order = _resolve_lazy_plan(self._graph, blackboard)
                plan = [self._graph.components[i] for i in lazy_order]
            else:
                plan = static_plan

            # 3. DAG-sorted Pipeline Execution with Profiling
            profiles = {}
            diffs: List[BlackboardDiff] = []
            for component in plan:
                comp_name = type(component).__name__
                t0 = time.perf_counter()
                
                # Runtime Validation (Strict Mode)
                if self.strict:
                    component.validate(blackboard)
                
                # Snapshot before (if diffing)
                if self.diff:
                    snap_before = snapshot_blackboard(blackboard)

                outputs = component.execute(blackboard)
                apply_updates(blackboard, outputs)
                
                # Snapshot after and compute diff
                if self.diff:
                    snap_after = snapshot_blackboard(blackboard)
                    diffs.append(diff_snapshots(comp_name, snap_before, snap_after, blackboard))

                profiles[comp_name] = profiles.get(comp_name, 0) + (time.perf_counter() - t0)

                if blackboard.meta.get("stop_execution"):
                    break
            
            # 4. Telemetry Output
            t_now = time.perf_counter()
            throughput = len(device_batch.get("actions", [0])) / (t_now - t_last)
            t_last = t_now
            
            blackboard.meta["learner_throughput"] = throughput
            blackboard.meta["component_profiles_ms"] = {
                k: round(v * 1000, 3) for k, v in profiles.items()
            }
            if self.lazy:
                blackboard.meta["lazy_skipped"] = len(static_plan) - len(plan)
            if self.diff:
                blackboard.meta["blackboard_diffs"] = diffs
            
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
