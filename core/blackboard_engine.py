from typing import Dict, Any, Iterable, Iterator, List, Set, Type, Optional
import torch
import time

from core.blackboard import Blackboard
from core.component import PipelineComponent
from core.contracts import Key, WriteMode
from core.execution_graph import (
    ExecutionGraph,
    build_execution_graph,
    _get_provides_with_modes,
)
from core.blackboard_diff import snapshot_blackboard, diff_snapshots, BlackboardDiff
from core.path_resolver import resolve_blackboard_path, write_blackboard_path
from core.shape_validation import validate_tensor


def apply_updates(blackboard: Blackboard, updates: Dict[str, Any]) -> None:
    """
    Applies a dictionary of path-based updates to the blackboard.
    Used by the engine and meta-components to apply returned mutations.
    """
    if not updates:
        return
    for path, value in updates.items():
        write_blackboard_path(blackboard, path, value)


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
            mode in (WriteMode.OVERWRITE, WriteMode.APPEND)
            for mode in provides_modes.values()
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
        target_keys: Optional[Set[Key]] = None,
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
        self.target_keys = target_keys

        # Build the DAG execution graph (includes dependency + cycle checks)
        self._graph: ExecutionGraph = build_execution_graph(
            components, initial_keys, target_keys
        )

    @property
    def execution_graph(self) -> ExecutionGraph:
        """Read-only access to the execution graph for introspection."""
        return self._graph

    def step(
        self,
        batch_iterator: Iterable[Dict[str, Any]],
        _restricted_plan: Optional[List[int]] = None,
    ) -> Iterator[Dict[str, Any]]:
        t_last = time.perf_counter()

        for batch in batch_iterator:
            # 1. Universal Time Mandate: Move Batch to Device explicitly here
            device_batch = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            blackboard = Blackboard(data=device_batch)

            # 2. Resolve execution plan (lazy or dynamic subgraph)
            if _restricted_plan is not None:
                plan_indices = _restricted_plan
            elif self.lazy:
                plan_indices = _resolve_lazy_plan(self._graph, blackboard)
            else:
                plan_indices = self._graph.execution_order

            plan = [self.components[i] for i in plan_indices]

            # 3. DAG-sorted Pipeline Execution
            blackboard = self._execute_batch(blackboard, plan)

            # 4. Telemetry Output
            t_now = time.perf_counter()
            throughput = len(device_batch.get("actions", [0])) / (t_now - t_last)
            t_last = t_now

            blackboard.meta["learner_throughput"] = throughput
            if self.lazy:
                blackboard.meta["lazy_skipped"] = len(
                    self._graph.execution_order
                ) - len(plan)

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

    def step_until(
        self, batch_iterator: Iterable[Dict[str, Any]], target_keys: Set[Key]
    ) -> Iterator[Dict[str, Any]]:
        """
        Similar to step(), but only executes the subgraph needed to reach target_keys.

        This is useful for evaluation (stopping before losses) or introspection.
        """
        subgraph_indices = self._graph.subgraph(target_keys)
        return self.step(batch_iterator, _restricted_plan=subgraph_indices)

    def run_until(self, batch: Dict[str, Any], target_keys: Set[Key]) -> Dict[str, Any]:
        """
        Runs a single batch through the subgraph needed to reach target_keys and returns results.

        Convenience for: next(engine.step_until([batch], target_keys))
        """
        return next(self.step_until([batch], target_keys))

    def _execute_batch(
        self, blackboard: Blackboard, plan: List[PipelineComponent]
    ) -> Blackboard:
        """Core execution loop for a pre-resolved plan on a single blackboard."""
        profiles = {}
        diffs: List[BlackboardDiff] = []

        for component in plan:
            comp_name = type(component).__name__
            t0 = time.perf_counter()

            if self.strict:
                component.validate(blackboard)

            if self.diff:
                snap_before = snapshot_blackboard(blackboard)

            # --- PURE TRANSFORM ENFORCEMENT ---
            # We pass a read-only (frozen) view of the blackboard to the component.
            # Any attempt to do blackboard.data['x'] = y inside execute() will now raise a TypeError.
            outputs = component.execute(blackboard.frozen())
            
            # Apply explicitly returned updates to the live (mutable) blackboard
            apply_updates(blackboard, outputs)

            if self.strict:
                provides_modes = _get_provides_with_modes(component)
                for key in provides_modes:
                    if key.path in outputs:
                        validate_tensor(key, outputs[key.path])

            if self.diff:
                snap_after = snapshot_blackboard(blackboard)
                diffs.append(
                    diff_snapshots(comp_name, snap_before, snap_after, blackboard)
                )

            profiles[comp_name] = profiles.get(comp_name, 0) + (
                time.perf_counter() - t0
            )

            if blackboard.meta.get("stop_execution"):
                break

        # Store metadata accumulated during graph run
        blackboard.meta["component_profiles_ms"] = {
            k: round(v * 1000, 3) for k, v in profiles.items()
        }
        if self.diff:
            blackboard.meta["blackboard_diffs"] = diffs

        return blackboard
