"""
Execution Graph — DAG-based pipeline scheduling.

Replaces the linear ``for component in components: execute()`` model with:
1. Explicit dependency edges derived from ``requires`` / ``provides`` contracts.
2. Kahn's-algorithm topological sort (stable: preserves user ordering as tiebreaker).
3. Backward-reachability pruning from terminal sinks.
4. Cycle detection with actionable error messages.
5. Consumer tracking for runtime lazy execution.

The ``ExecutionGraph`` is an immutable snapshot computed once at engine
construction time.  The ``BlackboardEngine`` iterates over its
``execution_order`` at runtime — no graph walking per step.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from core.component import PipelineComponent
from core.contracts import Key, WriteMode, check_shape_compatibility


# ──────────────────────────────────────────────────────────────────────
# Data structure
# ──────────────────────────────────────────────────────────────────────

_INITIAL: int = -1  # sentinel index for paths provided by initial_keys


@dataclass(frozen=True)
class ExecutionGraph:
    """Immutable dependency graph built from component contracts.

    Attributes:
        components:       The original component list (index-stable).
        edges:            ``{comp_idx: frozenset_of_prerequisite_indices}``.
                          An edge ``j in edges[i]`` means *i* depends on *j*.
        execution_order:  Topologically sorted component indices (pruned).
        pruned_indices:   Indices of components removed during pruning.
        provider_map:     ``{path: comp_idx}`` — who provides each path.
                          ``_INITIAL (-1)`` for paths in ``initial_keys``.
        consumer_map:     ``{comp_idx: frozenset_of_downstream_consumer_indices}``.
                          For each active component, which later active components
                          read at least one of its provided paths.
        terminal_indices: Indices of components that are execution targets
                          (produce target_keys or have no provides).
    """

    components: Tuple[PipelineComponent, ...]
    edges: Dict[int, FrozenSet[int]]
    execution_order: Tuple[int, ...]
    pruned_indices: FrozenSet[int]
    provider_map: Dict[Key, int]
    consumer_map: Dict[int, FrozenSet[int]]
    terminal_indices: FrozenSet[int]

    # ── convenience ──────────────────────────────────────────────────

    def active_components(self) -> List[PipelineComponent]:
        """Return components in execution order (no pruned nodes)."""
        return [self.components[i] for i in self.execution_order]

    def summary(self) -> str:
        """One-line human-readable summary."""
        n_total = len(self.components)
        n_active = len(self.execution_order)
        n_pruned = len(self.pruned_indices)
        return (
            f"ExecutionGraph: {n_total} components → "
            f"{n_active} active, {n_pruned} pruned"
        )


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _get_provides_keys(component: PipelineComponent) -> Set[Key]:
    """Extract the set of provided Keys from a component."""
    provides = component.provides
    if isinstance(provides, dict):
        return set(provides.keys())
    return provides

def _get_provides_with_modes(component: PipelineComponent) -> Dict[Key, WriteMode]:
    """Extract {Key: WriteMode} from a component's provides."""
    provides = component.provides
    if isinstance(provides, dict):
        return {
            k: (WriteMode(v) if isinstance(v, str) else v) for k, v in provides.items()
        }
    return {k: WriteMode.NEW for k in provides}


# ──────────────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────────────


def build_execution_graph(
    components: List[PipelineComponent],
    initial_keys: Set[Key],
    target_keys: Optional[Set[Key]] = None,
) -> ExecutionGraph:
    """Build, sort, and prune the execution graph.

    Args:
        components:   Pipeline components in the user's preferred order.
        initial_keys: Keys available before any component runs
                        (e.g. from the replay-buffer batch).
        target_keys: Keys that are execution targets (e.g. loss keys).
                    Components that produce these keys (or have no provides)
                    are guaranteed to execute. If None, no pruning occurs.

    Returns:
        An immutable ``ExecutionGraph``.

    Raises:
        RuntimeError: On missing dependencies or dependency cycles.
    """
    if target_keys is None:
        target_keys = set()

    n = len(components)
    comp_tuple = tuple(components)

    # ── 1. Build provider map ────────────────────────────────────────
    provider_map: Dict[Key, int] = {}
    duplicate_errors: List[Tuple[int, Key, str]] = []

    # Seed with initial keys
    for key in initial_keys:
        provider_map[key] = _INITIAL

    # Register each component's provides (with duplicate detection)
    for idx, comp in enumerate(components):
        provides_modes = _get_provides_with_modes(comp)
        for key, mode in provides_modes.items():
            existing_provider = key in provider_map

            # NEW: must NOT already exist, then register
            if mode == WriteMode.NEW:
                if existing_provider:
                    duplicate_errors.append((idx, key, f"{WriteMode.NEW.value} key already provided"))
                provider_map[key] = idx
            # OVERWRITE: MUST already exist (by initial_keys or previous component)
            elif mode == WriteMode.OVERWRITE:
                if not existing_provider:
                    duplicate_errors.append((idx, key, f"{WriteMode.OVERWRITE.value} but no existing provider"))
                provider_map[key] = idx
            # APPEND: can add to existing
            elif mode == WriteMode.APPEND:
                provider_map[key] = idx
            # OPTIONAL: may or may not exist, allow both
            elif mode == WriteMode.OPTIONAL:
                provider_map[key] = idx
            else:
                raise ValueError(f"Unknown WriteMode: {mode}")

    if duplicate_errors:
        lines = ["Duplicate provider detected in pipeline DAG:"]
        for idx, key, reason in duplicate_errors:
            name = type(components[idx]).__name__
            lines.append(f"  [{idx}] {name}: {key.path} — {reason}")
        raise RuntimeError("\n".join(lines))

    # ── 2. Build dependency edges ────────────────────────────────────
    edges: Dict[int, Set[int]] = {i: set() for i in range(n)}
    missing_deps: Dict[int, List[Key]] = {}

    for idx, comp in enumerate(components):
        for req in comp.requires:
            provider_idx = provider_map.get(req)
            if provider_idx is None:
                missing_deps.setdefault(idx, []).append(req)
            elif provider_idx != _INITIAL:
                edges[idx].add(provider_idx)

    if missing_deps:
        lines = ["Missing dependencies in pipeline DAG:"]
        for idx, keys in missing_deps.items():
            name = type(components[idx]).__name__
            lines.append(f"  [{idx}] {name}: needs {keys}")
        lines.append(f"  Available keys: {sorted(provider_map.keys(), key=lambda k: k.path)}")
        raise RuntimeError("\n".join(lines))

    # ── 3. Topological sort (Kahn's, stable) ─────────────────────────
    frozen_edges: Dict[int, FrozenSet[int]] = {
        i: frozenset(deps) for i, deps in edges.items()
    }
    topo_order = _topological_sort(n, edges)

    # ── 4. Prune unreachable components ──────────────────────────────
    # Find components that produce target keys OR have no provides (side-effects)
    target_component_indices = _find_target_components(
        components, provider_map, target_keys
    )
    reachable = _backward_reachability(n, frozen_edges, target_component_indices)
    pruned = frozenset(set(range(n)) - reachable)

    execution_order = tuple(i for i in topo_order if i not in pruned)

    # ── 5. Build consumer map (for lazy execution) ───────────────────
    consumer_map = _build_consumer_map(components, execution_order, provider_map)

    graph = ExecutionGraph(
        components=comp_tuple,
        edges=frozen_edges,
        execution_order=execution_order,
        pruned_indices=pruned,
        provider_map=provider_map,
        consumer_map=consumer_map,
        terminal_indices=frozenset(target_component_indices),
    )

    # ── 6. Contract Validation Layer ─────────────────────────────────
    # Perform build-time check of shape, dtype, and symbolic consistency.
    _validate_contracts(graph, initial_keys)

    print(graph.summary())
    return graph


# ──────────────────────────────────────────────────────────────────────
# Internal algorithms
# ──────────────────────────────────────────────────────────────────────


def _topological_sort(n: int, edges: Dict[int, Set[int]]) -> Tuple[int, ...]:
    """Kahn's algorithm with original-index tiebreaker for stability.

    Raises ``RuntimeError`` if the graph contains a cycle.
    """
    # Forward adjacency: prerequisite → set of dependents
    forward: Dict[int, Set[int]] = defaultdict(set)
    in_degree = [0] * n
    for node, prereqs in edges.items():
        in_degree[node] = len(prereqs)
        for prereq in prereqs:
            forward[prereq].add(node)

    # Min-heap keyed by original index → stable tiebreaker
    ready: List[int] = []
    for i in range(n):
        if in_degree[i] == 0:
            heapq.heappush(ready, i)

    order: List[int] = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for dependent in forward[node]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                heapq.heappush(ready, dependent)

    if len(order) != n:
        visited = set(order)
        raise RuntimeError(
            f"Dependency cycle detected among {n - len(order)} components. "
            f"Nodes in cycle: indices {[i for i in range(n) if i not in visited]}"
        )

    return tuple(order)


def _find_target_components(
    components: List[PipelineComponent],
    provider_map: Dict[Key, int],
    target_keys: Set[Key],
) -> Set[int]:
    """Identify components that are execution targets (never pruned).

    A component is a target if:
    - It produces one of the explicit target_keys, OR
    - It has no provides (side-effect component like a buffer writer).
    """
    targets: Set[int] = set()
    
    # Find components that produce target keys
    for key in target_keys:
        comp_idx = provider_map.get(key)
        if comp_idx is not None and comp_idx != _INITIAL:
            targets.add(comp_idx)
    
    # Side-effect components (no provides) are always targets
    for idx, comp in enumerate(components):
        if not _get_provides_keys(comp):
            targets.add(idx)

    return targets


def _backward_reachability(
    n: int,
    edges: Dict[int, FrozenSet[int]],
    seeds: Set[int],
) -> Set[int]:
    """Find all nodes reachable by walking backward from *seeds*.

    Returns the set of indices that must be kept (seeds + all ancestors).
    """
    reachable: Set[int] = set()
    stack = list(seeds)

    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        # Walk backward: this node depends on its prerequisites
        for prereq in edges.get(node, frozenset()):
            if prereq not in reachable:
                stack.append(prereq)

    return reachable


def _build_consumer_map(
    components: List[PipelineComponent],
    execution_order: Tuple[int, ...],
    provider_map: Dict[Key, int],
) -> Dict[int, FrozenSet[int]]:
    """Build a map of provider_idx → {consumer indices that read its outputs}.

    Only considers active (non-pruned) components in *execution_order*.
    """
    active_set = set(execution_order)
    consumers: Dict[int, Set[int]] = defaultdict(set)

    for idx in execution_order:
        comp = components[idx]
        for req in comp.requires:
            provider_idx = provider_map.get(req, _INITIAL)
            if provider_idx != _INITIAL and provider_idx in active_set:
                consumers[provider_idx].add(idx)

    return {idx: frozenset(s) for idx, s in consumers.items()}


def _validate_contracts(graph: ExecutionGraph, initial_keys: Set[Key]) -> None:
    """
    Validates the pipeline DAG at build-time.
    Stages:
    1. Dependency Resolution (Path check)
    2. Semantic Compatibility (Type check)
    3. Representation Consistency (Metadata/Parameters check)
    4. Shape Integrity (Shape, Dtype, and Symbolic consistency)
    """
    # Track the full Key object for every available path
    available_contracts: Dict[str, Key] = {k.path: k for k in initial_keys}

    for idx in graph.execution_order:
        component = graph.components[idx]
        missing = []
        incompatibilities = []

        for req in component.requires:
            # Stage 1: Dependency Resolution
            if req.path not in available_contracts:
                missing.append(req.path)
                continue

            found_key = available_contracts[req.path]

            # Stage 2: Semantic Compatibility
            # Generic semantic types must match or found_key must be a subclass
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

            # Stage 4: Shape Integrity (Shape, Dtype, Symbolic)
            shape_issues = check_shape_compatibility(provider=found_key, consumer=req)
            for issue in shape_issues:
                incompatibilities.append(f"SHAPE ERROR for '{req.path}': {issue}")

        if missing or incompatibilities:
            error_header = f"DAG Topology Error at Component [{idx}] '{type(component).__name__}':"
            error_msg = [error_header]
            if missing:
                error_msg.append(f"  Missing Dependencies: {missing}")
            if incompatibilities:
                error_msg.append("  Contract Violations:")
                for inc in incompatibilities:
                    error_msg.append(f"    - {inc}")

            error_msg.append(
                f"  Available keys in namespace: {sorted(list(available_contracts.keys()))}"
            )
            raise RuntimeError("\n".join(error_msg))

        # Update available keys with component provisions
        provides_modes = _get_provides_with_modes(component)
        for prov, mode in provides_modes.items():
            # OVERWRITE requires the key to already exist (checked in build_execution_graph step 1)
            # Register the key for downstream components
            available_contracts[prov.path] = prov

    print(f"DAG Contract Validation Passed: {len(graph.execution_order)} components verified.")
