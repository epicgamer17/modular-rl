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
from core.contracts import Key


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
        terminal_indices: Indices of components that are terminal sinks
                          (losses.*, meta.*, or side-effect components).
    """

    components: Tuple[PipelineComponent, ...]
    edges: Dict[int, FrozenSet[int]]
    execution_order: Tuple[int, ...]
    pruned_indices: FrozenSet[int]
    provider_map: Dict[str, int]
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

def _get_provides_paths(component: PipelineComponent) -> Set[str]:
    """Extract the set of provided path strings from a component."""
    provides = component.provides
    provides_items = (
        provides.items() if isinstance(provides, dict) else [(k, "new") for k in provides]
    )
    return {k.path for k, _ in provides_items}

def _get_provides_with_modes(component: PipelineComponent) -> Dict[str, str]:
    """Extract {path: write_mode} from a component's provides."""
    provides = component.provides
    provides_items = (
        provides.items() if isinstance(provides, dict) else [(k, "new") for k in provides]
    )
    return {k.path: mode for k, mode in provides_items}


# ──────────────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────────────

# Prefixes that mark a component as a terminal sink (never pruned).
_DEFAULT_TERMINAL_PREFIXES: Tuple[str, ...] = ("losses.", "meta.")


def build_execution_graph(
    components: List[PipelineComponent],
    initial_keys: Set[Key],
    terminal_prefixes: Optional[Tuple[str, ...]] = None,
) -> ExecutionGraph:
    """Build, sort, and prune the execution graph.

    Args:
        components:        Pipeline components in the user's preferred order.
        initial_keys:      Keys available before any component runs
                           (e.g. from the replay-buffer batch).
        terminal_prefixes: Path prefixes that mark a component as a
                           terminal sink.  Defaults to ``("losses.", "meta.")``.

    Returns:
        An immutable ``ExecutionGraph``.

    Raises:
        RuntimeError: On missing dependencies or dependency cycles.
    """
    if terminal_prefixes is None:
        terminal_prefixes = _DEFAULT_TERMINAL_PREFIXES

    n = len(components)
    comp_tuple = tuple(components)

    # ── 1. Build provider map ────────────────────────────────────────
    provider_map: Dict[str, int] = {}

    # Seed with initial keys
    for key in initial_keys:
        provider_map[key.path] = _INITIAL

    # Register each component's provides
    for idx, comp in enumerate(components):
        for path in _get_provides_paths(comp):
            provider_map[path] = idx

    # ── 2. Build dependency edges ────────────────────────────────────
    edges: Dict[int, Set[int]] = {i: set() for i in range(n)}
    missing_deps: Dict[int, List[str]] = {}

    for idx, comp in enumerate(components):
        for req in comp.requires:
            provider_idx = provider_map.get(req.path)
            if provider_idx is None:
                missing_deps.setdefault(idx, []).append(req.path)
            elif provider_idx != _INITIAL:
                edges[idx].add(provider_idx)

    if missing_deps:
        lines = ["Missing dependencies in pipeline DAG:"]
        for idx, paths in missing_deps.items():
            name = type(components[idx]).__name__
            lines.append(f"  [{idx}] {name}: needs {paths}")
        lines.append(f"  Available paths: {sorted(provider_map.keys())}")
        raise RuntimeError("\n".join(lines))

    # ── 3. Topological sort (Kahn's, stable) ─────────────────────────
    frozen_edges: Dict[int, FrozenSet[int]] = {
        i: frozenset(deps) for i, deps in edges.items()
    }
    topo_order = _topological_sort(n, edges)

    # ── 4. Prune unreachable components ──────────────────────────────
    terminal_indices = _find_terminal_sinks(
        components, frozen_edges, terminal_prefixes
    )
    reachable = _backward_reachability(n, frozen_edges, terminal_indices)
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
        terminal_indices=frozenset(terminal_indices),
    )

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


def _find_terminal_sinks(
    components: List[PipelineComponent],
    edges: Dict[int, FrozenSet[int]],
    terminal_prefixes: Tuple[str, ...],
) -> Set[int]:
    """Identify components that are terminal sinks (always kept).

    A component is terminal if:
    - It writes to a terminal-prefixed path (losses.*, meta.*), OR
    - It has no provides (side-effect component like a buffer writer).
    """
    terminals: Set[int] = set()
    for idx, comp in enumerate(components):
        provides_paths = _get_provides_paths(comp)

        # No provides → side-effect component, always keep
        if not provides_paths:
            terminals.add(idx)
            continue

        # Writes to a terminal prefix → always keep
        for path in provides_paths:
            if any(path.startswith(prefix) for prefix in terminal_prefixes):
                terminals.add(idx)
                break

    return terminals


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
    provider_map: Dict[str, int],
) -> Dict[int, FrozenSet[int]]:
    """Build a map of provider_idx → {consumer indices that read its outputs}.

    Only considers active (non-pruned) components in *execution_order*.
    """
    active_set = set(execution_order)
    consumers: Dict[int, Set[int]] = defaultdict(set)

    for idx in execution_order:
        comp = components[idx]
        for req in comp.requires:
            provider_idx = provider_map.get(req.path, _INITIAL)
            if provider_idx != _INITIAL and provider_idx in active_set:
                consumers[provider_idx].add(idx)

    return {idx: frozenset(s) for idx, s in consumers.items()}
