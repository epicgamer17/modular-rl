"""
Minimal runtime executor for the RL IR.
Handles topological sorting and sequential execution of graph nodes.
"""

import torch
from typing import Dict, Any, List, Set, Callable, Optional
from core.graph import Graph, NodeId, Node
from runtime.context import ExecutionContext
from runtime.refs import RuntimeValue, Value
from runtime.signals import NoOp, Skipped, MissingInput
from runtime.errors import ExecutionError
from runtime.errors import ExecutionError

from runtime.io.collator import ReplayCollator
import time

from runtime.operator_registry import (
    OPERATOR_REGISTRY,
    register_operator,
    ValidatedOperator,
)

# The executor does not register operators itself.
# Operators should be registered via runtime/bootstrap.py before execution.


def execute(
    graph: Graph,
    initial_inputs: Dict[NodeId, Any],
    context: Optional[ExecutionContext] = None,
    validate_purity: bool = True,
) -> Dict[NodeId, Any]:
    """
    Executes the graph using the provided initial inputs and runtime context.

    Steps:
    0. Validate that the graph is pure.
    1. Perform topological sort of the graph.
    2. Iterate through nodes in order.
    3. For each node, gather outputs from predecessors as inputs.
    4. Call the registered operator function.
    5. Store the output for downstream nodes.

    Boundary Definition:
    - Compile-time IR (Graph): Pure declarative configuration. Must NOT contain live objects.
    - Runtime Context (ExecutionContext): Container for mutable live objects (Models, Buffers, etc.).

    Args:
        graph: The static, declarative computation graph (Compile-time IR).
        initial_inputs: Raw data inputs for source nodes.
        context: The stateful runtime.io.environment (Runtime Context).
        tracer: Optional logger for execution traces.
        validate_purity: If True, performs a quick check to ensure the graph is pure.

    Returns:
        A dictionary mapping NodeId to their computed outputs.
    """
    # 0. Explicit Boundary Enforcement
    if validate_purity:
        from compiler.passes.semantic.serialization import validate_ir_purity

        purity_report = validate_ir_purity(graph)
        if purity_report.has_errors():
            from compiler.validation import SEVERITY_ERROR

            issues = purity_report.get_issues_by_severity(SEVERITY_ERROR)
            raise RuntimeError(
                f"Boundary Violation: Attempted to execute a 'dirty' graph with live objects in params.\n"
                f"{issues[0].message}"
            )

    context = context or ExecutionContext()

    # 1. Topological Sort (Kahn's Algorithm)
    order = _topological_sort(graph)

    # 2. Execution
    node_outputs: Dict[NodeId, Any] = {}

    # Seed initial inputs (for source nodes)
    for nid, val in initial_inputs.items():
        node_outputs[nid] = val

    for nid in order:
        if nid in node_outputs and nid in initial_inputs:
            # Source node already materialized
            continue

        node = graph.nodes[nid]

        # Gather inputs from predecessors
        # We distinguish between mapped (named port) and unmapped (control) inputs
        mapped_inputs = {}
        all_predecessor_outputs = {}  # For skip propagation

        for edge in graph.edges:
            if edge.dst == nid:
                if edge.src not in node_outputs:
                    raise RuntimeError(
                        f"Input {edge.src} for node {nid} not yet computed."
                    )

                val = node_outputs[edge.src]
                all_predecessor_outputs[edge.src] = val

                # If src_port is specified, extract that specific output.
                # Unwrap Value so we can inspect the underlying data (dicts
                # and custom structs are wrapped in Value by the executor).
                if edge.src_port:
                    inner = val.data if isinstance(val, Value) else val
                    if isinstance(inner, dict) and edge.src_port in inner:
                        extracted = inner[edge.src_port]
                        val = Value(extracted) if isinstance(val, Value) else extracted
                    elif not isinstance(inner, dict) and hasattr(inner, edge.src_port):
                        extracted = getattr(inner, edge.src_port)
                        val = Value(extracted) if isinstance(val, Value) else extracted
                    # else: single-output operators return the raw value directly;
                    # src_port is a label in that case, so pass val through unchanged.

                if edge.dst_port:
                    mapped_inputs[edge.dst_port] = val

        # 1.5 Inject default values and validate contract
        from runtime.registry import get_spec

        op_spec = get_spec(node.node_type)
        final_inputs = {}

        if op_spec:
            # A. Start with all provided inputs
            final_inputs = dict(mapped_inputs)

            # B. Enforce required ports from spec
            for port_name, p_spec in op_spec.inputs.items():
                if port_name not in final_inputs:
                    if p_spec.default is not None:
                        final_inputs[port_name] = Value(p_spec.default)
                    elif p_spec.required and not p_spec.variadic:
                        raise RuntimeError(
                            f"Contract Violation: Node '{nid}' ({node.node_type}) "
                            f"missing required port '{port_name}'."
                        )

            # C. Check for unknown ports (unless variadic)
            has_variadic = any(p.variadic for p in op_spec.inputs.values())
            if not has_variadic:
                allowed = set(op_spec.inputs.keys())
                received = set(final_inputs.keys())
                unknown = received - allowed
                for k in unknown:
                    del final_inputs[k]
        else:
            # No spec registered, fallback to all mapped inputs (legacy support)
            final_inputs = mapped_inputs

        # 2. Propagate Skip if any input is MissingInput or Error
        skip_reason = None
        for k, v in all_predecessor_outputs.items():
            if isinstance(v, (MissingInput, ExecutionError)):
                skip_reason = f"Predecessor {k} failed or missing input"
                break
            if isinstance(v, Skipped):
                skip_reason = f"Predecessor {k} was skipped"
                break

        # 3. Unwrap inputs for operator
        unwrapped_inputs = {}
        for k, v in final_inputs.items():
            # Unwrap Value objects, pass other things as is (might be raw if from initial_inputs)
            unwrapped_inputs[k] = v.data if isinstance(v, Value) else v

        # Execute operator
        if node.node_type not in OPERATOR_REGISTRY:
            raise RuntimeError(
                f"No operator registered for node type: {node.node_type}"
            )

        op_func = OPERATOR_REGISTRY[node.node_type]

        if skip_reason:
            output = Skipped(skip_reason)
            runtime_ms = 0.0
        else:
            from observability.tracing.event_schema import get_emitter, EventType, Event

            emitter = get_emitter()

            emitter.emit(
                Event(
                    type=EventType.NODE_ENTER,
                    name=node.node_type,
                    metadata={"node_id": str(nid)},
                )
            )

            start_time = time.perf_counter()
            output = op_func(node, unwrapped_inputs, context=context)
            runtime_ms = (time.perf_counter() - start_time) * 1000

            emitter.emit(
                Event(
                    type=EventType.NODE_EXIT,
                    name=node.node_type,
                    duration=runtime_ms,
                    metadata={"node_id": str(nid)},
                )
            )

        # Ensure output is a RuntimeValue
        if not isinstance(output, RuntimeValue):
            if output is None:
                output = NoOp()
            else:
                output = Value(output)

        final_skip_reason = skip_reason
        if isinstance(output, Skipped) and not final_skip_reason:
            final_skip_reason = output.reason

        node_outputs[nid] = output

    return node_outputs


def _topological_sort(graph: Graph) -> List[NodeId]:
    """Returns a list of NodeIds in topological order."""
    # Build in-degree map
    in_degree = {nid: 0 for nid in graph.nodes}
    for edges in graph.adjacency_list.values():
        for dst in edges:
            in_degree[dst] += 1

    # Queue for nodes with 0 in-degree
    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    order = []

    while queue:
        u = queue.pop(0)
        order.append(u)

        for v in graph.adjacency_list.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(order) != len(graph.nodes):
        raise ValueError("Graph contains cycles; cannot perform topological sort.")

    return order


# Registration should be triggered explicitly by the agent or entry point
# from ops.registry import register_all_operators
# register_all_operators()
