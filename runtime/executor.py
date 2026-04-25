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
from runtime.tracing import TraceLogger
from runtime.io.collator import ReplayCollator
import time

from runtime.operator_registry import (
    OPERATOR_REGISTRY,
    register_operator,
    ValidatedOperator,
)

from runtime.registry import register_spec, get_spec, register_base_specs

# Register built-in operator specs (Backward, AccumulateGrad, OptimizerStepEvery,
# MSELoss, Mean, WeightedSum, etc.) so strict compilation finds them without
# requiring agents to bootstrap them individually.
register_base_specs()


# Register built-in operators
from runtime.io.transfer import register_transfer_operators

register_transfer_operators(register_operator)

# Built-in operators
from core.graph import NODE_TYPE_SOURCE, NODE_TYPE_REPLAY_QUERY

register_operator(NODE_TYPE_SOURCE, lambda node, inputs, context=None: NoOp())


def op_replay_query(node, inputs, context=None):
    # Prioritize explicit buffer object (legacy/test support)
    rb = node.params.get("replay_buffer")

    if rb is None and context:
        buffer_id = node.params.get("buffer_id", "main")
        try:
            rb = context.get_buffer(buffer_id)
        except (KeyError, AttributeError):
            pass

    if rb is None:
        return Skipped(f"buffer_{buffer_id}_not_found")

    min_size = node.params.get("min_size", 0)

    if len(rb) < min_size:
        return Skipped(f"buffer_size_{len(rb)}_under_min_{min_size}")

    batch_size = node.params.get("batch_size", 32)
    filters = node.params.get("filters")
    temporal_window = node.params.get("temporal_window")
    contiguous = node.params.get("contiguous", False)
    collator = node.params.get("collator")
    if collator is None and node.schema_out and node.schema_out.fields:
        # Create a temporary collator from the output schema
        collator = ReplayCollator(node.schema_out)

    sampling_seed = (
        context.seed + context.learner_step
        if context and context.seed is not None
        else None
    )
    batch = rb.sample_query(
        batch_size=batch_size,
        filters=filters,
        temporal_window=temporal_window,
        contiguous=contiguous,
        seed=sampling_seed,
    )

    # TODO: should we always require a collator?
    if batch and collator:
        return collator(batch)
    return batch


register_operator(NODE_TYPE_REPLAY_QUERY, op_replay_query)

from core.graph import (
    NODE_TYPE_TARGET_SYNC,
    NODE_TYPE_EXPLORATION,
    NODE_TYPE_METRICS_SINK,
)
from ops.rl.sync import op_target_sync
from ops.rl.exploration import op_epsilon_greedy
from ops.rl.metrics import op_metrics_sink
from ops.math.schedule import op_linear_decay

register_operator(NODE_TYPE_TARGET_SYNC, op_target_sync)
register_operator(NODE_TYPE_EXPLORATION, op_epsilon_greedy)
register_operator(NODE_TYPE_METRICS_SINK, op_metrics_sink)
register_operator("LinearDecay", op_linear_decay)

from ops.loss.math import op_mse_loss
from ops.math.reduce import op_reduce_mean, op_weighted_sum
from ops.math.clip import op_clip

register_operator("MSELoss", op_mse_loss)
register_operator("Mean", op_reduce_mean)
register_operator("WeightedSum", op_weighted_sum)
register_operator("Clip", op_clip)

from ops.rl.learner import (
    op_backward,
    op_grad_buffer,
    op_accumulate_grad,
    op_optimizer_step_every,
    op_optimizer_step,
)

register_operator("Backward", op_backward)
register_operator("GradBuffer", op_grad_buffer)
register_operator("AccumulateGrad", op_accumulate_grad)
register_operator("OptimizerStepEvery", op_optimizer_step_every)
register_operator("Optimizer", op_optimizer_step)


def execute(
    graph: Graph,
    initial_inputs: Dict[NodeId, Any],
    context: Optional[ExecutionContext] = None,
    tracer: Optional[TraceLogger] = None,
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
        from compiler.passes.validate_ir_purity import validate_ir_purity

        purity_report = validate_ir_purity(graph)
        if purity_report.has_errors():
            from compiler.validation import SEVERITY_ERROR

            issues = purity_report.get_issues_by_severity(SEVERITY_ERROR)
            raise RuntimeError(
                f"Boundary Violation: Attempted to execute a 'dirty' graph with live objects in params.\n"
                f"{issues[0].message}"
            )

    context = context or ExecutionContext()
    if tracer:
        tracer.start_step()

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
            if tracer:
                tracer.record_node(
                    node_id=nid, inputs={}, outputs=node_outputs[nid], runtime_ms=0.0
                )
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
                if unknown:
                    raise RuntimeError(
                        f"Contract Violation: Node '{nid}' ({node.node_type}) "
                        f"received undeclared ports {unknown}. "
                        f"Allowed ports are {allowed}."
                    )
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
            start_time = time.perf_counter()
            output = op_func(node, unwrapped_inputs, context=context)
            runtime_ms = (time.perf_counter() - start_time) * 1000

        # Ensure output is a RuntimeValue
        if not isinstance(output, RuntimeValue):
            if output is None:
                output = NoOp()
            else:
                output = Value(output)

        final_skip_reason = skip_reason
        if isinstance(output, Skipped) and not final_skip_reason:
            final_skip_reason = output.reason

        if tracer:
            tracer.record_node(
                node_id=nid,
                inputs=unwrapped_inputs,
                outputs=output,
                runtime_ms=runtime_ms,
                skipped_reason=final_skip_reason,
            )

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
