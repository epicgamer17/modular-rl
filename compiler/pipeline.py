"""
Main Compiler entry point for the RL IR.
Orchestrates validation passes and optimization.
"""

from typing import Set, Optional, Any
from core.graph import Graph
from compiler.validation import ValidationReport, SEVERITY_ERROR, SEVERITY_WARN
from compiler.passes.structural import validate_structural
from compiler.passes.semantic import validate_semantic
from compiler.passes.shape import run_shape_analysis, validate_shape_semantics
from compiler.passes.optimization import run_transformations


def compile_graph(
    graph: Graph,
    strict: bool = False,
    model_handles: Optional[Set[str]] = None,
    buffer_handles: Optional[Set[str]] = None,
    context: str = "both",
    optimize: bool = True,
    autobatch: bool = False,
    autodiff_lowering: bool = True,
    optimization_report: Optional[Any] = None,
) -> Graph:
    """
    Compiles an RL IR graph by running all validation passes.

    Boundary Definition:
    - Compile-time IR (Graph): The output of this function is a pure declarative config.
      It must be serializable and contain NO live runtime objects.
    - Runtime Context (ExecutionContext): Handled separately; resolved at runtime via handles.

    Args:
        graph: The graph to compile.
        strict: If True, treats all warnings as errors.
        model_handles: Set of registered model handles in the target environment.
        buffer_handles: Set of registered buffer handles in the target environment.

    Returns:
        The validated graph.

    Raises:
        RuntimeError: If validation fails (due to errors or strict mode warnings).
    """
    from observability.tracing.event_schema import get_emitter, EventType
    import time
    
    emitter = get_emitter()
    start_time = time.time()
    
    emitter.emit_trace(
        name="compile_graph",
        type=EventType.COMPILER_START,
        metadata={"strict": strict, "context": context}
    )


    report = ValidationReport()

    # 1. Structural Pre-check (Metadata)
    with emitter.trace_pass("metadata_check"):
        from compiler.passes.structural.metadata import validate_metadata
        report.merge(validate_metadata(graph, strict=strict))

    # 2. Shape Inference
    with emitter.trace_pass("shape_inference"):
        graph = run_shape_analysis(graph)

    # 3. Optimization and Transformations (Autodiff, Autobatch, Pruning, Fusion)
    # Optimization (DNE) happens here to clean up the graph before detailed validation
    with emitter.trace_pass("transformations"):
        graph = run_transformations(
            graph,
            optimize=optimize,
            autobatch=autobatch,
            autodiff_lowering=autodiff_lowering,
            context=context,
            report=optimization_report
        )

    # 4. Detailed Validation
    # These passes run on the optimized graph
    with emitter.trace_pass("structural_validation"):
        # Connectivity, Ports, Handles
        report.merge(validate_structural(graph, model_handles, buffer_handles, strict=strict))

    with emitter.trace_pass("shape_validation"):
        # Shape consistency and gradient flow
        report.merge(validate_shape_semantics(graph, context=context))

    with emitter.trace_pass("semantic_validation"):
        # RL rules, Context, Domains, Purity
        report.merge(validate_semantic(graph, context=context))

    # Check for hard errors
    if report.has_errors():
        issues = report.get_issues_by_severity(SEVERITY_ERROR)
        error_details = "\n".join(
            [f"[{i.code}] Node {i.node_id}: {i.message}" for i in issues]
        )
        raise RuntimeError(f"Graph compilation failed with errors:\n{error_details}")

    # Strict mode: treat all warnings as fatal errors
    if strict and report.has_warnings():
        issues = report.get_issues_by_severity(SEVERITY_WARN)
        warn_details = "\n".join(
            [f"[{i.code}] Node {i.node_id}: {i.message}" for i in issues]
        )
        raise RuntimeError(
            f"Graph compilation failed in STRICT mode due to warnings:\n{warn_details}"
        )

    emitter.emit_trace(
        name="compile_graph",
        type=EventType.COMPILER_END,
        duration=time.time() - start_time,
        metadata={"nodes": len(graph.nodes), "edges": len(graph.edges)}
    )


    return graph

