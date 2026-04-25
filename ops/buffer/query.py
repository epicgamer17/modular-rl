"""
Operator for querying a replay buffer.
"""

from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.signals import Skipped, NoOp
from runtime.io.collator import ReplayCollator
from runtime.registry import OperatorSpec, TransitionBatch

REPLAY_QUERY_SPEC = OperatorSpec.create(
    "ReplayQuery",
    inputs={},
    outputs={"default": TransitionBatch},
    pure=False,
    stateful=True,
    allowed_contexts={"learner"},
    requires_buffers=["main"],
    math_category="buffer_io",
    differentiable=False,
    creates_grad=False,
    consumes_grad=False,
    updates_params=False,
)

SAMPLE_BATCH_SPEC = OperatorSpec.create(
    name="SampleBatch",
    inputs={},
    outputs={"batch": TransitionBatch},
    pure=False,
    stateful=True,
    reads_buffer=True,
    math_category="buffer_io",
    allowed_contexts={"actor", "learner"},
    differentiable=False,
    creates_grad=False,
    consumes_grad=False,
    updates_params=False,
)

def op_replay_query(node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None) -> Any:
    """
    Queries a replay buffer for a batch of transitions.
    """
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
