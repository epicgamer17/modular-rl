from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext

def op_sample_all(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> Dict[str, Any]:
    """SampleBatch(buffer_id) -> all transitions from buffer."""
    buffer_id = node.params.get("buffer_id", "main")
    buffer = context.get_buffer(buffer_id)
    return buffer.get_all()

def op_sample_batch(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> Dict[str, Any]:
    """SampleBatch(buffer_id, batch_size) -> random batch from buffer."""
    buffer_id = node.params.get("buffer_id", "main")
    batch_size = node.params.get("batch_size", 64)
    buffer = context.get_buffer(buffer_id)
    return buffer.sample(batch_size)
