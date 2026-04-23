from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from core.graph import NodeId, Node
import time

if TYPE_CHECKING:
    from runtime.executor import OPERATOR_REGISTRY
    from runtime.context import ExecutionContext

@dataclass
class NodeTrace:
    node_id: NodeId
    inputs: Dict[str, Any]
    outputs: Any
    runtime_ms: float
    skipped_reason: Optional[str] = None

class ExecutionTrace:
    def __init__(self):
        self.nodes: Dict[NodeId, NodeTrace] = {}
        
    def add_node_trace(self, trace: NodeTrace):
        self.nodes[trace.node_id] = trace

    def run(self, graph: 'Graph', context: 'ExecutionContext'):
        """
        Replays the execution trace deterministically by re-running operators
        with the recorded inputs and asserting matching outputs.
        """
        from runtime.executor import OPERATOR_REGISTRY
        from runtime.values import Value, Skipped, NoOp
        from core.graph import NODE_TYPE_SOURCE
        
        for nid, trace in self.nodes.items():
            if trace.skipped_reason:
                continue
                
            node = graph.nodes[nid]
            if node.node_type == NODE_TYPE_SOURCE:
                # Source nodes are inputs, they don't have operators to replay
                continue
                
            op_func = OPERATOR_REGISTRY[node.node_type]
            
            # Re-run operator
            result = op_func(node, trace.inputs, context=context)
            
            # Unwrap for comparison
            actual = result.data if isinstance(result, Value) else result
            expected = trace.outputs.data if isinstance(trace.outputs, Value) else trace.outputs
            
            # Basic equality check for replay verification
            # Note: Tensor comparison might need torch.allclose in real RL
            assert actual == expected, f"Replay mismatch at node {nid}: expected {expected}, got {actual}"

class TraceLogger:
    def __init__(self):
        self.traces: List[ExecutionTrace] = []
        self._current_trace: Optional[ExecutionTrace] = None
        
    def start_step(self):
        self._current_trace = ExecutionTrace()
        self.traces.append(self._current_trace)
        
    def record_node(self, node_id: NodeId, inputs: Dict[str, Any], outputs: Any, runtime_ms: float, skipped_reason: Optional[str] = None):
        if self._current_trace:
            self._current_trace.add_node_trace(NodeTrace(
                node_id=node_id,
                inputs=inputs,
                outputs=outputs,
                runtime_ms=runtime_ms,
                skipped_reason=skipped_reason
            ))

    def get_step(self, step: int) -> ExecutionTrace:
        return self.traces[step]
