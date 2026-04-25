"""
AutoBatching / Vectorization compiler pass.
Transforms single-step operations into batched operations across a batch dimension B.
"""

from typing import Dict, List, Set, Optional, Tuple, Union
from core.graph import Graph, Node, NodeId
from core.schema import Schema, Field, TensorSpec
from core.types import RLType, TensorType, DistributionType, RLTypeCategory
import copy

def vectorize_spec(spec: Union[TensorSpec, Schema]) -> Union[TensorSpec, Schema]:
    """Adds a batch dimension 'B' to the given specification."""
    if isinstance(spec, TensorSpec):
        # Prevent double batching
        if "batched" in spec.tags:
            return spec
            
        new_shape = (-1,) + spec.shape # Use -1 to represent dynamic batch size 'B'
        new_rl_type = spec.rl_type.vectorize() if spec.rl_type else None
        
        return TensorSpec(
            shape=new_shape,
            dtype=spec.dtype,
            tags=spec.tags + ["batched"],
            rl_type=new_rl_type
        )
    elif isinstance(spec, Schema):
        return Schema(fields=[Field(f.name, vectorize_spec(f.spec)) for f in spec.fields])
    return spec

def vectorize_graph(graph: Graph) -> Graph:
    """
    Transforms the entire graph to operate on batches of data.
    Effectively applies 'vmap' semantics to the computation graph.
    """
    new_nodes = {}
    
    # Iterate through all nodes and vectorize their schemas
    for nid, node in graph.nodes.items():
        # Some nodes might be exempt from autobatching (e.g., Sinks that handle their own batching)
        if "no_batch" in node.tags:
            new_nodes[nid] = copy.deepcopy(node)
            continue
            
        new_schema_in = vectorize_spec(node.schema_in)
        new_schema_out = vectorize_spec(node.schema_out)
        
        # Create the vectorized node
        new_nodes[nid] = Node(
            node_id=node.node_id,
            node_type=node.node_type,
            schema_in=new_schema_in,
            schema_out=new_schema_out,
            params=copy.deepcopy(node.params),
            tags=node.tags + ["vectorized"]
        )
        
    # Construct the new graph
    new_graph = Graph()
    new_graph.nodes = new_nodes
    new_graph.edges = copy.deepcopy(graph.edges)
    
    # Rebuild adjacency
    for edge in new_graph.edges:
        if edge.src not in new_graph._adjacency:
            new_graph._adjacency[edge.src] = set()
        new_graph._adjacency[edge.src].add(edge.dst)
        
    return new_graph
