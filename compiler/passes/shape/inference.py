from typing import Dict, List, Set, Optional
from core.graph import Graph, Node, NodeId
from runtime.registry import get_spec, Spec, Schema, Field, PortSpec, is_compatible
import copy


def infer_shapes(graph: Graph) -> Graph:
    """
    Infers output shapes for all nodes in the graph based on their inputs.
    Returns a new Graph with updated node schemas.
    """
    # 1. Build dependency info
    in_degree = {nid: 0 for nid in graph.nodes}
    for edge in graph.edges:
        if edge.dst in in_degree:
            in_degree[edge.dst] += 1

    # 2. Topological sort (Kahn's algorithm)
    queue = [nid for nid, d in in_degree.items() if d == 0]
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for edge in graph.edges:
            if edge.src == u and edge.dst in in_degree:
                v = edge.dst
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

    # 3. Propagate shapes
    new_nodes = {}
    for nid in topo_order:
        node = graph.nodes[nid]
        spec = get_spec(node.node_type)

        if not spec:
            new_nodes[nid] = node
            continue

        # Collect input specs from predecessors
        inputs_to_node = {}
        for edge in graph.edges:
            if edge.dst == nid:
                src_node = new_nodes.get(edge.src)
                if not src_node:
                    # Might happen if there's a cycle or not all predecessors in topo_order
                    src_node = graph.nodes[edge.src]

                src_spec = get_spec(src_node.node_type)
                if not src_spec:
                    continue

                # Source port determination
                if "default" in src_spec.outputs:
                    src_port_name = "default"
                elif len(src_spec.outputs) == 1:
                    src_port_name = next(iter(src_spec.outputs.keys()))
                else:
                    continue

                src_port_type = None
                # Check if it was already inferred in src_node.schema_out
                for field in src_node.schema_out.fields:
                    if field.name == src_port_name:
                        src_port_type = field.spec
                        break

                if not src_port_type:
                    src_port_type = src_spec.outputs[src_port_name].spec

                # Destination port determination
                dst_port_name = edge.dst_port
                if not dst_port_name:
                    # Auto-wiring logic (simplified)
                    compatible = [
                        p
                        for p, ps in spec.inputs.items()
                        if is_compatible(src_port_type, ps.spec)
                    ]
                    if len(compatible) == 1:
                        dst_port_name = compatible[0]

                if dst_port_name:
                    inputs_to_node[dst_port_name] = src_port_type

        # Run inference
        new_schema_out = copy.deepcopy(node.schema_out)

        # Populate schema_out from static spec if it's currently empty
        # TODO: do we need this isinstance check? can we get rid of that?
        is_empty_schema = (
            isinstance(new_schema_out, Schema) and not new_schema_out.fields
        )
        if is_empty_schema:
            new_schema_out = Schema(
                fields=[Field(name, ps.spec) for name, ps in spec.outputs.items()]
            )

        if spec.shape_fn:
            try:
                # shape_fn(inputs: Dict[str, Spec]) -> Dict[str, Spec]
                inferred = spec.shape_fn(inputs_to_node)
                current_map = new_schema_out.get_field_map()
                for port, new_spec in inferred.items():
                    current_map[port] = new_spec
                new_schema_out = Schema(
                    fields=[Field(k, v) for k, v in current_map.items()]
                )
            except Exception:
                # If inference fails, keep existing/static schema
                pass

        # Update schema_in for completeness
        new_schema_in = Schema(fields=[Field(k, v) for k, v in inputs_to_node.items()])

        new_nodes[nid] = Node(
            node_id=node.node_id,
            node_type=node.node_type,
            schema_in=new_schema_in,
            schema_out=new_schema_out,
            params=node.params,
            tags=node.tags,
        )
        
        from observability.tracing.event_schema import get_emitter, EventType, Event
        get_emitter().emit(Event(
            type=EventType.SHAPE_INFERENCE,
            name=node.node_type,
            metadata={
                "node_id": str(nid),
                "schema_in": str(new_schema_in),
                "schema_out": str(new_schema_out)
            }
        ))


    # Handle nodes that weren't in topo_order (cycles)
    for nid in graph.nodes:
        if nid not in new_nodes:
            new_nodes[nid] = graph.nodes[nid]

    # Reconstruct the graph
    new_graph = Graph()
    new_graph.nodes = new_nodes
    new_graph.edges = copy.deepcopy(graph.edges)
    # Rebuild adjacency
    for edge in new_graph.edges:
        if edge.src not in new_graph._adjacency:
            new_graph._adjacency[edge.src] = set()
        new_graph._adjacency[edge.src].add(edge.dst)

    return new_graph
