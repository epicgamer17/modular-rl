"""
Graph validation logic for the RL IR.
Ensures structural, type-level, and semantic correctness of graphs.
"""

from typing import List, Dict, Set, Optional
from core.graph import Graph, EdgeType, NodeId
from core.schema import TAG_ON_POLICY, TAG_ORDERED, TAG_OFF_POLICY

def validate_graph(graph: Graph) -> None:
    """
    Performs full validation on a Graph instance.
    
    Checks:
    - Structural: No cycles (must be a DAG).
    - Connectivity: All nodes must be reachable from at least one source.
    - Type Compatibility: Output schemas must match input schemas for data edges.
    - Semantic Constraints: Rule-based checks for specific node types/tags.
    - Side-Effect Constraints: Reject illegal placements (e.g., optimizer in Actor graph).
    - Operator Spec Enforcement: Required ports must be wired.
    
    Raises:
        ValueError: If any validation rule is violated.
    """
    # 1. Structural Checks
    _check_cycles(graph)
    _check_connectivity(graph)
    
    # 2. Semantic Constraints
    _check_semantic_constraints(graph)
    
    # 3. Side-Effect Constraints
    _check_side_effects(graph)
    
    # 4. Shape Inference and Validation
    _check_shapes(graph)
    
    # 5. Operator Spec Enforcement (Compile-time)
    _check_operator_specs(graph)
    
    # 6. Domain Consistency (Algorithm Assumptions)
    _check_domain_consistency(graph)

    # 7. Legacy Type Compatibility (only if schemas are explicitly set)
    _check_type_compatibility(graph)

def _check_domain_consistency(graph: Graph) -> None:
    """Ensures that operators are used within their intended algorithm family."""
    from runtime.specs import get_spec
    
    # Identify graph's domain from tags (e.g., 'DQN', 'PPO')
    graph_domains = {t.lower() for t in graph.tags}
    
    # Map high-level tags to family tags
    DOMAIN_MAPPING = {
        "ppo": "policy_gradient",
        "sac": "policy_gradient",
        "dqn": "q_learning",
        "nfsp": "q_learning"
    }
    
    target_families = {DOMAIN_MAPPING.get(d) for d in graph_domains if d in DOMAIN_MAPPING}
    target_families.discard(None)
    
    if not target_families:
        return # No domain specified, skip check
        
    for nid, node in graph.nodes.items():
        spec = get_spec(node.node_type)
        if not spec or not spec.domain_tags:
            continue
            
        # If the operator has domain tags, at least one must match the graph's target families
        if not any(tag in target_families for tag in spec.domain_tags):
            print(f"LINT WARNING: Node '{nid}' ({node.node_type}) is a {spec.domain_tags} operator "
                  f"being used in a {graph_domains} graph. Verify algorithm assumptions.")

def _check_shapes(graph: Graph) -> None:
    """Propagates shapes through the graph and validates compatibility."""
    from runtime.specs import get_spec
    from core.schema import TensorSpec
    
    # Map from (NodeId, PortName) -> TensorSpec
    inferred_specs: Dict[Tuple[NodeId, str], TensorSpec] = {}
    
    # Initialize Source nodes
    for nid, node in graph.nodes.items():
        if node.node_type == "Source":
            for field in node.schema_out.fields:
                inferred_specs[(nid, field.name)] = field.spec

    # Topological sort for shape propagation
    sorted_nodes = _topological_sort(graph)
    
    for nid in sorted_nodes:
        node = graph.nodes[nid]
        spec = get_spec(node.node_type)
        if not spec:
            continue
            
        # 1. Collect inputs for this node
        incoming_edges = [e for e in graph.edges if e.dst == nid]
        
        # 2. Check input compatibility with spec
        for e in incoming_edges:
            src_spec = inferred_specs.get((e.src, e.src_port))
            if not src_spec:
                continue # Might be a non-tensor edge or source not yet inferred
                
            dst_port_wrapper = spec.inputs.get(e.dst_port)
            if not dst_port_wrapper:
                continue
            
            dst_port_spec = dst_port_wrapper.spec
            if isinstance(dst_port_spec, TensorSpec):
                if not _is_shape_compatible(dst_port_spec.shape, src_spec.shape):
                    raise ValueError(
                        f"Shape Mismatch: Node '{nid}' port '{e.dst_port}' expects {dst_port_spec.shape}, "
                        f"but received {src_spec.shape} from '{e.src}' port '{e.src_port}'"
                    )
        
        # 3. Infer output shapes
        # For now, we assume outputs in spec are fixed or symbolic
        for port_name, out_wrapper in spec.outputs.items():
            if isinstance(out_wrapper.spec, TensorSpec):
                inferred_specs[(nid, port_name)] = out_wrapper.spec

def _is_shape_compatible(spec_shape: tuple, actual_shape: tuple) -> bool:
    """Checks if actual shape matches specification (handling symbolic dims)."""
    if len(spec_shape) != len(actual_shape):
        return False
    for s, a in zip(spec_shape, actual_shape):
        if s == -2: # AnyShape
            continue
        if s < 0: # Symbolic dimension like B
            continue
        if s != a:
            return False
    return True

def _topological_sort(graph: Graph) -> List[NodeId]:
    """Helper for topological sorting."""
    visited = set()
    stack = []
    
    def visit(nid: NodeId):
        if nid in visited:
            return
        visited.add(nid)
        for neighbor in graph.adjacency_list.get(nid, []):
            visit(neighbor)
        stack.append(nid)
        
    for nid in graph.nodes:
        visit(nid)
    return stack[::-1]

def _check_side_effects(graph: Graph) -> None:
    """Ensures that operators with side-effects are placed in appropriate graphs."""
    from runtime.specs import get_spec
    
    is_actor_graph = "Actor" in graph.tags or "Inference" in graph.tags
    
    for nid, node in graph.nodes.items():
        spec = get_spec(node.node_type)
        if not spec:
            continue
            
        if is_actor_graph:
            if spec.updates_params or spec.math_category == "optimizer":
                raise ValueError(
                    f"Compilation Error: Side-effect violation in Actor graph. "
                    f"Node '{nid}' ({node.node_type}) is an optimizer node or updates parameters, which is illegal in inference loops."
                )
            if spec.reads_buffer or spec.math_category == "buffer_io":
                raise ValueError(
                    f"Compilation Error: Side-effect violation in Actor graph. "
                    f"Node '{nid}' ({node.node_type}) is a buffer node or reads from a replay buffer, which is illegal in inference loops."
                )
            if spec.math_category == "loss":
                raise ValueError(
                    f"Compilation Error: Semantic violation in Actor graph. "
                    f"Node '{nid}' ({node.node_type}) is a loss node, which is illegal in inference loops."
                )

def _check_operator_specs(graph: Graph) -> None:
    """Verifies that nodes satisfy their OperatorSpec requirements."""
    from runtime.specs import get_spec
    
    for nid, node in graph.nodes.items():
        spec = get_spec(node.node_type)
        if not spec:
            continue
            
        # 1. Input Check
        incoming_edges = [edge for edge in graph.edges if edge.dst == nid]
        provided_ports = {edge.dst_port for edge in incoming_edges if edge.dst_port}
        
        required_ports = {
            name for name, port in spec.inputs.items() 
            if port.required and not port.variadic
        }
        missing_ports = required_ports - provided_ports
        
        if missing_ports:
            raise ValueError(
                f"Compilation Error: Node '{nid}' of type '{node.node_type}' "
                f"is missing required input ports: {missing_ports}"
            )
            
        # 2. Check for unknown ports (unless variadic is allowed)
        has_variadic = any(port.variadic for port in spec.inputs.values())
        if not has_variadic:
            allowed_ports = set(spec.inputs.keys())
            unknown_ports = provided_ports - allowed_ports
            if unknown_ports:
                raise ValueError(
                    f"Compilation Error: Node '{nid}' of type '{node.node_type}' "
                    f"received unknown input ports: {unknown_ports}. "
                    f"Allowed ports are: {allowed_ports}"
                )
        
        # TODO: Add type-check for ports if Edge and Schema provide enough info

def _check_cycles(graph: Graph) -> None:
    """Ensures the graph is a Directed Acyclic Graph (DAG)."""
    visited = set()
    path = set()
    
    def visit(nid: NodeId):
        if nid in path:
            raise ValueError(f"Cycle detected in graph at node: {nid}")
        if nid in visited:
            return
        
        path.add(nid)
        for neighbor in graph.adjacency_list.get(nid, []):
            visit(neighbor)
        path.remove(nid)
        visited.add(nid)

    for nid in graph.nodes:
        visit(nid)

def _check_connectivity(graph: Graph) -> None:
    """Basic connectivity check. For now, ensure no completely isolated nodes."""
    if not graph.nodes:
        return
        
    adj = graph.adjacency_list
    # Nodes with incoming edges
    has_incoming = set()
    for edges in adj.values():
        for dst in edges:
            has_incoming.add(dst)
            
    # Nodes with outgoing edges
    has_outgoing = set()
    for src, edges in adj.items():
        if edges:
            has_outgoing.add(src)
            
    for nid in graph.nodes:
        if nid not in has_incoming and nid not in has_outgoing and len(graph.nodes) > 1:
            raise ValueError(f"Isolated node detected: {nid}")

def _check_type_compatibility(graph: Graph) -> None:
    """Verifies that data flows between nodes have compatible schemas."""
    for edge in graph.edges:
        if edge.edge_type == EdgeType.DATA:
            src_node = graph.nodes[edge.src]
            dst_node = graph.nodes[edge.dst]
            
            # Skip if destination has no explicit schema requirements 
            # (likely handled by OperatorSpec)
            if not dst_node.schema_in.fields:
                continue
                
            if not src_node.schema_out.is_compatible(dst_node.schema_in):
                raise ValueError(
                    f"Type mismatch on edge {edge.src} -> {edge.dst}: "
                    f"Output schema {src_node.schema_out} is not compatible with "
                    f"Input schema {dst_node.schema_in}"
                )

def _check_semantic_constraints(graph: Graph) -> None:
    """Enforces semantic rules based on node tags."""
    for nid, node in graph.nodes.items():
        # PPO Rule: PPO nodes must be marked as OnPolicy
        if "PPO" in node.tags:
            if TAG_ON_POLICY not in node.tags:
                raise ValueError(f"Semantic error: PPO node {nid} must have {TAG_ON_POLICY} tag.")
            
            # Check for Replay buffer in predecessors
            _check_no_replay_predecessor(graph, nid)

        # GAE Rule: GAE nodes require Ordered input trajectories
        if "GAE" in node.tags:
            is_ordered = any(TAG_ORDERED in spec.tags for spec in node.schema_in.get_field_map().values())
            if not is_ordered:
                raise ValueError(f"Semantic error: GAE node {nid} requires {TAG_ORDERED} input trajectory.")

def _check_no_replay_predecessor(graph: Graph, target_nid: NodeId) -> None:
    """Helper to ensure a node doesn't have a Replay buffer in its upstream path."""
    # Build reverse adjacency for upstream traversal
    reverse_adj: Dict[NodeId, List[NodeId]] = {nid: [] for nid in graph.nodes}
    for src, dsts in graph.adjacency_list.items():
        for dst in dsts:
            reverse_adj[dst].append(src)
            
    visited = set()
    stack = [target_nid]
    
    while stack:
        nid = stack.pop()
        if nid in visited:
            continue
        visited.add(nid)
        
        node = graph.nodes[nid]
        if "Replay" in node.tags:
            raise ValueError(f"Semantic error: OnPolicy node {target_nid} cannot consume data from Replay buffer {nid}.")
            
        stack.extend(reverse_adj.get(nid, []))
