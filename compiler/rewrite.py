from dataclasses import dataclass
from typing import List, Callable, Dict, Any, Optional, Set
from core.graph import Graph, NodeId, Node, Edge
from runtime.registry import get_spec
import copy

def find_linear_chain(graph: Graph, types: List[str]) -> List[NodeId]:
    """
    Finds a sequence of nodes that form a linear chain with the specified types.
    Conditions:
    - Exact edge chain in the order of 'types'.
    - Single consumer: Each node (except the last) has exactly one outgoing edge.
    - No branching: Each node (except the first) has exactly one incoming edge.
    """
    if not types:
        return []

    # Map for easy lookup of incoming edges
    in_edges: Dict[NodeId, List[Edge]] = {nid: [] for nid in graph.nodes}
    for edge in graph.edges:
        if edge.dst in in_edges:
            in_edges[edge.dst].append(edge)

    # Map for easy lookup of outgoing edges
    out_edges: Dict[NodeId, List[Edge]] = {nid: [] for nid in graph.nodes}
    for edge in graph.edges:
        if edge.src in out_edges:
            out_edges[edge.src].append(edge)

    # Map for easy lookup of outgoing edges

    def is_standard(node_type: str) -> bool:
        spec = get_spec(node_type)
        return bool(spec and spec.pure and spec.deterministic and not spec.stateful)

    for start_node_id in graph.nodes:
        current_id = start_node_id
        
        # Check first node type
        first_node = graph.nodes[current_id]
        if first_node.node_type != types[0]:
            continue
            
        chain = [current_id]
        valid_chain = True
        non_standard_count = 0
        if not is_standard(first_node.node_type):
            non_standard_count += 1

        for i, next_type in enumerate(types[1:]):
            # Must have exactly one consumer
            curr_out = out_edges[current_id]
            if len(curr_out) != 1:
                valid_chain = False
                break
                
            next_id = curr_out[0].dst
            if next_id not in graph.nodes:
                valid_chain = False
                break

            # Next node must have exactly one producer
            next_in = in_edges[next_id]
            if len(next_in) != 1:
                valid_chain = False
                break
                
            # Next node must match type
            next_node = graph.nodes[next_id]
            if next_node.node_type != next_type:
                valid_chain = False
                break
                
            if not is_standard(next_node.node_type):
                non_standard_count += 1
                # Non-standard node can only be head (index 0) or tail (index len(types)-1)
                is_tail = (i == len(types) - 2)
                if not is_tail:
                    valid_chain = False
                    break
            
            # Additional safety: at most one non-standard node in the whole chain
            if non_standard_count > 1:
                valid_chain = False
                break
                
            chain.append(next_id)
            current_id = next_id
            
        if valid_chain and len(chain) == len(types):
            return chain
            
    return []

def rewrite(graph: Graph, match: List[NodeId], rule: 'FusionRule', report: Optional[Any] = None) -> Graph:
    """
    Transactionally replaces a subgraph with a fused node.
    Follows a non-mutating pattern.
    """
    if not match:
        return graph

    # 1. Copy graph
    new_graph = copy.deepcopy(graph)
    
    # Identify head and tail of the chain
    head_id = match[0]
    tail_id = match[-1]
    
    # 2. Insert fused node
    fused_id = NodeId(f"{head_id}_{tail_id}_fused")
    
    # Merge params from all nodes in the match
    merged_params = {}
    for nid in match:
        merged_params.update(graph.nodes[nid].params)
        
    new_graph.add_node(fused_id, rule.replacement, params=merged_params)
    
    # 3. Reconnect incoming edges to head
    # 4. Reconnect outgoing edges from tail
    new_edges = []
    match_set = set(match)
    
    for edge in graph.edges:
        # Edge coming from outside the match into the head
        if edge.dst == head_id and edge.src not in match_set:
            new_edges.append(Edge(src=edge.src, dst=fused_id, edge_type=edge.edge_type, dst_port=edge.dst_port))
        # Edge going from the tail to outside the match
        elif edge.src == tail_id and edge.dst not in match_set:
            new_edges.append(Edge(src=fused_id, dst=edge.dst, edge_type=edge.edge_type, dst_port=edge.dst_port))
        # Edge entirely outside the match
        elif edge.src not in match_set and edge.dst not in match_set:
            new_edges.append(edge)
            
    new_graph.edges = new_edges
    
    # 5. Delete old nodes
    for nid in match:
        if nid in new_graph.nodes:
            del new_graph.nodes[nid]
            
    # 6. Rebuild adjacency
    new_graph._adjacency = {}
    for e in new_graph.edges:
        if e.src not in new_graph._adjacency:
            new_graph._adjacency[e.src] = set()
        new_graph._adjacency[e.src].add(e.dst)
        
    # 7. Validate graph
    # Import locally to avoid circular dependencies
    from compiler.passes.validate_structure import validate_structure
    val_report = validate_structure(new_graph)
    if val_report.has_errors():
        # If the transformation produced a broken graph, reject it and return original
        return graph

    if report:
        # Avoid circular import by using Any for report type in signature
        from compiler.optimizer import OptimizationStep
        report.add_step(OptimizationStep(
            rule_name=rule.name,
            pattern=rule.pattern,
            replacement=rule.replacement,
            removed_nodes=list(match),
            new_node=fused_id
        ))

    return new_graph

@dataclass
class FusionRule:
    name: str
    pattern: List[str] # List of node types in sequence (e.g. ["QValuesSingle", "Argmax"])
    replacement: str  # New node type
    constraints: Optional[Callable[[List[Node]], bool]] = None
    min_profitability: float = 0.0 # Minimum savings in launch costs or other metrics

class RewriteEngine:
    """
    Engine for applying graph rewrite rules (fusion, substitution).
    """
    def __init__(self):
        self.rules: List[FusionRule] = []

    def add_rule(self, rule: FusionRule):
        self.rules.append(rule)

    def apply(self, graph: Graph, report: Optional[Any] = None) -> Graph:
        """
        Applies all registered rewrite rules to the graph.
        Currently handles simple linear sequence fusions.
        """
        current_graph = copy.deepcopy(graph)
        
        for rule in self.rules:
            current_graph = self._apply_rule(current_graph, rule, report=report)
            
        return current_graph

    def _apply_rule(self, graph: Graph, rule: FusionRule, report: Optional[Any] = None) -> Graph:
        """
        Applies a single rule to the graph.
        Uses find_linear_chain and rewrite for transactional updates.
        """
        changed = True
        while changed:
            changed = False
            
            # Use the new matcher to find candidates
            chain = find_linear_chain(graph, rule.pattern)
            
            if chain:
                # Check constraints if any
                nodes = [graph.nodes[nid] for nid in chain]
                
                # Check for backward boundary
                is_boundary = False
                for n in nodes:
                    spec = get_spec(n.node_type)
                    if spec and (spec.creates_grad or spec.consumes_grad):
                        is_boundary = True
                        break
                
                if is_boundary:
                    if report:
                        report.add_skipped_fusion(rule.name, chain, "backward boundary blocks fusion")
                    # Break the while loop to avoid infinite loop on same chain
                    changed = False
                elif rule.constraints and not rule.constraints(nodes):
                    # Skip if constraints fail
                    pass 
                elif not self._is_profitable(rule, nodes):
                    # Skip if not profitable
                    if report:
                        report.add_skipped_fusion(rule.name, chain, "Not profitable based on cost model")
                    pass
                else:
                    # Transactional rewrite
                    updated_graph = rewrite(graph, chain, rule, report=report)
                    if updated_graph is not graph:
                        graph = updated_graph
                        changed = True
            else:
                changed = False
                
        return graph

    def _is_profitable(self, rule: FusionRule, match_nodes: List[Node]) -> bool:
        """
        Heuristic to determine if fusion is profitable.
        Profit = (Sum of original launch costs) - (Replacement launch cost)
        """
        from runtime.registry import get_spec
        replacement_spec = get_spec(rule.replacement)
        if not replacement_spec:
            return True
            
        original_specs = [get_spec(n.node_type) for n in match_nodes]
        total_original_launch = sum(s.kernel_launch_cost for s in original_specs if s)
        
        savings = total_original_launch - replacement_spec.kernel_launch_cost
        return savings >= rule.min_profitability
