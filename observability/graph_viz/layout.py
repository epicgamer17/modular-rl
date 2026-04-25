import networkx as nx
from typing import Dict, Any, List

def compute_dag_layout(G: nx.DiGraph) -> Dict[Any, List[float]]:
    """
    Computes a hierarchical layout for a DAG.
    Prefers Graphviz 'dot' if available, falls back to custom layer-based layout.
    """
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        return graphviz_layout(G, prog='dot')
    except ImportError:
        # Custom layer-based layout
        if not nx.is_directed_acyclic_graph(G):
            return nx.spring_layout(G)
            
        layers = list(nx.topological_generations(G))
        pos = {}
        for y, layer in enumerate(layers):
            for x, node in enumerate(layer):
                # Spread nodes horizontally within the layer
                pos[node] = [x - len(layer) / 2, -y]
        return pos
