import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Union
from core.graph import Graph, EdgeType
from .styles import get_node_style, get_edge_style, GRAPH_COLORS

def render_graph(
    graph: Graph,
    mode: str = "static",
    highlight: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    High-value Graph Visualization for RL IR.
    
    Modes:
    1. "static": Standard DAG view colored by op type.
    2. "compiler": Highlights fused/pruned nodes (requires 'state' in node params).
    3. "runtime": Heatmap of execution frequency (requires 'exec_count' in node params).
    """
    G = nx.DiGraph()
    
    # Build NetworkX graph from IR
    for nid, node in graph.nodes.items():
        G.add_node(nid, type=node.node_type, params=node.params)
        
    for edge in graph.edges:
        G.add_edge(edge.src, edge.dst, type=edge.edge_type.value)
        
    # Layout: Use hierarchical layout if possible, otherwise spring
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog='dot')
    except ImportError:
        pos = nx.spring_layout(G, k=1.5, iterations=50)
    
    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    
    # Draw Nodes
    for nid, data in G.nodes(data=True):
        node_type = data.get("type", "default")
        params = data.get("params", {})
        
        state = "normal"
        if mode == "compiler":
            state = params.get("compiler_state", "normal")
        elif mode == "runtime":
            if params.get("active", False):
                state = "active"
        
        if highlight and nid in highlight:
            state = "active"
            
        style = get_node_style(node_type, state=state)
        
        # Runtime Heatmap logic
        if mode == "runtime" and "exec_count" in params:
            count = params["exec_count"]
            # Map count to alpha or size
            style["alpha"] = min(0.3 + (count / 100), 1.0)
        
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[nid],
            node_color=style["color"],
            node_shape=style["shape"],
            node_size=style["size"],
            edgecolors=style.get("edgecolor", "none"),
            linewidths=style.get("edgewidth", 0),
            alpha=style.get("alpha", 0.9)
        )

    # Draw Edges
    for u, v, data in G.edges(data=True):
        edge_type = data.get("type", "data")
        style = get_edge_style(edge_type)
        
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            edge_color=style["color"],
            style=style["style"],
            width=style["width"],
            arrowsize=20,
            alpha=0.6,
            connectionstyle="arc3,rad=0.1"
        )

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=9, font_color="#D2DAE2", font_weight="bold")

    plt.title(title or f"Graph View [{mode.upper()}]", fontsize=20, fontweight="bold", color="white", pad=30)
    plt.axis("off")
    
    # Legend for Op Types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=k, markerfacecolor=v, markersize=10)
        for k, v in list(GRAPH_COLORS.items())[:5]
    ]
    ax.legend(handles=legend_elements, loc='upper right', title="Op Types", frameon=True)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        plt.close()
    else:
        plt.show()
