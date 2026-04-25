from typing import Dict, Any

# Muted Scientific Palette for Graph Nodes
GRAPH_COLORS = {
    "Actor": "#4A69BD",          # Blue
    "Transform": "#E58E26",      # Orange
    "Source": "#78E08F",         # Green
    "Sink": "#EB2F06",           # Red
    "Control": "#60A3BC",        # Cyan
    "MetricsSink": "#6A89CC",    # Lighter Blue
    "ReplayQuery": "#B8E994",    # Lighter Green
    "Transfer": "#F6B93B",       # Yellow
    "default": "#D2DAE2",        # Light Grey
    "fused": "#82589F",          # Purple
    "pruned": "#4b4b4b",         # Dark Grey (dimmed)
    "active": "#3ae374",         # Bright Green (highlight)
}

NODE_STYLES = {
    "Actor": {"color": GRAPH_COLORS["Actor"], "shape": "s", "size": 1000},
    "Transform": {"color": GRAPH_COLORS["Transform"], "shape": "o", "size": 800},
    "Source": {"color": GRAPH_COLORS["Source"], "shape": "d", "size": 700},
    "Sink": {"color": GRAPH_COLORS["Sink"], "shape": "v", "size": 700},
    "Control": {"color": GRAPH_COLORS["Control"], "shape": "p", "size": 800},
    "default": {"color": GRAPH_COLORS["default"], "shape": "o", "size": 600},
}

EDGE_STYLES = {
    "data": {"color": "#60A3BC", "style": "-", "width": 1.5},
    "control": {"color": "#E58E26", "style": "--", "width": 1.0},
    "effect": {"color": "#EB2F06", "style": ":", "width": 1.0},
}

def get_node_style(node_type: str, state: str = "normal") -> Dict[str, Any]:
    """Get the visual style for a node based on type and state."""
    base_style = NODE_STYLES.get(node_type, NODE_STYLES["default"]).copy()
    
    if state == "fused":
        base_style["color"] = GRAPH_COLORS["fused"]
        base_style["size"] *= 1.2
    elif state == "pruned":
        base_style["color"] = GRAPH_COLORS["pruned"]
        base_style["alpha"] = 0.3
    elif state == "active":
        base_style["edgecolor"] = GRAPH_COLORS["active"]
        base_style["edgewidth"] = 3
        
    return base_style

def get_edge_style(edge_type: str) -> Dict[str, Any]:
    """Get visual style for an edge."""
    return EDGE_STYLES.get(edge_type, EDGE_STYLES["data"])
