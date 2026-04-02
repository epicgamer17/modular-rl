import os
import argparse
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as colors

from search.nodes import DecisionNode, ChanceNode


def visualize_mcts(root, output_path="stats/mcts_tree", max_depth=5):
    """
    Visualizes an MCTS tree using Graphviz.
    """
    try:
        import graphviz
    except ImportError:
        print(
            "Error: 'graphviz' library not found. Please install it: pip install graphviz"
        )
        return

    dot = graphviz.Digraph(comment="MCTS Tree", format="svg")
    dot.attr(rankdir="TB", size="10,10")

    # Color mapping for Q values (assuming Q is normalized or in a known range)
    # We'll use RdYlGn (Red-Yellow-Green) colormap
    try:
        from matplotlib import colormaps

        cmap = colormaps.get_cmap("RdYlGn")
    except ImportError:
        cmap = cm.get_cmap("RdYlGn")
    norm = colors.Normalize(vmin=-1.0, vmax=1.0)  # Standard range for many RL tasks

    def get_color(q):
        try:
            rgba = cmap(norm(q))
            return colors.to_hex(rgba)
        except:
            return "#ffffff"  # Fallback to white

    def get_penwidth(n, max_n):
        if max_n <= 0:
            return 1.0
        # Logarithmic scaling or linear scaling for visibility
        return 1.0 + 4.0 * (n / max_n)

    # First pass: find max visit count for normalization
    visit_counts = []

    def collect_visits(node, depth):
        if depth > max_depth:
            return
        visit_counts.append(node.visits)
        if hasattr(node, "children"):
            for child in node.children.values():
                collect_visits(child, depth + 1)

    collect_visits(root, 0)
    max_visits = max(visit_counts) if visit_counts else 1

    # Second pass: build the graph
    node_id_counter = 0

    def add_node(node, parent_id=None, edge_label="", depth=0):
        nonlocal node_id_counter
        if depth > max_depth:
            return

        node_id = f"node_{node_id_counter}"
        node_id_counter += 1

        # Calculate Q
        if hasattr(node, "value"):
            q = node.value()
            if hasattr(q, "item"):
                q = q.item()
        else:
            q = 0.0

        n = node.visits

        # Determine node shape and label
        shape = "ellipse" if isinstance(node, DecisionNode) else "box"
        player_info = f"P{node.to_play}" if hasattr(node, "to_play") else ""

        label = f"{edge_label}\nN={n}\nQ={q:.3f}\n{player_info}"

        color = get_color(q)
        penwidth = get_penwidth(n, max_visits)

        dot.node(
            node_id,
            label,
            shape=shape,
            style="filled",
            fillcolor=color,
            penwidth=str(penwidth),
        )

        if parent_id:
            dot.edge(parent_id, node_id)

        # Recurse to children
        if hasattr(node, "children"):
            for action, child in node.children.items():
                add_node(child, node_id, edge_label=f"A={action}", depth=depth + 1)

    add_node(root)

    # Ensure directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Render
    try:
        render_path = dot.render(output_path, cleanup=True)
        print(f"MCTS tree visualization saved to: {render_path}")
    except Exception as e:
        print(f"Error rendering MCTS tree: {e}")
        print("Saving DOT source instead...")
        with open(f"{output_path}.gv", "w") as f:
            f.write(dot.source)
        print(f"DOT source saved to: {output_path}.gv")


def create_mock_tree():
    """Creates a small mock MCTS tree for testing."""
    # Ensure class attributes are set for the mock
    DecisionNode.estimation_method = "zero"
    DecisionNode.discount = 0.9
    DecisionNode.value_prefix = False
    DecisionNode.stochastic = False

    root = DecisionNode(prior=1.0)
    root.visits = 100
    root.value_sum = 50.0  # Q=0.5
    root.to_play = 0

    # Level 1
    child1 = DecisionNode(prior=0.6, parent=root)
    child1.visits = 60
    child1.value_sum = 48.0  # Q=0.8
    child1.to_play = 1

    child2 = DecisionNode(prior=0.4, parent=root)
    child2.visits = 40
    child2.value_sum = -20.0  # Q=-0.5
    child2.to_play = 1

    root.children = {0: child1, 1: child2}

    # Level 2
    gchild1 = DecisionNode(prior=0.5, parent=child1)
    gchild1.visits = 50
    gchild1.value_sum = 45.0  # Q=0.9
    gchild1.to_play = 0

    child1.children = {0: gchild1}

    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MCTS tree using Graphviz.")
    parser.add_argument(
        "--output",
        type=str,
        default="stats/mcts_tree",
        help="Output path (without extension).",
    )
    parser.add_argument(
        "--depth", type=int, default=3, help="Maximum depth to visualize."
    )
    args = parser.parse_args()

    print("Creating mock MCTS tree for visualization...")
    mock_root = create_mock_tree()
    visualize_mcts(mock_root, output_path=args.output, max_depth=args.depth)
