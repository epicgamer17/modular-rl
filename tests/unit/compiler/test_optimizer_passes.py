import pytest
from core.graph import (
    Graph,
    NODE_TYPE_SOURCE,
    NODE_TYPE_SINK,
    NODE_TYPE_METRICS_SINK,
    NODE_TYPE_TARGET_SYNC,
)
from runtime.specs import (
    register_spec,
    OperatorSpec,
    TensorSpec,
    ObsTensor,
    ActionValuesTensor,
)
from compiler.compiler import compile_graph

pytestmark = pytest.mark.unit


def test_dead_node_elimination() -> None:
    """Verifies that nodes with no path to a Sink or side-effect are removed."""
    # Register necessary specs
    register_spec(NODE_TYPE_SINK, OperatorSpec.create(name=NODE_TYPE_SINK))
    register_spec("PureOp", OperatorSpec.create(name="PureOp", pure=True))
    register_spec(
        "SideEffectOp", OperatorSpec.create(name="SideEffectOp", side_effects=["log"])
    )

    g = Graph()
    g.add_node("src", NODE_TYPE_SOURCE)
    g.add_node("live_sink", NODE_TYPE_SINK)
    g.add_node("pure_used", "PureOp")
    g.add_node("pure_unused", "PureOp")
    g.add_node("side_effect_unused", "SideEffectOp")

    # Connect live branch
    g.add_edge("src", "pure_used")
    g.add_edge("pure_used", "live_sink")

    # pure_unused is NOT connected to anything live
    # side_effect_unused is NOT connected to sink, but HAS side effects

    optimized_g = compile_graph(g)

    # Live nodes should be kept
    assert "src" in optimized_g.nodes
    assert "live_sink" in optimized_g.nodes
    assert "pure_used" in optimized_g.nodes
    assert "side_effect_unused" in optimized_g.nodes

    # Dead nodes should be removed
    assert "pure_unused" not in optimized_g.nodes


def test_unreachable_metrics_branch_preserved() -> None:
    """Verifies that side-effect nodes (like metrics) are not removed even if sinkless."""
    register_spec(
        "MyMetrics", OperatorSpec.create(name="MyMetrics", side_effects=["metrics"])
    )

    g = Graph()
    g.add_node("src", NODE_TYPE_SOURCE)
    g.add_node("metrics", "MyMetrics")
    g.add_edge("src", "metrics")

    # No traditional 'Sink' node, but 'metrics' has side effects
    optimized_g = compile_graph(g)
    assert "metrics" in optimized_g.nodes
    assert "src" in optimized_g.nodes


def test_multi_step_dne() -> None:
    """Verifies that DNE removes long chains of unused pure nodes."""
    register_spec("PureChain", OperatorSpec.create(name="PureChain", pure=True))

    g = Graph()
    g.add_node("src", NODE_TYPE_SOURCE)
    g.add_node("p1", "PureChain")
    g.add_node("p2", "PureChain")
    g.add_node("p3", "PureChain")

    g.add_edge("src", "p1")
    g.add_edge("p1", "p2")
    g.add_edge("p2", "p3")
    # p3 is not connected to anything live

    optimized_g = compile_graph(g)

    # Since p3 is dead, p2 becomes dead, then p1, then src (if src is pure)
    assert "p3" not in optimized_g.nodes
    assert "p2" not in optimized_g.nodes
    assert "p1" not in optimized_g.nodes


def test_greedy_policy_fusion() -> None:
    """Verifies that QValuesSingle + Argmax are fused into GreedyPolicy."""
    # Register necessary specs for validation with correct ports to avoid pollution
    register_spec(
        "QValuesSingle",
        OperatorSpec.create(
            name="QValuesSingle",
            inputs={"obs": ObsTensor},
            outputs=ActionValuesTensor,
            pure=True,
            deterministic=True,
        ),
    )
    register_spec(
        "Argmax",
        OperatorSpec.create(
            name="Argmax",
            inputs={"q_values": ActionValuesTensor},
            outputs=TensorSpec(shape=(-1,), dtype="int64"),
            pure=True,
            deterministic=True,
        ),
    )
    register_spec(
        "GreedyPolicy",
        OperatorSpec.create(
            name="GreedyPolicy",
            inputs={"obs": ObsTensor},
            outputs=TensorSpec(shape=(-1,), dtype="int64"),
            pure=True,
            deterministic=True,
        ),
    )
    register_spec(NODE_TYPE_SINK, OperatorSpec.create(name=NODE_TYPE_SINK))

    g = Graph()
    g.add_node("src", NODE_TYPE_SOURCE)
    # Source must output ObsTensor to match QValuesSingle
    # register_spec(NODE_TYPE_SOURCE, ...) is already done in some places,
    # but let's ensure it's compatible.

    g.add_node("q", "QValuesSingle", params={"model": "q_net"})
    g.add_node("argmax", "Argmax", params={"dim": -1})
    g.add_node("sink", NODE_TYPE_SINK)

    g.add_edge("src", "q", dst_port="obs")
    g.add_edge("q", "argmax", dst_port="q_values")
    g.add_edge("argmax", "sink")

    optimized_g = compile_graph(g)

    # q and argmax should be gone, replaced by a fused node
    assert "q" not in optimized_g.nodes
    assert "argmax" not in optimized_g.nodes

    # Find the fused node
    fused_nodes = [
        nid for nid, n in optimized_g.nodes.items() if n.node_type == "GreedyPolicy"
    ]
    assert len(fused_nodes) == 1
    fused_node = optimized_g.nodes[fused_nodes[0]]

    # Check params
    assert fused_node.params["model"] == "q_net"
    assert fused_node.params["dim"] == -1
