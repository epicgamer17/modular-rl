import pytest
from core.graph import Graph
from compiler.rewrite import find_linear_chain
from runtime.registry import register_spec, OperatorSpec

pytestmark = pytest.mark.unit


def test_find_linear_chain_basic() -> None:
    g = Graph()
    register_spec("A", OperatorSpec.create(
        name="A", 
        pure=True, 
        deterministic=True,
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("B", OperatorSpec.create(
        name="B", 
        pure=True, 
        deterministic=True,
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    g.add_node("n1", "A")
    g.add_node("n2", "B")
    g.add_edge("n1", "n2")

    chain = find_linear_chain(g, ["A", "B"])
    assert chain == ["n1", "n2"]


def test_find_linear_chain_single_consumer_constraint() -> None:
    """Verifies that fusion fails if a node has multiple consumers."""
    g = Graph()
    register_spec("A", OperatorSpec.create(
        name="A", 
        pure=True, 
        deterministic=True,
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("B", OperatorSpec.create(
        name="B", 
        pure=True, 
        deterministic=True,
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("C", OperatorSpec.create(
        name="C", 
        pure=True, 
        deterministic=True,
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    g.add_node("n1", "A")
    g.add_node("n2", "B")
    g.add_node("n3", "C")
    g.add_edge("n1", "n2")
    g.add_edge("n1", "n3")  # n1 has 2 consumers

    chain = find_linear_chain(g, ["A", "B"])
    assert chain == []


def test_find_linear_chain_purity_constraint() -> None:
    """Verifies that fusion fails if there are multiple impure nodes or one in the middle."""
    register_spec(
        "ImpureOp", OperatorSpec.create(
            name="ImpureOp", 
            pure=False, 
            deterministic=True,
            allowed_contexts={"actor", "learner"},
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
        )
    )
    register_spec(
        "PureOp", OperatorSpec.create(
            name="PureOp", 
            pure=True, 
            deterministic=True,
            allowed_contexts={"actor"},
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
        )
    )

    # Case 1: Multiple impure nodes
    g1 = Graph()
    g1.add_node("n1", "ImpureOp")
    g1.add_node("n2", "ImpureOp")
    g1.add_edge("n1", "n2")
    assert find_linear_chain(g1, ["ImpureOp", "ImpureOp"]) == []

    # Case 2: Impure node in the middle
    g2 = Graph()
    g2.add_node("n1", "PureOp")
    g2.add_node("n2", "ImpureOp")
    g2.add_node("n3", "PureOp")
    g2.add_edge("n1", "n2")
    g2.add_edge("n2", "n3")
    assert find_linear_chain(g2, ["PureOp", "ImpureOp", "PureOp"]) == []


def test_find_linear_chain_determinism_constraint() -> None:
    """Verifies that fusion fails if multiple stochastic nodes are present."""
    register_spec(
        "StochasticOp",
        OperatorSpec.create(
            name="StochasticOp", 
            pure=True, 
            deterministic=False,
            allowed_contexts={"actor"},
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
        ),
    )
    register_spec(
        "PureOp", OperatorSpec.create(
            name="PureOp", 
            pure=True, 
            deterministic=True,
            allowed_contexts={"actor"},
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
        )
    )

    g = Graph()
    g.add_node("n1", "StochasticOp")
    g.add_node("n2", "StochasticOp")
    g.add_edge("n1", "n2")

    chain = find_linear_chain(g, ["StochasticOp", "StochasticOp"])
    assert chain == []


def test_find_linear_chain_stateless_constraint() -> None:
    """Verifies that fusion fails if multiple stateful nodes are present."""
    register_spec(
        "StatefulOp",
        OperatorSpec.create(
            name="StatefulOp", 
            pure=True, 
            deterministic=True, 
            stateful=True,
            allowed_contexts={"actor"},
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
        ),
    )
    register_spec(
        "PureOp", OperatorSpec.create(
            name="PureOp", 
            pure=True, 
            deterministic=True,
            allowed_contexts={"actor"},
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
        )
    )

    g = Graph()
    g.add_node("n1", "StatefulOp")
    g.add_node("n2", "StatefulOp")
    g.add_edge("n1", "n2")

    chain = find_linear_chain(g, ["StatefulOp", "StatefulOp"])
    assert chain == []


def test_find_linear_chain_no_branching_constraint() -> None:
    """Verifies that fusion fails if a node has multiple producers."""
    register_spec("A", OperatorSpec.create(
        name="A", 
        pure=True, 
        deterministic=True,
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("B", OperatorSpec.create(
        name="B", 
        pure=True, 
        deterministic=True,
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("C", OperatorSpec.create(
        name="C", 
        pure=True, 
        deterministic=True,
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    g = Graph()
    g.add_node("n1", "A")
    g.add_node("n2", "B")
    g.add_node("n3", "C")
    g.add_edge("n1", "n2")
    g.add_edge("n3", "n2")  # n2 has 2 producers

    chain = find_linear_chain(g, ["A", "B"])
    assert chain == []


def test_find_linear_chain_long() -> None:
    """Verifies that long chains can be matched."""
    g = Graph()
    g.add_node("n1", "A")
    g.add_node("n2", "B")
    g.add_node("n3", "C")
    g.add_edge("n1", "n2")
    g.add_edge("n2", "n3")

    chain = find_linear_chain(g, ["A", "B", "C"])
    assert chain == ["n1", "n2", "n3"]


def test_find_linear_chain_not_found() -> None:
    g = Graph()
    g.add_node("n1", "A")
    g.add_node("n2", "C")
    g.add_edge("n1", "n2")

    chain = find_linear_chain(g, ["A", "B"])
    assert chain == []
