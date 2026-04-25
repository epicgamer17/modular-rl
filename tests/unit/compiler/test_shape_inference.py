import pytest
from core.graph import Graph
from runtime.registry import register_spec, OperatorSpec, TensorSpec, BatchObs, BatchQ
from compiler.pipeline import compile_graph

pytestmark = pytest.mark.unit


def test_q_values_batch_infers_batch_q() -> None:
    """Verifies that QValuesBatch infers [B, A] from [B, D] input."""

    def q_batch_shape_fn(inputs):
        obs = inputs.get("obs")
        if obs and len(obs.shape) >= 1:
            # Infer batch size from input, assume 4 actions
            return {"default": TensorSpec(shape=(obs.shape[0], 4), dtype="float32")}
        return {}

    register_spec(
        "QBatch",
        OperatorSpec.create(
            name="QBatch", 
            inputs={"obs": BatchObs}, 
            shape_fn=q_batch_shape_fn,
            allowed_contexts={"actor", "learner"},
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
        ),
    )

    # Custom Source that provides a specific Batch size
    register_spec(
        "BatchSrc",
        OperatorSpec.create(
            name="BatchSrc", 
            outputs=TensorSpec(shape=(32, 128), dtype="float32"),
            allowed_contexts={"actor", "learner"},
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
        ),
    )

    g = Graph()
    g.add_node("src", "BatchSrc")
    g.add_node("q", "QBatch")
    g.add_edge("src", "q")

    compiled_g = compile_graph(g)

    q_node = compiled_g.nodes["q"]
    out_spec = q_node.schema_out.get_field_map()["default"]
    assert out_spec.shape == (32, 4)


def test_gather_preserves_rank() -> None:
    """Verifies that a gather-like op preserves the batch dimension."""

    def gather_shape_fn(inputs):
        data = inputs.get("data")
        if data and len(data.shape) >= 1:
            # Preserves batch dim, but reduces rank by 1
            return {"default": TensorSpec(shape=(data.shape[0],), dtype=data.dtype)}
        return {}

    from runtime.registry import PortSpec
    register_spec(
        "Gather",
        OperatorSpec.create(
            name="Gather",
            inputs={
                "data": BatchQ, 
                "indices": PortSpec(spec=TensorSpec(shape=(-1,), dtype="int64"), required=False)
            },
            shape_fn=gather_shape_fn,
            allowed_contexts={"actor", "learner"},
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
        ),
    )

    register_spec(
        "QSrc",
        OperatorSpec.create(
            name="QSrc", 
            outputs=TensorSpec(shape=(64, 10), dtype="float32"),
            allowed_contexts={"actor", "learner"},
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
        ),
    )

    g = Graph()
    g.add_node("src", "QSrc")
    g.add_node("g", "Gather")
    g.add_edge("src", "g", dst_port="data")

    compiled_g = compile_graph(g)

    g_node = compiled_g.nodes["g"]
    out_spec = g_node.schema_out.get_field_map()["default"]
    assert out_spec.shape == (64,)


def test_reduce_mean_outputs_scalar() -> None:
    """Verifies that reduce_mean outputs a scalar (0-d)."""

    def reduce_shape_fn(inputs):
        return {"default": TensorSpec(shape=(), dtype="float32")}

    register_spec(
        "ReduceMean",
        OperatorSpec.create(
            name="ReduceMean",
            inputs={"input": TensorSpec(shape=(-1, -1), dtype="float32")},
            shape_fn=reduce_shape_fn,
            allowed_contexts={"actor", "learner"},
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
        ),
    )

    register_spec(
        "InputSrc",
        OperatorSpec.create(
            name="InputSrc", 
            outputs=TensorSpec(shape=(10, 10), dtype="float32"),
            allowed_contexts={"actor", "learner"},
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
        ),
    )

    g = Graph()
    g.add_node("src", "InputSrc")
    g.add_node("reduce", "ReduceMean")
    g.add_edge("src", "reduce")

    compiled_g = compile_graph(g)

    reduce_node = compiled_g.nodes["reduce"]
    out_spec = reduce_node.schema_out.get_field_map()["default"]
    assert out_spec.shape == ()
