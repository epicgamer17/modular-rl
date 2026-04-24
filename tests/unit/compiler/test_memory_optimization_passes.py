import pytest

from agents.dqn.config import DQNConfig
from agents.dqn.graphs import build_learner_graph
from agents.dqn.specs import register_dqn_specs
from compiler.passes.memory_optimizations import optimize_memory
from core.graph import Graph, NODE_TYPE_SINK
from core.schema import Schema, Field, TensorSpec
from runtime.collator import ReplayCollator
from runtime.specs import OperatorSpec, clear_registry, register_base_specs, register_spec

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def setup_specs():
    clear_registry()
    register_base_specs()
    register_dqn_specs()
    register_spec(NODE_TYPE_SINK, OperatorSpec.create(name=NODE_TYPE_SINK))


def _dqn_collator(config: DQNConfig) -> ReplayCollator:
    schema = Schema(
        [
            Field("obs", TensorSpec(shape=(config.obs_dim,), dtype="float32")),
            Field("action", TensorSpec(shape=(), dtype="long")),
            Field("reward", TensorSpec(shape=(), dtype="float32")),
            Field("next_obs", TensorSpec(shape=(config.obs_dim,), dtype="float32")),
            Field("done", TensorSpec(shape=(), dtype="float32")),
        ]
    )
    return ReplayCollator(schema)


def test_training_nodes_are_annotated_for_checkpoint_and_no_grad():
    config = DQNConfig(obs_dim=4, act_dim=2)
    graph = build_learner_graph(config, _dqn_collator(config))

    optimized = optimize_memory(graph)

    assert optimized.nodes["td_loss"].params.get("activation_checkpoint") is True
    assert optimized.nodes["td_loss"].params.get("no_grad_region") is True


def test_shared_buffer_slots_are_reused_for_temporary_nodes():
    register_spec(
        "PureOp",
        OperatorSpec.create(
            name="PureOp",
            inputs={"x": TensorSpec(shape=(4,), dtype="float32")},
            outputs={"y": TensorSpec(shape=(4,), dtype="float32")},
            pure=True,
            deterministic=True,
        ),
    )

    g = Graph()
    g.add_node("src", "Source", schema_out=Schema([Field("x", TensorSpec(shape=(4,), dtype="float32"))]))
    g.add_node("a", "PureOp")
    g.add_node("b", "PureOp")
    g.add_node("c", "PureOp")
    g.add_node("sink", NODE_TYPE_SINK)
    g.add_edge("src", "a", dst_port="x")
    g.add_edge("a", "b", dst_port="x")
    g.add_edge("b", "c", dst_port="x")
    g.add_edge("c", "sink")

    optimized = optimize_memory(g)

    assert "buffer_slot" in optimized.nodes["a"].params
    assert "buffer_slot" in optimized.nodes["b"].params
    assert "buffer_slot" in optimized.nodes["c"].params
    assert len({optimized.nodes["a"].params["buffer_slot"], optimized.nodes["b"].params["buffer_slot"], optimized.nodes["c"].params["buffer_slot"]}) <= 2
