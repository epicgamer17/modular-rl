import pytest
from core.graph import Graph
from compiler.rewrite import RewriteEngine, find_linear_chain
from compiler.fusion_rules import METRICS_FOLD_RULE, RL_IR_FUSION_RULES
from runtime.registry import register_spec, OperatorSpec, Scalar, clear_registry

pytestmark = pytest.mark.unit

@pytest.fixture(autouse=True)
def setup_specs():
    # Register common types
    register_spec("Mean", OperatorSpec.create(
        name="Mean", 
        pure=True, 
        deterministic=True,
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("MetricsSink", OperatorSpec.create(
        name="MetricsSink", 
        inputs={"replay_size": Scalar("int64")}, 
        pure=False, 
        deterministic=True,
        allowed_contexts={"actor", "learner"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("Clip", OperatorSpec.create(
        name="Clip", 
        pure=True, 
        deterministic=True,
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("Cast", OperatorSpec.create(
        name="Cast", 
        pure=True, 
        deterministic=True,
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("GAE", OperatorSpec.create(
        name="GAE", 
        pure=True, 
        deterministic=True,
        allowed_contexts={"actor", "learner"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("Normalize", OperatorSpec.create(
        name="Normalize", 
        pure=True, 
        deterministic=True,
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("PPOActorLoss", OperatorSpec.create(
        name="PPOActorLoss", 
        pure=True, 
        deterministic=True,
        allowed_contexts={"learner"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("ReplayQuery", OperatorSpec.create(
        name="ReplayQuery", 
        pure=True, 
        deterministic=True, 
        stateful=True,
        allowed_contexts={"learner"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("Collate", OperatorSpec.create(
        name="Collate", 
        pure=True, 
        deterministic=True,
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("Encoder", OperatorSpec.create(
        name="Encoder", 
        pure=True, 
        deterministic=True,
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("PolicyHead", OperatorSpec.create(
        name="PolicyHead", 
        pure=True, 
        deterministic=True,
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("Sample", OperatorSpec.create(
        name="Sample", 
        pure=True, 
        deterministic=False,
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    
    # Register replacement types
    register_spec("MetricsFolded", OperatorSpec.create(
        name="MetricsFolded", 
        pure=False, 
        deterministic=True,
        allowed_contexts={"actor", "learner"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("ClampedCast", OperatorSpec.create(
        name="ClampedCast", 
        pure=True, 
        deterministic=True,
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("PPOAdvantageLoss", OperatorSpec.create(
        name="PPOAdvantageLoss", 
        pure=True, 
        deterministic=True,
        allowed_contexts={"learner"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("ReplaySample", OperatorSpec.create(
        name="ReplaySample", 
        pure=True, 
        deterministic=True, 
        stateful=True,
        allowed_contexts={"learner"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("PolicyHeadFused", OperatorSpec.create(
        name="PolicyHeadFused", 
        pure=True, 
        deterministic=False,
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))

def test_metrics_fold_fusion() -> None:
    """Verifies Mean -> MetricsSink fusion (Tail is impure)."""
    g = Graph()
    g.add_node("m", "Mean")
    g.add_node("s", "MetricsSink")
    g.add_edge("m", "s")
    
    chain = find_linear_chain(g, METRICS_FOLD_RULE.pattern)
    assert chain == ["m", "s"]

def test_policy_head_fusion() -> None:
    """Verifies Encoder -> PolicyHead -> Sample fusion (Tail is stochastic)."""
    g = Graph()
    g.add_node("enc", "Encoder")
    g.add_node("head", "PolicyHead")
    g.add_node("sample", "Sample")
    g.add_edge("enc", "head")
    g.add_edge("head", "sample")
    
    chain = find_linear_chain(g, ["Encoder", "PolicyHead", "Sample"])
    assert chain == ["enc", "head", "sample"]

def test_stochastic_path_rejection() -> None:
    """Verifies that Stochastic -> Stochastic path still fails."""
    # Register another stochastic op
    register_spec("EntropyNoise", OperatorSpec.create(
        name="EntropyNoise", 
        pure=True, 
        deterministic=False,
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    
    g = Graph()
    g.add_node("s1", "Sample")
    g.add_node("s2", "EntropyNoise")
    g.add_edge("s1", "s2")
    
    chain = find_linear_chain(g, ["Sample", "EntropyNoise"])
    assert chain == [] # Two non-standard nodes should fail
