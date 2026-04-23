import pytest
from core.graph import Graph, Edge, NODE_TYPE_SOURCE, NODE_TYPE_ACTOR, NODE_TYPE_TRANSFORM
from core.schema import Schema, Field, TensorSpec, TAG_ON_POLICY, TAG_OFF_POLICY
from core.types import DistributionType, PolicySnapshotType, TensorType
from compiler.analyzer import analyze_graph

pytestmark = pytest.mark.unit

def test_on_policy_vs_off_policy_violation():
    """Test that passing off-policy data to an on-policy node is detected."""
    # Define an off-policy output type
    off_policy_type = TensorType(shape=(1,), dtype="float32", tags={TAG_OFF_POLICY})
    off_policy_schema = Schema(fields=[Field("data", TensorSpec(shape=(1,), dtype="float32", rl_type=off_policy_type))])
    
    # Define an on-policy input type
    on_policy_type = TensorType(shape=(1,), dtype="float32", tags={TAG_ON_POLICY})
    on_policy_schema = Schema(fields=[Field("data", TensorSpec(shape=(1,), dtype="float32", rl_type=on_policy_type))])
    
    graph = Graph()
    
    # add_node takes (node_id, node_type, schema_in, schema_out, ...)
    graph.add_node("src", NODE_TYPE_SOURCE, schema_out=off_policy_schema)
    graph.add_node("dst", "OnPolicyNode", schema_in=on_policy_schema)
    graph.add_edge("src", "dst")
    
    analysis = analyze_graph(graph)
    assert not analysis.is_valid()
    assert any("Type Mismatch" in e for e in analysis.errors)

def test_logits_vs_probs_violation():
    """Test that passing logits where probabilities are expected is detected."""
    # Define a distribution type with logits
    logits_type = DistributionType(dist_type="Categorical", is_logits=True)
    logits_schema = Schema(fields=[Field("logits", TensorSpec(shape=(10,), dtype="float32", rl_type=logits_type))])
    
    # Define an input that expects probabilities
    # We'll use the field name "action_probs" to trigger the check in the analyzer
    probs_schema = Schema(fields=[Field("action_probs", TensorSpec(shape=(10,), dtype="float32", rl_type=logits_type))])
    
    graph = Graph()
    graph.add_node("src", NODE_TYPE_SOURCE, schema_out=logits_schema)
    graph.add_node("dst", "ProbsExpectedNode", schema_in=probs_schema)
    graph.add_edge("src", "dst")
    
    analysis = analyze_graph(graph)
    assert any("expects probabilities" in e for e in analysis.errors)

def test_stale_rollout_warning():
    """Test that using a stale policy snapshot triggers a warning."""
    # Define a stale snapshot type (version < 0)
    stale_snapshot = PolicySnapshotType(version=-1)
    snapshot_schema = Schema(fields=[Field("snapshot", TensorSpec(shape=(1,), dtype="float32", rl_type=stale_snapshot))])
    
    graph = Graph()
    graph.add_node("src", NODE_TYPE_SOURCE, schema_out=snapshot_schema)
    graph.add_node("dst", "ConsumerNode", schema_in=snapshot_schema)
    graph.add_edge("src", "dst")
    
    analysis = analyze_graph(graph)
    assert any("Stale Rollout Warning" in w for w in analysis.warnings)
