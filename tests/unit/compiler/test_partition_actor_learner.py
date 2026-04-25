import pytest
from core.graph import Graph, NODE_TYPE_SOURCE, NODE_TYPE_SINK
from compiler.partition import partition_graph
from runtime.registry import register_spec, OperatorSpec, Scalar, clear_registry

pytestmark = pytest.mark.unit

def setup_function():
    """Clear the global registry before each test to ensure isolation."""
    clear_registry()

def test_partition_actor_learner_basic():
    """Verifies that a global graph is correctly partitioned into actor and learner subgraphs."""
    # Register specs with allowed contexts
    register_spec("QNetwork", OperatorSpec.create("QNetwork", allowed_contexts={"actor", "learner"}, differentiable=True, creates_grad=False, consumes_grad=False, updates_params=False))
    register_spec("GreedyPolicy", OperatorSpec.create("GreedyPolicy", allowed_contexts={"actor"}))
    register_spec("ReplayBuffer", OperatorSpec.create("ReplayBuffer", allowed_contexts={"learner"}, differentiable=False, creates_grad=False, consumes_grad=False, updates_params=False))
    register_spec("Optimizer", OperatorSpec.create("Optimizer", allowed_contexts={"learner"}, differentiable=True, creates_grad=False, consumes_grad=True, updates_params=True))

    graph = Graph()
    graph.add_node("obs", NODE_TYPE_SOURCE) # Default to actor
    graph.add_node("q_net", "QNetwork")     # Shared, defaults to actor
    graph.add_node("policy", "GreedyPolicy") # Actor only
    graph.add_node("replay", "ReplayBuffer") # Learner only
    graph.add_node("opt", "Optimizer")       # Learner only

    # Internal actor edges
    graph.add_edge("obs", "q_net")
    graph.add_edge("q_net", "policy")

    # Crossing edge: Actor -> Learner (Replay Write)
    graph.add_edge("policy", "replay")

    # Internal learner edge
    graph.add_edge("replay", "opt")

    partitions = partition_graph(graph)
    actor_g = partitions["actor"]
    learner_g = partitions["learner"]

    # 1. Verify Node Placement
    assert "obs" in actor_g.nodes
    assert "q_net" in actor_g.nodes
    assert "policy" in actor_g.nodes
    assert "replay" not in actor_g.nodes
    assert "opt" not in actor_g.nodes

    assert "replay" in learner_g.nodes
    assert "opt" in learner_g.nodes
    assert "policy" not in learner_g.nodes
    assert "obs" not in learner_g.nodes

    # 2. Verify Channels
    # Actor should have a channel OUT for 'policy -> replay'
    assert "policy_replay_channel_out" in actor_g.nodes
    assert actor_g.nodes["policy_replay_channel_out"].node_type == "ChannelOut"
    
    # Learner should have a channel IN for 'policy -> replay'
    assert "policy_replay_channel_in" in learner_g.nodes
    assert learner_g.nodes["policy_replay_channel_in"].node_type == "ChannelIn"

    # 3. Verify Edge Rewriting
    # Actor: policy -> channel_out
    actor_edge_targets = [e.dst for e in actor_g.edges if e.src == "policy"]
    assert "policy_replay_channel_out" in actor_edge_targets

    # Learner: channel_in -> replay
    learner_edge_sources = [e.src for e in learner_g.edges if e.dst == "replay"]
    assert "policy_replay_channel_in" in learner_edge_sources

def test_partition_optimizer_absent_from_actor():
    """Ensures that learner-only nodes like Optimizer never end up in the actor partition."""
    register_spec("Optimizer", OperatorSpec.create("Optimizer", allowed_contexts={"learner"}, differentiable=True, creates_grad=False, consumes_grad=True, updates_params=True))
    
    graph = Graph()
    graph.add_node("opt", "Optimizer")
    
    partitions = partition_graph(graph)
    assert "opt" not in partitions["actor"].nodes
    assert "opt" in partitions["learner"].nodes
