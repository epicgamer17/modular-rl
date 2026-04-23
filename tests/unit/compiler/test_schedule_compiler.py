import pytest
from core.graph import Graph, NODE_TYPE_ACTOR, NODE_TYPE_REPLAY_QUERY
from core.nodes import create_policy_actor_def, create_replay_query_def
from core.schema import Schema, TAG_ON_POLICY
from compiler.scheduler import compile_schedule

pytestmark = pytest.mark.unit

def test_schedule_compiler_deterministic_derivation():
    """
    Test 14.2: Ensure execution plan changes deterministically while IR graph is unchanged.
    """
    # 1. Setup a standard Graph
    graph = Graph()
    actor_def = create_policy_actor_def(Schema(fields=[]), Schema(fields=[]))
    graph.add_node("actor_1", NODE_TYPE_ACTOR, tags=[TAG_ON_POLICY])
    graph.add_node("actor_2", NODE_TYPE_ACTOR, tags=[TAG_ON_POLICY])
    
    # 2. Compile with default hints
    plan_1 = compile_schedule(graph)
    assert plan_1.batching_strategy == "parallel"
    assert "step" in plan_1.sync_points
    assert plan_1.prefetch_depth == 0
    
    # 3. Change "Hints" (Compiler Input) but NOT the Graph IR
    plan_2 = compile_schedule(graph, user_hints={"batching_strategy": "serial", "prefetch_depth": 5})
    
    # 4. Assertions
    # Graph remains unchanged
    assert len(graph.nodes) == 2
    assert graph.nodes["actor_1"].node_type == NODE_TYPE_ACTOR
    
    # Execution plan changed deterministically
    assert plan_2.batching_strategy == "serial"
    assert plan_2.prefetch_depth == 5
    # Implicit constraints (like on-policy sync) should still be preserved if not overridden
    assert "step" in plan_2.sync_points 

def test_schedule_compiler_auto_prefetch():
    """Verify that ReplayQuery triggers auto-prefetch depth."""
    graph = Graph()
    graph.add_node("query", NODE_TYPE_REPLAY_QUERY)
    
    plan = compile_schedule(graph)
    assert plan.prefetch_depth == 2
    
    # Ensure it's 0 if no replay query
    graph_no_replay = Graph()
    plan_no_replay = compile_schedule(graph_no_replay)
    assert plan_no_replay.prefetch_depth == 0
