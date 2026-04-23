import pytest
import torch
import torch.nn as nn
from core.graph import Graph, NODE_TYPE_ACTOR, NODE_TYPE_TARGET_SYNC
from runtime.context import ExecutionContext
from runtime.scheduler import SchedulePlan, ScheduleExecutor
from runtime.runtime import ActorRuntime, LearnerRuntime
from runtime.executor import register_operator, OPERATOR_REGISTRY

# Set module level marker as per RULE[testing-standards.md]
pytestmark = pytest.mark.unit

class DummyEnv:
    def reset(self, seed=None): return [0.0], {}
    def step(self, action): return [0.0], 0.0, False, False, {}

def test_clocks_actor_learner_ratio(monkeypatch):
    """
    Test 5A: 100 actor steps + learner frequency 0.5 (1 learner every 2 actor steps).
    Verifies that global counters in ExecutionContext correctly track distributed execution.
    """
    plan = SchedulePlan(
        actor_frequency=2,
        learner_frequency=1
    )
    
    # Register dummy actor operator
    monkeypatch.setitem(OPERATOR_REGISTRY, NODE_TYPE_ACTOR, lambda n, i, context=None: {"action": 0})
    
    interact_graph = Graph()
    interact_graph.add_node("actor", NODE_TYPE_ACTOR)
    
    train_graph = Graph()
    
    env = DummyEnv()
    actor = ActorRuntime(interact_graph, env)
    learner = LearnerRuntime(train_graph)
    
    executor = ScheduleExecutor(plan, actor, learner)
    ctx = ExecutionContext()
    
    executor.run(total_actor_steps=100, context=ctx)
    
    assert ctx.actor_step == 100, f"Expected 100 actor steps, got {ctx.actor_step}"
    assert ctx.learner_step == 50, f"Expected 50 learner steps, got {ctx.learner_step}"
    assert ctx.env_step == 100
    assert ctx.global_step == 100

def test_clocks_target_sync_trigger(monkeypatch):
    """
    Test 5B: Target sync every 10 learner steps triggers at 10, 20, 30...
    Verifies that scheduler correctly uses context counters for periodic tasks.
    """
    plan = SchedulePlan(
        actor_frequency=1,
        learner_frequency=1
    )
    
    sync_calls = 0
    last_sync_step = -1
    def mock_sync_op(node, inputs, context=None):
        nonlocal sync_calls, last_sync_step
        if context and context.learner_step > 0 and context.learner_step % 10 == 0:
            if context.learner_step != last_sync_step:
                sync_calls += 1
                context.sync_step += 1
                last_sync_step = context.learner_step

    monkeypatch.setitem(OPERATOR_REGISTRY, NODE_TYPE_TARGET_SYNC, mock_sync_op)
    monkeypatch.setitem(OPERATOR_REGISTRY, NODE_TYPE_ACTOR, lambda n, i, context=None: {"action": 0})
    
    interact_graph = Graph()
    interact_graph.add_node("actor", NODE_TYPE_ACTOR)
    
    train_graph = Graph()
    # Node owns the frequency (10)
    train_graph.add_node("sync_node", NODE_TYPE_TARGET_SYNC, params={"sync_frequency": 10, "sync_on": "learner_step"})
    
    env = DummyEnv()
    actor = ActorRuntime(interact_graph, env)
    learner = LearnerRuntime(train_graph)
    
    executor = ScheduleExecutor(plan, actor, learner)
    ctx = ExecutionContext()
    
    # Run 30 actor steps
    executor.run(total_actor_steps=30, context=ctx)
    
    assert ctx.learner_step == 30
    assert ctx.sync_step == 3, f"Expected 3 sync steps, got {ctx.sync_step}"
    assert sync_calls == 3

def test_clocks_episode_tracking(monkeypatch):
    """Verifies that episode_step and episode_count are correctly tracked."""
    plan = SchedulePlan(actor_frequency=1, learner_frequency=0)
    
    class FiniteEnv:
        def __init__(self): self.steps = 0
        def reset(self, seed=None): 
            self.steps = 0
            return [0.0], {}
        def step(self, action):
            self.steps += 1
            done = self.steps >= 5
            return [0.0], 0.0, done, False, {}

    monkeypatch.setitem(OPERATOR_REGISTRY, NODE_TYPE_ACTOR, lambda n, i, context=None: {"action": 0})
    interact_graph = Graph()
    interact_graph.add_node("actor", NODE_TYPE_ACTOR)
    
    env = FiniteEnv()
    actor = ActorRuntime(interact_graph, env)
    learner = LearnerRuntime(Graph())
    
    executor = ScheduleExecutor(plan, actor, learner)
    ctx = ExecutionContext()
    
    # Run 12 steps. 
    # Ep 1: steps 0,1,2,3,4 (done) -> 5 steps
    # Ep 2: steps 0,1,2,3,4 (done) -> 5 steps
    # Ep 3: steps 0,1 -> 2 steps
    executor.run(total_actor_steps=12, context=ctx)
    
    assert ctx.actor_step == 12
    assert ctx.episode_count == 3 # Start(1) + Done(1) + Done(2) = 3
    assert ctx.episode_step == 2 # 2 steps into Ep 3
