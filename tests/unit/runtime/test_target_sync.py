import pytest
import torch
import torch.nn as nn
from core.graph import Graph, NODE_TYPE_TARGET_SYNC
from runtime.context import ExecutionContext
from runtime.state import ModelRegistry
from runtime.runner import SchedulePlan, ScheduleRunner
from runtime.engine import ActorRuntime, LearnerRuntime
from compiler.planner import compile_schedule

pytestmark = pytest.mark.unit

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)
        self.fc.weight.data.fill_(1.0)

def test_target_sync_hard_update():
    online = SimpleNet()
    target = SimpleNet()
    target.fc.weight.data.fill_(0.0) # Start different
    
    registry = ModelRegistry()
    registry.register("online", online)
    registry.register("target", target)
    
    graph = Graph()
    graph.add_node("sync", NODE_TYPE_TARGET_SYNC, params={
        "model_handle": "online",
        "target_handle": "target",
        "tau": 1.0,
        "sync_type": "periodic_hard"
    })
    
    ctx = ExecutionContext(model_registry=registry)
    from runtime.executor import execute
    execute(graph, {}, context=ctx)
    
    assert torch.equal(target.fc.weight, online.fc.weight), "Hard update failed"

def test_target_sync_soft_update():
    online = SimpleNet()
    target = SimpleNet()
    target.fc.weight.data.fill_(0.0)
    online.fc.weight.data.fill_(1.0)
    
    registry = ModelRegistry()
    registry.register("online", online)
    registry.register("target", target)
    
    graph = Graph()
    graph.add_node("sync", NODE_TYPE_TARGET_SYNC, params={
        "model_handle": "online",
        "target_handle": "target",
        "tau": 0.5,
        "sync_type": "soft"
    })
    
    ctx = ExecutionContext(model_registry=registry)
    from runtime.executor import execute
    execute(graph, {}, context=ctx)
    
    # target = 0.5 * 0 + 0.5 * 1 = 0.5
    assert torch.allclose(target.fc.weight, torch.tensor([[0.5]])), f"Soft update failed: {target.fc.weight}"

def test_schedule_executor_triggers_sync():
    online = SimpleNet()
    target = SimpleNet()
    target.fc.weight.data.fill_(0.0)
    
    registry = ModelRegistry()
    registry.register("online", online)
    registry.register("target", target)
    
    # Train graph with sync node
    train_graph = Graph()
    train_graph.add_node("sync", NODE_TYPE_TARGET_SYNC, params={
        "model_handle": "online",
        "target_handle": "target",
        "sync_frequency": 2,
        "sync_on": "learner_step"
    })
    
    # Runtime mock-ups
    class DummyActor(ActorRuntime):
        def __init__(self): pass
        def step(self, context=None):
            if context: context.actor_step += 1
            return {}
        
    class DummyLearner(LearnerRuntime):
        def __init__(self, graph): self.train_graph = graph
        def update_step(self, batch=None, context=None):
            if context: context.learner_step += 1
            # We must explicitly call execute in the mock to run the TargetSync node
            from runtime.executor import execute
            execute(self.train_graph, {}, context=context)
        
    plan = SchedulePlan(
        actor_frequency=1,
        learner_frequency=1
    )
    
    executor = ScheduleRunner(plan, DummyActor(), DummyLearner(train_graph))
    
    ctx = ExecutionContext(model_registry=registry)
    
    # Step 1: No sync yet (learner_step=1, freq=2)
    executor.run(total_actor_steps=1, context=ctx)
    assert ctx.learner_step == 1
    assert not torch.equal(target.fc.weight, online.fc.weight)
    
    # Step 2: Sync should trigger (learner_step=2)
    executor.run(total_actor_steps=1, context=ctx)
    assert ctx.learner_step == 2
    assert torch.equal(target.fc.weight, online.fc.weight), "Scheduled sync failed"

def test_compiler_sets_no_target_sync_in_plan():
    graph = Graph()
    graph.add_node("sync", NODE_TYPE_TARGET_SYNC)
    
    plan = compile_schedule(graph)
    # Attributes should no longer exist on the plan object
    assert not hasattr(plan, "target_sync_frequency")
    assert not hasattr(plan, "target_sync_tau")

def test_target_sync_frequency_check():
    """Test 3A & 3B: Every 100 learner steps exactly one sync, no sync at 99."""
    online = SimpleNet()
    target = SimpleNet()
    target.fc.weight.data.fill_(0.0)
    
    registry = ModelRegistry()
    registry.register("online", online)
    registry.register("target", target)
    
    graph = Graph()
    graph.add_node("sync", NODE_TYPE_TARGET_SYNC, params={
        "model_handle": "online",
        "target_handle": "target",
        "sync_frequency": 100
    })
    
    from runtime.executor import execute
    
    # Step 99: No sync
    ctx = ExecutionContext(learner_step=99, model_registry=registry)
    execute(graph, {}, context=ctx)
    assert not torch.equal(target.fc.weight, online.fc.weight)
    
    # Step 100: Sync
    ctx = ExecutionContext(learner_step=100, model_registry=registry)
    execute(graph, {}, context=ctx)
    assert torch.equal(target.fc.weight, online.fc.weight)

def test_target_sync_hard_sync_equality():
    """Test 3C: Weights identical immediately after hard sync."""
    online = SimpleNet()
    target = SimpleNet()
    online.fc.weight.data.normal_()
    target.fc.weight.data.fill_(0.0)
    
    registry = ModelRegistry()
    registry.register("online", online)
    registry.register("target", target)
    
    graph = Graph()
    graph.add_node("sync", NODE_TYPE_TARGET_SYNC, params={
        "model_handle": "online",
        "target_handle": "target",
        "sync_type": "periodic_hard"
    })
    
    from runtime.executor import execute
    execute(graph, {}, context=ExecutionContext(learner_step=0, model_registry=registry))
    
    assert torch.equal(target.fc.weight, online.fc.weight)
    for p1, p2 in zip(target.parameters(), online.parameters()):
        assert torch.equal(p1, p2)
