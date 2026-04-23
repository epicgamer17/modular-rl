import pytest
import torch
import torch.nn as nn
from core.graph import Graph, NODE_TYPE_TARGET_SYNC
from runtime.context import ExecutionContext
from runtime.scheduler import SchedulePlan, ScheduleExecutor
from runtime.runtime import ActorRuntime, LearnerRuntime
from compiler.scheduler import compile_schedule

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
    
    graph = Graph()
    graph.add_node("sync", NODE_TYPE_TARGET_SYNC, params={
        "source_net": online,
        "target_net": target,
        "tau": 1.0,
        "sync_type": "periodic_hard"
    })
    
    ctx = ExecutionContext()
    from runtime.executor import execute
    execute(graph, {}, context=ctx)
    
    assert torch.equal(target.fc.weight, online.fc.weight), "Hard update failed"

def test_target_sync_soft_update():
    online = SimpleNet()
    target = SimpleNet()
    target.fc.weight.data.fill_(0.0)
    online.fc.weight.data.fill_(1.0)
    
    graph = Graph()
    graph.add_node("sync", NODE_TYPE_TARGET_SYNC, params={
        "source_net": online,
        "target_net": target,
        "tau": 0.5,
        "sync_type": "soft"
    })
    
    ctx = ExecutionContext()
    from runtime.executor import execute
    execute(graph, {}, context=ctx)
    
    # target = 0.5 * 0 + 0.5 * 1 = 0.5
    assert torch.allclose(target.fc.weight, torch.tensor([[0.5]])), f"Soft update failed: {target.fc.weight}"

def test_schedule_executor_triggers_sync():
    online = SimpleNet()
    target = SimpleNet()
    target.fc.weight.data.fill_(0.0)
    
    # Train graph with sync node
    train_graph = Graph()
    train_graph.add_node("sync", NODE_TYPE_TARGET_SYNC, params={
        "source_net": online,
        "target_net": target
    })
    
    # Runtime mock-ups
    class DummyActor(ActorRuntime):
        def __init__(self): pass
        def step(self, context=None): return {}
        
    class DummyLearner(LearnerRuntime):
        def __init__(self, graph): self.train_graph = graph
        def update_step(self, context=None): pass
        
    plan = SchedulePlan(
        target_sync_frequency=2,
        target_sync_on="learner_step"
    )
    
    executor = ScheduleExecutor(plan, DummyActor(), DummyLearner(train_graph))
    
    ctx = ExecutionContext()
    
    # Step 1: No sync yet
    executor.run(total_actor_steps=1, context=ctx)
    assert ctx.learner_step == 1
    assert not torch.equal(target.fc.weight, online.fc.weight)
    
    # Step 2: Sync should trigger
    executor.run(total_actor_steps=1, context=ctx)
    assert ctx.learner_step == 2
    assert torch.equal(target.fc.weight, online.fc.weight), "Scheduled sync failed"

def test_compiler_sets_sync_defaults():
    graph = Graph()
    graph.add_node("sync", NODE_TYPE_TARGET_SYNC)
    
    plan = compile_schedule(graph)
    assert plan.target_sync_frequency == 100, "Default sync frequency should be 100"
    assert plan.target_sync_tau == 1.0
