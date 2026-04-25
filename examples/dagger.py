"""
DAgger (Dataset Aggregation) Implementation using the RL IR.
Demonstrates multi-actor interaction, expert labeling, and explicit ports.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from typing import Optional, Dict, Any
from core.graph import (
    Graph, 
    NODE_TYPE_SOURCE, 
    NODE_TYPE_ACTOR,
    NODE_TYPE_REPLAY_QUERY,
    NODE_TYPE_METRICS_SINK
)
from core.schema import Schema, Field, TensorSpec
from runtime.executor import register_operator
from runtime.context import ExecutionContext
from runtime.state import ReplayBuffer, ModelRegistry, BufferRegistry, OptimizerState
from runtime.runtime import ActorRuntime, LearnerRuntime
from runtime.scheduler import SchedulePlan, ScheduleExecutor
from runtime.values import MissingInput, Value
from runtime.collator import ReplayCollator

# 1. Networks
class StudentNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )
    def forward(self, x):
        return self.net(x)

class ExpertPolicy:
    def __call__(self, obs):
        # Simple CartPole expert: if angle is positive, push right
        return 1 if obs[2] > 0 else 0

# 2. Operators
def op_student_actor(node, inputs, context=None):
    obs = inputs.get("obs")
    if obs is None:
        return MissingInput("obs")
        
    model_handle = node.params.get("model_handle", "student")
    net = context.get_model(model_handle)
    
    with torch.no_grad():
        logits = net(obs)
        return torch.argmax(logits, dim=-1)

def op_expert_actor(node, inputs, context=None):
    obs = inputs.get("obs")
    if obs is None:
        return MissingInput("obs")
        
    expert = node.params["expert"]
    # obs is [B, ...]
    actions = [expert(o) for o in obs]
    return torch.tensor(actions, dtype=torch.int64)

def op_sl_loss(node, inputs, context=None):
    batch = inputs.get("batch")
    if batch is None:
        return MissingInput("batch")
        
    model_handle = node.params.get("model_handle", "student")
    student_net = context.get_model(model_handle)
    
    obs_batch = batch.obs
    act_batch = batch.action.long()
    
    logits = student_net(obs_batch)
    return nn.functional.cross_entropy(logits, act_batch)

def op_optimizer_step(node, inputs, context=None):
    opt_state = node.params["opt_state"]
    loss = inputs.get("loss")
    if loss is None:
        return MissingInput("loss")
        
    opt_state.step(loss)
    return loss.item()

register_operator("StudentActor", op_student_actor)
register_operator("ExpertActor", op_expert_actor)
register_operator("SLLoss", op_sl_loss)
register_operator("Optimizer", op_optimizer_step)

# 3. DAgger Training Loop
def run_dagger_demo(total_iterations=5, steps_per_iter=500):
    from runtime.environment import wrap_env
    raw_env = gym.make("CartPole-v1")
    env = wrap_env(raw_env)
    
    obs_dim = env.obs_spec.shape[0]
    act_dim = 2 # CartPole-v1
    
    student_net = StudentNetwork(obs_dim, act_dim)
    expert = ExpertPolicy()
    sl_buffer = ReplayBuffer(capacity=10000)
    
    model_registry = ModelRegistry()
    model_registry.register("student", student_net)
    
    buffer_registry = BufferRegistry()
    buffer_registry.register("main", sl_buffer)
    
    opt = OptimizerState(optim.Adam(student_net.parameters(), lr=1e-3))
    
    obs_spec = TensorSpec(shape=(obs_dim,), dtype="float32")
    act_spec = TensorSpec(shape=(), dtype="int64")
    sl_schema = Schema(fields=[
        Field("obs", obs_spec),
        Field("action", act_spec)
    ])
    collator = ReplayCollator(sl_schema)
    
    # 1. Graph Definitions
    interact_graph = Graph()
    interact_graph.add_node("obs_in", NODE_TYPE_SOURCE)
    interact_graph.add_node("actor", "StudentActor", params={"model_handle": "student"})
    interact_graph.add_node("expert", "ExpertActor", params={"expert": expert})
    interact_graph.add_edge("obs_in", "actor", dst_port="obs")
    interact_graph.add_edge("obs_in", "expert", dst_port="obs")
    
    train_graph = Graph()
    train_graph.add_node("sampler", NODE_TYPE_REPLAY_QUERY, params={
        "buffer_id": "main", "batch_size": 64, "min_size": 100, "collator": collator
    })
    train_graph.add_node("loss", "SLLoss", params={"model_handle": "student"})
    train_graph.add_node("opt", "Optimizer", params={"opt_state": opt})
    
    train_graph.add_node("metrics", NODE_TYPE_METRICS_SINK, params={"log_frequency": 100, "buffer_id": "main"})
    
    train_graph.add_edge("sampler", "loss", dst_port="batch")
    train_graph.add_edge("loss", "opt", dst_port="loss")
    train_graph.add_edge("loss", "metrics", dst_port="loss")
    
    # 2. Runtime Setup
    def dagger_record(single_step):
        # Extract expert label from node results
        expert_val = single_step.metadata.get("actor_results", {}).get("expert")
        if expert_val and hasattr(expert_val, "data"):
            # Buffer storage for a single transition
            sl_buffer.add({
                "obs": single_step.obs,
                "action": torch.tensor(expert_val.data)
            })
 
        if single_step.done:
            step_idx = single_step.metadata.get("step_index", 0)
            print(
                f"Step {step_idx} | Episode Return: {actor_runtime.last_episode_return:.2f}"
            )
            
    actor_runtime = ActorRuntime(interact_graph, env, recording_fn=dagger_record)
    learner_runtime = LearnerRuntime(train_graph)
    
    # 3. Scheduling
    # DAgger usually collects a whole dataset, then trains for many epochs.
    # Here we simplify: collect steps_per_iter, then do many learner steps.
    plan = SchedulePlan(actor_frequency=steps_per_iter, learner_frequency=20)
    ctx = ExecutionContext(model_registry=model_registry, buffer_registry=buffer_registry)
    executor = ScheduleExecutor(plan, actor_runtime, learner_runtime)
    
    print(f"Starting DAgger with Explicit Ports and Registries")
    executor.run(total_actor_steps=total_iterations * steps_per_iter, context=ctx)
    print("DAgger Modern Demo Finished.")

if __name__ == "__main__":
    run_dagger_demo()
