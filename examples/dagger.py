"""
DAgger (Dataset Aggregation) Implementation using the RL IR.
Demonstrates multi-actor interaction and declarative scheduling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random
import numpy as np
from typing import Optional, Dict, Any
from core.graph import Graph, NODE_TYPE_SOURCE, NODE_TYPE_ACTOR
from runtime.executor import register_operator
from runtime.context import ExecutionContext
from runtime.state import ReplayBuffer, OptimizerState
from runtime.runtime import ActorRuntime, LearnerRuntime
from runtime.scheduler import ScheduleExecutor
from compiler.scheduler import compile_schedule

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
        return 1 if obs[2] > 0 else 0

# 2. Operators
def op_student_actor(node, inputs, context=None):
    obs = list(inputs.values())[0]
    net = node.params["net"]
    with torch.no_grad():
        logits = net(obs.unsqueeze(0))
        return torch.argmax(logits).item()

def op_expert_actor(node, inputs, context=None):
    obs = list(inputs.values())[0]
    expert = node.params["expert"]
    return expert(obs)

register_operator("StudentActor", op_student_actor)
register_operator("ExpertActor", op_expert_actor)
register_operator(NODE_TYPE_SOURCE, lambda n, i, context=None: None)

# 3. DAgger Training Loop
def run_dagger_demo(total_iterations=5, steps_per_iter=500):
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    student_net = StudentNetwork(obs_dim, act_dim)
    expert = ExpertPolicy()
    sl_buffer = ReplayBuffer(capacity=10000)
    opt = optim.Adam(student_net.parameters(), lr=1e-3)
    opt_state = OptimizerState(opt)
    
    # 1. Graph Definitions
    graph = Graph()
    graph.add_node("obs_in", NODE_TYPE_SOURCE)
    # Student drives the environment, Expert provides labels
    graph.add_node("actor", "StudentActor", params={"net": student_net})
    graph.add_node("expert", "ExpertActor", params={"expert": expert})
    graph.add_edge("obs_in", "actor")
    graph.add_edge("obs_in", "expert")
    
    # 2. Custom ActorRuntime to handle DAgger labeling
    def dagger_record(step_data):
        expert_action = step_data["metadata"]["actor_results"]["expert"]
        sl_buffer.add({
            "obs": step_data["obs"],
            "action": torch.tensor(expert_action)
        })
        
    actor_runtime = ActorRuntime(graph, env, recording_fn=dagger_record)
    
    # 3. LearnerRuntime
    class DAggerLearner(LearnerRuntime):
        def update_step(self, batch: Optional[Dict[str, Any]] = None, context: Optional[ExecutionContext] = None):
            if len(self.replay_buffer) > 32:
                batch = self.replay_buffer.sample(64)
                obs_batch = torch.stack([t["obs"] for t in batch])
                act_batch = torch.stack([t["action"] for t in batch]).long()
                
                logits = student_net(obs_batch)
                loss = nn.functional.cross_entropy(logits, act_batch)
                opt_state.step(loss)
                
    learner_runtime = DAggerLearner(Graph(), replay_buffer=sl_buffer)
    
    # 4. Compiled Scheduling
    plan = compile_schedule(graph, user_hints={
        "actor_frequency": steps_per_iter,
        "learner_frequency": 20 # 20 steps per iteration
    })
    
    executor = ScheduleExecutor(plan, actor_runtime, learner_runtime)
    
    print(f"Starting DAgger with Compiled Schedule: {plan.to_dict()}")
    # Total steps = iterations * steps_per_iter
    executor.run(total_actor_steps=total_iterations * steps_per_iter)
    print("DAgger Modern Demo Finished.")

if __name__ == "__main__":
    run_dagger_demo()
