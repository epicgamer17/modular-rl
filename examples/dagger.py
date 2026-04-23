"""
DAgger (Dataset Aggregation) Implementation using the RL IR.
Demonstrates multi-actor interaction and dataset aggregation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random
import numpy as np
from core.graph import Graph, NODE_TYPE_SOURCE
from runtime.executor import execute, register_operator
from runtime.state import ReplayBuffer, OptimizerState
from runtime.controller import RolloutController

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
    """A simple heuristic expert for CartPole."""
    def __call__(self, obs):
        # CartPole heuristic: if pole is leaning right, move right; else move left
        # obs[2] is pole angle
        return 1 if obs[2] > 0 else 0

# 2. Operators
def op_student_actor(node, inputs):
    obs = list(inputs.values())[0]
    net = node.params["net"]
    with torch.no_grad():
        logits = net(obs.unsqueeze(0))
        return torch.argmax(logits).item()

def op_expert_actor(node, inputs):
    obs = list(inputs.values())[0]
    expert = node.params["expert"]
    return expert(obs)

register_operator("StudentActor", op_student_actor)
register_operator("ExpertActor", op_expert_actor)
register_operator(NODE_TYPE_SOURCE, lambda n, i: None)

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
    
    # 1. Setup Graph with dual actors
    graph = Graph()
    graph.add_node("obs_in", NODE_TYPE_SOURCE)
    graph.add_node("actor", "StudentActor", params={"net": student_net})
    graph.add_node("expert", "ExpertActor", params={"expert": expert})
    graph.add_edge("obs_in", "actor")
    graph.add_edge("obs_in", "expert")
    
    # 2. DAgger Recording Function
    def dagger_record(step_data):
        # The environment was driven by 'actor' (student)
        # But we record the label from 'expert'
        expert_action = step_data["metadata"]["actor_results"]["expert"]
        sl_buffer.add({
            "obs": step_data["obs"],
            "action": torch.tensor(expert_action)
        })
        
    controller = RolloutController(graph, env, recording_fn=dagger_record)
    
    losses = []
    for iter_idx in range(total_iterations):
        # Phase A: Data Collection (Aggregation)
        controller.collect_trajectory(max_steps=steps_per_iter)
        print(f"Iteration {iter_idx}: Buffer Size = {len(sl_buffer)}")
        
        # Phase B: Training
        if len(sl_buffer) > 64:
            iter_losses = []
            for _ in range(20): # 20 gradient steps per iteration
                batch = sl_buffer.sample(64)
                obs_batch = torch.stack([t["obs"] for t in batch])
                act_batch = torch.stack([t["action"] for t in batch]).long()
                
                logits = student_net(obs_batch)
                loss = nn.functional.cross_entropy(logits, act_batch)
                
                opt_state.step(loss)
                iter_losses.append(loss.item())
            losses.append(np.mean(iter_losses))
            print(f"Iteration {iter_idx}: Avg Loss = {losses[-1]:.4f}")
            
    return losses

if __name__ == "__main__":
    print("Starting DAgger demo on CartPole...")
    losses = run_dagger_demo()
    if losses:
        assert losses[-1] < losses[0] or losses[-1] < 0.5, "Loss should generally decrease"
    print("DAgger System Execution Successful!")
