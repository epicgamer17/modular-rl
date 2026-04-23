"""
NFSP (Neural Fictitious Self-Play) Implementation using the RL IR.
Demonstrates dual-buffer management and complex orchestration.
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
from runtime.state import ReplayBuffer, ParameterStore, OptimizerState
from runtime.runtime import ActorRuntime, LearnerRuntime
from runtime.scheduler import ScheduleExecutor
from compiler.scheduler import compile_schedule

# 1. Networks
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )
    def forward(self, x):
        return self.net(x)

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

# 2. Operators
def op_mixture_actor(node, inputs, context=None):
    obs = list(inputs.values())[0]
    q_net = node.params["q_net"]
    policy_net = node.params["policy_net"]
    eta = node.params["eta"]
    act_dim = node.params["act_dim"]
    
    # Choose between Best Response (RL) and Average Policy (SL)
    mode = "best_response" if random.random() < eta else "average_policy"
    
    with torch.no_grad():
        if mode == "best_response":
            q_values = q_net(obs.unsqueeze(0))
            action = torch.argmax(q_values).item()
        else:
            probs = policy_net(obs.unsqueeze(0))
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            
    return {"action": action, "mode": mode}

def op_sl_loss(node, inputs, context=None):
    batch = list(inputs.values())[0]
    if not batch: return torch.tensor(0.0, requires_grad=True)
    
    policy_net = node.params["policy_net"]
    obs = batch["obs"]
    actions = batch["action"].long()
    
    probs = policy_net(obs)
    dist = torch.distributions.Categorical(probs)
    return -dist.log_prob(actions).mean()

register_operator("MixtureActor", op_mixture_actor)
register_operator("SLLoss", op_sl_loss)
register_operator(NODE_TYPE_SOURCE, lambda n, i, context=None: None)

# 3. Training Loop
def run_nfsp_demo(total_steps=1000):
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # RL State
    q_net = QNetwork(obs_dim, act_dim)
    rl_buffer = ReplayBuffer(capacity=10000)
    
    # SL State
    policy_net = PolicyNetwork(obs_dim, act_dim)
    sl_buffer = ReplayBuffer(capacity=10000)
    sl_opt = OptimizerState(optim.Adam(policy_net.parameters(), lr=1e-3))
    
    # 1. Graph Definitions
    interact_graph = Graph()
    interact_graph.add_node("obs_in", NODE_TYPE_SOURCE)
    interact_graph.add_node("actor", "MixtureActor", params={
        "q_net": q_net, "policy_net": policy_net, "eta": 0.1, "act_dim": act_dim
    })
    interact_graph.add_edge("obs_in", "actor")
    
    train_graph = Graph()
    train_graph.add_node("batch_in", NODE_TYPE_SOURCE)
    train_graph.add_node("loss", "SLLoss", params={"policy_net": policy_net})
    train_graph.add_edge("batch_in", "loss")
    
    # 2. Runtime Setup
    # Custom recording to handle dual buffers
    def nfsp_record(step_data):
        # Always record to RL buffer
        rl_buffer.add(step_data)
        # Record to SL buffer only if best response was used
        if step_data["metadata"]["actor_results"]["actor"]["mode"] == "best_response":
            sl_buffer.add({
                "obs": step_data["obs"],
                "action": torch.tensor(step_data["metadata"]["actor_results"]["actor"]["action"])
            })
            
    actor_runtime = ActorRuntime(interact_graph, env, recording_fn=nfsp_record)
    
    class NFSP_SL_Learner(LearnerRuntime):
        def update_step(self, batch: Optional[Dict[str, Any]] = None, context: Optional[ExecutionContext] = None):
            if len(sl_buffer) > 64:
                batch = sl_buffer.sample(64)
                collated = {k: torch.stack([t[k] for t in batch]) for k in batch[0].keys()}
                loss = super().update_step(collated)["loss"]
                sl_opt.step(loss)
                
    learner_runtime = NFSP_SL_Learner(train_graph, replay_buffer=sl_buffer)
    
    # 3. Compiled Scheduling
    plan = compile_schedule(interact_graph, user_hints={"actor_frequency": 1, "learner_frequency": 1})
    executor = ScheduleExecutor(plan, actor_runtime, learner_runtime)
    
    print(f"Starting NFSP with Compiled Schedule: {plan.to_dict()}")
    executor.run(total_actor_steps=total_steps)
    print("NFSP Modern Demo Finished.")

if __name__ == "__main__":
    run_nfsp_demo()
