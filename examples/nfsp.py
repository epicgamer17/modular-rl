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
from core.graph import (
    Graph,
    NODE_TYPE_SOURCE,
    NODE_TYPE_ACTOR,
    NODE_TYPE_TARGET_SYNC,
    NODE_TYPE_REPLAY_QUERY
)
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

def op_td_loss(node, inputs, context=None):
    batch = list(inputs.values())[0]
    if not batch: return torch.tensor(0.0, requires_grad=True)
    
    q_net = node.params["q_net"]
    target_net = node.params["target_net"]
    gamma = node.params["gamma"]
    
    obs = batch["obs"]
    actions = batch["action"].long()
    rewards = batch["reward"]
    next_obs = batch["next_obs"]
    dones = batch["done"]
    
    current_q = q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        max_next_q = target_net(next_obs).max(1)[0]
        target_q = rewards + (1 - dones) * gamma * max_next_q
        
    return nn.functional.mse_loss(current_q, target_q)

def op_sl_loss(node, inputs, context=None):
    batch = list(inputs.values())[0]
    if not batch: return torch.tensor(0.0, requires_grad=True)
    
    policy_net = node.params["policy_net"]
    obs = batch["obs"]
    actions = batch["action"].long()
    
    probs = policy_net(obs)
    dist = torch.distributions.Categorical(probs)
    return -dist.log_prob(actions).mean()

def op_optimizer_step(node, inputs, context=None):
    opt_state = node.params["opt_state"]
    loss = list(inputs.values())[0]
    if hasattr(loss, "requires_grad") and loss.requires_grad:
        opt_state.step(loss)
    return loss.item()

register_operator("MixtureActor", op_mixture_actor)
register_operator("TDLoss", op_td_loss)
register_operator("SLLoss", op_sl_loss)
register_operator("Optimizer", op_optimizer_step)
register_operator(NODE_TYPE_SOURCE, lambda n, i, context=None: None)
register_operator(NODE_TYPE_REPLAY_QUERY, lambda n, i, context=None: None)

# 3. Training Loop
def run_nfsp_demo(total_steps=1000):
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # RL State
    q_net = QNetwork(obs_dim, act_dim)
    target_q_net = QNetwork(obs_dim, act_dim)
    target_q_net.load_state_dict(q_net.state_dict())
    rl_buffer = ReplayBuffer(capacity=10000)
    rl_opt = OptimizerState(optim.Adam(q_net.parameters(), lr=1e-3))
    
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
    # RL Sub-graph
    train_graph.add_node("rl_sampler", NODE_TYPE_REPLAY_QUERY, params={"batch_size": 64})
    train_graph.add_node("rl_loss", "TDLoss", params={
        "q_net": q_net, "target_net": target_q_net, "gamma": 0.99
    })
    train_graph.add_node("rl_opt_node", "Optimizer", params={"opt_state": rl_opt})
    train_graph.add_edge("rl_sampler", "rl_loss")
    train_graph.add_edge("rl_loss", "rl_opt_node")
    
    # SL Sub-graph
    train_graph.add_node("sl_sampler", NODE_TYPE_REPLAY_QUERY, params={"batch_size": 64})
    train_graph.add_node("sl_loss", "SLLoss", params={"policy_net": policy_net})
    train_graph.add_node("sl_opt_node", "Optimizer", params={"opt_state": sl_opt})
    train_graph.add_edge("sl_sampler", "sl_loss")
    train_graph.add_edge("sl_loss", "sl_opt_node")
    
    # 4. NEW: Declarative Target Sync for RL
    train_graph.add_node("rl_sync", NODE_TYPE_TARGET_SYNC, params={
        "source_net": q_net,
        "target_net": target_q_net,
        "sync_type": "periodic_hard"
    })
    
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
    
    class NFSP_Learner(LearnerRuntime):
        def update_step(self, batch: Optional[Dict[str, Any]] = None, context: Optional[ExecutionContext] = None):
            # 1. RL Update
            if len(rl_buffer) > 64:
                batch_rl = rl_buffer.sample(64)
                collated_rl = {k: torch.stack([t[k] for t in batch_rl]) if isinstance(batch_rl[0][k], torch.Tensor) else batch_rl[0][k] for k in batch_rl[0].keys()}
                # Map inputs for the graph execution
                # Since we have two samplers, we might need to be careful if we used execute()
                # But here we use super().update_step which uses execute() on the whole graph
                # If we want to execute them separately, we can.
                super().update_step({"rl_sampler": collated_rl}, context=context)
            
            # 2. SL Update
            if len(sl_buffer) > 64:
                batch_sl = sl_buffer.sample(64)
                collated_sl = {k: torch.stack([t[k] for t in batch_sl]) if isinstance(batch_sl[0][k], torch.Tensor) else batch_sl[0][k] for k in batch_sl[0].keys()}
                super().update_step({"sl_sampler": collated_sl}, context=context)
                
    learner_runtime = NFSP_Learner(train_graph, replay_buffer=rl_buffer) # Primary buffer is RL
    
    # 3. Compiled Scheduling
    plan = compile_schedule(
        train_graph, 
        user_hints={
            "actor_frequency": 1, 
            "learner_frequency": 1,
            "target_sync_frequency": 100, # Sync RL target every 100 learner steps
            "target_sync_on": "learner_step"
        }
    )
    executor = ScheduleExecutor(plan, actor_runtime, learner_runtime)
    
    print(f"Starting NFSP with Compiled Schedule: {plan.to_dict()}")
    executor.run(total_actor_steps=total_steps)
    print("NFSP Modern Demo Finished.")

if __name__ == "__main__":
    run_nfsp_demo()
