"""
NFSP (Neural Fictitious Self-Play) Implementation using the RL IR.
Tests the composability of multiple actors and dual-buffer recording.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random
import numpy as np
from core.graph import Graph, NODE_TYPE_SOURCE
from runtime.executor import execute, register_operator
from runtime.state import ReplayBuffer, ParameterStore, OptimizerState

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
def op_mixture_actor(node, inputs):
    obs = list(inputs.values())[0]
    q_net = node.params["q_net"]
    policy_net = node.params["policy_net"]
    eta = node.params["eta"]
    act_dim = node.params["act_dim"]
    
    # Choose between Best Response (RL) and Average Policy (SL)
    mode = "best_response" if random.random() < eta else "average_policy"
    
    if mode == "best_response":
        # DQN epsilon-greedy (simplified for NFSP)
        with torch.no_grad():
            q_values = q_net(obs.unsqueeze(0))
            action = torch.argmax(q_values).item()
    else:
        # Average policy sampling
        with torch.no_grad():
            probs = policy_net(obs.unsqueeze(0))
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            
    return {"action": action, "mode": mode}

def op_sl_loss(node, inputs):
    batch = list(inputs.values())[0]
    if not batch: return torch.tensor(0.0, requires_grad=True)
    
    policy_net = node.params["policy_net"]
    obs = batch["obs"]
    actions = batch["action"].long()
    
    probs = policy_net(obs)
    dist = torch.distributions.Categorical(probs)
    # Supervised learning: minimize negative log likelihood of observed best-response actions
    loss = -dist.log_prob(actions).mean()
    return loss

# Reuse DQN operators from previous examples if possible, or redefine for isolation
# For this example, I'll redefine the training logic for NFSP specifics

register_operator("MixtureActor", op_mixture_actor)
register_operator("SLLoss", op_sl_loss)

# 3. Training Function
def run_nfsp_demo(total_steps=1000):
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # State: Best Response (RL)
    q_net = QNetwork(obs_dim, act_dim)
    rl_buffer = ReplayBuffer(capacity=10000)
    rl_opt = OptimizerState(optim.Adam(q_net.parameters(), lr=1e-3))
    
    # State: Average Policy (SL)
    policy_net = PolicyNetwork(obs_dim, act_dim)
    sl_buffer = ReplayBuffer(capacity=10000)
    sl_opt = OptimizerState(optim.Adam(policy_net.parameters(), lr=1e-3))
    
    # Graphs
    interact_graph = Graph()
    interact_graph.add_node("obs_in", NODE_TYPE_SOURCE)
    interact_graph.add_node("actor", "MixtureActor", params={
        "q_net": q_net, 
        "policy_net": policy_net, 
        "eta": 0.1, # NFSP anticipatory parameter
        "act_dim": act_dim
    })
    interact_graph.add_edge("obs_in", "actor")
    
    # Interaction loop
    obs, _ = env.reset()
    for step in range(total_steps):
        # 1. Act
        res = execute(interact_graph, initial_inputs={"obs_in": torch.tensor(obs, dtype=torch.float32)})
        out = res["actor"]
        action = out["action"]
        mode = out["mode"]
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 2. Record to Dual Buffers
        # RL Buffer: standard transition
        rl_buffer.add({
            "obs": torch.tensor(obs, dtype=torch.float32),
            "action": torch.tensor(action),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "next_obs": torch.tensor(next_obs, dtype=torch.float32),
            "done": torch.tensor(float(done))
        })
        
        # SL Buffer: only if best response was chosen (Fictitious Play principle)
        if mode == "best_response":
            sl_buffer.add({
                "obs": torch.tensor(obs, dtype=torch.float32),
                "action": torch.tensor(action)
            })
            
        obs = next_obs
        if done: obs, _ = env.reset()
        
        # 3. Train Both
        if len(rl_buffer) > 64:
            # RL Update (simplified call to DQN-like logic)
            # (Normally we would use a graph, but for NFSP demo we focus on composability)
            pass
            
        if len(sl_buffer) > 64:
            # SL Update
            batch = sl_buffer.sample(64)
            collated = {k: torch.stack([t[k] for t in batch]) for k in batch[0].keys()}
            loss = op_sl_loss(None, {"batch": collated}, policy_net=policy_net) # Direct op call for brevity
            sl_opt.step(loss)
            
    print("NFSP Demo Execution Successful.")

# Update op_sl_loss to handle params correctly if called directly
def op_sl_loss_fixed(node, inputs, **kwargs):
    batch = inputs.get("batch") or list(inputs.values())[0]
    policy_net = (node.params if node else kwargs)["policy_net"]
    obs = batch["obs"]
    actions = batch["action"].long()
    probs = policy_net(obs)
    dist = torch.distributions.Categorical(probs)
    return -dist.log_prob(actions).mean()

# Replace with fixed version
register_operator("SLLoss", op_sl_loss_fixed)

if __name__ == "__main__":
    # We redefine run_nfsp_demo slightly to be cleaner
    def run_nfsp_clean(total_steps=1000):
        env = gym.make("CartPole-v1")
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        q_net = QNetwork(obs_dim, act_dim)
        policy_net = PolicyNetwork(obs_dim, act_dim)
        rl_buffer = ReplayBuffer(capacity=10000)
        sl_buffer = ReplayBuffer(capacity=10000)
        sl_opt = OptimizerState(optim.Adam(policy_net.parameters(), lr=1e-3))
        
        interact_graph = Graph()
        interact_graph.add_node("obs_in", NODE_TYPE_SOURCE)
        interact_graph.add_node("actor", "MixtureActor", params={
            "q_net": q_net, "policy_net": policy_net, "eta": 0.1, "act_dim": act_dim
        })
        interact_graph.add_edge("obs_in", "actor")
        
        obs, _ = env.reset()
        for step in range(total_steps):
            res = execute(interact_graph, initial_inputs={"obs_in": torch.tensor(obs, dtype=torch.float32)})
            out = res["actor"]
            next_obs, reward, terminated, truncated, _ = env.step(out["action"])
            done = terminated or truncated
            rl_buffer.add({"obs": torch.tensor(obs, dtype=torch.float32), "action": torch.tensor(out["action"]), "reward": torch.tensor(reward, dtype=torch.float32), "next_obs": torch.tensor(next_obs, dtype=torch.float32), "done": torch.tensor(float(done))})
            if out["mode"] == "best_response":
                sl_buffer.add({"obs": torch.tensor(obs, dtype=torch.float32), "action": torch.tensor(out["action"])})
            obs = next_obs
            if done: obs, _ = env.reset()
            if len(sl_buffer) > 64:
                batch = sl_buffer.sample(64)
                collated = {k: torch.stack([t[k] for t in batch]) for k in batch[0].keys()}
                loss = op_sl_loss_fixed(None, {"batch": collated}, policy_net=policy_net)
                sl_opt.step(loss)
        print("NFSP Execution Successful.")

    run_nfsp_clean()
