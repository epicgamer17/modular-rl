"""
PPO Implementation using the RL IR.
Demonstrates on-policy enforcement and stale policy detection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random
import numpy as np
from torch.func import functional_call
from core.graph import Graph, NODE_TYPE_SOURCE, NODE_TYPE_ACTOR, NODE_TYPE_TRANSFORM
from runtime.executor import execute, register_operator
from runtime.state import ReplayBuffer, ParameterStore, OptimizerState
from core.schema import TAG_ON_POLICY, TAG_ORDERED

# 1. Define Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.actor(x), self.critic(x)

# 2. Register Operators
def op_source(node, inputs, context=None):
    return None

def op_policy_actor(node, inputs, context=None):
    obs = list(inputs.values())[0]
    ac_net = node.params["ac_net"]
    param_store = node.params["param_store"]
    
    # Use frozen snapshot if available in context
    snapshot = context.get_actor_snapshot(node.node_id) if context else None
    
    if snapshot:
        params = snapshot.parameters
        version = snapshot.policy_version
    else:
        params = dict(ac_net.named_parameters())
        version = param_store.version
        
    with torch.inference_mode():
        probs, _ = functional_call(ac_net, params, (obs.unsqueeze(0),))
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
    return {
        "action": action.item(),
        "log_prob": log_prob.item(),
        "policy_version": version
    }

def op_gae(node, inputs, context=None):
    trajectory = list(inputs.values())[0]
    gamma = node.params["gamma"]
    gae_lambda = node.params["gae_lambda"]
    ac_net = node.params["ac_net"]
    
    obs = trajectory["obs"]
    rewards = trajectory["reward"]
    dones = trajectory["done"]
    next_obs = trajectory["next_obs"]
    
    with torch.no_grad():
        _, values = ac_net(obs)
        _, next_values = ac_net(next_obs[-1].unsqueeze(0))
        values = torch.cat([values.view(-1), next_values.view(-1)])
        
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        
    returns = advantages + values[:-1]
    return {"advantages": advantages, "returns": returns}

def op_ppo_objective(node, inputs, context=None):
    traj_data = inputs.get("traj_in")
    gae_data = inputs.get("gae")
    if not traj_data or not gae_data:
        return torch.tensor(0.0, requires_grad=True)
    
    data = {**traj_data, **gae_data}
    ac_net = node.params["ac_net"]
    param_store = node.params["param_store"]
    clip_epsilon = node.params["clip_epsilon"]
    
    # STALE POLICY DETECTION using Context or Data
    data_version = data.get("policy_version")
    if data_version is not None and data_version != param_store.version:
        if node.params.get("strict_on_policy", True):
            raise ValueError(f"STALE POLICY DETECTED: Data version {data_version} != Policy version {param_store.version}")

    obs = data["obs"]
    actions = data["action"].long()
    old_log_probs = data["log_prob"]
    advantages = data["advantages"]
    returns = data["returns"]
    
    probs, values = ac_net(obs)
    dist = torch.distributions.Categorical(probs)
    new_log_probs = dist.log_prob(actions)
    
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    
    actor_loss = -torch.min(surr1, surr2).mean()
    critic_loss = nn.functional.mse_loss(values.view(-1), returns.view(-1))
    entropy_loss = -dist.entropy().mean()
    
    return actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

def op_optimizer_step(node, inputs, context=None):
    opt_state = node.params["opt_state"]
    param_store = node.params["param_store"]
    loss = list(inputs.values())[0]
    
    if loss.requires_grad:
        opt_state.step(loss)
        param_store.update_parameters({}) # Increment version
    return loss.item()

register_operator("PolicyActor", op_policy_actor)
register_operator("GAE", op_gae)
register_operator("PPOObjective", op_ppo_objective)
register_operator("Optimizer", op_optimizer_step)
register_operator(NODE_TYPE_SOURCE, op_source)

# 3. Training Function
def run_ppo_demo(total_steps=1000):
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    ac_net = ActorCritic(obs_dim, act_dim)
    param_store = ParameterStore(dict(ac_net.named_parameters()))
    opt = optim.Adam(ac_net.parameters(), lr=1e-3)
    opt_state = OptimizerState(opt)
    
    # Graphs
    interact_graph = Graph()
    interact_graph.add_node("obs_in", NODE_TYPE_SOURCE)
    interact_graph.add_node("actor", "PolicyActor", params={"ac_net": ac_net, "param_store": param_store}, tags=[TAG_ON_POLICY])
    interact_graph.add_edge("obs_in", "actor")
    
    train_graph = Graph()
    train_graph.add_node("traj_in", NODE_TYPE_SOURCE)
    train_graph.add_node("gae", "GAE", params={"ac_net": ac_net, "gamma": 0.99, "gae_lambda": 0.95}, tags=[TAG_ORDERED])
    train_graph.add_node("ppo", "PPOObjective", params={"ac_net": ac_net, "param_store": param_store, "clip_epsilon": 0.2})
    train_graph.add_node("opt", "Optimizer", params={"opt_state": opt_state, "param_store": param_store})
    
    train_graph.add_edge("traj_in", "gae")
    train_graph.add_edge("traj_in", "ppo")
    train_graph.add_edge("gae", "ppo")
    train_graph.add_edge("ppo", "opt")
    
    obs, _ = env.reset()
    trajectory = []
    
    for step in range(total_steps):
        # 1. Interaction
        res = execute(interact_graph, initial_inputs={"obs_in": torch.tensor(obs, dtype=torch.float32)})
        out = res["actor"]
        
        next_obs, reward, terminated, truncated, _ = env.step(out["action"])
        done = terminated or truncated
        
        trajectory.append({
            "obs": torch.tensor(obs, dtype=torch.float32),
            "action": torch.tensor(out["action"]),
            "log_prob": torch.tensor(out["log_prob"]),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "next_obs": torch.tensor(next_obs, dtype=torch.float32),
            "done": torch.tensor(float(done)),
            "policy_version": out["policy_version"]
        })
        
        obs = next_obs
        if done or len(trajectory) >= 32:
            # 2. Training
            # Collate trajectory
            collated = {k: torch.stack([t[k] for t in trajectory]) if isinstance(trajectory[0][k], torch.Tensor) else trajectory[0][k] for k in trajectory[0].keys()}
            
            try:
                execute(train_graph, initial_inputs={"traj_in": collated})
            except ValueError as e:
                print(f"Caught expected PPO error: {e}")
                break
                
            trajectory = []
            if done: obs, _ = env.reset()
            
    print("PPO Demo Finished.")

if __name__ == "__main__":
    run_ppo_demo()
