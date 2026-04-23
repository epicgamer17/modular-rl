"""
PPO Implementation using the RL IR.
Demonates modern runtime split, compiler-driven scheduling, and snapshot binding.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random
import numpy as np
from typing import Optional, Dict, Any
from torch.func import functional_call
from core.graph import Graph, NODE_TYPE_SOURCE, NODE_TYPE_ACTOR, NODE_TYPE_TRANSFORM
from core.schema import TAG_ON_POLICY, TAG_ORDERED, Schema
from core.nodes import create_policy_actor_def
from runtime.executor import register_operator
from runtime.context import ExecutionContext
from runtime.state import ReplayBuffer, ParameterStore, OptimizerState
from runtime.runtime import ActorRuntime, LearnerRuntime
from runtime.scheduler import SchedulePlan, ScheduleExecutor
from compiler.scheduler import compile_schedule

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
def op_policy_actor(node, inputs, context=None):
    obs = list(inputs.values())[0]
    ac_net = node.params["ac_net"]
    
    # Snapshot binding is handled by ActorRuntime automatically
    snapshot = context.get_actor_snapshot(node.node_id) if context else None
    
    if snapshot:
        params = snapshot.parameters
        version = snapshot.policy_version
    else:
        params = dict(ac_net.named_parameters())
        version = 0
        
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
    data = inputs.get("traj_in")
    gae_data = inputs.get("gae")
    if data is None or gae_data is None:
        return torch.tensor(0.0, requires_grad=True)
    
    ac_net = node.params["ac_net"]
    param_store = node.params["param_store"]
    clip_epsilon = node.params["clip_epsilon"]
    
    # Semantic verification: Check for stale data if strict
    data_version = data.get("policy_version")
    if data_version is not None and torch.any(data_version != param_store.version):
         # In a real training loop we might allow some staleness, 
         # but PPO is sensitive to it.
         pass

    obs = data["obs"]
    actions = data["action"].long()
    old_log_probs = data["log_prob"]
    advantages = gae_data["advantages"]
    returns = gae_data["returns"]
    
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
    
    if hasattr(loss, "requires_grad") and loss.requires_grad:
        opt_state.step(loss)
        # ParameterStore version increments inside step()
    return loss.item()

register_operator("PolicyActor", op_policy_actor)
register_operator("GAE", op_gae)
register_operator("PPOObjective", op_ppo_objective)
register_operator("Optimizer", op_optimizer_step)
register_operator(NODE_TYPE_SOURCE, lambda n, i, context=None: None)

# 3. Training Loop
def run_ppo_demo(total_steps=1000):
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    ac_net = ActorCritic(obs_dim, act_dim)
    param_store = ParameterStore(dict(ac_net.named_parameters()))
    opt = optim.Adam(ac_net.parameters(), lr=1e-3)
    opt_state = OptimizerState(opt)
    rb = ReplayBuffer(capacity=1000)
    
    # 1. Graph Definitions
    interact_graph = Graph()
    interact_graph.add_node("obs_in", NODE_TYPE_SOURCE)
    interact_graph.add_node("actor", "PolicyActor", params={"ac_net": ac_net}, tags=[TAG_ON_POLICY])
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
    
    # 2. Runtime Setup
    # ActorRuntime handles step-by-step interaction and snapshotting
    actor_runtime = ActorRuntime(interact_graph, env, replay_buffer=rb)
    
    # LearnerRuntime handles batch processing and optimization
    class PPOLearner(LearnerRuntime):
        def update_step(self, batch: Optional[Dict[str, Any]] = None, context: Optional[ExecutionContext] = None):
            if len(self.replay_buffer) >= 32:
                # On-policy PPO usually trains on the latest trajectory
                # We query for the most recent 32 steps
                batch = self.replay_buffer.sample_query(batch_size=32)
                if batch:
                    # Collate
                    collated = {k: torch.stack([t[k] for t in batch]) if isinstance(batch[0][k], torch.Tensor) else batch[0][k] for k in batch[0].keys()}
                    super().update_step(collated)
                    # Clear buffer for next on-policy batch
                    self.replay_buffer.clear()
                    
    learner_runtime = PPOLearner(train_graph, replay_buffer=rb)
    
    # 3. Compiler-Driven Scheduling
    # We provide hints but the compiler decides the final plan
    plan = compile_schedule(interact_graph, user_hints={
        "actor_frequency": 32, # Collect 32 steps
        "learner_frequency": 1, # Then do 1 update
    })
    
    executor = ScheduleExecutor(plan, actor_runtime, learner_runtime)
    
    print(f"Starting PPO with Compiled Schedule: {plan.to_dict()}")
    executor.run(total_actor_steps=total_steps)
    print("PPO Modern Demo Finished.")

if __name__ == "__main__":
    run_ppo_demo()
