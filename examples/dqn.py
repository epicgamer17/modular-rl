"""
DQN Implementation using the RL IR.
Demonstrates off-policy learning with ReplayQuery and prefetching.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random
import numpy as np
from typing import Optional, Dict, Any
from torch.func import functional_call
from core.graph import Graph, NODE_TYPE_SOURCE, NODE_TYPE_ACTOR, NODE_TYPE_REPLAY_QUERY
from core.schema import TAG_OFF_POLICY, Schema
from runtime.executor import register_operator
from runtime.context import ExecutionContext
from runtime.state import ReplayBuffer, ParameterStore, OptimizerState
from runtime.runtime import ActorRuntime, LearnerRuntime
from runtime.scheduler import ScheduleExecutor
from compiler.scheduler import compile_schedule


# 1. Define Q-Network
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, act_dim),
        )

    def forward(self, x):
        return self.net(x)


# 2. Register Operators
def op_q_actor(node, inputs, context=None):
    obs = list(inputs.values())[0]
    q_net = node.params["q_net"]
    epsilon = node.params["epsilon"]
    act_dim = node.params["act_dim"]

    if random.random() < epsilon:
        return random.randint(0, act_dim - 1)

    snapshot = context.get_actor_snapshot(node.node_id) if context else None
    params = snapshot.parameters if snapshot else dict(q_net.named_parameters())

    with torch.inference_mode():
        q_values = functional_call(q_net, params, (obs.unsqueeze(0),))
        return torch.argmax(q_values).item()


def op_td_loss(node, inputs, context=None):
    batch_dict = list(inputs.values())[0]
    if not batch_dict:
        return torch.tensor(0.0, requires_grad=True)

    q_net = node.params["q_net"]
    target_net = node.params["target_net"]
    gamma = node.params["gamma"]

    states = batch_dict["obs"]
    actions = batch_dict["action"].long()
    rewards = batch_dict["reward"]
    next_states = batch_dict["next_obs"]
    dones = batch_dict["done"]

    current_q = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        max_next_q = target_net(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * gamma * max_next_q

    return nn.functional.mse_loss(current_q, target_q)


def op_optimizer_step(node, inputs, context=None):
    opt_state = node.params["opt_state"]
    loss = list(inputs.values())[0]
    if hasattr(loss, "requires_grad") and loss.requires_grad:
        opt_state.step(loss)
    return loss.item()


register_operator("QActor", op_q_actor)
register_operator("TDLoss", op_td_loss)
register_operator("Optimizer", op_optimizer_step)
register_operator(NODE_TYPE_SOURCE, lambda n, i, context=None: None)
register_operator(
    NODE_TYPE_REPLAY_QUERY, lambda n, i, context=None: None
)  # Handled by LearnerRuntime


# 3. Training Loop
def train_dqn(total_steps=1000):
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    q_net = QNetwork(obs_dim, act_dim)
    target_net = QNetwork(obs_dim, act_dim)
    target_net.load_state_dict(q_net.state_dict())

    rb = ReplayBuffer(capacity=50000)
    param_store = ParameterStore(dict(q_net.named_parameters()))
    opt = optim.Adam(q_net.parameters(), lr=1e-3)
    opt_state = OptimizerState(opt)

    # 1. Graph Definitions
    interact_graph = Graph()
    interact_graph.add_node("obs_in", NODE_TYPE_SOURCE)
    interact_graph.add_node(
        "actor", "QActor", params={"q_net": q_net, "epsilon": 0.1, "act_dim": act_dim}
    )
    interact_graph.add_edge("obs_in", "actor")

    train_graph = Graph()
    # ReplayQuery is now a first-class node
    train_graph.add_node("sampler", NODE_TYPE_REPLAY_QUERY)
    train_graph.add_node(
        "loss",
        "TDLoss",
        params={"q_net": q_net, "target_net": target_net, "gamma": 0.99},
    )
    train_graph.add_node("opt", "Optimizer", params={"opt_state": opt_state})
    train_graph.add_edge("sampler", "loss")
    train_graph.add_edge("loss", "opt")

    # 2. Runtime Setup
    actor_runtime = ActorRuntime(interact_graph, env, replay_buffer=rb)

    class DQNLearner(LearnerRuntime):
        def update_step(
            self,
            batch: Optional[Dict[str, Any]] = None,
            context: Optional[ExecutionContext] = None,
        ):
            if len(self.replay_buffer) > 64:
                # Use query system
                batch = self.replay_buffer.sample_query(batch_size=32)
                if batch:
                    collated = {
                        k: (
                            torch.stack([t[k] for t in batch])
                            if isinstance(batch[0][k], torch.Tensor)
                            else batch[0][k]
                        )
                        for k in batch[0].keys()
                    }
                    super().update_step(collated)

                # Target Update
                if random.random() < 0.01:
                    # TODO: make this update with training steps
                    target_net.load_state_dict(q_net.state_dict())

    learner_runtime = DQNLearner(train_graph, replay_buffer=rb)

    # 3. Compiler-Driven Scheduling
    plan = compile_schedule(
        train_graph, user_hints={"actor_frequency": 1, "learner_frequency": 1}
    )
    executor = ScheduleExecutor(plan, actor_runtime, learner_runtime)

    print(f"Starting DQN with Compiled Schedule: {plan.to_dict()}")
    executor.run(total_actor_steps=total_steps)
    print("DQN Modern Demo Finished.")


if __name__ == "__main__":
    train_dqn()
