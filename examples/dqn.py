"""
DQN Implementation using the RL IR.
Demonstrates off-policy learning with ReplayQuery and prefetching.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from typing import Optional, Dict, Any
from torch.func import functional_call
from core.graph import (
    Graph,
    NODE_TYPE_SOURCE,
    NODE_TYPE_ACTOR,
    NODE_TYPE_REPLAY_QUERY,
    NODE_TYPE_TARGET_SYNC,
    NODE_TYPE_EXPLORATION,
    NODE_TYPE_METRICS_SINK,
)
from core.schema import TAG_OFF_POLICY, Schema, TensorSpec, Field
from runtime.collator import ReplayCollator
from runtime.executor import register_operator
from runtime.context import ExecutionContext
from runtime.state import ReplayBuffer, ParameterStore, OptimizerState, ModelRegistry, BufferRegistry
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
def op_q_values_single(node, inputs, context=None):
    obs = inputs.get("obs")
    if obs is None:
        from runtime.values import MissingInput
        return MissingInput("obs")
    model_handle = node.params.get("model_handle", "online_q")
    q_net = context.get_model(model_handle) if context else node.params.get("q_net")

    snapshot = context.get_actor_snapshot(node.node_id) if (context and hasattr(context, "get_actor_snapshot")) else None
    state = snapshot.state if snapshot else {**dict(q_net.named_parameters()), **dict(q_net.named_buffers())}

    with torch.inference_mode():
        # Expects [D], returns [A]
        q_values = functional_call(q_net, state, (obs.unsqueeze(0),))
        return q_values.squeeze(0)


def op_q_values_batch(node, inputs, context=None):
    batch = inputs.get("batch")
    if batch is None:
        from runtime.values import MissingInput
        return MissingInput("batch")
    obs = batch["obs"] if isinstance(batch, dict) else batch
    
    model_handle = node.params.get("model_handle", "online_q")
    q_net = context.get_model(model_handle) if context else node.params.get("q_net")

    # In training mode, we usually use the live parameters, not a snapshot
    state = {**dict(q_net.named_parameters()), **dict(q_net.named_buffers())}

    # Training mode: expects [B, D], returns [B, A]
    return q_net(obs) # Simplified for live net, or use functional_call(q_net, state, (obs,))


def op_td_loss(node, inputs, context=None):
    batch_dict = inputs.get("batch")
    if batch_dict is None:
        from runtime.values import MissingInput
        return MissingInput("batch")

    q_handle = node.params.get("q_handle", "online_q")
    target_handle = node.params.get("target_handle", "target_q")
    
    q_net = context.get_model(q_handle)
    target_net = context.get_model(target_handle)
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
    loss = inputs.get("loss")
    if loss is None:
        from runtime.values import MissingInput
        return MissingInput("loss")
    
    # loss is guaranteed to be a real tensor here
    opt_state.step(loss)
    return loss.item()


register_operator("QValuesSingle", op_q_values_single)
register_operator("QValuesBatch", op_q_values_batch)
register_operator("TDLoss", op_td_loss)
register_operator("Optimizer", op_optimizer_step)
register_operator(NODE_TYPE_SOURCE, lambda n, i, context=None: None)


# 3. Training Loop
def train_dqn(total_steps=5000):
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    q_net = QNetwork(obs_dim, act_dim)
    target_net = QNetwork(obs_dim, act_dim)
    target_net.load_state_dict(q_net.state_dict())

    rb = ReplayBuffer(capacity=50000)
    # Include both parameters and buffers in the store
    initial_state = {**dict(q_net.named_parameters()), **dict(q_net.named_buffers())}
    param_store = ParameterStore(initial_state)
    opt = optim.Adam(q_net.parameters(), lr=1e-3)
    opt_state = OptimizerState(opt)

    # 0.1 Register Models
    model_registry = ModelRegistry()
    model_registry.register("online_q", q_net)
    model_registry.register("target_q", target_net)
    
    buffer_registry = BufferRegistry()
    buffer_registry.register("main", rb)

    # 0. Define Schema for Collation
    obs_spec = TensorSpec(shape=(obs_dim,), dtype="float32")
    act_spec = TensorSpec(shape=(), dtype="long")
    reward_spec = TensorSpec(shape=(), dtype="float32")
    done_spec = TensorSpec(shape=(), dtype="float32")

    replay_schema = Schema(
        [
            Field("obs", obs_spec),
            Field("action", act_spec),
            Field("reward", reward_spec),
            Field("next_obs", obs_spec),
            Field("done", done_spec),
        ]
    )
    collator = ReplayCollator(replay_schema)

    # 1. Graph Definitions
    interact_graph = Graph()
    interact_graph.add_node("obs_in", NODE_TYPE_SOURCE)
    interact_graph.add_node("q_values", "QValuesSingle", params={"model_handle": "online_q"})
    interact_graph.add_node(
        "epsilon_decay", 
        "LinearDecay", 
        params={"start_val": 1.0, "end_val": 0.1, "total_steps": 1000}
    )
    interact_graph.add_node(
        "actor", NODE_TYPE_EXPLORATION, params={"act_dim": act_dim}
    )
    interact_graph.add_edge("obs_in", "q_values", dst_port="obs")
    interact_graph.add_edge("q_values", "actor", dst_port="q_values")
    interact_graph.add_edge("epsilon_decay", "actor", dst_port="epsilon")

    train_graph = Graph()
    train_graph.add_node(
        "sampler",
        NODE_TYPE_REPLAY_QUERY,
        params={
            "buffer_id": "main",
            "batch_size": 128,
            "min_size": 100,
            "collator": collator,
        },
    )
    train_graph.add_node(
        "loss",
        "TDLoss",
        params={"q_handle": "online_q", "target_handle": "target_q", "gamma": 0.99},
    )
    train_graph.add_node("opt", "Optimizer", params={"opt_state": opt_state})

    # 4. NEW: Declarative Target Sync
    train_graph.add_node(
        "sync",
        NODE_TYPE_TARGET_SYNC,
        params={
            "source_handle": "online_q",
            "target_handle": "target_q",
            "sync_type": "periodic_hard",
            "sync_frequency": 100,
        },
    )

    train_graph.add_edge("sampler", "loss", dst_port="batch")
    train_graph.add_edge("loss", "opt", dst_port="loss")
    train_graph.add_edge("opt", "sync")

    # 5. Metrics Collection
    # We add a node to compute Q-values for the batch to track average Q
    train_graph.add_node("q_values", "QValuesBatch", params={"model_handle": "online_q"})
    train_graph.add_node("metrics", NODE_TYPE_METRICS_SINK, params={
        "log_frequency": 10,
        "buffer_id": "main" # MetricsSink can use this to track size
    })
    
    # 6. Wiring Metrics
    train_graph.add_edge("sampler", "q_values", dst_port="batch")
    train_graph.add_edge("q_values", "metrics", dst_port="avg_q")
    train_graph.add_edge("loss", "metrics", dst_port="loss")
    train_graph.add_edge("sampler", "metrics", dst_port="batch")

    # 2. Runtime Setup
    actor_runtime = ActorRuntime(interact_graph, env, replay_buffer=rb)

    learner_runtime = LearnerRuntime(train_graph, replay_buffer=rb)

    # 3. Compiler-Driven Scheduling
    plan = compile_schedule(
        train_graph,
        user_hints={
            "actor_frequency": 1,
            "learner_frequency": 1,
        },
    )
    
    ctx = ExecutionContext(model_registry=model_registry, buffer_registry=buffer_registry)
    executor = ScheduleExecutor(plan, actor_runtime, learner_runtime)

    print(f"Starting DQN with Compiled Schedule: {plan.to_dict()}")
    executor.run(total_actor_steps=total_steps, context=ctx)
    print("DQN Modern Demo Finished.")


if __name__ == "__main__":
    train_dqn()
