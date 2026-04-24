"""
NFSP (Neural Fictitious Self-Play) Implementation using the RL IR.
Demonstrates dual-buffer management and explicit port-based orchestration.
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
    NODE_TYPE_TARGET_SYNC,
    NODE_TYPE_REPLAY_QUERY,
    NODE_TYPE_METRICS_SINK,
)
from core.schema import TensorSpec, Schema, Field
from runtime.executor import register_operator
from runtime.context import ExecutionContext
from runtime.state import ReplayBuffer, ModelRegistry, BufferRegistry, OptimizerState
from runtime.runtime import ActorRuntime, LearnerRuntime
from runtime.scheduler import ScheduleExecutor, SchedulePlan
from compiler.scheduler import compile_schedule
from runtime.values import MissingInput, Value
from runtime.collator import ReplayCollator


# 1. Networks
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, act_dim)
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
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


# 2. Operators
def op_mixture_actor(node, inputs, context=None):
    obs = inputs.get("obs")
    if obs is None:
        return MissingInput("obs")

    q_handle = node.params.get("q_handle", "online_q")
    policy_handle = node.params.get("policy_handle", "policy")

    q_net = context.get_model(q_handle)
    policy_net = context.get_model(policy_handle)
    eta = inputs.get("eta", node.params.get("eta", 0.1))

    batch_size = obs.shape[0]
    rng = context.rng if context else None

    # Selection modes for each environment in batch
    modes = []
    actions = []

    with torch.no_grad():
        q_values = q_net(obs)  # [B, act_dim]
        probs = policy_net(obs)  # [B, act_dim]

        for i in range(batch_size):
            rand_val = rng.random() if rng else 0.0
            if rand_val < eta:
                modes.append("best_response")
                actions.append(torch.argmax(q_values[i]).item())
            else:
                modes.append("average_policy")
                dist = torch.distributions.Categorical(probs[i])
                actions.append(dist.sample().item())

    return {"action": torch.tensor(actions, dtype=torch.int64), "mode": modes}


def op_td_loss(node, inputs, context=None):
    batch = inputs.get("batch")
    if batch is None:
        return MissingInput("batch")

    q_handle = node.params.get("q_handle", "online_q")
    target_handle = node.params.get("target_handle", "target_q")

    q_net = context.get_model(q_handle)
    target_net = context.get_model(target_handle)
    gamma = node.params["gamma"]

    obs = batch["obs"]
    actions = batch["action"].long()
    rewards = batch["reward"]
    next_obs = batch["next_obs"]
    dones = batch["done"]

    current_q = q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        max_next_q = target_net(next_obs).max(1)[0]
        target_q = rewards + (1 - dones.float()) * gamma * max_next_q

    return nn.functional.mse_loss(current_q, target_q)


def op_sl_loss(node, inputs, context=None):
    batch = inputs.get("batch")
    if batch is None:
        return MissingInput("batch")

    policy_handle = node.params.get("policy_handle", "policy")
    policy_net = context.get_model(policy_handle)

    obs = batch["obs"]
    actions = batch["action"].long()

    probs = policy_net(obs)
    dist = torch.distributions.Categorical(probs)
    return -dist.log_prob(actions).mean()


def op_optimizer_step(node, inputs, context=None):
    opt_state = node.params["opt_state"]
    loss = inputs.get("loss")
    if loss is None:
        return MissingInput("loss")

    opt_state.step(loss)
    return loss.item()


register_operator("MixtureActor", op_mixture_actor)
register_operator("TDLoss", op_td_loss)
register_operator("SLLoss", op_sl_loss)
register_operator("Optimizer", op_optimizer_step)
from runtime.values import NoOp

register_operator(NODE_TYPE_SOURCE, lambda n, i, context=None: NoOp())


# 3. Training Loop
def run_nfsp_demo(total_steps=5000):
    from runtime.environment import wrap_env

    raw_env = gym.make("CartPole-v1")
    env = wrap_env(raw_env)

    obs_dim = env.obs_spec.shape[0]
    act_dim = 2  # CartPole-v1

    # Networks
    q_net = QNetwork(obs_dim, act_dim)
    target_q_net = QNetwork(obs_dim, act_dim)
    target_q_net.load_state_dict(q_net.state_dict())
    policy_net = PolicyNetwork(obs_dim, act_dim)

    # Optimizers
    rl_opt = OptimizerState(optim.Adam(q_net.parameters(), lr=1e-3))
    sl_opt = OptimizerState(optim.Adam(policy_net.parameters(), lr=1e-3))

    # Buffers
    rl_buffer = ReplayBuffer(capacity=10000)
    sl_buffer = ReplayBuffer(capacity=10000)

    # Registry
    model_registry = ModelRegistry()
    model_registry.register("online_q", q_net)
    model_registry.register("target_q", target_q_net)
    model_registry.register("policy", policy_net)

    buffer_registry = BufferRegistry()
    buffer_registry.register("rl", rl_buffer)
    buffer_registry.register("sl", sl_buffer)

    # Collator Schema
    obs_spec = TensorSpec(shape=(obs_dim,), dtype="float32")
    act_spec = TensorSpec(shape=(), dtype="int64")
    reward_spec = TensorSpec(shape=(), dtype="float32")

    rl_schema = Schema(
        fields=[
            Field("obs", obs_spec),
            Field("action", act_spec),
            Field("reward", reward_spec),
            Field("next_obs", obs_spec),
            Field("done", TensorSpec(shape=(), dtype="bool")),
        ]
    )
    sl_schema = Schema(fields=[Field("obs", obs_spec), Field("action", act_spec)])

    rl_collator = ReplayCollator(rl_schema)
    sl_collator = ReplayCollator(sl_schema)

    # 1. Graph Definitions
    interact_graph = Graph()
    interact_graph.add_node("obs_in", NODE_TYPE_SOURCE)
    interact_graph.add_node(
        "actor",
        "MixtureActor",
        params={
            "q_handle": "online_q",
            "policy_handle": "policy",
            "eta": 0.1,
            "act_dim": act_dim,
        },
    )
    interact_graph.add_edge("obs_in", "actor", dst_port="obs")

    train_graph = Graph()
    # RL Sub-graph
    train_graph.add_node(
        "rl_sampler",
        NODE_TYPE_REPLAY_QUERY,
        params={
            "buffer_id": "rl",
            "batch_size": 64,
            "min_size": 100,
            "collator": rl_collator,
        },
    )
    train_graph.add_node(
        "rl_loss",
        "TDLoss",
        params={"q_handle": "online_q", "target_handle": "target_q", "gamma": 0.99},
    )
    train_graph.add_node("rl_opt_node", "Optimizer", params={"opt_state": rl_opt})
    train_graph.add_edge("rl_sampler", "rl_loss", dst_port="batch")
    train_graph.add_edge("rl_loss", "rl_opt_node", dst_port="loss")

    # SL Sub-graph
    train_graph.add_node(
        "sl_sampler",
        NODE_TYPE_REPLAY_QUERY,
        params={
            "buffer_id": "sl",
            "batch_size": 64,
            "min_size": 100,
            "collator": sl_collator,
        },
    )
    train_graph.add_node("sl_loss", "SLLoss", params={"policy_handle": "policy"})
    train_graph.add_node("sl_opt_node", "Optimizer", params={"opt_state": sl_opt})
    train_graph.add_edge("sl_sampler", "sl_loss", dst_port="batch")
    train_graph.add_edge("sl_loss", "sl_opt_node", dst_port="loss")

    # Target Sync
    train_graph.add_node(
        "rl_sync",
        NODE_TYPE_TARGET_SYNC,
        params={
            "source_handle": "online_q",
            "target_handle": "target_q",
            "sync_frequency": 100,
            "sync_on": "learner_step",
        },
    )

    # Metrics
    train_graph.add_node(
        "metrics",
        NODE_TYPE_METRICS_SINK,
        params={"log_frequency": 100, "buffer_id": "rl"},
    )
    train_graph.add_edge("rl_loss", "metrics", dst_port="loss")

    # 2. Runtime Setup
    # The RL buffer is written automatically by ActorRuntime (unbatched transitions).
    # recording_fn handles the SL-buffer branch and logging, which the runtime
    # can't know about generically.
    def nfsp_record(single_step):
        actor_val = single_step["metadata"]["actor_results"].get("actor")
        if actor_val and hasattr(actor_val, "data"):
            mode = actor_val.data.get("mode")
            if mode == "best_response":
                sl_buffer.add(
                    {
                        "obs": single_step["obs"],
                        "action": single_step["action"],
                    }
                )

        if single_step["done"]:
            step_idx = single_step["metadata"]["step_index"]
            print(
                f"Step {step_idx} | Episode Return: {actor_runtime.last_episode_return:.2f}"
            )

    actor_runtime = ActorRuntime(
        interact_graph, env, recording_fn=nfsp_record, replay_buffer=rl_buffer
    )
    learner_runtime = LearnerRuntime(train_graph)

    # 3. Scheduling
    plan = SchedulePlan(actor_frequency=1, learner_frequency=1)
    ctx = ExecutionContext(
        model_registry=model_registry, buffer_registry=buffer_registry
    )
    executor = ScheduleExecutor(plan, actor_runtime, learner_runtime)

    print(f"Starting NFSP with Explicit Ports and Registries")
    executor.run(total_actor_steps=total_steps, context=ctx)
    print("NFSP Modern Demo Finished.")


if __name__ == "__main__":
    run_nfsp_demo()
