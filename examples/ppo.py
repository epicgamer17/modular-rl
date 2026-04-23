"""
PPO Implementation using the RL IR.
Demonstrates on-policy scheduling, explicit ports, and model registries.
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
    NODE_TYPE_METRICS_SINK,
)
from core.schema import Schema, Field, TensorSpec, TAG_ON_POLICY, TAG_ORDERED
from runtime.executor import register_operator
from runtime.context import ExecutionContext
from runtime.state import ReplayBuffer, ModelRegistry, BufferRegistry, OptimizerState
from runtime.runtime import ActorRuntime, LearnerRuntime
from runtime.scheduler import SchedulePlan, ScheduleExecutor
from runtime.values import MissingInput, Value
from runtime.collator import ReplayCollator


# 1. Define Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.actor(x), self.critic(x)


# 2. Operators
def op_policy_actor(node, inputs, context=None):
    obs = inputs.get("obs")
    if obs is None:
        return MissingInput("obs")

    model_handle = node.params.get("model_handle", "ppo_net")
    ac_net = context.get_model(model_handle)

    # Snapshot binding is handled by ActorRuntime automatically via ExecutionContext
    snapshot = context.get_actor_snapshot(node.node_id) if context else None

    if snapshot:
        params = snapshot.parameters
        version = snapshot.policy_version
    else:
        params = dict(ac_net.named_parameters())
        version = 0

    with torch.inference_mode():
        # Pass both parameters and buffers (PPO AC net might have buffers)
        # For now assume no buffers or get them from context
        probs, _ = functional_call(ac_net, params, (obs.unsqueeze(0),))
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

    return {
        "action": action.item(),
        "log_prob": log_prob.item(),
        "policy_version": version,
    }


def op_gae(node, inputs, context=None):
    batch = inputs.get("batch")
    if batch is None:
        return MissingInput("batch")

    gamma = node.params["gamma"]
    gae_lambda = node.params["gae_lambda"]
    model_handle = node.params.get("model_handle", "ppo_net")
    ac_net = context.get_model(model_handle)

    obs = batch["obs"]
    rewards = batch["reward"]
    dones = batch["done"].float()
    next_obs = batch["next_obs"]

    with torch.no_grad():
        _, values = ac_net(obs)
        _, next_values = ac_net(next_obs[-1].unsqueeze(0))
        values = torch.cat([values.view(-1), next_values.view(-1)])

    advantages = torch.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = (
            delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        )

    returns = advantages + values[:-1]
    return {"advantages": advantages, "returns": returns}


def op_ppo_objective(node, inputs, context=None):
    batch = inputs.get("batch")
    gae_data = inputs.get("gae")
    if batch is None or gae_data is None:
        return MissingInput("batch/gae")

    model_handle = node.params.get("model_handle", "ppo_net")
    ac_net = context.get_model(model_handle)
    clip_epsilon = node.params["clip_epsilon"]

    obs = batch["obs"]
    actions = batch["action"].long()
    old_log_probs = batch["log_prob"]
    advantages = gae_data["advantages"]
    returns = gae_data["returns"]

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
    loss = inputs.get("loss")
    if loss is None:
        return MissingInput("loss")

    opt_state.step(loss)
    return loss.item()


register_operator("PolicyActor", op_policy_actor)
register_operator("GAE", op_gae)
register_operator("PPOObjective", op_ppo_objective)
register_operator("Optimizer", op_optimizer_step)


# 3. Training Loop
def run_ppo_demo(total_steps=5000):
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    ac_net = ActorCritic(obs_dim, act_dim)
    model_registry = ModelRegistry()
    model_registry.register("ppo_net", ac_net)

    opt = OptimizerState(optim.Adam(ac_net.parameters(), lr=1e-3))
    rb = ReplayBuffer(capacity=512)
    buffer_registry = BufferRegistry()
    buffer_registry.register("main", rb)

    obs_spec = TensorSpec(shape=(obs_dim,), dtype="float32")
    act_spec = TensorSpec(shape=(), dtype="int64")

    ppo_schema = Schema(
        fields=[
            Field("obs", obs_spec),
            Field("action", act_spec),
            Field("reward", TensorSpec(shape=(), dtype="float32")),
            Field("next_obs", obs_spec),
            Field("done", TensorSpec(shape=(), dtype="bool")),
            Field("log_prob", TensorSpec(shape=(), dtype="float32")),
            Field("policy_version", TensorSpec(shape=(), dtype="int64")),
        ]
    )
    collator = ReplayCollator(ppo_schema)

    # 1. Graph Definitions
    interact_graph = Graph()
    interact_graph.add_node("obs_in", NODE_TYPE_SOURCE)
    interact_graph.add_node(
        "actor", "PolicyActor", params={"model_handle": "ppo_net"}, tags=[TAG_ON_POLICY]
    )
    interact_graph.add_edge("obs_in", "actor", dst_port="obs")

    train_graph = Graph()
    train_graph.add_node(
        "sampler",
        NODE_TYPE_REPLAY_QUERY,
        # TODO: correct PPO minibatch logic
        params={
            "buffer_id": "main",
            "batch_size": 512,
            "min_size": 512,
            "collator": collator,
        },
    )
    train_graph.add_node(
        "gae",
        "GAE",
        params={"model_handle": "ppo_net", "gamma": 0.99, "gae_lambda": 0.95},
        tags=[TAG_ORDERED],
    )
    train_graph.add_node(
        "ppo", "PPOObjective", params={"model_handle": "ppo_net", "clip_epsilon": 0.2}
    )
    train_graph.add_node("opt", "Optimizer", params={"opt_state": opt})

    train_graph.add_node(
        "metrics",
        NODE_TYPE_METRICS_SINK,
        params={"log_frequency": 32, "buffer_id": "main"},
    )

    train_graph.add_edge("sampler", "gae", dst_port="batch")
    train_graph.add_edge("sampler", "ppo", dst_port="batch")
    train_graph.add_edge("gae", "ppo", dst_port="gae")
    train_graph.add_edge("ppo", "opt", dst_port="loss")
    train_graph.add_edge("ppo", "metrics", dst_port="loss")

    # 2. Runtime Setup
    def ppo_record(step_data):
        # Flatten actor_results into top level for the collator
        results = step_data["metadata"]["actor_results"]
        actor_val = results.get("actor")
        if (
            actor_val
            and hasattr(actor_val, "data")
            and isinstance(actor_val.data, dict)
        ):
            step_data.update(actor_val.data)
        rb.add(step_data)

    actor_runtime = ActorRuntime(interact_graph, env, recording_fn=ppo_record)

    # Standard LearnerRuntime but with a small trick for On-Policy:
    # Clear the buffer after each successful execution of the train_graph.
    class OnPolicyLearner(LearnerRuntime):
        def update_step(self, context: Optional[ExecutionContext] = None):
            # Try to run the graph
            results = super().update_step(context=context)
            # If the optimizer actually ran (produced a value), clear the on-policy buffer
            opt_res = results.get("opt")
            if opt_res and opt_res.has_data:
                context.get_buffer("main").clear()

    learner_runtime = OnPolicyLearner(train_graph)

    # 3. Scheduling
    # Run 32 steps of actor, then 1 step of learner
    plan = SchedulePlan(actor_frequency=512, learner_frequency=1)
    ctx = ExecutionContext(
        model_registry=model_registry, buffer_registry=buffer_registry
    )
    executor = ScheduleExecutor(plan, actor_runtime, learner_runtime)

    print(f"Starting PPO with Explicit Ports and Registries")
    executor.run(total_actor_steps=total_steps, context=ctx)
    print("PPO Modern Demo Finished.")


if __name__ == "__main__":
    run_ppo_demo()
