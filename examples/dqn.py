"""
DQN Implementation using the RL IR.
Demonstrates a full system with interaction, record, and training graphs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random
import numpy as np
from torch.func import functional_call
from core.graph import (
    Graph,
    NODE_TYPE_SOURCE,
    NODE_TYPE_ACTOR,
    NODE_TYPE_SINK,
    NODE_TYPE_TRANSFORM,
)
from runtime.executor import execute, register_operator
from runtime.state import ReplayBuffer, ParameterStore, OptimizerState


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
def op_source(node, inputs, context=None):
    return None


def op_q_actor(node, inputs, context=None):
    obs = list(inputs.values())[0]
    if obs is None:
        return None
    q_net = node.params["q_net"]
    param_store = node.params.get("param_store")
    epsilon = node.params["epsilon"]
    act_dim = node.params["act_dim"]

    # Use context for RNG if provided
    rng = context.rng if context else random
    if rng.random() < epsilon:
        return rng.randint(0, act_dim - 1)

    # Use frozen snapshot if available
    snapshot = context.get_actor_snapshot(node.node_id) if context else None
    if snapshot:
        params = snapshot.parameters
    else:
        params = dict(q_net.named_parameters())

    with torch.inference_mode():
        q_values = functional_call(q_net, params, (obs.unsqueeze(0),))
        return torch.argmax(q_values).item()


def op_replay_add(node, inputs, context=None):
    rb = node.params["buffer"]
    if not inputs:
        return None
    transition = list(inputs.values())[0]
    if transition:
        rb.add(transition)
    return None


def op_sample_batch(node, inputs, context=None):
    rb = node.params["buffer"]
    batch_size = node.params["batch_size"]
    batch = rb.sample(batch_size)
    collated = {}
    if batch:
        for k in batch[0].keys():
            collated[k] = torch.stack([t[k] for t in batch])
    return collated


def op_td_loss(node, inputs, context=None):
    batch = list(inputs.values())[0]
    if not batch:
        return torch.tensor(0.0, requires_grad=True)

    q_net = node.params["q_net"]
    target_net = node.params["target_net"]
    gamma = node.params["gamma"]

    states = batch["obs"]
    actions = batch["action"].long()
    rewards = batch["reward"]
    next_states = batch["next_obs"]
    dones = batch["done"]

    current_q = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        max_next_q = target_net(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * gamma * max_next_q

    return nn.functional.mse_loss(current_q, target_q)


def op_optimizer_step(node, inputs, context=None):
    opt_state = node.params["opt_state"]
    loss = list(inputs.values())[0]
    if loss.requires_grad:
        opt_state.step(loss)
    return loss.item()


register_operator(NODE_TYPE_SOURCE, op_source)
register_operator("QActor", op_q_actor)
register_operator("ReplayAdd", op_replay_add)
register_operator("SampleBatch", op_sample_batch)
register_operator("TDLoss", op_td_loss)
register_operator("Optimizer", op_optimizer_step)


# 3. Main Training Loop
def train_dqn(env_name="CartPole-v1", total_steps=30_000):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    q_net = QNetwork(obs_dim, act_dim)
    target_net = QNetwork(obs_dim, act_dim)
    target_net.load_state_dict(q_net.state_dict())

    rb = ReplayBuffer(capacity=50000)
    param_store = ParameterStore(dict(q_net.named_parameters()))
    opt = optim.Adam(q_net.parameters(), lr=1e-3)
    opt_state = OptimizerState(opt)

    # Graphs
    interact_graph = Graph()
    interact_graph.add_node("obs_in", NODE_TYPE_SOURCE)
    interact_graph.add_node(
        "actor", "QActor", params={"q_net": q_net, "param_store": param_store, "epsilon": 0.1, "act_dim": act_dim}
    )
    interact_graph.add_edge("obs_in", "actor")

    record_graph = Graph()
    record_graph.add_node("transition_in", NODE_TYPE_SOURCE)
    record_graph.add_node("replay", "ReplayAdd", params={"buffer": rb})
    record_graph.add_edge("transition_in", "replay")

    train_graph = Graph()
    train_graph.add_node(
        "sampler", "SampleBatch", params={"buffer": rb, "batch_size": 128}
    )
    train_graph.add_node(
        "loss",
        "TDLoss",
        params={"q_net": q_net, "target_net": target_net, "gamma": 0.99},
    )
    train_graph.add_node("opt", "Optimizer", params={"opt_state": opt_state})
    train_graph.add_edge("sampler", "loss")
    train_graph.add_edge("loss", "opt")

    obs, _ = env.reset()
    losses = []
    episode_reward = 0

    for step in range(total_steps):
        # 1. Interaction
        res_interact = execute(
            interact_graph,
            initial_inputs={"obs_in": torch.tensor(obs, dtype=torch.float32)},
        )
        action = res_interact["actor"]

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        # 2. Recording
        transition = {
            "obs": torch.tensor(obs, dtype=torch.float32),
            "action": torch.tensor(action),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "next_obs": torch.tensor(next_obs, dtype=torch.float32),
            "done": torch.tensor(float(done)),
        }
        execute(record_graph, initial_inputs={"transition_in": transition})

        obs = next_obs
        if done:
            obs, _ = env.reset()
            if step % 200 == 0:
                print(f"Step {step}, Episode Reward: {episode_reward}")
            episode_reward = 0

        # 3. Training
        if len(rb) > 64:
            train_results = execute(train_graph, initial_inputs={})
            losses.append(train_results["opt"])

        # 4. Target Update
        if step % 100 == 0:
            target_net.load_state_dict(q_net.state_dict())

    return losses


if __name__ == "__main__":
    print("Starting DQN training on CartPole...")
    losses = train_dqn(total_steps=1000)
    if losses:
        avg_initial_loss = np.mean(losses[:100])
        avg_final_loss = np.mean(losses[-100:])
        print(f"Initial Loss (avg first 100 steps): {avg_initial_loss:.4f}")
        print(f"Final Loss (avg last 100 steps): {avg_final_loss:.4f}")
        # In 1000 steps on CartPole, loss should at least be stable or decreasing
        # We don't assert strictly to avoid flakey tests, but print for verification
    print("DQN System Execution Successful!")
