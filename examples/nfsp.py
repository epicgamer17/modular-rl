"""
NFSP (Neural Fictitious Self-Play) implementation.
Demonstrates two alternating agents (Best Response and Average Policy).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from core.graph import Graph, NODE_TYPE_SOURCE
from runtime.operator_registry import register_operator
from runtime.state import ReplayBuffer, ModelRegistry, BufferRegistry, OptimizerRegistry, OptimizerState
from runtime.context import ExecutionContext
from runtime.engine import ActorRuntime, LearnerRuntime
from runtime.runner import SchedulePlan, ScheduleRunner
from runtime.signals import MissingInput

# 1. Models
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

# 2. NFSP Specific Operators
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

    modes = []
    actions = []

    with torch.no_grad():
        q_values = q_net(obs)
        probs = policy_net(obs)

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

register_operator("MixtureActor", op_mixture_actor)

# 3. Training Loop
def run_nfsp():
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # 1. Graph Definitions
    interact_graph = Graph()
    interact_graph.add_node("obs_in", NODE_TYPE_SOURCE)
    interact_graph.add_node("actor", "MixtureActor", params={"eta": 0.1})
    interact_graph.add_edge("obs_in", "actor", dst_port="obs")

    # DQN Learner (Best Response)
    dqn_graph = Graph()
    dqn_graph.add_node("batch_in", NODE_TYPE_SOURCE)
    dqn_graph.add_node("loss", "TDLoss", params={"gamma": 0.99})
    dqn_graph.add_node("opt", "Optimizer", params={"optimizer_handle": "dqn_opt"})
    dqn_graph.add_edge("batch_in", "loss", dst_port="batch")
    dqn_graph.add_edge("loss", "opt", dst_port="loss")

    # SL Learner (Average Policy)
    sl_graph = Graph()
    sl_graph.add_node("batch_in", NODE_TYPE_SOURCE)
    sl_graph.add_node("loss", "SLLoss", params={"policy_handle": "policy"})
    sl_graph.add_node("opt", "Optimizer", params={"optimizer_handle": "sl_opt"})
    sl_graph.add_edge("batch_in", "loss", dst_port="batch")
    sl_graph.add_edge("loss", "opt", dst_port="loss")

    # 2. Registries
    online_q = QNetwork(obs_dim, act_dim)
    target_q = QNetwork(obs_dim, act_dim)
    policy = PolicyNetwork(obs_dim, act_dim)
    
    model_registry = ModelRegistry()
    model_registry.register("online_q", online_q)
    model_registry.register("target_q", target_q)
    model_registry.register("policy", policy)
    
    dqn_opt = OptimizerState(optim.Adam(online_q.parameters(), lr=1e-3))
    sl_opt = OptimizerState(optim.Adam(policy.parameters(), lr=1e-3))
    
    optimizer_registry = OptimizerRegistry()
    optimizer_registry.register("dqn_opt", dqn_opt)
    optimizer_registry.register("sl_opt", sl_opt)
    
    rl_buffer = ReplayBuffer(capacity=10000)
    sl_buffer = ReplayBuffer(capacity=10000)
    
    buffer_registry = BufferRegistry()
    buffer_registry.register("rl", rl_buffer)
    buffer_registry.register("sl", sl_buffer)

    # 3. Execution Context
    ctx = ExecutionContext(
        model_registry=model_registry,
        optimizer_registry=optimizer_registry,
        buffer_registry=buffer_registry
    )

    # 4. Runtimes
    def recording_fn(step_data):
        # step_data is TransitionBatch
        mode = step_data.metadata["actor_results"]["actor"]["mode"]
        # In NFSP, we add to RL buffer always, but SL buffer only if mode was 'best_response'
        rl_buffer.add(step_data.to_dict())
        
        # Batch size 1 assumption for simplicity in this demo
        if mode[0] == "best_response":
            sl_buffer.add({
                "obs": step_data.obs,
                "action": step_data.action
            })

    actor_runtime = ActorRuntime(interact_graph, env, recording_fn=recording_fn)
    dqn_learner = LearnerRuntime(dqn_graph)
    sl_learner = LearnerRuntime(sl_graph)

    # 5. Composite Learner for ScheduleRunner
    class NFSP_Learner:
        def update_step(self, batch=None, context=None):
            # Batch arg is ignored — we sample from the internal buffers directly.
            rl_batch = rl_buffer.sample(32)
            sl_batch = sl_buffer.sample(32)
            
            res_dqn = dqn_learner.update_step(batch=rl_batch, context=context)
            res_sl = sl_learner.update_step(batch=sl_batch, context=context)
            return {"dqn_loss": res_dqn["opt"], "sl_loss": res_sl["opt"]}

    runner = ScheduleRunner(
        SchedulePlan(actor_frequency=1, learner_frequency=4),
        actor_runtime,
        NFSP_Learner()
    )

    # 6. Run
    print("Starting NFSP demo...")
    runner.run(total_actor_steps=1000, context=ctx)
    print("NFSP demo finished.")

if __name__ == "__main__":
    run_nfsp()
