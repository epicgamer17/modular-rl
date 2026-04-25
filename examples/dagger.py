"""
DAgger (Dataset Aggregation) implementation.
Demonstrates running student while recording expert labels.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from core.graph import Graph, NODE_TYPE_SOURCE
from runtime.operator_registry import register_operator
from runtime.state import (
    ReplayBuffer,
    ModelRegistry,
    BufferRegistry,
    OptimizerRegistry,
    OptimizerState,
    CallableRegistry,
)
from runtime.context import ExecutionContext
from runtime.engine import ActorRuntime, LearnerRuntime
from runtime.runner import SchedulePlan, ScheduleRunner
from compiler.planner import compile_schedule

# 1. Simple Models
class SimpleMLP(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )
    def forward(self, x):
        return self.net(x)

def expert_policy(obs):
    # Trivial expert for CartPole: move towards pole lean
    return 1 if obs[2] > 0 else 0

# 2. Training Loop
def run_dagger():
    from ops.registry import register_all_operators
    register_all_operators()
    
    env = gym.make("CartPole-v1")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # 1. Graph Definitions
    interact_graph = Graph()
    interact_graph.add_node("obs_in", NODE_TYPE_SOURCE)
    interact_graph.add_node("student_forward", "QForward", params={"model_handle": "student"})
    interact_graph.add_node("actor", "GreedyAction")
    interact_graph.add_node("expert", "ExpertActor", params={"expert_handle": "expert"})
    interact_graph.add_edge("obs_in", "student_forward", dst_port="obs")
    interact_graph.add_edge("student_forward", "actor", dst_port="input")
    interact_graph.add_edge("obs_in", "expert", dst_port="obs")

    train_graph = Graph()
    train_graph.add_node("batch_in", NODE_TYPE_SOURCE)
    train_graph.add_node("loss", "SLLoss", params={"model_handle": "student"})
    train_graph.add_node("opt", "Optimizer", params={"optimizer_handle": "main_opt"})
    train_graph.add_node("metrics", "MetricsSink", params={"buffer_id": "main"})
    
    train_graph.add_edge("batch_in", "loss", dst_port="batch")
    train_graph.add_edge("loss", "opt", dst_port="loss")
    train_graph.add_edge("loss", "metrics", dst_port="loss")
    train_graph.add_edge("opt", "metrics", dst_port="opt_stats")


    # 2. Registries
    student = SimpleMLP(obs_dim, act_dim)
    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    
    model_registry = ModelRegistry()
    model_registry.register("student", student)
    
    optimizer_registry = OptimizerRegistry()
    optimizer_registry.register("main_opt", OptimizerState(optimizer))
    
    # DAgger Buffer
    sl_buffer = ReplayBuffer(capacity=10000)
    buffer_registry = BufferRegistry()
    buffer_registry.register("main", sl_buffer)

    # 3. Execution Context
    callable_registry = CallableRegistry()
    callable_registry.register("expert", expert_policy)

    ctx = ExecutionContext(
        model_registry=model_registry,
        optimizer_registry=optimizer_registry,
        buffer_registry=buffer_registry,
        callable_registry=callable_registry,
    )

    # 4. Runtimes
    def recording_fn(step_data):
        # step_data is TransitionBatch
        # metadata contains all node results from the interact graph
        expert_action = step_data.metadata["actor_results"]["expert"]
        sl_buffer.add({
            "obs": step_data.obs,
            "action": expert_action
        })

    actor_runtime = ActorRuntime(interact_graph, env, recording_fn=recording_fn)
    learner_runtime = LearnerRuntime(train_graph)

    # 5. Schedule
    plan = SchedulePlan(actor_frequency=1, learner_frequency=10)
    runner = ScheduleRunner(plan, actor_runtime, learner_runtime)

    # 6. Run
    from observability.dispatcher import setup_default_observability
    setup_default_observability()

    print("Starting DAgger demo...")
    try:
        runner.run(total_actor_steps=1000, context=ctx)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Generating plots...")
    finally:
        print("DAgger demo finished.")
        
        # 7. Plot Results
        from observability.plotting.rl_plots import plot_metric
        plot_metric("episode_return", title="DAgger: Episodic Return", save_path="dagger_return.png")
        plot_metric("loss", title="DAgger: Imitation Loss", save_path="dagger_loss.png")
        print("Plots saved to dagger_return.png and dagger_loss.png")


if __name__ == "__main__":
    run_dagger()
