import torch
import torch.optim as optim
from typing import Optional, Dict, Any
from agents.dqn.config import DQNConfig
from agents.dqn.model import QNetwork
from agents.dqn.graphs import build_actor_graph, build_learner_graph
from agents.dqn.operators import register_dqn_operators
from core.schema import Schema, TensorSpec, Field
from runtime.io.collator import ReplayCollator
from runtime.state import (
    ReplayBuffer,
    OptimizerState,
    ModelRegistry,
    BufferRegistry,
    OptimizerRegistry,
)
from runtime.context import ExecutionContext
from runtime.engine import ActorRuntime, LearnerRuntime
from runtime.runner import ScheduleRunner
from compiler.planner import compile_schedule


class DQNAgent:
    def __init__(self, config: DQNConfig):
        self.config = config

        # 1. Models
        self.q_net = QNetwork(config.obs_dim, config.act_dim, config.hidden_dim)
        self.target_net = QNetwork(config.obs_dim, config.act_dim, config.hidden_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        # 2. Optimization
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.opt_state = OptimizerState(self.optimizer)

        # 3. Buffer & Registries
        self.rb = ReplayBuffer(capacity=config.buffer_capacity)

        self.model_registry = ModelRegistry()
        self.model_registry.register(config.model_handle, self.q_net)
        self.model_registry.register(config.target_handle, self.target_net)

        self.buffer_registry = BufferRegistry()
        self.buffer_registry.register(config.buffer_id, self.rb)

        self.optimizer_registry = OptimizerRegistry()
        self.optimizer_registry.register("main_opt", self.opt_state)

        # 4. Schema & Collator
        self.replay_schema = Schema(
            [
                Field("obs", TensorSpec(shape=(config.obs_dim,), dtype="float32")),
                Field("action", TensorSpec(shape=(), dtype="long")),
                Field("reward", TensorSpec(shape=(), dtype="float32")),
                Field("next_obs", TensorSpec(shape=(config.obs_dim,), dtype="float32")),
                Field("done", TensorSpec(shape=(), dtype="float32")),
            ]
        )
        self.collator = ReplayCollator(self.replay_schema)

        # 5. Graphs
        self.actor_graph = build_actor_graph(config)
        self.learner_graph = build_learner_graph(config, self.collator)

        # 6. Register Operators (idempotent)
        from ops.registry import register_dqn_operators_with_base
        register_dqn_operators_with_base()


    def get_execution_context(self, seed: int = 42) -> ExecutionContext:
        return ExecutionContext(
            seed=seed,
            model_registry=self.model_registry,
            buffer_registry=self.buffer_registry,
            optimizer_registry=self.optimizer_registry,
        )

    def compile(self, strict: bool = False):
        """
        Compiles both actor and learner graphs.
        """
        from compiler.pipeline import compile_graph

        model_handles = {self.config.model_handle, self.config.target_handle}
        buffer_handles = {self.config.buffer_id}

        # Ensure model_handles is a set and contains the expected strings
        assert isinstance(
            model_handles, set
        ), f"model_handles must be a set, got {type(model_handles)}"

        self.actor_graph = compile_graph(
            self.actor_graph,
            strict=strict,
            model_handles=model_handles,
            buffer_handles=buffer_handles,
            context="actor",
        )

        self.learner_graph = compile_graph(
            self.learner_graph,
            strict=strict,
            model_handles=model_handles,
            buffer_handles=buffer_handles,
            context="learner",
        )
