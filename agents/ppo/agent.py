import torch
import torch.optim as optim
from typing import Optional, Dict, Any

from runtime.state import ReplayBuffer, ModelRegistry, BufferRegistry, OptimizerState, OptimizerRegistry
from runtime.engine import ActorRuntime
from runtime.runner import SchedulePlan, ScheduleRunner
from runtime.context import ExecutionContext

from .config import PPOConfig
from .model import ActorCritic
from .operators import register_ppo_operators
from .graphs import create_interact_graph, create_train_graph
from .rollout import create_ppo_recording_fn
from .learner import OnPolicyLearner

class PPOAgent:
    """
    Modular PPO Agent.
    
    Orchestrates the model, graphs, runtimes, and scheduling for PPO.
    """
    def __init__(self, config: PPOConfig, env: Any):
        """
        Initialize the PPO Agent.
        
        Args:
            config: PPO configuration.
            env: The environment to interact with.
        """
        self.config = config
        self.env = env
        
        # 1. Register PPO specific operators
        register_ppo_operators()
        
        # 2. Setup Model
        self.ac_net = ActorCritic(
            obs_dim=config.obs_dim, 
            act_dim=config.act_dim, 
            hidden_dim=config.hidden_dim
        )
        self.model_registry = ModelRegistry()
        self.model_registry.register(config.model_handle, self.ac_net)
        
        # 3. Setup Optimizer
        self.opt = OptimizerState(
            optim.Adam(self.ac_net.parameters(), lr=config.learning_rate, eps=config.adam_epsilon),
            grad_clip=config.max_grad_norm
        )
        self.optimizer_registry = OptimizerRegistry()
        self.optimizer_registry.register(config.optimizer_handle, self.opt)
        
        # 4. Setup Buffer
        # PPO uses RolloutBuffer, not ReplayBuffer
        from .buffer import RolloutBuffer
        self.rb = RolloutBuffer(
            rollout_steps=config.rollout_steps,
            num_envs=config.num_envs,
            obs_dim=config.obs_dim
        )
        self.buffer_registry = BufferRegistry()
        self.buffer_registry.register(config.buffer_id, self.rb)
        
        # 5. Setup Graphs
        from .graphs import create_ppo_update_graph
        self.interact_graph = create_interact_graph(config)
        self.train_graph = create_train_graph(config)
        self.update_graph = create_ppo_update_graph(config)
        
        # 6. Setup Runtimes
        recording_fn = create_ppo_recording_fn(self.rb)
        self.actor_runtime = ActorRuntime(
            self.interact_graph, 
            env, 
            recording_fn=recording_fn
        )
        self.learner_runtime = OnPolicyLearner(
            self.update_graph, 
            config=config,
            ac_net=self.ac_net,
            actor_runtime=self.actor_runtime,
            buffer_id=config.buffer_id
        )
        
        # 7. Setup Context and Executor
        self.ctx = ExecutionContext(
            model_registry=self.model_registry,
            buffer_registry=self.buffer_registry,
            optimizer_registry=self.optimizer_registry,
        )
        
        plan = SchedulePlan(
            actor_frequency=1, # PPO usually runs 1 rollout then N epochs
            learner_frequency=1
        )
        self.runner = ScheduleRunner(
            plan, 
            self.actor_runtime, 
            self.learner_runtime
        )

    def train(self, total_steps: int):
        """
        Train the agent.
        
        Args:
            total_steps: Total number of actor steps to perform.
        """
        print(f"Starting PPO Training for {total_steps} steps...")
        self.runner.run(total_actor_steps=total_steps, context=self.ctx)
        print("PPO Training Finished.")
