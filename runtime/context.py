"""
Execution Context for the RL IR runtime.
Maintains global clocks, policy versions, and RNG state to ensure
consistency across asynchronous and distributed execution.
"""

from typing import Dict, Any, List, Optional
import torch
import random
import uuid
from runtime.state import (
    ModelRegistry,
    BufferRegistry,
    OptimizerRegistry,
    GradientRegistry,
    CallableRegistry,
)

class ActorSnapshot:
    """
    A frozen snapshot of an actor's state at a specific point in time.
    Ensures immutability during rollout and training.
    """
    def __init__(
        self, 
        policy_version: int, 
        state: Dict[str, torch.Tensor], 
        config: Optional[Dict[str, Any]] = None
    ):
        self.policy_version = policy_version
        # Deep copy state (parameters and buffers) to ensure immutability
        self.state = {k: v.detach().clone() for k, v in state.items()}
        self.config = config.copy() if config else {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "config": self.config
        }

class ExecutionContext:
    """
    The source of truth for a single execution step or rollout.
    Ensures that all components (actors, processors, buffers) 
    operate on a consistent snapshot of the world.

    Boundary Definition:
    - Runtime Context: This class and its registries are the designated container for 
      MUTABLE LIVE OBJECTS (Models, Optimizers, Buffers).
    - Compile-time IR (Graph): Must remain PURE and DECLARATIVE. It references 
      objects in this context via string handles.
    """
    def __init__(
        self,
        step_id: int = 0,
        policy_versions: Optional[Dict[str, int]] = None,
        device: str = "cpu",
        seed: int = 42,
        shard_id: int = 0,
        actor_step: int = 0,
        env_step: int = 0,
        learner_step: int = 0,
        sync_step: int = 0,
        episode_step: int = 0,
        episode_count: int = 0,
        global_step: int = 0,
        model_registry: Optional[ModelRegistry] = None,
        buffer_registry: Optional[BufferRegistry] = None,
        optimizer_registry: Optional[OptimizerRegistry] = None,
        gradient_registry: Optional[GradientRegistry] = None,
        callable_registry: Optional[CallableRegistry] = None,
    ):
        self.step_id = step_id
        self.policy_versions = policy_versions or {}
        self.device = device
        self.seed = seed
        self.shard_id = shard_id
        self.model_registry = model_registry or ModelRegistry()
        self.buffer_registry = buffer_registry or BufferRegistry()
        self.optimizer_registry = optimizer_registry or OptimizerRegistry()
        self.gradient_registry = gradient_registry or GradientRegistry()
        self.callable_registry = callable_registry or CallableRegistry()
        
        # Clocks
        self.actor_step = actor_step
        self.env_step = env_step
        self.learner_step = learner_step
        self.sync_step = sync_step
        self.episode_step = episode_step
        self.episode_count = episode_count
        self.global_step = global_step
        
        # RNG state isolation
        # Use shard_id to decorrelate RNG streams across parallel workers
        actual_seed = seed + (shard_id * 1000000)
        self.rng = random.Random(actual_seed)
        self.torch_generator = torch.Generator(device=device)
        self.torch_generator.manual_seed(seed)
        
        # Metadata for traceability
        self.trace_id = str(uuid.uuid4())
        self.actor_snapshots: Dict[str, ActorSnapshot] = {}
        self.device_placement: Dict[str, str] = {}
        self.trace_lineage: List[str] = []
        
        # Target Sync State
        self.sync_state: Dict[str, Any] = {
            "last_learner_sync": 0,
            "last_env_sync": 0
        }

    def get_model(self, handle: str) -> torch.nn.Module:
        """Resolves a ModelHandle string to a live PyTorch module."""
        return self.model_registry.get(handle)

    def get_buffer(self, handle: str) -> Any:
        """Resolves a BufferHandle string to a live ReplayBuffer."""
        return self.buffer_registry.get(handle)

    def get_optimizer(self, handle: str) -> Any:
        """Resolves an OptimizerHandle string to a live OptimizerState."""
        return self.optimizer_registry.get(handle)

    def get_gradients(self, handle: str) -> Optional[torch.Tensor]:
        """Resolves a gradient handle to a stored gradient buffer."""
        return self.gradient_registry.get(handle)

    def get_callable(self, handle: str):
        """Resolves a callable handle (e.g., expert policy) to a live function."""
        return self.callable_registry.get(handle)

    def bind_actor(self, actor_id: str, snapshot: ActorSnapshot):
        """Binds an actor to a specific immutable snapshot for this context."""
        self.actor_snapshots[actor_id] = snapshot

    def get_actor_snapshot(self, actor_id: str) -> Optional[ActorSnapshot]:
        """Retrieves the frozen snapshot for a specific actor."""
        return self.actor_snapshots.get(actor_id)

    def derive(self, step_id: Optional[int] = None) -> 'ExecutionContext':
        """Creates a child context with incremented step or inherited properties."""
        new_ctx = ExecutionContext(
            step_id=step_id if step_id is not None else self.step_id + 1,
            policy_versions=self.policy_versions.copy(),
            device=self.device,
            seed=self.rng.randint(0, 10**6),
            shard_id=self.shard_id,
            actor_step=self.actor_step,
            env_step=self.env_step,
            learner_step=self.learner_step,
            sync_step=self.sync_step,
            episode_step=self.episode_step,
            episode_count=self.episode_count,
            global_step=self.global_step,
            model_registry=self.model_registry,
            buffer_registry=self.buffer_registry,
            optimizer_registry=self.optimizer_registry,
            gradient_registry=self.gradient_registry,
            callable_registry=self.callable_registry,
        )
        new_ctx.trace_lineage = self.trace_lineage + [self.trace_id]
        # Inherit actor snapshots
        new_ctx.actor_snapshots = self.actor_snapshots.copy()
        # Inherit sync state
        new_ctx.sync_state = self.sync_state.copy()
        return new_ctx

    def to_dict(self) -> Dict[str, Any]:
        """Serializes context for attachment to traces/DataRefs."""
        return {
            "step_id": self.step_id,
            "trace_id": self.trace_id,
            "policy_versions": self.policy_versions,
            "actor_snapshots": {k: v.to_dict() for k, v in self.actor_snapshots.items()},
            "trace_lineage": self.trace_lineage
        }
