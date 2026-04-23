"""
Execution Context for the RL IR runtime.
Maintains global clocks, policy versions, and RNG state to ensure
consistency across asynchronous and distributed execution.
"""

from typing import Dict, Any, List, Optional
import torch
import random
import uuid

class ActorSnapshot:
    """
    A frozen snapshot of an actor's state at a specific point in time.
    Ensures immutability during rollout and training.
    """
    def __init__(
        self, 
        policy_version: int, 
        parameters: Dict[str, torch.Tensor], 
        config: Optional[Dict[str, Any]] = None
    ):
        self.policy_version = policy_version
        # Deep copy parameters to ensure immutability
        self.parameters = {k: v.detach().clone() for k, v in parameters.items()}
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
    """
    def __init__(
        self,
        step_id: int = 0,
        policy_versions: Optional[Dict[str, int]] = None,
        device: str = "cpu",
        seed: int = 42,
        global_step: int = 0,
        env_step: int = 0,
        learner_step: int = 0
    ):
        self.step_id = step_id
        self.policy_versions = policy_versions or {}
        self.device = device
        self.seed = seed
        
        # Clocks
        self.global_step = global_step
        self.env_step = env_step
        self.learner_step = learner_step
        
        # RNG state isolation
        self.rng = random.Random(seed)
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
            global_step=self.global_step,
            env_step=self.env_step,
            learner_step=self.learner_step
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
