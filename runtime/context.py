"""
Execution Context for the RL IR runtime.
Maintains global clocks, policy versions, and RNG state to ensure
consistency across asynchronous and distributed execution.
"""

from typing import Dict, Any, List, Optional
import torch
import random
import uuid

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
        seed: int = 42
    ):
        self.step_id = step_id
        self.policy_versions = policy_versions or {}
        self.device = device
        self.seed = seed
        
        # RNG state isolation
        self.rng = random.Random(seed)
        self.torch_rng = torch.Generator(device=device)
        self.torch_rng.manual_seed(seed)
        
        # Metadata for traceability
        self.trace_id = str(uuid.uuid4())
        self.actor_snapshots: Dict[str, int] = {}
        self.device_placement: Dict[str, str] = {}
        self.trace_lineage: List[str] = []

    def snapshot_actor(self, actor_id: str, version: int):
        """Records the version of an actor used in this context."""
        self.actor_snapshots[actor_id] = version

    def derive(self, step_id: Optional[int] = None) -> 'ExecutionContext':
        """Creates a child context with incremented step or inherited properties."""
        new_ctx = ExecutionContext(
            step_id=step_id if step_id is not None else self.step_id + 1,
            policy_versions=self.policy_versions.copy(),
            device=self.device,
            seed=self.rng.randint(0, 10**6)
        )
        new_ctx.trace_lineage = self.trace_lineage + [self.trace_id]
        return new_ctx

    def to_dict(self) -> Dict[str, Any]:
        """Serializes context for attachment to traces/DataRefs."""
        return {
            "step_id": self.step_id,
            "trace_id": self.trace_id,
            "policy_versions": self.policy_versions,
            "actor_snapshots": self.actor_snapshots,
            "trace_lineage": self.trace_lineage
        }
