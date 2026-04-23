"""
Scheduler and Loop nodes for the RL IR runtime.
Handles periodic tasks, main execution loops, and parallel rollout pooling.
"""

from typing import List, Callable, Any, Optional, Dict
import threading
import time
from runtime.runtime import ActorRuntime, LearnerRuntime
from runtime.context import ExecutionContext
from dataclasses import dataclass, field


@dataclass
class SchedulePlan:
    """
    Declarative execution plan for RL loops.
    """

    actor_frequency: int = 1
    learner_frequency: int = 1
    prefetch_depth: int = 1
    batching_strategy: str = "serial"  # "serial", "parallel", "vectorized"
    sync_points: List[str] = field(default_factory=list)  # ["step", "episode", "epoch"]

    def to_dict(self):
        return {
            "actor_frequency": self.actor_frequency,
            "learner_frequency": self.learner_frequency,
            "prefetch_depth": self.prefetch_depth,
            "batching_strategy": self.batching_strategy,
            "sync_points": self.sync_points,
        }


from typing import List, Callable, Any, Optional, Dict, Union


class ScheduleExecutor:
    """
    Executes a SchedulePlan by coordinating Actor and Learner runtimes.
    Supports parallel and serial execution strategies.
    """

    def __init__(
        self,
        plan: SchedulePlan,
        actor_runtime: Union[ActorRuntime, List[ActorRuntime]],
        learner_runtime: LearnerRuntime,
    ):
        self.plan = plan
        self.actor_runtimes = (
            actor_runtime if isinstance(actor_runtime, list) else [actor_runtime]
        )
        self.learner_runtime = learner_runtime
        self._running = False

    def run(self, total_actor_steps: int, context: Optional[ExecutionContext] = None):
        self._running = True
        context = context or ExecutionContext()
        target_actor_step = context.actor_step + total_actor_steps

        while context.actor_step < target_actor_step and self._running:
            # 1. Actor Execution (Strategy-aware)
            self._execute_actors(context)

            # Check for env-step based target sync (Formalized: scan nodes)
            self._perform_target_sync(context, on_type="env_step")

            # 2. Learner Step(s)
            for _ in range(self.plan.learner_frequency):
                self.learner_runtime.update_step(context=context)
                
                # Check for learner-step based target sync (Formalized: scan nodes)
                # Note: If TargetSync is in train_graph, it runs during update_step anyway.
                # This call is for TargetSync nodes NOT in the main training loop.
                self._perform_target_sync(context, on_type="learner_step")

            if context.actor_step >= target_actor_step:
                break

    def _perform_target_sync(self, context: ExecutionContext, on_type: str = "learner_step"):
        """Finds and executes TargetSync nodes in the learner graph."""
        from core.graph import NODE_TYPE_TARGET_SYNC
        
        # SEARCH for TargetSync nodes in the learner runtime's graph
        sync_nodes = [nid for nid, node in self.learner_runtime.train_graph.nodes.items() 
                      if node.node_type == NODE_TYPE_TARGET_SYNC 
                      and node.params.get("sync_on", "learner_step") == on_type]
        
        if sync_nodes:
            # If sync nodes exist, they are executed during the normal training step.
            # But here we are triggering an EXPLICIT sync.
            # This might be redundant if learner_frequency == 1 and sync_frequency == 1.
            # To avoid redundancy, we only execute them here if they are NOT meant to run every learner step.
            
            # For now, let's just use the manual update logic if no nodes are found, 
            # or execute the sync nodes specifically.
            for nid in sync_nodes:
                node = self.learner_runtime.train_graph.nodes[nid]
                from runtime.executor import OPERATOR_REGISTRY
                op_func = OPERATOR_REGISTRY[NODE_TYPE_TARGET_SYNC]
                op_func(node, {}, context=context)

    def _execute_actors(self, context: ExecutionContext) -> int:
        """Executes actor steps based on the batching strategy."""
        num_steps = 0

        if self.plan.batching_strategy == "parallel" and len(self.actor_runtimes) > 1:
            # Multi-threaded parallel execution
            threads = []
            for runtime in self.actor_runtimes:
                for _ in range(self.plan.actor_frequency):
                    t = threading.Thread(target=runtime.step, args=(context,))
                    threads.append(t)
                    t.start()
                    num_steps += 1
            for t in threads:
                t.join()
        else:
            # Serial execution (standard or single actor)
            for runtime in self.actor_runtimes:
                for _ in range(self.plan.actor_frequency):
                    runtime.step(context=context)
                    num_steps += 1

        return num_steps

    def stop(self):
        self._running = False
