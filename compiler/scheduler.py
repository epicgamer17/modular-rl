"""
Schedule Compiler for the RL IR.
Translates a Graph and user hints into a deterministic SchedulePlan.
This pass ensures the runtime is purely an executor and doesn't make scheduling decisions.
"""

from typing import Dict, Any, List
from core.graph import Graph, NODE_TYPE_ACTOR, NODE_TYPE_REPLAY_QUERY
from runtime.scheduler import SchedulePlan

def compile_schedule(graph: Graph, user_hints: Dict[str, Any] = None) -> SchedulePlan:
    """
    Compiler pass that derives an optimal SchedulePlan from the Graph IR.
    
    Decisions made here:
    1. actor_frequency: Based on whether the graph is on-policy or off-policy.
    2. prefetch_depth: Automatically enabled if ReplayQuery nodes are present.
    3. batching_strategy: Parallelized if multiple actor nodes exist.
    4. sync_points: Injected at step/episode boundaries based on semantic tags.
    """
    hints = user_hints or {}
    
    # Analyze Graph Properties
    actor_nodes = [n for n in graph.nodes.values() if n.node_type == NODE_TYPE_ACTOR]
    has_on_policy = any("OnPolicy" in n.tags for n in graph.nodes.values())
    has_replay = any(n.node_type == NODE_TYPE_REPLAY_QUERY for n in graph.nodes.values())
    
    # 1. Strategy Selection
    strategy = hints.get("batching_strategy")
    if strategy is None:
        strategy = "parallel" if len(actor_nodes) > 1 else "serial"
        
    # 2. Frequency Logic
    # On-policy usually defaults to 1 interaction per training step if not specified
    actor_freq = hints.get("actor_frequency", 1)
    learner_freq = hints.get("learner_frequency", 1)
    
    # 3. Prefetch Logic
    # Auto-enable prefetching if we have a replay buffer to hide sampling latency
    prefetch = hints.get("prefetch_depth")
    if prefetch is None:
        prefetch = 2 if has_replay else 0
        
    # 4. Sync Points
    # Ensure parameter sync happens at least every step for on-policy algorithms
    sync = hints.get("sync_points", [])
    if has_on_policy and "step" not in sync:
        sync.append("step")
        

    return SchedulePlan(
        actor_frequency=actor_freq,
        learner_frequency=learner_freq,
        prefetch_depth=prefetch,
        batching_strategy=strategy,
        sync_points=sync,
    )
