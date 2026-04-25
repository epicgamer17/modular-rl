from typing import Dict, Any, Optional, List
from core.graph import Node
from runtime.context import ExecutionContext

def op_loop(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> List[Dict[str, Any]]:
    """Loop(iterations, body_graph) -> execute body_graph N times"""
    from runtime.executor import execute

    iterations = node.params.get("iterations", 1)
    body_graph = node.params.get("body_graph")
    if not body_graph:
        return []

    results = []
    for _ in range(iterations):
        res = execute(body_graph, initial_inputs=inputs, context=context)
        results.append(res)
    return results

def op_minibatch_iterator(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> List[Dict[str, Any]]:
    """MinibatchIterator(batch, advantages, returns, body_graph) -> loop over minibatches"""
    from runtime.executor import execute
    import numpy as np
    from core.batch import TransitionBatch

    batch = inputs.get("batch")
    advantages = inputs.get("advantages")
    returns = inputs.get("returns")
    minibatch_size = node.params.get("minibatch_size", 64)
    body_graph = node.params.get("body_graph")

    if not batch or not body_graph:
        return []

    total_size = batch.obs.shape[0]
    indices = np.arange(total_size)
    np.random.shuffle(indices)

    results = []
    for start in range(0, total_size, minibatch_size):
        end = start + minibatch_size
        mb_indices = indices[start:end]

        # Create a new TransitionBatch for the minibatch
        minibatch = TransitionBatch(
            obs=batch.obs[mb_indices],
            action=batch.action[mb_indices],
            reward=batch.reward[mb_indices],
            next_obs=batch.next_obs[mb_indices],
            done=batch.done[mb_indices],
            log_prob=batch.log_prob[mb_indices] if hasattr(batch, "log_prob") and batch.log_prob is not None else None,
            value=batch.value[mb_indices] if hasattr(batch, "value") and batch.value is not None else None,
            terminated=(
                batch.terminated[mb_indices] if hasattr(batch, "terminated") and batch.terminated is not None else None
            ),
            truncated=(
                batch.truncated[mb_indices] if hasattr(batch, "truncated") and batch.truncated is not None else None
            ),
            policy_version=(
                batch.policy_version[mb_indices]
                if hasattr(batch, "policy_version") and batch.policy_version is not None
                else None
            ),
            advantages=advantages[mb_indices] if advantages is not None else None,
            returns=returns[mb_indices] if returns is not None else None,
        )

        res = execute(
            body_graph, initial_inputs={"traj_in": minibatch}, context=context
        )
        results.append(res)
    return results
