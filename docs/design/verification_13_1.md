# Step 13.1 Verification Report: Replay as a First-Class Query System

This report verifies the evolution of the Replay Buffer from a simple storage mechanism into a queryable dataset system using the `ReplayQuery` IR node.

## 1. Feature Mapping

| Component | Responsibility | Semantic Role |
| :--- | :--- | :--- |
| **ReplayQuery (IR)** | Formal representation of a data request | Declarative Intent |
| **ReplayBuffer.sample_query** | Execution of constraints (filters, temporal, contiguous) | Query Processor |
| **_check_filters** | Deep metadata and version matching | Semantic Validator |

## 2. Alignment with Semantic Integrity

- **Replay as Dataset**: Data is no longer "sampled" blindly. It is queried based on its semantic properties (e.g., "only on-policy data from v2").
- **Declarative constraints**: The IR node explicitly carries constraints like `temporal_window` and `contiguous`, which are then enforced by the runtime.
- **Causal Consistency**: Querying by `policy_version` ensures that Learners can strictly target data produced by specific model versions, which is critical for algorithms like PPO or NFSP.

## 3. Query Capabilities Verified (Test 13.1)

The implementation was verified against four core query types:
- **Metadata Filtering**: Successfully isolated "expert" data from "on-policy" data using arbitrary metadata keys.
- **Temporal Windowing**: Verified that `temporal_window=N` correctly restricts sampling to the most recent $N$ transitions.
- **Contiguous Sampling**: Proved that `contiguous=True` returns a sequence of transitions in the order they were added, which is essential for RNN training or N-step returns.
- **Version Constraints**: Demonstrated precise isolation of data traces belonging to specific `policy_version` IDs, even when stored in a shared buffer.

## 4. Implementation Details
- [x] **IR Integration**: Added `NODE_TYPE_REPLAY_QUERY` to the core graph schema.
- [x] **Executor Binding**: Registered the `op_replay_query` operator to bridge the graph IR with the stateful `ReplayBuffer`.
- [x] **Thread Safety**: Maintained lock-protected access during query execution to support parallel actors and learners.

> [!TIP]
> By making Replay a query system, we enable advanced patterns like "prioritized experience replay" to be implemented as a higher-level query optimization rather than a hardcoded buffer logic.
