# ADR-0029: Introduce Service Nodes for Stateful Resources

## Status
Proposed

## Context
In a pure dataflow system, most nodes are transient and stateless. However, RL systems rely on several long-lived, stateful resources that do not fit the "transform" model (ADR-0029):
- **Replay Buffers**: Store potentially millions of transitions across the entire training run.
- **Parameter Stores**: Manage the master copies of model weights.
- **Loggers/Metrics Sinks**: Long-lived connections to external diagnostic services.

Attempting to model these as standard compute components leads to awkward state management and lifecycle issues, especially in a distributed environment.

## Options Considered

### Option 1: Model Resources as Standard Components
- **Pros**
    - Uniformity; everything uses the same Component/Transform API.
- **Cons**
    - **Awkward State Handling**: Standard components are usually re-initialized or cleared across execution steps, making them poor candidates for persistent storage.
    - **Pickling Issues**: Large stateful objects (like replay buffers) cannot be easily moved through the Graph IR edges if they are treated as standard nodes.

### Option 2: Use Service Nodes (Chosen)
- **Pros**
    - **Explicit State Ownership**: Service nodes have a lifecycle that matches the training session, not just a single execution step.
    - **Request/Reply Semantics**: Other nodes interact with Service Nodes via explicit requests (e.g., "Add these transitions to the buffer" or "Give me a batch of weights").
    - **Isolation**: The complexity of the storage backend (Redis, Shared Memory, File System) is hidden inside the Service Node.
- **Cons**
    - Adds more node categories to the **Unified Graph IR (ADR-0026)**.
    - Requires defining specific runtime APIs for node-to-service communication.

## Decision
We propose adopting this approach because the system will formally introduce **Service Nodes** to represent long-lived, stateful resources.

Service Nodes differ from standard Compute Nodes in that:
1. They maintain persistent internal state across graph executions.
2. They are typically instantiated once at the start of a session.
3. They provide specific "endpoints" that other nodes in the graph can call to mutate or query their state.

## Consequences

### Positive
- **Better Encapsulation**: Replay buffer logic is completely isolated from the training graph's dataflow.
- **Easier Distributed Evolution**: A `ReplayNode` can start as a local object and be swapped for a remote Ray-based service without changing the internal structure of the `Learner` or `Actor`.
- **System Stability**: Clearer lifecycle management prevents the accidental deletion or corruption of persistent training data.

### Negative / Tradeoffs
- **API Complexity**: Developers need to understand the difference between a `TransformNode` and a `ServiceNode` and use the correct communication patterns.

## Notes
The `ReplayNode` is the canonical example of a Service Node in this architecture.