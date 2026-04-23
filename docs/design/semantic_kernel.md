# Semantic Kernel Nouns

Component:
A modular unit of logic that interacts with the Blackboard via declared input (requires) and output (provides) contracts.

Node:
A structural unit in the execution graph that encapsulates a component and manages its runtime execution context.

Edge:
A directed dependency between nodes derived from overlapping output and input contracts at graph-build time.

Actor:
An engine configuration and hardware-wrapped execution loop specialized for environment interaction and trajectory generation.

Source:
A provider of initial data, such as a batch from a replay buffer or a raw observation from an environment.

Transform:
A stateless mathematical mapping that produces new blackboard state from existing inputs without managing execution state.

Objective:
A target key or loss scalar that serves as a sink for backward-reachability pruning in the execution graph.

Service:
A long-lived, stateful resource node that persists across multiple graph executions to manage shared resources like buffers.

Controller:
The BlackboardEngine orchestrator that manages the execution lifecycle and dynamic resolution of the dependency graph.

Schema:
A formal contract (Key) that defines the semantic type, shape, and metadata expected at a specific blackboard path.

DataRef:
A standardized dotted-path string (e.g., "data.observations") used to address and validate entries within the Blackboard's structure.

ExecutionPlan:
A topologically sorted sequence of component indices resolved to satisfy a set of requested targets for a given step.

## The Data Contract

What crosses graph edges? The system recognizes four fundamental types of data that cross edges in the execution graph.

ScalarValue:
A primitive numeric value (int, float, bool) or static metadata entry used for telemetry, control flags, and hyperparameter scheduling.

DataRef:
A symbolic handle to a multidimensional tensor governed by a Schema that ensures mathematical compatibility across graph edges.

StreamHandle:
An asynchronous pointer to a continuous data producer that feeds sequences of Blackboard states from external sources.

ResourceHandle:
A stable reference to a persistent Service Node that provides components with access to shared state and side-effecting logic.

## Implementation Mapping

The following table maps the Semantic Kernel abstractions to their concrete implementations in the current codebase:

| Semantic Concept | Implementation | Primary Location |
| :--- | :--- | :--- |
| **Component** | `PipelineComponent` (ABC) | `core/component.py` |
| **Node** | Component instance in graph | `core/execution_graph.py` |
| **Edge** | `edges` mapping in `ExecutionGraph` | `core/execution_graph.py` |
| **Actor** | `ActorWorker` | `executors/workers/actor_worker.py` |
| **Source** | `initial_keys` / `batch_iterator` | `core/blackboard_engine.py` |
| **Transform** | Component `execute()` logic | `core/component.py` |
| **Objective** | `target_keys` | `core/blackboard_engine.py` |
| **Service** | Stateful resource (e.g. Buffer) | `data/storage/circular.py` |
| **Controller** | `BlackboardEngine` | `core/blackboard_engine.py` |
| **Schema** | `Key` dataclass | `core/contracts.py` |
| **DataRef** | `Key.path` (dotted string) | `core/contracts.py` |
| **ExecutionPlan** | `ExecutionGraph.execution_order` | `core/execution_graph.py` |
| **ScalarValue** | `int`, `float`, `bool` primitives | `core/blackboard.py` |
| **StreamHandle** | `Iterable[Dict[str, Any]]` | `core/blackboard_engine.py` |
| **ResourceHandle**| Injected external objects | `registries/*.py` |

## Node Lifecycle

| Node Category | Initialization | Inputs | Outputs | State | Failure | Reset |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Component** | `__init__` with static config and resource handles. | Read-only Blackboard view via `requires`. | Dictionary of path-based updates to the Blackboard. | Stateless (Math) or Static (Network Weights). | Contract violation or runtime logic error (NaN). | None; intended as deterministic transforms. |
| **Service** (Replay) | Once-per-session instantiation with shared memory. | Method calls (`insert`, `sample`) carrying payloads. | Sampled data batches or update acknowledgments. | Yes; persistent memory across multiple steps. | Memory exhaustion (OOM) or empty-state sampling. | `clear()` method to purge all stored state. |
| **Actor** (Worker) | Orchestrated worker process with a local Engine. | `StreamHandle` (Env) and `ResourceHandle` (Weights). | Completed trajectories and episode-level telemetry. | Yes; encapsulates environment and local engine state. | Environment crash or invalid model outputs. | Re-run `play_sequence` loop with new iterator. |
| **Source** (Env) | Wrapped within an Observation component. | External hardware signals or simulation triggers. | Raw facts (obs, rewards) for the initial Blackboard. | Yes; represents the external physical/simulated state. | Simulation error, timeout, or illegal action. | `env.reset()` at episode or task boundaries. |

## Graph Edge Classes

The following classes define the allowed types of edges that connect nodes within the execution graph:

Data:
An edge representing the flow of a Tensor or ScalarValue payload from a provider to a consumer via a standardized Blackboard path.

Control:
An edge used to signal execution branching or termination conditions, typically mapped to flags within the `meta` domain.

Resource:
An edge connecting a compute node to a stateful Service, allowing the component to perform side-effecting operations like sampling or updates.

Dependency:
A structural link that enforces a specific execution order between nodes, ensuring that prerequisites are satisfied before a node runs.
