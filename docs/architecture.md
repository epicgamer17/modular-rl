# Architecture

This document describes how the pieces of the RL IR Semantic Kernel fit together. It covers module boundaries, data flow, and dependency rules.

---

## Module Boundaries

The system is organized into three primary layers:

```
┌─────────────────────────────────────────────────────────────┐
│                      Applications                            │
│         (DQN, PPO, NFSP, DAgger examples)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Compiler Layer                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │ compiler.py │ │ analyzer.py │ │ optimizer.py            ││
│  │             │ │             │ │                         ││
│  │ - Validation│ │ - Analysis  │ │ - Fusion               ││
│  │ - Passes    │ │ - Issues    │ │ - Pruning              ││
│  └─────────────┘ └─────────────┘ └─────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Core IR Layer                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │ graph.py    │ │ schema.py   │ │ nodes.py                ││
│  │             │ │             │ │                         ││
│  │ - Graph     │ │ - Schema    │ │ - NodeDef              ││
│  │ - Node      │ │ - TensorSpec│ │ - NodeInstance         ││
│  │ - Edge      │ │ - Trajectory│ │ - Registry             ││
│  └─────────────┘ └─────────────┘ └─────────────────────────┘│
│  ┌─────────────┐                                             │
│  │ inspect.py  │  Introspection tools                        │
│  └─────────────┘                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Runtime Layer                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │ executor.py │ │ context.py  │ │ runtime.py              ││
│  │             │ │             │ │                         ││
│  │ - Execution │ │ - ExecCtx   │ │ - ActorRuntime         ││
│  │ - TopoSort  │ │ - Snapshots │ │ - LearnerRuntime       ││
│  └─────────────┘ └─────────────┘ └─────────────────────────┘│
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │ state.py    │ │ dataref.py  │ │ values.py               ││
│  │             │ │             │ │                         ││
│  │ - ReplayBuf │ │ - DataRef   │ │ - RuntimeValue         ││
│  │ - ParamStore│ │ - BufferRef │ │ - Value/Skipped        ││
│  │ - Registry  │ │             │ │ - NoOp/MissingInput    ││
│  └─────────────┘ └─────────────┘ └─────────────────────────┘│
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │ scheduler.py│ │ collator.py │ │ operators/              ││
│  │             │ │             │ │                         ││
│  │ - Schedule  │ │ - ReplayCol │ │ - exploration          ││
│  │ - Executor  │ │             │ │ - target_sync          ││
│  └─────────────┘ └─────────────┘ │ - metrics              ││
│                                   │ - schedule             ││
│                                   │ - transfer             ││
│                                   └─────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Core (`core/`)

The immutable IR layer containing only data structures:

| Module | Responsibility |
|--------|----------------|
| `graph.py` | Graph, Node, Edge definitions; topological sort |
| `schema.py` | Schema, TensorSpec, TrajectorySpec for type safety |
| `nodes.py` | NodeDef, NodeInstance, Registry for operator definitions |
| `inspect.py` | Introspection tools for debugging |

**Constraint**: Core has zero runtime dependencies. No execution, no state, no side effects.

### Runtime (`runtime/`)

The mutable execution layer:

| Module | Responsibility |
|--------|----------------|
| `executor.py` | Graph execution, operator dispatch, topological sort |
| `context.py` | ExecutionContext, ActorSnapshot, clocks, RNG |
| `runtime.py` | ActorRuntime, LearnerRuntime |
| `state.py` | ReplayBuffer, ParameterStore, ModelRegistry, BufferRegistry |
| `values.py` | RuntimeValue system (Value, NoOp, Skipped, MissingInput) |
| `dataref.py` | DataRef, BufferRef, StreamRef with location tracking |
| `scheduler.py` | SchedulePlan, ScheduleExecutor |
| `collator.py` | Schema-driven batch collation |
| `operators/` | Built-in operators (exploration, target_sync, metrics, schedule, transfer) |

**Constraint**: Runtime may import Core but never modifies graph structure at runtime.

### Compiler (`compiler/`)

The analysis and transformation layer:

| Module | Responsibility |
|--------|----------------|
| `compiler.py` | Orchestrates validation passes |
| `analyzer.py` | Static analysis for issues |
| `optimizer.py` | Graph optimizations (fusion, pruning) |
| `scheduler.py` | Compile schedule from graph topology |
| `passes/` | Individual validation passes |

**Constraint**: Compiler may read but never executes graphs. Output is validated graph or error.

---

## Data Flow

### Build-Time Flow

```
User Code (examples/)
       │
       ▼
   Graph Definition
  (nodes + edges with dst_port)
       │
       ▼
  compile_graph()
       │
       ├─► validate_metadata()
       ├─► infer_shapes()
       ├─► autodiff()           (if autodiff_lowering=True)
       ├─► vectorize_graph()    (if autobatch=True)
       ├─► optimize_graph()     (DNE, fusion)
       ├─► validate_structure()
       ├─► validate_ports()
       ├─► validate_rl_semantics()
       ├─► validate_handles()
       ├─► validate_purity()
       ├─► validate_grad_semantics()
       ├─► validate_ir_purity()
       ├─► collect_trainable_parameters()
       └─► analyze_gradients()
       │
       ▼
  Validated Graph with parameters
```

### Runtime Flow

```
ScheduleExecutor
       │
       ├─────────────────────┐
       ▼                     ▼
  ActorRuntime          LearnerRuntime
       │                     │
       ▼                     ▼
  execute(interact)   execute(train)
       │                     │
       ├─► Topo Sort    ◄───┤
       ├─► For each node:   │
       │    ├─ Gather inputs│ (by dst_port)
       │    ├─ Check skip   │
       │    ├─ Lookup op    │
       │    ├─ Execute      │
       │    └─ Wrap output  │
       │                     │
       ▼                     ▼
  ReplayBuffer ◄───────────
  (shared via registry)
```

### Single Execution Flow

```
execute(graph, initial_inputs, context)
       │
       ▼
┌──────────────────┐
│ Topological Sort │  Kahn's algorithm
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
 ┌─────┐   ┌─────┐
 │Node1│   │Node2│  (parallel if no dep)
 └──┬──┘   └──┬──┘
    │         │
    └────┬────┘
         ▼
  ┌──────────────────────┐
  │ Collect inputs from  │
  │ upstream edges       │
  │ (key = dst_port)     │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ Auto-skip check:    │
  │ if Skipped/Missing  │
  │ input: skip node    │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ Call operator from   │
  │ OPERATOR_REGISTRY    │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ Auto-wrap output in  │
  │ Value() or NoOp()    │
  └──────────┬───────────┘
             │
             ▼
  node_outputs[node_id] = output
```

### Context Flow

```
ExecutionContext
       │
       ├── step_id           (global timestep)
       ├── global_step       (accumulated across runtimes)
       ├── env_step          (environment interactions)
       ├── learner_step      (training updates)
       ├── sync_step         (target network syncs)
       ├── actor_step        (rollout steps)
       ├── episode_step      (steps in current episode)
       ├── episode_count     (total episodes)
       │
       ├── seed              (base RNG seed)
       ├── shard_id          (parallel worker ID)
       ├── rng               (random.Random, seed + shard_id*1000000)
       │
       ├── model_registry    (handles → nn.Module)
       ├── buffer_registry   (handles → ReplayBuffer)
       │
       ├── actor_snapshots   (frozen params+buffers per actor)
       └── sync_state        (last sync timestamps)
```

---

## Dependency Rules

### Layer Dependencies

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│   Examples     │ ──► │   Compiler     │ ──► │    Core        │
│ (Application)  │     │ (Analysis)     │     │   (IR only)    │
└────────────────┘     └───────┬────────┘     └────────────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │     Runtime        │
                    │   (Execution)      │
                    └────────────────────┘
```

**Rule 1**: Dependencies flow downward only. Examples → Compiler → Core ← Runtime.

**Rule 2**: Core has zero imports from Runtime or Compiler.

### Import Rules

```python
# core/graph.py - ONLY standard library + core
from dataclasses import dataclass, field
from typing import Dict, List, Any, NewType, Set, Optional
from core.schema import Schema  # Core can import Core

# runtime/executor.py - Core + Runtime
from core.graph import Graph, NodeId, Node  # Can import Core
from runtime.context import ExecutionContext  # Can import Runtime

# compiler/compiler.py - Core + Compiler
from core.graph import Graph  # Can import Core
from compiler.validation import ValidationReport  # Can import Compiler

# CANNOT DO:
# - core/graph.py imports runtime/executor.py  (violation)
# - compiler/compiler.py imports runtime/state.py  (violation)
```

### State Mutation Rules

**Rule 3**: Graphs are immutable after construction.

```python
# ILLEGAL - modifying graph after creation
graph.nodes["foo"] = new_node  # Don't do this
```

**Rule 4**: ExecutionContext is immutable per slice, derive new for changes.

```python
# CORRECT - derive new context
child = context.derive(step_id=context.step_id + 1)
# Original context unchanged
```

**Rule 5**: ParameterStore updates are explicit.

```python
# CORRECT - explicit update
param_store.update_state(new_params)  # Increments version
```

### Named Port Rules

**Rule 6**: Edges must use explicit `dst_port` for unambiguous data flow.

```python
# CORRECT - explicit port
graph.add_edge("sampler", "loss", dst_port="batch")

# Avoid - implicit positional (allowed for backward compat)
graph.add_edge("sampler", "loss")
```

### Runtime Value Rules

**Rule 7**: Use RuntimeValue wrappers instead of None for control flow.

```python
# CORRECT - explicit skip
if buffer_too_small:
    return Skipped("buffer_size_under_min")

# CORRECT - explicit missing input
if "batch" not in inputs:
    return MissingInput("batch")

# Avoid - None as signal
if not batch:
    return None  # Don't do this
```

---

## Execution Boundaries

### Actor / Learner Separation

```
┌─────────────────────┐     ┌─────────────────────┐
│    ActorRuntime     │     │   LearnerRuntime    │
├─────────────────────┤     ├─────────────────────┤
│ interact_graph      │     │ train_graph         │
│ - SourceNode (obs)  │     │ - ReplayQuery       │
│ - QValuesSingle     │────►│ - QValuesBatch      │
│ - Exploration       │     │ - TDLoss            │
│ - (no persistence)  │     │ - Optimizer         │
│                     │     │ - TargetSync        │
└─────────┬───────────┘     └──────────┬──────────┘
          │                             │
          │     ┌─────────────────┐    │
          └────►│  ReplayBuffer   │◄───┘
                │  (shared via    │
                │   BufferRegistry)│
                └─────────────────┘
```

**Rule 8**: Actors write to shared ReplayBuffer; Learners read from it.

### Snapshot Isolation

```
ParameterStore (version=5)
        │
        ├─────────────────────┐
        ▼                     ▼
  ActorRuntime(1)      ActorRuntime(2)  (parallel shards)
        │                     │
        ▼                     ▼
  ActorSnapshot(v5)   ActorSnapshot(v5)
  (frozen params      (frozen params
   + buffers)          + buffers)
        │                     │
        ▼                     ▼
  Rollout continues    Rollout continues
  even if PS updates   even if PS updates
  to version 6         to version 6
```

**Rule 9**: Each ActorRuntime binds to a snapshot at step start. Background ParameterStore updates cannot affect in-flight rollouts.

---

## Validation Pipeline

The compiler runs these passes in order:

| Pass | Checks | Failures |
|------|--------|----------|
| `validate_metadata` | Node types have specs defined | Hard error |
| `infer_shapes` | Propagate TensorSpec through graph | Warning |
| `optimize_graph` | Remove dead code, fuse nodes | N/A |
| `validate_structure` | No cycles, all nodes reachable | Hard error |
| `validate_ports` | Schema compatibility at edges | Hard error |
| `validate_rl_semantics` | On-policy tags, sync rules | Hard error |
| `validate_handles` | Model/buffer handles exist | Hard error |
| `validate_purity` | No illegal side effects | Hard error |

**Rule 10**: Compilation fails on first hard error. Warnings are collectible but non-fatal (unless strict mode).

---

## Key Invariants

1. **Determinism**: Same Graph + same DataRef versions + same ExecutionContext (including seed/shard_id) → identical execution
2. **Immutability**: Graph structure never changes after construction
3. **Version Causality**: Parameter versions form a monotonic counter
4. **Effect Isolation**: External effects never fused with Pure transforms
5. **Snapshot Immutability**: ActorSnapshots are deep-copied, never modified
6. **State Encapsulation**: All mutable state lives in Runtime layer, accessed via registries
7. **Named Ports**: Data flow is explicit via dst_port, not implicit list ordering
8. **RuntimeValue**: Control flow uses explicit wrappers (Value, NoOp, Skipped, MissingInput), not None
9. **JSON Serializability**: Node params must be primitives (no nn.Module, Optimizer, ReplayBuffer)
10. **Canonical Handles**: Parameters use `model_handle`, `target_handle`, `optimizer_handle`, `buffer_id`

---

## Operator Specifications

### OperatorSpec Structure

```python
@dataclass(frozen=True)
class OperatorSpec:
    name: str
    version: str = "1.0.0"
    
    # IO Ports
    inputs: Dict[str, PortSpec]   # {port_name: PortSpec}
    outputs: Dict[str, PortSpec]
    
    # Execution Properties
    pure: bool                    # No side effects
    stateful: bool                # Maintains internal state
    deterministic: bool           # Same inputs → same outputs
    
    # Dependencies
    requires_models: List[str]    # Model handles needed
    requires_buffers: List[str]   # Buffer handles needed
    requires_optimizer: bool      # Needs optimizer state
    
    allowed_contexts: Set[str]    # {"actor", "learner"}
    
    # Gradient Properties (learner ops)
    differentiable: bool = True
    creates_grad: bool = False
    consumes_grad: bool = False
    updates_params: bool = False
    parameter_handles: List[str] = []
    
    # Meta
    tags: Set[str] = set()
    shape_fn: Optional[Callable] = None
    
    # Cost Model
    estimated_flops: int = 0
    memory_reads: int = 0
    kernel_launch_cost: float = 0.0
```

### PortSpec Structure

```python
@dataclass(frozen=True)
class PortSpec:
    spec: Spec                    # TensorSpec or Schema
    required: bool = True
    variadic: bool = False        # Accepts multiple inputs
    default: Any = None
    description: str = ""
```

---

## Rewrite System

### FusionRule

Fuses adjacent node chains into single operators:

```python
FusionRule(
    name="fuse_a_b",
    pattern=["OpA", "OpB"],      # Chain to match
    replacement="FusedOp"         # Single fused node
)
```

### RewriteEngine

```python
engine = RewriteEngine()
engine.add_rule(FusionRule(...))
optimized = engine.apply(graph)
```

Ensures numerical equivalence between original and fused graphs.

---

## Autobatching

### Vectorization Pass

Transforms single-step graphs to batched:

```
Single: obs [D] → policy → action [A]
Batched: obs [B, D] → policy → action [B, A]
```

- Adds "batched" tag to TensorSpec
- Updates RLType.shape with "B" prefix
- Skips nodes tagged "no_batch"

---

## Validation Codes

### Metadata (M)
| Code | Description |
|------|-------------|
| M001 | Unregistered operator type |

### Purity (P)
| Code | Description |
|------|-------------|
| P001 | nn.Module in node params |
| P002 | Optimizer in node params |
| P003 | ReplayBuffer in node params |
| P004 | Callable/lambda in node params |

### Structure (S)
| Code | Description |
|------|-------------|
| S001 | Cycle detected in graph |
| S002 | Disconnected node |

### RL Semantics (R)
| Code | Description |
|------|-------------|
| R001 | Off-policy data to on-policy node |

### Gradient (G)
| Code | Description |
|------|-------------|
| G001 | Optimizer without preceding Backward node |
| G002 | Backward without optimizer |
| G003 | Same parameter updated twice in one step |
| G004 | Inference graph (actor) updates parameters |
| G005 | Actor graph contains gradient nodes |

### Explainability (E)
| Code | Description |
|------|-------------|
| E204 | Port mismatch - shows connection path and Expected/Got |
| E310 | Missing field in schema |
| E311 | Field type/dtype mismatch |

---

## Gradient System

### Autodiff Lowering

The compiler can automatically insert gradient nodes:

```
Original: Loss → Optimizer
Transformed: Loss → Backward → GradBuffer → Optimizer
```

Enabled via `compile_graph(..., autodiff_lowering=True)`.

### Gradient Analysis

`analyze_gradients()` produces a report with:

- `params_with_grad`: Parameters receiving gradient flow
- `params_without_grad`: Parameters detached from loss
- `unused_branches`: Differentiable nodes with no consumers
- `warnings`: Specific issues like "Detached gradient flow"

### Gradient Validation

Validates gradient lifecycle safety:

- Backward must precede Optimizer
- No duplicate parameter updates
- Gradient nodes forbidden in actor context
- Parameter updates forbidden in inference

---

## Memory Optimizations

### Activation Checkpointing

Located in `compiler/passes/memory_optimizations.py`:

- `apply_activation_checkpointing(graph)` - reduces memory by recomputing activations
- Recomputes forward pass tensors during backward instead of storing them

---

## Optimization Pipeline

### OptimizationReport

Tracks all optimizations applied to a graph:

```python
report = OptimizationReport()
optimized = optimize_graph(graph, report=report)

for step in report.steps:
    print(f"Rule: {step.rule_name}")
    print(f"Pattern: {step.pattern}")
    print(f"Replacement: {step.replacement}")
    print(f"Removed: {step.removed_nodes}")
```

### Dead Node Elimination

Removes nodes not reachable from any sink node.

### Fusion

Combines adjacent pure/deterministic nodes into single operators while preserving numerical output.

---

## Graph Parameters

Graph objects store parameter metadata after compilation:

```python
graph.parameters = {
    "online_q": ["q_values_node", "loss_node"],
    "target_q": ["sync_node"]
}
```

Used for:

- Partitioning actor vs learner
- Identifying which nodes use which models
- Gradient analysis

---

## Explainable Errors

The validation system provides detailed error messages:

**Port Mismatch (E204)**:
```
q_net.obs <- sampler
Expected: SingleObs[float32, shape=(4,)]
Got:      TransitionBatch[obs: float32, action: int64, ...]
```

**Field Mismatch (E311)**:
```
loss.batch <- sampler
Field 'action' mismatch:
Expected: int64
Got:      float32
```

**Missing Field (E310)**:
```
loss.batch <- sampler
Field 'reward' missing from schema
```

---

## Type System

### RLType Hierarchy

```
RLType (base)
├── TensorType (shape, dtype, tags)
├── DistributionType (dist_type, is_logits)
├── TrajectoryType (length)
├── EpisodeType
├── PolicySnapshotType (version)
├── ReplayBatchType
├── ScalarMetricType
├── RNGKeyType
└── HiddenStateType
```

### Compatibility Rules

- OnPolicy vs OffPolicy: Incompatible
- Logits vs Probs: Incompatible
- Stale snapshot (version < 0): Warning
- Same category + compatible tags: Compatible

---

## Agent Architecture

### DQNAgent

```
DQNAgent
├── q_net: QNetwork
├── target_net: QNetwork  
├── optimizer: Adam
├── opt_state: OptimizerState
├── rb: ReplayBuffer
├── model_registry: ModelRegistry
│   ├── "q_net" → q_net
│   └── "target_q" → target_net
├── buffer_registry: BufferRegistry
│   └── "main" → rb
├── optimizer_registry: OptimizerRegistry
│   └── "main_opt" → opt_state
├── replay_schema: Schema
├── collator: ReplayCollator
├── actor_graph: Graph (interact)
└── learner_graph: Graph (train)
```

### Graph Construction

1. Build graphs using builder functions
2. Inject handles (not objects) into node params
3. Register operators and specs
4. Compile with `compile_graph(autodiff_lowering=True)`
5. Validate handles exist in registries
6. Parameters populated in `graph.parameters`

---

## Module Map

| Module | Contents |
|--------|----------|
| `core/graph.py` | Graph, Node, Edge, NodeId, EdgeType |
| `core/schema.py` | Schema, Field, TensorSpec, tags |
| `core/types.py` | RLType hierarchy |
| `core/nodes.py` | NodeDef, NodeInstance, Registry |
| `runtime/executor.py` | execute(), register_operator(), _topological_sort() |
| `runtime/context.py` | ExecutionContext, ActorSnapshot |
| `runtime/runtime.py` | ActorRuntime, LearnerRuntime |
| `runtime/state.py` | ReplayBuffer, ParameterStore, *Registry |
| `runtime/values.py` | RuntimeValue wrappers (Value, NoOp, Skipped, MissingInput) |
| `runtime/specs.py` | OperatorSpec, PortSpec, register_spec(), get_spec() |
| `runtime/collator.py` | ReplayCollator |
| `runtime/operators/` | Built-in operators (exploration, target_sync, metrics, schedule, transfer) |
| `compiler/compiler.py` | compile_graph() with autodiff, autobatch, optimization |
| `compiler/rewrite.py` | RewriteEngine, FusionRule, find_linear_chain() |
| `compiler/optimizer.py` | optimize_graph(), OptimizationReport, dead_node_elimination() |
| `compiler/analyzer.py` | Static analysis |
| `compiler/passes/` | Validation & transformation passes |
| `compiler/passes/autodiff.py` | autodiff() - inserts Backward/GradBuffer |
| `compiler/passes/analyze_gradients.py` | analyze_gradients() - gradient flow analysis |
| `compiler/passes/collect_trainable_parameters.py` | collect_trainable_parameters() |
| `compiler/passes/memory_optimizations.py` | apply_activation_checkpointing() |
| `agents/dqn/` | DQNAgent, graphs, operators, specs |
| `agents/ppo/` | PPOAgent, graphs, operators, specs |