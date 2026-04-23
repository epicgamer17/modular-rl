# RL IR Semantic Kernel (Final Stable Version)

## 1. IR Structural Layer

### Node (abstract)
Minimal executable unit with typed inputs/outputs and an Effect class.

### ActorNode
Decision-producing Node executing:
f(inputs, policy, state, rng_seed)

### SourceNode
External input Node with no dependencies.

### TransformNode
Pure computational mapping Node.

### Edge
Directed dependency carrying DataRefs.

### Graph
Static computation structure of Nodes and Edges.

### DataRef
Immutable typed value with metadata:
- ownership
- locality
- version
- determinism

### ExecutionPlan
Unified schedule:
- Static Graph Execution Layer
- Temporal Interaction Loop Layer

### NodeStatefulness
- Stateless
- Stateful

---

## 2. Data Schema Layer

State, Action, Reward, Transition, Trajectory, Policy, ValueFunction

(All attached to DataRefs, not executable)

---

## 3. Execution Semantics Layer

### ActorNode Execution Contract
Executes:
f(inputs, policy, state, rng_seed)

- stochasticity via policy or rng_seed
- reproducibility via versioned inputs + fixed seed
- output must match Action schema

---

### ExecutionPlan Composition
Unified hierarchical schedule:
- Interaction Loop contains repeated Graph executions per timestep

---

## 4. Effect System

### Pure
Deterministic, no side effects

### Stochastic
Randomized output via rng_seed

### External
Boundary interaction with environment/system

### StateMutating
Alters Node internal state

---

## 5. Execution Rule

Graph Execution = computation
Interaction Loop = time evolution

---

## 6. Stability Statement

All major RL families (PPO, DQN, SAC, NFSP, MCTS, DAgger) map cleanly without structural extension.

# RL IR Graph Execution & Compilation Model

This section defines how an RL IR Graph is compiled into an ExecutionPlan and executed at runtime, assuming the semantic kernel (Nodes, DataRefs, Effects, ExecutionPlan, Actor execution contract) is already defined.

---

## 1. Graph Execution Model

### 1.1 Scheduling Strategy

Graph execution is based on a **dependency-resolved, effect-aware scheduling system**:

- Nodes are scheduled using **topological ordering over Edges**
- Cycles are allowed only when explicitly broken by:
  - Interaction Loop boundaries
  - Stateful Nodes with delayed dependency resolution
- Scheduling is constrained by:
  - DataRef readiness
  - Effect class ordering constraints

Execution priority:
1. Pure Nodes (TransformNode)
2. Stochastic Nodes (ActorNode sampling)
3. External Effect Nodes (environment interaction)
4. StateMutating Nodes (memory updates)

---

### 1.2 Batching Rules

Batching is defined over **structurally identical Nodes**:

Nodes are batchable if:
- Same Node type
- Same Effect class
- Compatible DataRef schema shapes

Batching behaviors:
- ActorNodes with identical policy structure are batched for inference
- TransformNodes with identical operators are fused into vectorized execution
- SourceNodes are batched only if external source supports joint sampling

Batch boundaries are preserved across:
- Different ExecutionPlan timesteps
- Different Interaction Loop iterations

---

### 1.3 Parallel Execution Model

Graph execution is parallelized over **independent subgraphs**:

- Nodes with no shared unresolved dependencies execute in parallel
- Parallelism is constrained by:
  - DataRef version dependencies
  - StateMutating Node ordering constraints

Execution model:
- Stateless Pure Nodes → fully parallel
- Stochastic Nodes → parallel with synchronized RNG context
- Stateful Nodes → sequential per state shard
- External Effect Nodes → serialized at boundary interface

---

### 1.4 Stateful vs Stateless Handling

#### Stateless Nodes
- Fully parallelizable
- No execution memory retained
- Safe for fusion and caching

#### Stateful Nodes
- Execution is partitioned by **state identity key**
- State is updated only after Node execution completes
- Execution order must respect:
  - intra-node state consistency
  - cross-timestep ordering constraints

Stateful execution implies:
- partial serialization within execution shard
- no cross-batch state mixing unless explicitly allowed

---

## 2. DataRef Materialization Rules

### 2.1 Symbolic vs Materialized Values

DataRefs exist in two forms:

- **Symbolic DataRef**
  - Represents unresolved computation
  - Exists in compiled ExecutionPlan
  - Carries dependency references only

- **Materialized DataRef**
  - Fully computed value
  - Produced during execution
  - Can be cached or reused

Materialization occurs when:
- All upstream dependencies are resolved
- Node execution is triggered in ExecutionPlan

---

### 2.2 Caching and Reuse Semantics

Caching is defined over:
- (Node ID, input DataRef version set, Effect class)

Cache validity conditions:
- All input DataRef versions unchanged
- No intervening StateMutating effect occurred
- Stochastic seed identical (if applicable)

Reuse rules:
- Pure Nodes → always cacheable
- Stochastic Nodes → cacheable only with fixed RNG seed
- Stateful Nodes → cache invalid unless state snapshot identical

---

### 2.3 Versioning Interaction with Execution

DataRef versioning defines execution consistency:

- Each DataRef carries a monotonic version identifier
- ExecutionPlan binds to specific version sets
- Version mismatch invalidates cached computation paths

Version propagation:
- Edges propagate version constraints forward
- Interaction Loop increments global version clock per step

---

## 3. ExecutionPlan Generation

### 3.1 Graph → ExecutionPlan Lowering

Compilation proceeds in two phases:

#### Phase 1: Static Lowering
- Graph is decomposed into dependency DAG layers
- Nodes are grouped into execution tiers
- Effect ordering constraints are embedded

#### Phase 2: Temporal Binding
- Interaction Loop structure is attached
- Stateful Nodes are bound to execution shards
- Stochastic execution contexts are initialized

Result:
- A layered ExecutionPlan with:
  - intra-step Graph execution layers
  - inter-step Interaction Loop cycles

---

### 3.2 Interaction Loop Integration

Interaction Loop defines temporal execution:

Each step:
1. SourceNodes produce initial State DataRefs
2. Graph Execution Plan runs to completion or partial completion
3. ActorNodes produce Actions
4. External environment step occurs
5. Rewards and next States are injected as new DataRefs

Graph execution is **nested inside each loop iteration**.

---

### 3.3 Stochastic Effect Handling

Stochasticity is resolved at ExecutionPlan boundary:

- Each stochastic Node receives:
  - deterministic RNG seed derived from:
    - global step index
    - Node ID
    - DataRef version hash

Rules:
- No implicit randomness allowed outside ExecutionPlan
- RNG context is explicitly passed into ActorNode execution contract
- Stochastic execution is reproducible if seed + inputs are fixed

---

## 4. Runtime Architecture (Conceptual)

Execution system is divided into three logical subsystems:

### 4.1 Scheduler

Responsibilities:
- Converts Graph → ExecutionPlan
- Resolves dependency ordering
- Assigns batching groups
- Enforces Effect constraints

---

### 4.2 Executor

Responsibilities:
- Executes Nodes according to ExecutionPlan
- Materializes DataRefs
- Applies caching logic
- Manages RNG context for stochastic Nodes
- Handles Stateful Node updates

---

### 4.3 Environment Interface Layer

Responsibilities:
- Provides external state transitions
- Supplies reward signals
- Receives Actions from ActorNodes
- Isolated from Graph semantics (black-box boundary)

---

Separation principle:
- Scheduler defines *what to run*
- Executor defines *how to run it*
- Environment defines *what changes externally*

---

## 5. Early Optimization Opportunities

### 5.1 Graph Pruning

- Remove unreachable Nodes based on ExecutionPlan slicing
- Eliminate redundant branches with identical DataRef outputs
- Prune inactive policy branches in ActorNodes

---

### 5.2 Operator Fusion

- Merge sequential TransformNodes into single fused kernels
- Fuse ActorNode pre-processing with policy evaluation where schema-compatible
- Collapse linear dependency chains within Graph layers

---

### 5.3 Batching Across ActorNodes

- Batch ActorNodes sharing:
  - identical policy structure
  - identical Action schema
  - compatible observation shapes

- Cross-episode batching allowed only when:
  - no Stateful dependency exists
  - no stochastic divergence in policy execution

---

## 6. RL Algorithm Compatibility Validation

### PPO
- Rollout batching supported via Interaction Loop batching
- Policy/value fusion supported via operator fusion
- Advantage computation remains pure TransformNode layer

### DQN
- Replay buffer handled via external interface layer
- Q-function fully batchable
- Target network separation preserved via versioning

### SAC
- Stochastic Actor batching supported
- Entropy term handled as pure transform
- Continuous action sampling compatible with stochastic RNG model

### NFSP
- Dual policy ActorNode batching supported independently
- Reservoir sampling handled via Stateful Node rules

### MCTS
- Tree expansion modeled as Stateful execution shards
- Simulation batching supported across rollouts
- Stochastic rollout policy fully reproducible via seed control

### DAgger
- Expert queries treated as External Effect batching
- Dataset aggregation supported via versioned DataRefs

---

## 7. Summary

This execution model defines a deterministic, effect-aware compilation pipeline:

Graph → ExecutionPlan → Batched Parallel Execution → Interaction Loop Cycles → Environment Interaction

It ensures:
- reproducibility via DataRef versioning + RNG control
- scalability via batching and fusion
- correctness via strict effect ordering
- compatibility across all major RL families