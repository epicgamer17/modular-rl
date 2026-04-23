# RL IR Semantic Kernel

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

# RL IR Execution & Compilation Model

This section finalizes the execution semantics of the RL IR system for implementation-level correctness under distributed, stochastic, and multi-agent workloads.

It assumes:
- Graph / ExecutionPlan model is already defined
- Effect system is finalized
- DataRef versioning is authoritative

---

## 1. Final Execution Model

The execution system is a **context-driven, multi-granularity graph execution engine embedded inside an Interaction Loop runtime**.

Core pipeline:

Graph → ExecutionPlan → ExecutionContext → Executor → Materialized DataRefs → Interaction Loop advancement

Key invariant:
> All execution is context-bound and version-consistent; no Node executes without a fully defined ExecutionContext.

---

## 2. ExecutionContext (CRITICAL)

### Definition

ExecutionContext is a unified runtime state object carried through Scheduler → Executor → Node execution.

It defines all dynamic execution constraints for a single execution slice.

---

### Fields

#### step_index
- Global Interaction Loop timestep identifier
- Defines temporal ordering across episodes and trajectories

---

#### rng_state
- Deterministic random state for stochastic Nodes
- Derived from:
  - step_index
  - Node ID
  - DataRef version hash
- Ensures reproducibility across distributed execution

---

#### version_clock
- Monotonic logical clock for DataRef versions
- Updated after:
  - Node execution completion
  - Interaction Loop step completion

---

#### execution_shard_id
- Identifies parallel execution partition
- Used for:
  - distributed scheduling
  - state partitioning
  - ActorNode grouping

---

#### batching_group_id
- Identifies vectorized execution group
- Used for:
  - fused ActorNode execution
  - TransformNode batching
  - inference optimization

---

### Propagation Rule

ExecutionContext flows:

Scheduler → Executor → Node invocation

Rules:
- Immutable per execution slice
- Can be forked for parallel shards
- Must remain version-consistent across all Nodes in a batch

---

## 3. ActorNode Binding Semantics (CRITICAL)

ActorNode execution depends on **three binding layers**:

---

### 3.1 Policy Binding

#### When policy DataRefs are resolved

Policy DataRefs are resolved at:

- **ExecutionPlan instantiation time**
  - binds versioned policy graph

- NOT at Graph compilation time
- NOT dynamically inside Node logic

Resolution rule:
> Policy is a runtime-resolved dependency pinned per ExecutionContext step

---

### 3.2 State Binding

State binding depends on NodeStatefulness:

#### Stateless ActorNode
- No state binding required
- Fully determined by inputs + policy + rng_state

#### Stateful ActorNode
- State is bound at:
  - ExecutionContext shard initialization
- Updated only after Node execution completes

State isolation rule:
- State is shard-local
- Cannot leak across execution_shard_id boundaries

---

### 3.3 Binding Time Model

| Component | Binding Time |
|----------|-------------|
| Policy DataRef | Runtime (per step) |
| State (stateful nodes) | Shard initialization + runtime update |
| RNG state | ExecutionContext creation |

---

### 3.4 Reproducibility Guarantee

An ActorNode execution is reproducible iff:
- ExecutionContext is identical (step_index, rng_state, shard_id)
- Policy DataRef version is identical
- Input DataRefs are identical
- State snapshot (if Stateful) is identical

---

### 3.5 Distributed Execution Guarantee

Across distributed shards:
- Each shard receives a deterministic ExecutionContext slice
- RNG streams are partitioned by shard_id
- No shared mutable state exists outside controlled Stateful Nodes

---

## 4. ExecutionGranularity (CRITICAL)

### Definition

ExecutionGranularity defines how many Graph executions occur per Interaction Loop step and how they are structured.

---

### 4.1 Granularity Levels

#### Level 1: Single-pass execution
- One Graph execution per Interaction Loop step
- Used in:
  - DQN
  - standard PPO rollout step

---

#### Level 2: Multi-pass execution
- Multiple sequential Graph executions within a single step
- Used in:
  - SAC (policy + Q updates)
  - PPO (policy + value + advantage recomputation)

---

#### Level 3: Nested execution loops
- Graph execution contains internal sub-loops
- Used in:
  - MCTS (tree expansion + simulation rollouts)
  - NFSP (best response + average policy updates)

---

### 4.2 Formal Representation

ExecutionGranularity is defined as:

- outer_loop_count (Interaction Loop steps)
- inner_graph_pass_count (Graph executions per step)
- optional nested_loop_depth (for recursive algorithms like MCTS)

---

### 4.3 Execution Rule

At each Interaction Loop step:

for i in 1..inner_graph_pass_count:
    execute Graph using ExecutionContext

Then:
- environment step is applied once per outer loop iteration unless explicitly overridden

---

### 4.4 Key Constraint

> Graph execution is always atomic per pass; partial execution is not allowed unless explicitly defined by ExecutionPlan segmentation.

---

## 5. RL Algorithm Validation

### PPO
- ✔ Multi-pass execution required (policy + value + advantage)
- ✔ ExecutionGranularity Level 2
- ✔ Reproducible rollout batches supported

---

### DQN
- ✔ Single-pass execution sufficient
- ✔ replay buffer externalized correctly
- ✔ no binding ambiguity

---

### SAC
- ✔ multi-pass (policy + Q + entropy)
- ✔ stochastic ActorNode fully context-bound

---

### NFSP
- ✔ dual-policy execution requires shard-separated state
- ✔ reservoir sampling stable under Stateful Node rules

---

### MCTS
- ✔ requires nested execution loops
- ✔ ExecutionGranularity Level 3 required
- ✔ stateful tree shards correctly supported

---

### DAgger
- ✔ external expert queries handled via External Effect
- ✔ dataset aggregation consistent across steps

---

No algorithm requires structural reinterpretation.

---

## 6. Remaining Architectural Risks

### 6.1 Distributed Execution Scaling
- ExecutionContext consistency requires strict synchronization of:
  - version_clock
  - rng_state partitioning
- Risk: drift in large-scale async clusters

---

### 6.2 Asynchronous Actor Execution
- Current model assumes synchronous step-based ActorNode execution
- Risk: incompatible with fully async policy serving systems

---

### 6.3 Model-Based RL Extensions
- No explicit environment model inside IR
- Planning-heavy systems (e.g., Dreamer-like architectures) require:
  - repeated internal simulated Interaction Loops
- Risk: recursion explosion without execution budgeting

---

### 6.4 State Explosion in Stateful Nodes
- MCTS and NFSP may generate large internal state shards
- Requires external state compression policy (not defined in IR)

---

### 6.5 Partial Execution / Early Exit Semantics
- Current ExecutionPlan assumes full pass completion
- No formal support for:
  - early termination
  - adaptive compute skipping

---

## 7. Final Statement

The system is now execution-complete under the following constraints:

- deterministic distributed execution is guaranteed via ExecutionContext
- stochasticity is fully seeded and versioned
- multi-pass RL algorithms are supported via ExecutionGranularity
- ActorNode behavior is fully resolved at runtime without ambiguity
- Graph execution is strictly separated from interaction dynamics

Remaining gaps are scalability and asynchronous execution extensions, not core IR correctness.