# RL IR Semantic Kernel (Minimal Final Spec)

This document defines the minimal, irreducible intermediate representation (IR) for reinforcement learning systems. It treats RL as a special case of a general executable graph IR, prioritizing semantics over implementation details.

```mermaid
graph TD
    subgraph "Fundamental IR Primitives"
        Node["Node (Executable Unit)"]
        DataRef["DataRef (Immutable Typed Value)"]
        Edge["Edge (Dependency)"]
        Graph["Graph (Static DAG)"]
    end

    subgraph "Node Specializations"
        ActorNode["ActorNode (Action Producer)"]
        SourceNode["SourceNode (Input Producer)"]
        TransformNode["TransformNode (Pure Mapping)"]
    end

    subgraph "Execution Layer"
        ExecutionContext["ExecutionContext (Runtime Descriptor)"]
        ExecutionPlan["ExecutionPlan (Compiled Structure)"]
    end

    Node -->|Processes| DataRef
    Node -->|Subject to| Effect["Effect (Constraint)"]
    Graph -->|Composed of| Node
    Graph -->|Connected by| Edge
    Edge -->|Flows| DataRef
    ActorNode --|> Node
    SourceNode --|> Node
    TransformNode --|> Node
    ExecutionPlan -.->|Derived from| Graph
    Node -.->|Reads| ExecutionContext
```

---

## 1. Fundamental IR Primitives
These are the only irreducible constructs. Everything else is derived.

### 1.1 Node (Abstract)
The minimal executable unit.
- **Properties:**
    - Typed inputs / outputs (**DataRefs**)
    - Associated **Effect** class
    - Execution function defined by implementation, not IR
- **Constraint:** No semantics beyond execution boundary.

### 1.2 ActorNode
Specialized Node producing actions.
- **Signature:** `f(inputs, policy, state, rng_seed) → Action DataRef`
- **Constraints:**
    - Stochasticity only via `rng_seed` or `policy`
    - Must emit `Action-schema` DataRef
    - May be stateless or stateful (explicit flag)

### 1.3 SourceNode
External input producer.
- **Properties:**
    - No dependencies
    - Emits DataRefs from environment or system boundary

### 1.4 TransformNode
Pure mapping Node.
- **Properties:**
    - Deterministic function over inputs
    - No side effects
    - Fully cacheable under version equivalence

### 1.5 Edge
Directed dependency relation between Nodes via DataRefs.
- **Defines:**
    - Execution ordering constraints
    - Data flow dependencies

### 1.6 Graph
Static DAG (or controlled cyclic extension in learning graphs).
- **Composed of:** Nodes and Edges.
- **Constraint:** No runtime semantics.

### 1.7 DataRef
Immutable typed value.
- **Metadata:**
    - `version_id`
    - Provenance (Node lineage)
    - Ownership scope
    - Locality hint
    - Determinism class

> [!IMPORTANT]
> **Core Rule:** DataRef identity is **version + lineage**, not value alone.

### 1.8 ExecutionPlan (Optional Fundamental Layer)
Compiled execution structure derived from Graph.
- **Contains:**
    - Execution ordering
    - Batching groups
    - Granularity schedule

### 1.9 ExecutionContext (Optional Fundamental Layer)
Runtime-scope execution descriptor.
- **Fields:**
    - `step_index`
    - `rng_state`
    - `version_clock`
    - `execution_shard_id`
    - `batching_group_id`

> [!IMPORTANT]
> **Rule:** ExecutionContext is immutable per execution slice.

---

## 2. Derived Constructs (NOT Fundamental)
These are expressible entirely in terms of Nodes + Graph + DataRefs + Effects.

| Construct | Derivation Basis |
| :--- | :--- |
| **Actor Behavior Systems** | ActorNode + DataRef + ExecutionContext (policies, action sampling) |
| **Learning Graphs** | TransformNode compositions over DataRefs (gradients, updates, optimizers) |
| **Optimization System** | Pure transformation over Graph/ExecutionPlan (fusion, pruning, caching) |
| **Objective Graphs** | TransformGraphs producing scalar/vector DataRefs (rewards, losses, entropy) |
| **Credit Assignment** | DataRef lineage traversal + TransformNodes (advantage, TD errors) |
| **Replay Systems** | SourceNodes + TransformGraphs (buffers, sampling, prioritization) |
| **RL Algorithms** | Graph compositions + ExecutionPlan schedules + Learning graphs |

---

## 3. Explicitly Outside the Kernel
These are intentionally excluded from semantic scope.

- **Environment Internals:** Physics engines, simulators, game rules (Accessed via SourceNode / External Effect boundary).
- **Optimization Heuristics:** Batching heuristics, compiler passes, hardware-specific fusion strategies (Belong to compiler/runtime layer).
- **Distributed Execution:** Networking, sharding, fault tolerance, cluster scheduling (Not part of kernel semantics).
- **RL Algorithm-Specific Structures:** Replay buffer implementation forms, target networks, epsilon schedules, exploration heuristics (Must be representable, but not fundamental).

---

## 4. Effect System (Core Constraint Layer)

Effects are constraints on execution, not behaviors.

| Effect Class | Constraints |
| :--- | :--- |
| **Pure** | Deterministic; no side effects |
| **Stochastic** | RNG-dependent; must use `ExecutionContext.rng_state` |
| **External** | Environment/system boundary interaction |
| **StateMutating** | Modifies Node-local state only |

---

## 5. Execution Rules (Global Invariants)

### 5.1 Determinism
Execution is deterministic iff:
- Same Graph
- Same DataRef versions
- Same ExecutionContext (including `rng_state`)

### 5.2 Effect Ordering
Ordering constraints for execution:
1. **Pure Nodes:** Freely parallel.
2. **Stochastic Nodes:** RNG-grouped.
3. **Stateful Nodes:** Serialized per shard.
4. **External Nodes:** Interaction boundary sync.

### 5.3 Execution Atomicity
- Node execution is atomic.
- Graph pass execution is atomic per ExecutionPlan step.
- Partial commits are forbidden.

### 5.4 Version Consistency
- DataRef versions define all equivalence.
- No execution may mix mismatched versions in the same context.

### 5.5 Reproducibility Guarantee
A full run is reproducible iff:
- Graph identical
- ExecutionPlan identical
- ExecutionContext lineage identical
- RNG derivation identical
- DataRef version graph identical

---

## 6. Definitions

### What This Kernel IS
- A minimal execution IR
- Independent of RL algorithms
- Independent of optimization strategy
- Independent of distributed systems
- Capable of expressing all RL systems as derived graphs

### What This Kernel IS NOT
- A learning framework
- A distributed runtime
- An RL algorithm library
- A compiler implementation
- A simulation environment

---

## 7. Core Invariants Summary

> [!TIP]
> - **Everything executable** is a Node.
> - **Everything meaningful** is a DataRef.
> - **Everything causal** is a Graph.
> - **Everything runtime-dependent** is an ExecutionContext.
> - **Everything behavioral** is an Effect constraint.
> - **Everything else** is derived.

---

## 8. Why This Matters
This reduction achieves:
1. **Strict separation** of semantics vs. implementation.
2. **Elimination** of RL-specific primitives.
3. **Uniform representation** of all learning systems.
4. **Composability** across all algorithm families.
5. **Portability** across execution substrates.

> [!NOTE]
> **In effect:** RL becomes a special case of a general executable graph IR, not a collection of algorithms.
