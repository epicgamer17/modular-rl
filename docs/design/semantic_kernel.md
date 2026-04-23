# RL IR Semantic Kernel

## 1. IR Structural Layer

### Node (abstract)
Minimal executable unit in the IR graph with typed inputs and outputs, independent of RL semantics.

---

### ActorNode
Node that produces outputs conditioned on input DataRefs and an attached decision mechanism.
- May be deterministic or stochastic
- May use policies, heuristics, or external sampling
- Does not encode execution semantics internally

---

### SourceNode
Node that introduces external or initial DataRefs with no input dependencies.
- Default state: stateless

---

### TransformNode
Node that maps input DataRefs to output DataRefs via computation.
- May be stateless or stateful (via NodeStatefulness axis)

---

### Edge
Directed dependency carrying DataRefs between Nodes.

---

### Graph
Collection of Nodes and Edges forming a complete computation structure.

---

### DataRef (refined)
Immutable reference to a typed value with metadata:
- Ownership: origin Node or external source
- Locality: local / shared / external
- Version/Freshness: logical generation index

---

### ExecutionPlan (refined)
Compiled representation derived from Graph.

Two distinct forms:
1. Graph Execution Plan
   - Schedules Node evaluation via dependency ordering
2. Interaction Loop Plan
   - Defines temporal RL interaction cycles (step/episode structure)

---

### NodeStatefulness (orthogonal axis)
Defines whether a Node carries persistent state.

- Stateless: output depends only on inputs
- Stateful: output depends on inputs + internal memory

---

## 2. Data Schema Layer (RL semantics as types)

These are NOT nodes. They are typed DataRefs.

### State
System/environment configuration at a time step.

### Action
Decision output produced by ActorNode.

### Reward
Scalar or structured feedback signal.

### Transition
Tuple: (State, Action, Reward, NextState)

### Trajectory
Ordered sequence of Transitions.

### Policy
Mapping from State → Action distribution or function.

### ValueFunction
Mapping from State or State-Action → expected return.

---

## 3. Execution Semantics Layer

### Graph Execution Semantics
- Executes Nodes via ExecutionPlan
- Resolves dependencies through Edges
- Applies NodeStatefulness only at execution time

---

### Interaction Loop Semantics
- Executes RL cycles over time steps
- Per step:
  1. Read State from SourceNodes/environment
  2. Execute ActorNodes → Actions
  3. Environment advances externally
  4. Produce Reward and next State
- Does not modify Graph structure

---

### Scheduling Semantics
- Graph scheduling: dependency resolution over Nodes
- Interaction scheduling: temporal RL loop execution
- Strictly separated concerns

---

## 4. Unification Rule

### Node unification
ActorNode, SourceNode, TransformNode are all specializations of Node:
- SourceNode: no dependencies
- TransformNode: pure computation
- ActorNode: decision-producing computation

### RL concepts are NOT Nodes
State, Action, Reward, Policy, ValueFunction are:
- Data schemas attached to DataRefs
- Never executed as graph primitives
- Never part of scheduling logic

---

## 5. Minimal Change Summary

- ActorNode generalized (deterministic/stochastic/externally driven)
- Introduced NodeStatefulness axis
- Split ExecutionPlan into graph vs interaction semantics
- Extended DataRef with metadata (ownership, locality, freshness)

---

## 6. Remaining Ambiguities

- Environment boundary not formally represented
- Learning/update phase not structurally modeled
- Multi-agent interaction semantics undefined
- Temporal abstraction (options/macro-actions) absent
- Policy update semantics unclear (data vs structure)