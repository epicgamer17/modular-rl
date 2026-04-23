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