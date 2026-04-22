# ADR-0001: Use a Blackboard + Contract-Validated DAG as the Core Execution Model

## Status
Accepted

## Context
The library is intended to support many reinforcement learning algorithms (DQN, Rainbow, PPO, MuZero, future variants) within one modular framework. These algorithms share some primitives (encoders, policy heads, value heads, losses, replay storage, selectors, target builders), but differ significantly in:
- Required data flow
- Update schedules
- Number of forward passes
- Training targets
- Stateful vs stateless components
- Online vs target networks
- Search/planning integrations
- Multi-agent semantics
- Temporal unrolling requirements

Traditional framework designs typically choose one of these approaches:
1. **Monolithic algorithm classes**: Each algorithm owns a custom training loop and manually wires dependencies.
2. **Static pipelines / sequential stages**: Data passes through a fixed ordered list of steps.
3. **Callback/event systems**: Components subscribe to hooks and mutate shared state opportunistically.

These approaches become increasingly difficult to maintain as algorithm variety grows. The project also has additional constraints:
- Strong tensor shape correctness is required.
- Components should be reusable across algorithms.
- Data dependencies should be explicit.
- Execution should be inspectable and testable.
- Errors should happen at graph-build time when possible.
- Side effects (optimizer step, replay writes, target sync) must be explicit.
- New algorithms should be created mostly by composition, not rewriting loops.

## Options Considered

### Option 1: Monolithic Algorithm Classes
Each registry algorithm owns handwritten orchestration logic.
- **Pros**
    - Simple for first few algorithms
    - Easy to optimize per algorithm
    - Familiar design
- **Cons**
    - Logic duplicated across algorithms
    - Hard to reuse subgraphs
    - Hard to verify dependencies
    - Shape contracts become ad hoc
    - Adding variants becomes expensive

### Option 2: Sequential Pipeline Architecture
Run components in a fixed order.
- **Pros**
    - Simple mental model
    - Easy batching
    - Easy debugging for linear workflows
- **Cons**
    - RL workflows are not purely linear
    - Branching dependencies awkward
    - Multi-loss graphs awkward
    - Optional components difficult
    - Search/planning loops unnatural
    - Over-computation likely

### Option 3: Event / Callback Architecture
Hooks such as `on_step`, `on_batch`, `on_update`.
- **Pros**
    - Flexible extension points
    - Easy plugin model
- **Cons**
    - Hidden control flow
    - Implicit dependencies
    - Ordering bugs
    - Difficult validation
    - Difficult reproducibility

### Option 4: Blackboard + Contract-Validated DAG (Chosen)
Components declare:
- required inputs (**requires**)
- outputs (**provides**)
- contracts (shape, semantics, dtype)
- side effects if any

A central blackboard stores facts/tensors. An execution graph resolves dependencies and schedules only necessary components.
- **Pros**
    - Explicit dataflow
    - Reusable components
    - Supports branching graphs
    - Build-time validation possible
    - Strong shape/semantic guarantees
    - Natural fit for actor/learner/search subsystems
    - Enables pruning unused components
    - Easier testing and introspection
- **Cons**
    - Higher implementation complexity
    - Requires robust contract system
    - Users must learn graph model
    - Side effects must be modeled carefully

## Decision
We will use a Blackboard + Contract-Validated DAG as the core architecture. The blackboard is the shared fact store. Components are independent producers/consumers of typed keys. The execution graph computes the minimal valid subgraph required to satisfy requested targets. Contracts validate tensor semantics and shapes before execution. Algorithms become compositions of components rather than handwritten imperative loops.

## Consequences

### Positive
- New algorithms can be assembled from reusable primitives.
- Shared modules (encoders, heads, losses, selectors) become first-class reusable units.
- Shape mismatches and missing dependencies fail early.
- Execution plans are inspectable.
- Actor, learner, replay, and search systems can use the same abstraction.
- Easier to unit test isolated components.
- Enables future graph optimization / compilation.

### Negative / Tradeoffs
- More upfront engineering complexity.
- Requires disciplined side-effect modeling.
- Some workflows need explicit lifecycle phases.
- Debugging graph metadata may replace debugging imperative loops.
- Performance tuning may require graph-aware caching/fusion.

## Notes
This ADR is the foundational decision that motivates later ADRs such as:
- Explicit side-effect components
- Separate actor/learner blackboards
- Replay as a typed boundary
- Shape contracts with semantic dimensions
- Minimal target-driven execution planning