# Project Design Overview

## Purpose
A system for defining reinforcement learning algorithms as composable, validated building blocks, where entire training pipelines are represented as typed computation graphs.

It enables RL research where:

complex algorithms can be assembled from interchangeable parts
correctness is enforced through explicit data and shape contracts
experimentation is safe, modular, and structurally consistent across implementations

## Design Philosophy
- **Small Decision Groups**: Minimize friction by keeping decision-making tight and focused.
- **Clear Focus**: Maintain a sharp focus on the core value proposition: a typed, compiled RL graph.
- **Doing Less, Better**: Prioritize high-quality, robust core implementation over a vast but fragile feature set.
- **Simplicity Over Complexity**: Keep abstractions simple and extensible; value readability and maintainability.
- **Defensive Engineering**: Test early, test incrementally, and never trust implicit state.
- **Structural Before Fancy**: Prioritize robust architectural foundations (pipelines, contracts) before adding complex algorithmic features.
- **Spinning Straw into Gold**: Knowing what to build first; focus on critical path items that enable the rest of the system.

## Core Goals
- Enable RL systems to be expressed as compiled, validated computation graphs where data flow and execution order are explicitly defined.
- Ensure all transformations—from environment steps to gradient updates—are validated through strict schema, shape, and semantic contracts.
- Unify execution scheduling (triggers), stateful resources (service nodes), and data pipelines into a single programmable execution environment.
- Support both sequence-based (MuZero-style) and transition-based (DQN-style) storage and sampling.
- Allow distributed execution (multi-process, potentially multi-node) without breaking determinism or contract validity.
- Provide a compile-time and runtime execution system that can optimize, validate, and schedule RL computation graphs across different execution backends.
- Support static and dynamic execution planning, including dependency pruning, partial execution, and graph-level optimization based on target outputs.
- Support advanced execution optimizations, including parallel running of independent components, component-level caching, and target-based graph pruning.
- Eliminate "feature bleed" by consolidating cross-cutting mechanics (PER, Epsilon-Greedy exploration) into single-owner components.
- Enforce algorithmic integrity through explicit capability and effect systems (e.g., preventing on-policy algorithms from running on stale data).
- Provide a rich, hierarchical semantic type system (Box, Discrete, Continuous, Trajectory[T]) with automated shape propagation and type-matching bridging.
- Enable high modularity through federated component assembly, allowing for easy transitions between imitation learning, DAgger, and standard RL.

## Non-Goals
- No hard dependency on a single RL algorithm family (e.g., MuZero-only or PPO-only design).
- No assumption that replay must be implemented as a single monolithic buffer abstraction.
- No tight coupling between environment APIs and learning code.
- No implicit state mutation across pipeline boundaries (all transformations must be explicit components).
- No reliance on Python object graphs (dataclasses like Sequence/Transition) in storage or training paths.

## Core Concepts
- **Contract (Semantic + Shape)**: Strict tensor definitions ensuring axial and semantic consistency across the system. (ADR-0015, ADR-0019)
- **Execution Graph (DAG)**: A dependency-resolved graph of nodes that ensures only required computation is executed for a given target. (ADR-0001, ADR-0003)
- **Blackboard Storage**: A tensor-only state store where all transition data and intermediate results are managed via semantic keys. (ADR-0001, ADR-0018)
- **Pipeline Components / Transforms**: Stateless, composable transformations that operate on tensors within a contract boundary. (ADR-0007, ADR-0013)
- **Executors**: Specialized runtime engines (Sequential, Workspace) that manage the dispatching and execution of the computation graph. (ADR-0030)

## Current Architecture Summary
The system is structured around the interaction between actors, learners, and a shared blackboard:

1. **Actors produce trajectories** driven by environment interaction and production contracts.
2. **Replay storage (Blackboard)** acts as a schema boundary and a persistent contract between actors and learners.
3. **Learners execute a DAG** of components to reconstruct training views and compute losses, resolving only required dependencies.
4. **Executors** manage the execution of the graph against the current workspace or blackboard state.

## Key Design Principles
- **Contract-first Execution**
  - No component operates without an explicit, validated schema contract.
- Tensor-only storage boundary
  - Python objects never enter replay storage or sampling layers.
- Symmetric actor/learner pipeline
  - Both sides operate on the same underlying contract but different projections.
- Temporal explicitness
  - Every stored fact is indexed by (episode_id, step_id) with explicit done semantics.
- Composability over inheritance
  - All transformations are pipeline components, not subclass hierarchies.
- Deterministic replay reconstruction
  - Same episode + same contract version → identical training tensors.
- **Defensive Design**: Test early, test incrementally, and never trust implicit state; enforce contracts and use asserts with descriptive messages.
- **Structural Before Fancy**: Prioritize robust architectural foundations (pipelines, contracts) before adding complex algorithmic features.
## Current Modules
- core/
  - Execution graph, contracts, and shape validation enforcing DAG correctness.
- modules/
  - Pure neural network components (backbones, heads, world models).
- registries/
  - Algorithm definitions that assemble components into runnable RL systems.
- components/
  - Losses, targets, selectors, environment wrappers. These form DAG nodes.
- data/
  - Currently contains replay storage, sampling, and preprocessing pipelines (to be refactored into contract-aligned storage + pipeline boundary).
- search/
  - MCTS and rollout engines for planning-based algorithms.
- tests/
  - Validation of shapes, contracts, and integration correctness.

## Roadmap
- Introduce formal Replay Contract schema (episode_id, step_id, done, semantic tensors)
- Replace Sequence/Transition objects with tensor-only episode writers
- Move n-step, GAE, normalization, masking into components/targets (DAG nodes)
- Replace buffer logic with blackboard tensor store + indexed sampler
- Add contract validation at actor → storage and storage → learner boundaries
- Implement DAG-based batch construction (execution graph resolves required transformations)
- Introduce versioned replay schema for backward compatibility
- **Modular Action Encoders**: Refactor action/chance encoders into pluggable components (Spatial, EfficientZero, Identity).
- **Evaluator Matrix**: Implement multi-agent matrix evaluation for testers to calculate relative win-rates across student and test agents.
- **Pipeline Compilation**: Transform declarative graphs into optimized execution plans with parallel node execution and C-level compilation paths.
- **Global & Temporal Storage**: Introduce Global (shared storage) and Temporal (history tracking) blackboard domains.
- **Standardized Key Strategy**: Enforce a unified naming convention (e.g., `data.observations`) across all actor, learner, and replay domains.
- **Declarative DSL**: Introduce a high-level Python DSL for expressive graph definitions.
- **Integrated AMP**: Provide automated mixed-precision support as a first-class component wrapper.
- **Windowing & Triggers**: Explore Apache Beam-inspired streaming primitives (windowing, triggers) for complex replay and training schedules.

## Algorithmic Expansion Goals
- **Enhanced Policy Objectives**: Add support for TRPO, Vanilla Policy Gradient (PG), V-trace, AWR, V-MPO, and Phasic Policy Gradient (PPG).
- **MuZero Advancements**: Implement Stochastic MuZero features, Gumbel Policy Improvement, and improved Chance Encoders.
- **Robust Architectures**: Formalize support for ResNet (pre/post activation), GNN, GRU, and Transformer-based backbones.
- **Offline & Imitation**: Native support for Offline World Models, Reanalyzation pipelines, and imitation learning (DAgger/BC) as first-class citizens.