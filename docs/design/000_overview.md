# Project Design Overview

## Purpose
A system for building reinforcement learning algorithms as a modular component library and compiled execution model, where complete learning systems are assembled from composable primitives rather than monolithic end-to-end implementations.

It enables RL systems where:

- Algorithms (e.g. DQN, PPO, SAC, MuZero, NFSP, DAgger) are constructed by composing shared components rather than rewriting full pipelines
- Hybrid methods (e.g. PPO + search, MuZero + DAgger, SAC + demonstrations, search-guided PPO) emerge by recombining existing components without redefining global structure
- Custom components can be introduced locally and integrated into existing systems with minimal friction
- Decision-making systems (policy acting, search, scripted control, mixtures, hierarchical control, human actions, model-based planning) are first-class composable units
- Data collection is expressed as structured interaction pipelines rather than assumed fixed rollout loops

The system represents RL computation as a validated execution graph, where structure, temporal semantics, data provenance, and correctness are explicit properties of the system:

- All transformations (interaction, replay, planning, optimization, sampling, target generation, scheduling) are governed by explicit schema, shape, and semantic contracts
- Composition is constrained through defined interfaces (requires/provides relationships, capability constraints, effect constraints), preventing invalid algorithmic wiring
- A structured type system (e.g. Box, Discrete, Continuous, Trajectory[T], Batch[T], Actor[I,O]) supports propagation of shape and compatibility constraints across the graph
- Temporal semantics (ordered sequences, episodic boundaries, truncation vs termination, freshness, causality) are represented explicitly rather than hidden in code

It provides a unified execution model for RL systems that is independent of backend or hardware constraints while preserving correctness, inspectability, and performance.

It supports a more debuggable form of RL research, where behavior can be analyzed at the level of components, streams, and subgraphs, and where full algorithm structure is explicit rather than embedded across training scripts.

Ultimately, the system is designed to maximize Time-to-Science. It bridges the historical gap between rapid research prototyping and hyper-optimized distributed execution. By acting as the "glue rather than the engine," researchers can invent and debug custom mathematical operations natively in PyTorch, and then instantly scale to DeepMind-level cluster speeds (using C++ backends like Reverb, EnvPool, or MCTX) via a single line of configuration—without ever rewriting their algorithm's logic.

## Design Philosophy
- **Small Decision Groups**: Minimize friction by keeping decision-making tight and focused.
- **Clear Focus**: Maintain a sharp focus on the core value proposition: a typed, compiled RL systems graph.
- **Doing Less, Better**: Prioritize a strong reusable core over a wide but shallow feature surface.
- **Simplicity Over Complexity**: Keep abstractions minimal, orthogonal, and composable.
- **Defensive Engineering**: Validate assumptions early; distrust implicit state and hidden coupling.
- **Structural Before Fancy**: Prioritize runtime, contracts, scheduling, and graph semantics before advanced algorithms.
- **Spinning Straw into Gold**: Build primitives that unlock many systems rather than bespoke implementations.
- **Research-Grade Hackability**: Preserve the ability to express new ideas without fighting the framework.

## Core Goals
- Enable RL systems to be expressed as compiled, validated execution graphs with explicit data flow, control flow, and execution order
- Treat acting, search, planning, replay, optimization, and scheduling as first-class composable graph components
- Ensure all transformations (environment steps, sampling, updates, losses, planning passes) are validated through strict schema, shape, and semantic contracts
- Unify execution scheduling, stateful services, and data pipelines into a single runtime model
- Support transition, trajectory, sequence, and tree-structured data representations
- Support both sequence-based systems (e.g. MuZero, recurrent PPO) and transition-based systems (e.g. DQN, SAC)
- Allow distributed execution (multi-process / multi-node) without violating determinism or contract correctness
- Provide compile-time and runtime validation, scheduling, and optimization across execution backends
- Support static and dynamic execution planning, including dependency pruning, partial execution, and graph-level optimization
- Optimize execution via batching, parallelism, caching, prefetching, and target-driven graph execution
- Eliminate cross-cutting duplication by centralizing reusable mechanisms (e.g. exploration, PER, search wrappers, sync logic) into single-ownership components
- Enforce algorithmic integrity through capability and effect constraints (e.g. preventing invalid on-policy/off-policy mixing, preventing shuffled recurrent batches, preventing stale-policy misuse)
- Provide a rich semantic type system (Box, Discrete, Continuous, Trajectory[T], Sequence[T], Tree[T], Actor[I,O]) with automatic shape propagation and compatibility checking
- Enable modular reuse across RL paradigms (online RL, offline RL, imitation learning, DAgger, self-play, model-based RL, hybrid systems)
- Preserve rapid experimentation by allowing local custom components that integrate cleanly into the graph model
- **Declarative Data Access**: Treat Replay and data fetching as Declarative Query Nodes (IR) rather than imperative Python objects, allowing the compiler to optimize data routing, pre-fetching, and backend delegation.
- **The Adapter/Provider Pattern**: Delegate physical execution of performance-critical bottlenecks (Search, Environments, Distributed Replay) to world-class external engines (e.g., MCTX, EnvPool, Reverb, TorchRL) without polluting the pure mathematical IR.
- **Component-Level Fusion**: Use pattern matching in the compiler to fuse standard subgraphs (e.g., standard UCT search) into highly optimized backend calls, while explicitly avoiding brittle "whole-algorithm" fusion.
- **Graceful Degradation (Native Fallback)**: Ensure that when researchers build custom, non-standard nodes that external backends do not support, the compiler seamlessly falls back to a pure PyTorch execution path (leveraging torch.compile for automatic kernel generation) to guarantee custom research always runs.
- **Zero-Config Baselines (Graph Factories)**: Prevent "configuration fatigue" by providing standard baseline recipes (e.g., standard PPO or SAC graphs) out of the box, allowing users to modify a single node without having to wire the entire graph themselves.



## Non-Goals
- No hard dependency on a single RL algorithm family (e.g. MuZero-only or PPO-only design)
- No assumption that data collection is a fixed rollout(policy, env) loop
- No assumption that replay is a single monolithic buffer abstraction
- No tight coupling between environment APIs and learning code
- No hidden mutation across pipeline boundaries; stateful effects must be explicit components
- No reliance on Python object graphs in storage or performance-critical paths
- No requirement for a custom standalone language syntax in early versions
- No black-box trainer APIs that obscure system structure
- **No reimplementing hyper-optimized low-level C++ structures**: We will not write custom C++ segment trees, MCTS parallelization loops, or gRPC communication layers. We wrap world-class libraries instead of competing with them.
- **No Whole-Algorithm Black-Box Fusion**: The compiler will never look at a graph and replace it with an external Acme.PPO or RLlib.SAC call. **Fusion** strictly happens at the component level to preserve the user's ability to tweak the surrounding algorithm.
- **No algorithm-awareness inside core operators**: Mathematical operators (ops/) will never know if they belong to PPO, DQN, or SAC. They remain purely functional Lego bricks.

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