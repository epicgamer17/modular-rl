# Architecture Decision Records (ADRs)

This folder contains all major design decisions.

## Format
Each ADR follows a structured template:
- Context
- Options
- Decision
- Consequences

## Index
- ADR-0001: Use a Blackboard + Contract-Validated DAG as the Core Execution Model
- ADR-0002: Use Component-System Modularity Instead of Inheritance-Centered Algorithm Classes
- ADR-0003: Contract-Driven Execution Graph for RL Pipelines
- ADR-0004: Tensor-Only Replay Storage
- ADR-0005: Canonical Replay Metadata (episode_id, step_id, done)
- ADR-0006: Actor–Learner Separation via Shared Replay Contract
- ADR-0007: Pipeline Components as DAG Nodes (ECS-style RL Systems)
- ADR-0008: Removal of Lazy Decompression from Replay Path
- ADR-0009: Removal of Sequence/Transition Objects from Replay Path
- ADR-0010: Explicit Temporal Alignment Contract (No Implicit Broadcasting)
- ADR-0011: Sampler Must Return Indices Only
- ADR-0012: Placement of Normalization (Output Processor, Not DAG Component)
- ADR-0013: Unified Component System for Actors and Learners
- ADR-0014: Static Graph Validation Before Execution
- ADR-0015: Semantic Typing for Blackboard Keys
- ADR-0016: Components as Pure Dataflow Units
- ADR-0017: Explicit Declaration of Side Effects
- ADR-0018: Tensor-Native Blackboard Data
- ADR-0019: Semantic Axis Shape Contracts
- ADR-0020: Opt-In Tensor Broadcasting
- ADR-0021: Replay Buffer as a Blackboard Boundary
- ADR-0022: Explicit Target Calculation as DAG Components
- ADR-0023: Derived Execution Graphs from Terminal Targets
- ADR-0024: Metrics as Blackboard Outputs
- ADR-0025: Unified Graph IR Across Macro and Micro Layers
- ADR-0026: Treat Blackboard as Executor Backend, Not Core Architecture
- ADR-0027: Use Recursive Composite Nodes for Hierarchical Graphs
- ADR-0028: Separate Transforms from Runtime Nodes
- ADR-0029: Introduce Service Nodes for Stateful Resources
- ADR-0030: Use Multiple Executors Over Same Graph IR
- ADR-0031: Preserve Contract System Across All Layers
- ADR-0032: Replay Buffer as Resource Node, Not Components
- ADR-0033: Use Trigger Policies for Runtime Scheduling
- ADR-0034: Migrate Existing Engine via Adapters, Not Rewrite
- ADR-0035: Prefer Python Composition Before YAML DSL
- ADR-0036: Keep Workspace Local, Avoid Global Shared Blackboard
- ADR-0037: Use Functions for Transforms by Default
- ADR-0038: Keep User-Facing API Graph-Centric, Hide Execution Memory
- ADR-0039: Promote Existing DAG Compiler Into Global Graph Compiler
- ADR-0040: Deterministic Instance-Bound Contracts
- ADR-0041: Explicit Write Intent via WriteModes
- ADR-0042: Layered Validation (Declarative vs Programmatic)
- ADR-0043: Polymorphic Semantic Type Matching
- ADR-0044: Explicit Return-Based Dataflow
- ADR-0045: Universal Pipeline Executors for All Worker Roles
- ADR-0046: Consolidation of Mechanical State (PER & Exploration)
- ADR-0047: Modular Action and Chance Encoders
- ADR-0048: Standardized Hyperparameter Scheduling
- ADR-0049: Federated Agent Orchestration
- ADR-0050: Global and Temporal Blackboard Storage
- ADR-0051: Advanced Semantic Type System and Shape Inference
- ADR-0052: Capability and Effect Systems for Algorithmic Integrity
- ADR-0053: Graph Compilation and Optimized Execution
- ADR-0054: Multi-Agent and Recurrent Contract Extensions
- ADR-0055: Unified Key Standardization Across Blackboard Domains
