# Blackboard & Pipeline System Features

This document provides a comprehensive list of capabilities and features within the `core/`, `data/`, and `search/` modules, as well as architectural constraints defined in the system.

---

## 1. Core Framework (`core/`)
The core framework provides the infrastructure for building RL learners as a directed acyclic graph (DAG) of components.

### **Blackboard Engine & Orchestration**
*   **Centralized State (`Blackboard`)**: A unified data exchange categorized into `data`, `predictions`, `targets`, `losses`, and `meta`.
*   **Frozen Views**: Enforces immutability during component execution by passing read-only snapshots to prevent side-effect bugs.
*   **DAG-Based Execution**: Automatically determines the correct execution order of components based on their data dependencies.
*   **Execution Graph Pruning**: Identifies and skips "dead" components whose outputs are not consumed by any terminal sink (losses or telemetry).
*   **Lazy Execution**: Dynamically skips components at runtime if their required outputs already exist on the blackboard (e.g., cached results).
*   **Build-time Validation**: Detects dependency cycles, missing inputs, and semantic mismatches before the first training step.
*   **Performance Profiling**: Per-component timing and telemetry (ms) injected into the blackboard metadata.
*   **Dotted Path Resolution**: Utilities for accessing and writing to nested dictionary structures via string paths (e.g., `data.obs.image`).

### **Contract & Type System**
*   **Semantic Typing (`SemanticType`)**: Data is categorized by meaning (e.g., `Observation`, `Action`, `ValueTarget`) rather than just shapes.
*   **Comprehensive Semantic Library**: Built-in support for `Trajectory`, `HiddenState`, `ToPlay`, `Done`, `Advantage`, `LogProb`, `Return`, `Mask`, `Priority`, `GradientScale`, `Weight`, `Epsilon`, and `Metric`.
*   **Polymorphic Compatibility**: Type-safe data flow where specialized types (e.g., `DiscreteValue`) can satisfy base type requirements (`ValueEstimate`) via `is_compatible()`.
*   **Structural Parameterization**: Semantic types can be parameterized with data structures like `Logits`, `Probs`, `Scalar`, or `LogProbs`.
*   **Shape Contracts (`ShapeContract`)**: Declarative schema for tensor rank, axis labeling (B, T, F, A, C, H, W), dtypes, and time-dimension consistency.
*   **Write Modes**: Explicit signaling of mutation intent:
    *   `NEW`: Creates a new entry (default).
    *   `OVERWRITE`: Modifies existing data.
    *   `APPEND`: Adds to a collection.
    *   `OPTIONAL`: Conditionally produces data.

### **Validation Layer**
*   **Layered Validation**: Combines lightweight declarative constraints for documentation/visualization with deep programmatic enforcement.
*   **Centralized Assertions**: Shared helpers for batch alignment, rank sanity, distributional vs. scalar compatibility, and bin count matching.
*   **Metadata Validation**: Enforces that all providers and consumers agree on semantic metadata (e.g., matching `bins`, `vmin`, `vmax`, or `num_classes`), preventing subtle algorithmic bugs.
*   **Strategy Delegation**: Components can delegate complex math validation to specific representation strategies (e.g., validating logits range).

---

## 2. Data & Replay Systems (`data/`)
The data module handles experience storage, transformation, and sampling.

### **Replay Buffers & Storage**
*   **Flexible Writers**:
    *   `CircularWriter`: Standard FIFO circular buffer logic.
    *   `SharedCircularWriter`: PyTorch Shared Memory support for multi-process data collection (MuZero style).
    *   `ReservoirWriter`: Reservoir sampling for unbiased data retention over long horizons.
    *   `PPOWriter`: Strict sequential writing for on-policy algorithms with overflow protection.
*   **Contiguous Storage**: Replay storage backed by pre-allocated tensors for maximum performance.
*   **Concurrency Backends**: Abstracted lock and tensor creation for `Local` (Ray/Single-thread) or `TorchMP` (Shared Memory) workflows.

### **Sampling & Processing**
*   **Prioritized Sampling (PER)**: Efficient prioritized experience replay using Sum-Tree and Min-Tree structures.
*   **Sequence Sampling**: Support for sampling continuous trajectories with padding and masking for RNN/Transformer training.
*   **N-Step Returns**: Built-in processors for computing n-step bootstrapped targets and discounting.
*   **Replay Compression**: Transparent bit-packing and float16/int8 compression for reducing memory footprint of large observation buffers.
*   **Input/Output Normalization**: Standardized processors for augmenting observations and post-processing network predictions.

---

## 3. Search & MCTS Backends (`search/`)
The search module provides modular backends for Monte Carlo Tree Search and related algorithms.

### **Multi-Backend Architecture**
*   **Python Search (`PySearch`)**: Highly flexible, modular implementation for rapid prototyping and debugging.
*   **C++ Search (`CPPSearch`)**: High-performance, multi-threaded implementation optimized for latency-critical environments.
*   **AOS Search (`AOSSearch`)**: Highly vectorized "Array-of-Structures" implementation using batched tensor operations for massive parallelism.

### **Search Components**
*   **Modular MCTS**: Decoupled modules for:
    *   **Selection**: Greedy, Sample-based, or PUCT (Polynomial UCB).
    *   **Scoring**: PUCT formulas, V-value interpolation, and value normalization (Min-Max Stats).
    *   **Backpropagation**: Support for standard mean, MuZero-style discounting, and custom value-mixing.
*   **Root Injection**: Specialized logic for Dirichlet noise and temperature-based exploration.
*   **Search Pruning**: Algorithmic pruning strategies (e.g., Alpha-Beta style) to focus search on high-value branches.
*   **Dynamic Masking**: Enforces legal move constraints during tree expansion and selection.
*   **Batched Tree Search**: Capable of running thousands of simultaneous searches across batches of states.

---

## 4. Architectural Constraints & Roadmap
*   **Deterministic Contracts**: Component requirements and provisions must be computed once at initialization and remain immutable.
*   **Pure Transform Enforcement**: Components are strictly forbidden from in-place mutations of the blackboard; all changes must be returned as explicit updates.
*   **Device Awareness**: The engine handles moving batches to the target device (`cuda`, `cpu`, `mps`) before execution, keeping component logic hardware-agnostic.
*   **Graph-Build Validation**: All contract consistency checks (shapes, types, metadata) occur at build-time to fail fast on invalid configurations.
*   **Future Optimizations**:
    *   **Component Fusion**: Combining small mathematical transforms into single kernels.
    *   **Adaptive Execution**: Skipping branches of the DAG based on dynamic valves (e.g., `stop_execution`).
    *   **Transparent Dataflow**: Per-step tracing and bottleneck analysis of every mutation.
