# ADR-0037: Use Functions for Transforms by Default

## Status
Proposed

## Context
As defined in **ADR-0029 (Separate Transforms from Runtime Nodes)**, the system distinguishes between mathematical logic and execution boundaries. In practice, the vast majority of mathematical logic units (e.g., computing a value loss, GAE, or observation normalization) are stateless operations that take tensors and return tensors.

Forcing every one of these simple operations into a class-based hierarchy (e.g., inheriting from a `BaseTransform`) introduces significant boilerplate code and unnecessary object-oriented complexity.

## Options Considered

### Option 1: Class-Based Transforms (Uniform OOP)
- **Pros**
    - Uniformity; every transform in the system follows the same object structure.
    - Metadata (contracts, names) is easily attached as class attributes.
- **Cons**
    - **High Boilerplate**: Small operations (like a 3-line math formula) require a full class definition and an `__init__` method.
    - **Verbosity**: Reading a file full of classes is slower than reading a set of clean mathematical functions.

### Option 2: Functions by Default (Chosen)
- **Pros**
    - **Lightweight**: Writing a transform is as simple as defining a Python function.
    - **Familiarity**: Most RL researchers think and write in terms of mathematical functions.
    - **Readability**: Logic is front-and-center, not buried in class structures.
- **Cons**
    - Attaching the mandatory **Contracts (ADR-0032)** to functions requires a convention (e.g., decorators or standard docstring formats).

## Decision
We propose adopting this approach because the system will prefer **Plain Python Functions** as the default representation for stateless **Transforms**.

1. **Standard Transforms**: Implemented as pure functions that accept tensors and return tensors.
2. **Stateful/Configurable Transforms**: Implemented as classes only when they require initialization parameters (e.g., a neural network with weights) or maintain internal state (rare, see ADR-0017).
3. **Metadata Attachment**: Mandatory metadata—such as `keys_in`, `keys_out`, and `semantic_types`—will be attached to functions via a standardized decorator (e.g., `@transform(inputs=..., outputs=...)`).

## Consequences

### Positive
- **Simpler Authoring**: extremely low barrier to entry for adding new mathematical logic to the system.
- **Better Readability**: Specialized files (like `losses.py`) look like a collection of mathematical operators rather than a complex object hierarchy.
- **Easier Unit Testing**: Pure functions are the easiest units of code to test in isolation.

### Negative / Tradeoffs
- **Convention Reliance**: Developers must follow the specific decorator convention to ensure the **Graph Engine (ADR-0024)** can correctly identify and wire the functions.

## Notes
Transform functions are intended to be analogous to functional ML APIs (like `torch.nn.functional`), emphasizing "what the math does" rather than "how the object is configured."