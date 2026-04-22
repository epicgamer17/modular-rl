# ADR-0018: Tensor-Native Blackboard Data

## Status
Accepted

## Context
The Blackboard is the central nervous system of the RL pipeline. To maintain high throughput and architectural clarity, the data flowing through it must be easy for machine learning frameworks (like PyTorch) to manipulate. 

Using arbitrary Python objects (e.g., custom `Transition` classes, deep-nested dictionaries, or mutable objects) inside the Blackboard causes several issues:
- **Serialization Overhead**: Pickling/unpickling complex objects for multiprocessing is extremely slow.
- **Device Incompatibility**: Moving a list of Python objects to the GPU is not possible; only tensors can be transferred efficiently.
- **Opaque Logic**: The validation engine (ADR-0015) cannot verify the "shape" or "type" of a custom Python object, leading to late-arriving runtime errors.
- **Memory Leaks**: Storing objects with hidden references (like `Transition` objects holding whole episodes) can cause massive memory leaks.

## Options Considered
### Option 1: Arbitrary Python Objects (Flexibility)
- **Pros**
    - Easy for quick prototyping.
    - No need to convert environment data into tensors immediately.
- **Cons**
    - Kill performance in distributed settings.
    - Makes graph validation impossible.

### Option 2: Tensor-Native Data (Chosen)
- **Pros**
    - **Efficiency**: Tensors are the "native language" of the GPU and are highly optimized for zero-copy sharing in multiprocessing.
    - **Validation**: Tensors have explicit shapes and dtypes that can be verified statically (ADR-0015, ADR-0016).
    - **Interoperability**: Standardized data format makes components truly interchangeable.
- **Cons**
    - Requires explicit conversion logic at the environment boundaries (wrappers).

## Decision
We will implement all values stored in the Blackboard must be **Tensor-Native**.

Permissible data types:
1. **`torch.Tensor`**: The primary data carrier.
2. **Numeric Scalars**: Simple `int`, `float`, or `bool` values.
3. **Static Config Objects**: Immutable, serializable configuration objects (e.g., `NamedTuple` or `dataclass` of primitives).

Strictly forbidden in the Blackboard:
1. **Custom Mutable Classes**: Any object that maintains its own internal state or references.
2. **"Transition" or "Sequence" Objects**: High-level abstractions that bundle multiple pieces of data (these must be flattened into individual tensor keys).
3. **Deep Deeply-Nested Dictionaries**: Use the Blackboard's flat key structure (or shallow namespaces) instead.

## Consequences
### Positive
- **Performance**: Near-zero overhead for moving the entire Blackboard state across process boundaries or to the GPU.
- **Safety**: Enabling ADR-0015's shape-checking pass, catching model mismatches at startup.
- **Consistency**: Extends the philosophy of **ADR-0005 (Tensor-Only Replay Storage)** to the entire execution pipeline.

### Negative / Tradeoffs
- **Initial Setup**: Writing environment wrappers to convert raw NumPy/Python data into clean tensors is mandatory.

## Notes
This decision reinforces the "Raw Math" philosophy of the framework, ensuring that neither the Actor nor the Learner needs to understand complex Python inheritance hierarchies to process data.