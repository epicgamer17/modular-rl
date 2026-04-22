# ADR-0015: Semantic Typing for Blackboard Keys

## Status
Accepted

## Context
In complex Reinforcement Learning systems, data passed through the system often lacks clear identity. Using raw strings as keys in a shared container (like a Blackboard) leads to several issues:
- **Ambiguity**: Does `"obs"` mean the current observation, the next observation, or a sequence?
- **Naming Collisions**: Different components might use the same key name for different purposes.
- **Contract Fragility**: Typos in strings (e.g., `"obserations"`) cause silent failures or late-arriving `KeyErrors`.
- **Vanishing Semantics**: A `torch.Tensor` keyed as `"state"` doesn't tell the system if it's a hidden state, a world model state, or a raw environment state.

## Options Considered
### Option 1: Raw String Keys (Standard Practice)
- **Pros**
    - Low boilerplate.
    - Familiar to users of most RL libraries.
- **Cons**
    - No static verification.
    - High risk of naming collisions in large graphs.

### Option 2: Typed Blackboard Keys / Semantic Types
- **Pros**
    - **No Collisions**: Keys are unique by their typed definition, not just their name.
    - **Clearer Contracts**: Components specify exactly what semantic data they need (e.g., `ValueEstimate[Scalar]`).
    - **Reusable Validation**: Validation logic (shape checks, range checks) can be attached to the semantic type once.
    - **Auto-Documentation**: The system can automatically generate data-flow diagrams with rich metadata.
- **Cons**
    - Higher initial boilerplate (defining types).
    - Requires developer discipline to maintain the type system.

## Decision
We will implement blackboard keys will be strictly defined as typed objects or associated with **Semantic Types**. 

Instead of:
```python
component.read("obs")
```

The system will use:
```python
component.read(Key("trajectory.observations", ObservationTensor))
```
(or a similar structured type system).

Every key in the Blackboard must have a declared semantic type that defines its mathematical meaning and expected structure.

## Consequences
### Positive
- **Contract Enforcement**: The `ExecutionEngine` can verify that a provider of `PolicyLogits` is correctly connected to a consumer of `PolicyLogits` by checking both the key name and the semantic type.
- **Visualizability**: Tooling can visualize the "data types" flowing between components, making the architecture easier to reason about.
- **Refactoring Safety**: Changing a key's definition is caught by the compiler/type-checker rather than failing at runtime.

### Negative / Tradeoffs
- **Development Overhead**: Adding a new piece of data to the system requires defining a `Key` or `SemanticType`.
- **Verbosity**: Configuration files and code will be more verbose as they must specify types.

## Notes
This ADR is a prerequisite for **ADR-0015 (Static Graph Validation Before Execution)**, as the validation pass relies on these semantic contracts to function.