# ADR-0019: Semantic Axis Shape Contracts

## Status
Accepted

## Context
In Reinforcement Learning, tensors frequently undergo complex transformations that change their semantic structure. A common source of silent, catastrophic bugs is the accidental broadcasting of tensors with mismatched axes. 

For example, adding a tensor of shape `(32, 1)` (intended as a per-batch scalar) to a tensor of shape `(32, 10)` (intended as a batch of action logits) might "work" mathematically but indicates a logical error if the scalar was supposed to be a per-action bias. Raw shapes like `(32, 128)` provide no information about whether `128` represents a flattened observation, a hidden state, or a sequence length.

## Options Considered
### Option 1: Raw Dimensionality Checks (Standard)
- **Pros**
    - Easy to implement using standard `assert tensor.shape == ...`.
- **Cons**
    - Fails to distinguish between different semantic meanings for the same dimensionality.
    - broadcasting bugs remain hidden.

### Option 2: Semantic Axis Contracts (Chosen)
- **Pros**
    - **Ambiguity Removal**: Shape `(B, T)` is clearly a batch of time-steps, while `(B, A)` is a batch of action-specific values.
    - **Early Validation**: The compilation pass (ADR-0015) can verify that a component expecting `[B, T]` isn't receiving `[B, A]`.
    - **Self-Documenting**: Semantic types become descriptive (e.g., `Advantages: [B, T]`, `PolicyLogits: [B, T, A]`).
- **Cons**
    - Requires a formalized axis naming convention.
    - Metadata overhead in component definitions.

## Decision
We will implement all tensor contracts in the system must explicitly declare their **Semantic Axes**.

A contract must define:
1. **NDims**: The expected number of dimensions.
2. **Semantic Axes**: An ordered list of axis labels (e.g., `B` for Batch, `T` for Time, `A` for Action, `C` for Channels, `H` for Height, `W` for Width).
3. **Event Shape**: The "core" shape of the data item (e.g., for an image, the event shape is `[C, H, W]`, while for a scalar value, it is `[]`).
4. **Broadcast Policy**: How the tensor is expected to expand or reduce across batch/time dimensions.

Example Semantic Types:
- `AdvantageTensor`: `[B, T]`
- `ObservationTensor (Image)`: `[B, T, C, H, W]`
- `ActionDistLogits`: `[B, T, A]`

## Consequences
### Positive
- **Broadcasting Guardrails**: The system can automatically detect and raise errors for operations that would result in unintended semantic broadcasting.
- **Improved Readability**: Code using these contracts is significantly easier to understand as the intent of every tensor dimension is explicit.
- **Axis-Aware Tooling**: Enables automatic reshaping or permuting utilities that work based on axis names (similar to `einops` but integrated into the contract system).

### Negative / Tradeoffs
- **Complexity**: Developers must learn and adhere to the shared axis naming convention.
- **Rigidity**: Highly dynamic or non-standard tensor structures may require new axis labels or complex contract definitions.

## Notes
This decision builds upon **ADR-0016 (Semantic Typing for Blackboard Keys)** and **ADR-0011 (Explicit Temporal Alignment Contract)**, providing the "inner structure" for those semantic types.