# ADR-0020: Opt-In Tensor Broadcasting

## Status
Accepted

## Context
Standard tensor libraries (PyTorch, NumPy) allow for "silent broadcasting," where smaller tensors are automatically expanded to match the shape of larger tensors during operations. While convenient for general-purpose math, this is extremely dangerous in Reinforcement Learning pipelines.

For example, if a `GAEProcessor` produces advantages of shape `[B, T]` but a downstream `ValueLoss` incorrectly receives a baseline of shape `[B, 1]`, simple subtraction `advantages - baseline` will succeed via broadcasting, silently applying a single value across all time-steps. If the `baseline` was actually intended to be per-time-step, this introduces a subtle bias that is nearly impossible to find via standard debugging.

## Options Considered
### Option 1: Standard Library Broadcasting (Status Quo)
- **Pros**
    - Familiar and flexible.
    - No extra configuration needed.
- **Cons**
    - Primary source of "silent logic bugs" in RL.
    - Mismatched semantic axes (ADR-0020) are not caught.

### Option 2: Opt-In Broadcasting Validation (Chosen)
- **Pros**
    - **Safety**: Errors are raised whenever shapes don't match exactly, unless the developer explicitly stated their intent.
    - **Clarity**: Contracts must define which dimensions are allowed to broadcast (e.g., `[B, (T=1)]`).
    - **Robustness**: Protects against unexpected environment or batch size changes.
- **Cons**
    - requires more verbose contract definitions.
    - may require explicit `unsqueeze` or `expand` calls in components.

## Decision
We will implement the validation engine (ADR-0015) will **block silent broadcasting by default**.

Any attempt to pass a tensor with a mismatched shape (even if mathematically broadcastable) will trigger a contract violation unless the **Semantic Axis Contract (ADR-0020)** explicitly marks the dimension as broadcastable.

Implicitly, this means:
- `[B, T]` is NOT compatible with `[B, 1]` or `[B]`.
- To allow broadcasting, the contract must use an explicit flag or syntax: e.g., `advantages: [B, T], baseline: [B, T_broadcastable]`.

## Consequences
### Positive
- **Bug Prevention**: Catches dimensionality mismatches that previously would have only been found via weeks of hyperparameter tuning and "RL voodoo."
- **Forced Intent**: Developers are forced to think about the semantic alignment (B, T, A) of their tensors at the point of implementation.
- **Simplified Debugging**: If the code runs, you are guaranteed that the tensors "fit" together exactly as described in the architecture.

### Negative / Tradeoffs
- **Boilerplate**: Some components may need more explicit reshaping logic or metadata declarations.

## Notes
This decision is the final defensive layer in the contract-driven architecture, ensuring that the **Semantic Axes (ADR-0020)** aren't just labels, but strictly enforced runtime (and compile-time) constraints.