# ADR-0041: Explicit Write Intent via WriteModes

## Status
Accepted

## Context
When components write to a shared Blackboard, the framework needs to understand the *intent* of the write to perform safety checks and optimizations. Simply declaring that a component "provides" a key is insufficient for:
- Detecting accidental overwrites (two components writing to the same key).
- Managing collections (appending to a list of metrics).
- Handling optional outputs that may not be produced every step.

## Options Considered

### Option 1: Implicit Write Semantics
- **Pros**: Simplest to implement; first writer wins or last writer wins.
- **Cons**: High risk of silent bugs where one component unintentionally clobbers data produced by another.

### Option 2: Explicit Write Modes (Chosen)
- **Pros**: Allows the engine to validate write safety and manage data lifecycles.
- **Cons**: Requires components to return a dictionary of `Key -> WriteMode` rather than just a set of keys.

## Decision
Components must communicate their intent for writing data using explicit **Write Modes**. The `provides()` method must return a mapping of `Key` to a `WriteMode` string.

Standard Modes:
- **`"new"`**: (Default) The component expects to be the primary creator of this data. The engine will error if this key already exists on the blackboard.
- **`"overwrite"`**: The component explicitly intends to modify an existing value (e.g., a gradient scaler or a normalization transform).
- **`"append"`**: The component adds data to a collection (list/dict). Useful for accumulating metrics or trajectories.
- **`"optional"`**: The key may or may not be produced depending on internal logic (e.g., periodic logging).

## Consequences

### Positive
- **Collision Detection**: The `BlackboardEngine` can detect invalid "double writes" during graph construction.
- **Traceability**: Harder to lose data when multiple components interact with the same semantic path.
- **Optimization**: The engine can use these modes to manage memory more effectively (e.g., knowing when a buffer can be safely cleared).

### Negative / Tradeoffs
- **API Complexity**: The `provides` return type changes from `Set[Key]` to `Dict[Key, str]`.

## Notes
Refines the "Write Modes" section of the Component Constraints.
