# ADR-0043: Polymorphic Semantic Type Matching

## Status
Accepted

## Context
RL pipelines often process data that can be represented in multiple ways (e.g., a "Value Target" could be a simple scalar or a categorical distribution). If components are too specific in their requirements, composition becomes difficult. If they are too generic, type safety is lost.

## Options Considered

### Option 1: Exact Type Matching
- **Pros**: Simple to implement.
- **Cons**: Brittle; a component expecting `ValueTarget` would reject `ValueTarget[Logits]`.

### Option 2: Duck Typing
- **Pros**: Flexible.
- **Cons**: Runtime errors; passing a `Policy` where a `Value` is expected might not fail until deep into a loss calculation.

### Option 3: Polymorphic Matching via Inheritance (Chosen)
- **Pros**: Matches programmer intuition (Python classes); allows specialized types to satisfy generic requirements.
- **Cons**: Requires a well-designed class hierarchy for `SemanticType`.

## Decision
The Contract System will use **Standard Python Inheritance** and `issubclass()` to resolve semantic compatibility.

Rules:
1. **Hierarchical Types**: `SemanticType` classes are organized in a hierarchy (e.g., `ValueTarget[Logits]` inherits from `ValueTarget`).
2. **One-Way Compatibility**: A provider of a *specific* type (e.g., `DiscreteValue`) satisfies a consumer's requirement for a *generic* type (e.g., `ValueEstimate`), but not vice versa.
3. **The `is_compatible()` Method**: While `issubclass()` is the default, `SemanticType` objects may implement a custom `is_compatible(other_type)` method to handle complex matching logic (e.g., matching metadata like bin counts).

## Consequences

### Positive
- **Flexibility**: Generic components (like a generic Logger) can accept any `SemanticType`, while specific losses (like `CrossEntropyLoss`) can require specific representations.
- **Safety**: Prevents semantically invalid data flow (e.g., using `Policy` data as a `Value` target) even if the tensor shapes happen to match.
- **Extensibility**: New representations can be added by simply subclassing the appropriate base `SemanticType`.

### Negative / Tradeoffs
- **Maintenance**: Requires keeping the `SemanticType` hierarchy clean and meaningful.

## Notes
Extends the "Semantic Type System" section of the Component Constraints.
