# ADR-0040: Deterministic Instance-Bound Contracts

## Status
Accepted

## Context
In a dynamic DAG-based execution model, the system must perform graph validation (dependency resolution, pruning, etc.) before execution begins. For this validation to be reliable, a component's interface—what it requires and what it provides—must be stable.

Issues with class-level or dynamic contracts:
- **Class-level attributes**: Prevent components from being configured differently (e.g., two instances of a network component reading from different blackboard paths).
- **Runtime branching**: If a component's contract changes based on intermediate results, the static validation pass cannot guarantee correctness for all execution steps.

## Options Considered

### Option 1: Class-level Contract Constants
- **Pros**: Low overhead, easy to read.
- **Cons**: Impossible to parameterize component instances (e.g., changing input keys via config).

### Option 2: Dynamic Runtime Contracts
- **Pros**: Maximum flexibility; components can request data only when they need it.
- **Cons**: Breaks static DAG validation; lead to "shifting sand" bugs where a pipeline works for 10 steps and then fails because a component suddenly requires a new key.

### Option 3: Deterministic Instance-Bound Contracts (Chosen)
- **Pros**: Enables both parameterization and static validation.
- **Cons**: Requires mapping logic in `__init__`.

## Decision
We will enforce that all components must declare their contracts as **instance properties** (`@property`) that are **deterministic after initialization**.

Rules:
1. **Instance Binding**: Contracts (`requires` and `provides`) must be `@property` methods on the instance, allowing them to return values based on `self.config` or `__init__` arguments.
2. **Post-Init Immutability**: The values returned by these properties must not change after the component is initialized.
3. **No Execution Branching**: Property methods should not contain logical branching based on blackboard state or runtime flags. All contract logic should be resolved during `__init__` and stored in private attributes (e.g., `self._requires`).

## Consequences

### Positive
- **Static Validation**: The `BlackboardEngine` can build and verify the entire execution graph once at startup.
- **Parameterization**: Multiple instances of the same component class can exist in a single graph with different input/output mappings.
- **Clarity**: The component's interface is clearly defined and discoverable by inspection tools without running the execution loop.

### Negative / Tradeoffs
- **Initialization Boilerplate**: Developers must compute the final keys in `__init__` rather than simply defining them at the class level.

## Notes
Derived from Principle 3 and 5 of the Component Constraints.
