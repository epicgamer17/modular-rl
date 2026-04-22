# ADR-0055: Unified Key Standardization Across Blackboard Domains

## Status
Proposed

## Context
Currently, blackboard keys for identical semantic concepts vary across different domains (e.g., `data.obs` in actors vs `data.observations` in learners). This inconsistency leads to mapping errors, boilerplate in adapters, and difficulty in reasoning about data flow.

## Options Considered

### Option 1: Per-Component Mapping (Current)
- **Pros**: Maximum flexibility for individual components.
- **Cons**: High boilerplate; inconsistent logs and telemetry.

### Option 2: Global Key Standardization (Chosen)
- **Pros**: Eliminates ambiguity; simplifies debugging and telemetry.
- **Cons**: Requires a large-scale refactor of existing components.

## Decision
We propose enforcing a strict, unified naming convention for all blackboard keys.

1. **Standard Keys**: Adopt canonical names for core concepts (e.g., `observations`, `actions`, `rewards`, `terminated`, `truncated`).
2. **Plurality**: Prefer plural names for collections to distinguish them from scalar metadata.
3. **Domain Prefixes**: Standardize on `data.` for raw facts, `targets.` for computed training goals, `predictions.` for network outputs, and `meta.` for telemetry.
4. **Consistency**: The same key (e.g., `data.observations`) MUST be used in actors, learners, and tracers to ensure seamless data flow through the replay boundary.

## Consequences

### Positive
- **Simplicity**: No more "was it `obs` or `observations`?" questions.
- **Interoperability**: Components can be swapped across actor and learner pipelines with fewer adapter mappings.
- **Cleanliness**: Consistent logs and visualization scripts.

### Negative / Tradeoffs
- **Breaking Change**: Existing components and experiments must be updated to the new naming scheme.
