# ADR-0035: Prefer Python Composition Before YAML DSL

## Status
Proposed

## Context
Designing the configuration layer for a complex system is a significant undertaking. In many modern frameworks, this is handled via a Domain Specific Language (DSL) defined in YAML or JSON (e.g., Hydra, Ray Rllib Configs). 

While declarative configurations are powerful, committing to a specific DSL before the underlying architectural semantics (Transforms, Nodes, Graphs, Contracts) have stabilized can lead to premature abstraction. If the DSL lacks the expressive power of the underlying code, researchers become frustrated; if the DSL is too complex, it becomes another "language" that must be maintained and documented.

## Options Considered

### Option 1: YAML-First Configuration
- **Pros**
    - High portability; configurations can be shared easily across different runs.
    - Better separation of code and hyperparameters.
- **Cons**
    - **Premature Schema Lock-in**: Changing the internal graph structure requires updating thousands of lines of config files and maintaining complex migration scripts.
    - **Debugging Friction**: Errors in YAML are harder to trace back to the relevant Python execution node.

### Option 2: Python Object Composition First (Chosen)
- **Pros**
    - **Velocity**: Researchers can use standard Python inheritance, loops, and logic to compose their execution graphs.
    - **Tooling**: IDE features like auto-complete, static analysis, and refactoring tools work perfectly on the "configuration" code.
    - **Faster Learning Loop**: The architecture can be evolved rapidly without the overhead of maintaining a separate configuration parser.
- **Cons**
    - **Less Portability**: Configurations are not trivially serializable without a dedicated storage layer (e.g., OmegaConf or custom pickling).

## Decision
We propose adopting this approach because the system will **prefer explicit Python object composition** for defining execution plans and configurations until core semantics have fully stabilized.

Instead of writing:
```yaml
# AVOID PREMATURE DSL
learner:
  type: PPO
  nodes: [...]
```

Developers will write:
```python
# PREFERRED DURING EVOLUTION
plan = Graph(
    nodes=[
        LocalNode(transform=PPOLoss(...)),
        LocalNode(transform=ValueLoss(...)),
    ]
)
```

A declarative DSL (like YAML) may be introduced later as a layer on top of these stable Python objects, but it will not be the primary interface during the architectural expansion phase.

## Consequences

### Positive
- **Architectural Flexibility**: The system can be refactored deeply without being constrained by a static config schema.
- **Developer Flow**: Researchers familiar with RL theory but not the framework's specifics can use standard Python idioms to express their ideas.
- **Reduced Technical Debt**: No need to maintain complex custom YAML loaders or "magic" string-to-class mapping logic in the early stages.

### Negative / Tradeoffs
- **Delayed Declarative Features**: Features that rely on pure text configs (like remote submission of untrusted configs or some third-party hyperparameter sweeps) will be more difficult to implement in the short term.

## Notes
A declarative DSL can always be added once the "alphabet" of components and the "grammar" of the graph are well-understood.