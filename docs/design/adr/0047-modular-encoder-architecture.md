# ADR-0047: Modular Action and Chance Encoders

## Status
Proposed

## Context
Neural network architectures (Backbones, World Models, Heads) often hardcode how actions or chance variables are encoded (e.g., one-hot vs embedding). This limits flexibility for algorithms like EfficientZero, LightZero, or continuous-action MuZero.

## Options Considered

### Option 1: Monolithic Heads/Dynamics
- **Pros**: Simpler implementation; fewer classes.
- **Cons**: Difficult to experiment with different encoding strategies (Spatial vs Non-spatial).

### Option 2: Modular Encoders (Chosen)
- **Pros**: Break up action and chance encoders into standalone, pluggable classes.
- **Cons**: Increases the number of components to manage.

## Decision
All action and chance encoding logic will be encapsulated in modular "Encoder" classes:

1. **Pluggable Encoders**: Action encoders (Spatial, Identity, Continuous, EfficientZero, LightZero) and Chance encoders should be independent components.
2. **Dynamics Ownership**: Dynamics models should "own" action fusion via these encoders, taking in raw actions and applying the configured encoder.
3. **Naming Scheme**: Adopt a robust naming scheme for action representations (e.g., `OneHotActionEncoder`, `ContinuousActionEncoder`).

## Consequences

### Positive
- **Flexibility**: Easily swap between discrete, continuous, and multi-discrete action representations.
- **Testability**: Action encoders can be unit-tested in isolation (e.g., verifying spatial encoders for board games like Tic-Tac-Toe).

### Negative / Tradeoffs
- **Overhead**: Minor increase in configuration complexity to specify which encoder to use.
