# ADR-0054: Multi-Agent and Recurrent Contract Extensions

## Status
Proposed

## Context
Standard RL contracts often assume a single agent and non-recurrent state. Multi-agent systems (PettingZoo) and recurrent architectures (LSTM/Transformer) introduce new dimensions and state-management requirements that are not currently formalized in the contract system.

## Options Considered

### Option 1: Ad-hoc State Management
- **Pros**: Quick to implement for specific algos.
- **Cons**: Brittle; prone to bugs related to state-resetting across episode boundaries or multi-agent indexing.

### Option 2: Formalized Semantic Contracts (Chosen)
- **Pros**: Ensures that recurrent state and multi-agent indexing are validated by the compiler.
- **Cons**: Requires extending the semantic type and axis logic.

## Decision
We propose formalizing support for multi-agent and recurrent architectures within the contract system.

1. **Multi-Agent Contracts**: Support the "Player" dimension natively in tensors. Components should explicitly handle broadcasting across players or indexing specific agents via `PlayerID` contracts.
2. **Recurrent Support**: Formalize "Memory Specs" for LSTM, GRU, and Transformer components. Contracts will explicitly track hidden and cell states, ensuring they are correctly unrolled across time steps and reset at episode boundaries.
3. **Trajectory Semantics**: Introduce `Trajectory[T]` as a semantic type to explicitly model the temporal unroll requirement for recurrent updates.

## Consequences

### Positive
- **Correctness**: Automated verification that multi-agent indexing and recurrent unrolling are consistent.
- **Standardization**: Provides a common language for building multi-player or memory-based agents.

### Negative / Tradeoffs
- **Complexity**: Tensors with (Batch, Player, Time, Dim) axes require more careful broadcasting logic.
