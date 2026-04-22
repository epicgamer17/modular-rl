# ADR-0008: Removal of Lazy Decompression from Replay Path

## Status
Proposed

## Context
Lazy decompression introduces:
- Python-level per-sample overhead
- Caching complexity
- Breaks tensor-only guarantees

## Options Considered
### Option 1: Lazy decompression wrapper
- Pros
    - Memory efficient
- Cons
    - Breaks vectorized sampling
    - Not multiprocessing-friendly

### Option 2: Eager decoding into tensor buffers
- Pros
    - Fully vectorizable
    - Compatible with shared memory
- Cons
    - Higher upfront memory cost

## Decision
We propose adopting this approach because remove lazy decompression. Store either:
- raw tensors, or
- pre-decoded compressed blocks resolved at write time

## Consequences
### Positive
- Faster sampling
- Simplified replay logic

### Negative / Tradeoffs
- Higher memory footprint at storage time

## Notes
Compression is moved outside replay-critical path.