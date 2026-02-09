# Replay Buffers

Experience replay implementations supporting various sampling strategies, prioritization schemes, and distributed training setups.

## Structure

```
replay_buffers/
├── modular_buffer.py       # Main modular replay buffer
├── buffer_factories.py     # Factory methods for buffer creation
├── processors.py           # Data processing pipelines
├── samplers.py             # Sampling strategies
├── segment_tree.py         # Efficient priority sampling
├── writers.py              # Data writing utilities
├── game.py                 # Game trajectory storage
├── concurrency.py          # Thread-safe operations
├── utils.py                # Buffer utilities
└── deprecated/             # Old buffer implementations
```

## Modular Buffer

The `ModularReplayBuffer` provides a flexible interface for different replay strategies:

```python
from replay_buffers.modular_buffer import ModularReplayBuffer
from replay_buffers.samplers import PrioritizedSampler
from replay_buffers.processors import NStepProcessor

buffer = ModularReplayBuffer(
    capacity=100000,
    sampler=PrioritizedSampler(alpha=0.6, beta=0.4),
    processor=NStepProcessor(n=3, gamma=0.99)
)

# Add transition
buffer.add(obs, action, reward, next_obs, done)

# Sample batch
batch = buffer.sample(batch_size=32)

# Update priorities (for PER)
buffer.update_priorities(indices, priorities)
```

## Sampling Strategies

### Uniform Sampler
Standard uniform sampling from buffer.

### Prioritized Sampler (PER)
Sample based on TD-error magnitude:

```python
from replay_buffers.samplers import PrioritizedSampler

sampler = PrioritizedSampler(
    alpha=0.6,  # Priority exponent
    beta=0.4    # Importance sampling exponent
)
```

### N-Step Returns
Accumulate rewards over multiple steps:

```python
from replay_buffers.processors import NStepProcessor

processor = NStepProcessor(n=3, gamma=0.99)
```

## Buffer Types by Algorithm

| Algorithm | Buffer Type | Features |
|-----------|-------------|----------|
| DQN | Standard | Uniform sampling |
| Rainbow | Prioritized | PER + N-step |
| Ape-X | Distributed | Multiple actors, prioritized |
| MuZero | Reanalyse | Policy reanalysis, sequences |
| NFSP | Reservoir | Reservoir sampling for SL buffer |

## Ape-X Buffer

Distributed prioritized experience replay:

```python
from replay_buffers import ApeXBuffer

buffer = ApeXBuffer(
    capacity=1000000,
    alpha=0.6,
    num_actors=4
)

# Actors add experiences
buffer.add_remote(actor_id, transition)

# Learner samples
batch = buffer.sample(512)
```

## MuZero Buffer

Stores sequences for unrolling and supports reanalysis:

```python
from replay_buffers.sequence import GameBuffer

buffer = GameBuffer(
    capacity=1000000,
    unroll_steps=5,
    td_steps=10
)

# Add complete episode
buffer.add_game(states, actions, rewards, policies)

# Sample sequences with reanalysis
batch = buffer.sample_with_reanalysis(batch_size, model)
```

## Thread Safety

`concurrency.py` provides thread-safe operations:

```python
from replay_buffers.concurrency import ThreadSafeBuffer

safe_buffer = ThreadSafeBuffer(buffer)
```

## Performance

- Segment trees provide O(log n) priority sampling
- Efficient memory layout for batch sampling
- Support for GPU prefetching
