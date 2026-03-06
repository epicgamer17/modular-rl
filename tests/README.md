# Tests

The test suite is organized by architecture area.

## Layout

- `tests/agents/` - action selectors, policies, tester behavior
- `tests/configs/` - config parsing and strict validation checks
- `tests/executors/` - local/MP executor routing, IPC, throughput metrics
- `tests/losses/` - vectorized loss pipeline behavior
- `tests/modules/` - network blocks, backbones, world-model module tests
- `tests/puffer/` - Puffer actor and batched integration checks
- `tests/replay_buffers/` - sampler, segment tree, n-step, and concurrency tests
- `tests/search/` - MCTS core, parity, scoring, and visit invariants
- `tests/stats/` - stat tracker and latent visualization tests
- `tests/trainers/` - trainer smoke and integration tests

## Naming Convention

All collected test files use:

- `test_<component>_<behavior>.py`

Examples:

- `test_search_muzero_mcts_core.py`
- `test_replay_buffer_segment_tree.py`
- `test_trainer_muzero_end_to_end_smoke.py`

## Running Tests

Run everything:

```bash
pytest tests/
```

Run a focused area:

```bash
pytest tests/search -v
pytest tests/replay_buffers -v
pytest tests/trainers -v
```

Run a single file:

```bash
pytest tests/stats/test_stats_latent_visualizations.py -v
```
