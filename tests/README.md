# Tests

Comprehensive test suite covering unit tests, integration tests, smoke tests, and verification scripts.

## Test Categories

### Unit Tests

Test individual components in isolation:

| Test File | Coverage |
|-----------|----------|
| `test_muzero_mcts_core.py` | MCTS search algorithms |
| `test_muzero_nstep.py` | N-step return calculation |
| `test_muzero_backprop.py` | Value backpropagation |
| `test_batched_search.py` | Batch MCTS operations |
| `test_buffer_concurrency.py` | Thread-safe buffers |
| `test_stats.py` | Statistics tracking |
| `test_torch_cat.py` | PyTorch utilities |
| `test_pickling.py` | Serialization |

### Smoke Tests

Basic functionality checks:

- `test_muzero_smoke.py` - MuZero basic training loop
- `agents/` - Agent-specific smoke tests
- `replay_buffers/` - Buffer functionality
- `losses/` - Loss computation

### Integration Tests

End-to-end system tests:

- `ray/` - Distributed training with Ray
- `benchmarks/` - Performance benchmarks
- `performance/` - Speed and memory tests

### Verification Scripts

Manual verification and debugging tools:

- `verify_learner_manual.py` - Verify MuZero learner
- `verify_refactor_manual.py` - Verify refactoring correctness
- `verify_reward_signs.py` - Check reward alignment
- `verify_mp.py` - Test multiprocessing
- `debug_muzero_alignment.py` - Debug alignment issues

### Latent Space Tests

Visualization tests for learned representations:

- `test_latent_pca.py` - PCA projection
- `test_latent_tsne.py` - t-SNE visualization
- `test_latent_umap.py` - UMAP projection

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test
```bash
pytest tests/test_muzero_smoke.py -v
```

### Run with Coverage
```bash
pytest tests/ --cov=agents --cov=modules --cov=search
```

### Run Smoke Tests Only
```bash
pytest tests/test_muzero_smoke.py -v --tb=short
```

## Test Structure

```python
# Example test structure
def test_component_functionality():
    # Arrange
    component = create_component()
    input_data = load_test_data()
    
    # Act
    result = component.process(input_data)
    
    # Assert
    assert result.shape == expected_shape
    assert torch.allclose(result, expected_output)
```

## Test Data

Test fixtures and data stored in:
- `tests/checkpoints/` - Model checkpoints for tests
- `tests/videos/` - Test recordings
- `tests/tmp/` - Temporary test files

## Continuous Integration

Tests run automatically on:
- Every commit
- Pull requests
- Nightly builds

## Adding Tests

When adding new features:

1. Add unit tests in `tests/test_<feature>.py`
2. Add integration tests if it interacts with other components
3. Update smoke tests if it's a core algorithm
4. Run full test suite before committing

## Debugging

Use verification scripts for debugging:

```bash
python tests/verify_reward_signs.py
python tests/debug_muzero_alignment.py
```
