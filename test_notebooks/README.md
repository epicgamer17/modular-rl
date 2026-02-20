# Test Notebooks

Interactive Jupyter notebooks for testing and debugging agents.

## Installation

Test notebooks require Jupyter:

```bash
pip install jupyter
pip install -e .
```

## Available Notebooks

| Notebook | Purpose | Status |
|----------|---------|--------|
| `muzero.ipynb` | MuZero debugging and testing | ✅ |
| `rainbow.ipynb` | Rainbow DQN experiments | ✅ |
| `ppo.ipynb` | PPO agent testing | ✅ |
| `nfsp.ipynb` | NFSP multi-agent testing | ✅ |

## Usage

Start Jupyter and open the desired notebook:

```bash
jupyter notebook test_notebooks/
```

## Notebook Structure

Each notebook typically includes:

1. **Setup** - Import libraries, create configs
2. **Environment Testing** - Verify environment behavior
3. **Agent Initialization** - Create and inspect agent
4. **Training** - Run short training sessions
5. **Evaluation** - Test trained agent
6. **Debugging** - Diagnose issues

## Checkpoints

Pre-trained checkpoints are stored in `test_notebooks/checkpoints/` for quick testing without retraining.

## Tips

- Use small `num_simulations` for faster MuZero testing
- Reduce `buffer_size` for quicker Rainbow DQN experiments
- Enable verbose logging to see agent decisions
- Use the `%debug` magic for post-mortem debugging
